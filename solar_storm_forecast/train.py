"""Training loop with early stopping, LR schedule, and checkpointing."""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from .config import Config
from .dataset import (
    SolarDataset,
    build_flare_sampler,
    get_eval_transform,
    get_train_augmentation,
    load_omni2,
    precompute_stats,
)
from .evaluate import evaluate_epoch
from .model import SolarStormModel, combined_loss
from .utils import (
    CSVLogger,
    RobustScaler,
    get_amp_autocast,
    get_device,
    get_grad_scaler,
    get_logger,
    load_channel_stats,
    seed_everything,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _to_device(batch: Dict, device: torch.device) -> Dict:
    """Move tensor values in a batch dict to *device*."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    amp_autocast = get_amp_autocast(device)
    amp_scaler = get_grad_scaler(device)
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with amp_autocast:
            outputs = model(
                images=batch["images"],
                omni=batch["omni"],
                image_mask=batch["image_mask"],
            )
            loss = combined_loss(
                outputs,
                target_log_flux=batch["target_log_flux"],
                target_log_dst=batch["target_log_dst"],
                cfg=cfg,
            )

        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        amp_scaler.step(optimizer)
        amp_scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train(cfg: Optional[Config] = None) -> str:
    """Full training pipeline. Returns path to best checkpoint.

    Steps:
    1. Seed everything
    2. Load / compute channel stats & OMNI scaler
    3. Build datasets & data loaders
    4. Instantiate model, optimizer, and scheduler
    5. Training loop with early stopping
    6. Save best checkpoint by ``val_mae_flare``
    """
    if cfg is None:
        cfg = Config()

    seed_everything(cfg.seed)
    device = get_device()
    logger.info("Device: %s", device)

    # ── 1. OMNI2 ──────────────────────────────────────────────────────
    omni_df = load_omni2(cfg)

    # Fit RobustScaler on training-period OMNI2 data
    omni_scaler = RobustScaler().fit(omni_df)
    scaler_path = os.path.join(cfg.output_dir, cfg.omni_scaler_file)
    omni_scaler.save(scaler_path)
    logger.info("OMNI2 scaler saved to %s", scaler_path)

    # ── 2. Channel stats ──────────────────────────────────────────────
    stats_path = os.path.join(cfg.output_dir, cfg.channel_stats_file)
    if os.path.exists(stats_path):
        channel_stats = load_channel_stats(stats_path)
        logger.info("Loaded existing channel stats from %s", stats_path)
    else:
        channel_stats = precompute_stats(cfg)

    # ── 3. Datasets & loaders ─────────────────────────────────────────
    train_ds = SolarDataset(
        cfg, split="train",
        omni_df=omni_df,
        channel_stats=channel_stats,
        omni_scaler=omni_scaler,
        augmentation=get_train_augmentation(cfg),
    )
    val_ds = SolarDataset(
        cfg, split="test",
        omni_df=omni_df,
        channel_stats=channel_stats,
        omni_scaler=omni_scaler,
        augmentation=get_eval_transform(),
    )

    train_sampler = build_flare_sampler(train_ds, cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── 4. Model / optimiser / scheduler ──────────────────────────────
    model = SolarStormModel(cfg).to(device)

    # Attempt torch.compile (PyTorch ≥2.0)
    try:
        model = torch.compile(model)
        logger.info("torch.compile applied successfully")
    except Exception as exc:
        logger.warning("torch.compile unavailable (%s); continuing without it", exc)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.lr_t0, T_mult=cfg.lr_t_mult
    )

    # ── 5. Logging ────────────────────────────────────────────────────
    csv_cols = [
        "epoch", "train_loss", "val_loss",
        "val_mae_log", "val_mae_flare", "val_picp_90", "lr",
    ]
    csv_logger = CSVLogger(
        os.path.join(cfg.output_dir, "training_log.csv"), csv_cols
    )

    # ── 6. Training loop ──────────────────────────────────────────────
    best_metric = float("inf")
    epochs_without_improvement = 0
    best_ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, cfg
        )
        scheduler.step()

        # Validate
        val_metrics = evaluate_epoch(model, val_loader, device, cfg)
        val_loss = val_metrics.get("loss", 0.0)
        val_mae_log = val_metrics.get("MAE_log", 0.0)
        val_mae_flare = val_metrics.get("MAE_flare", float("inf"))
        val_picp = val_metrics.get("PICP_90", 0.0)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_MAE_log=%.4f  "
            "val_MAE_flare=%.4f  PICP_90=%.3f  lr=%.2e  (%.1fs)",
            epoch, cfg.epochs, train_loss, val_mae_log,
            val_mae_flare, val_picp, current_lr, elapsed,
        )

        csv_logger.log({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_mae_log": f"{val_mae_log:.6f}",
            "val_mae_flare": f"{val_mae_flare:.6f}",
            "val_picp_90": f"{val_picp:.4f}",
            "lr": f"{current_lr:.2e}",
        })

        # Check improvement
        current_metric = val_mae_flare
        if current_metric < best_metric:
            best_metric = current_metric
            epochs_without_improvement = 0
            _save_checkpoint(model, optimizer, scheduler, epoch, best_metric,
                             best_ckpt_path)
            logger.info("  ✓ New best (val_mae_flare=%.4f) → saved %s",
                        best_metric, best_ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                logger.info("Early stopping after %d epochs without improvement.",
                            cfg.early_stopping_patience)
                break

    logger.info("Training complete. Best val_mae_flare=%.4f", best_metric)
    return best_ckpt_path


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    metric: float,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Handle torch.compile wrapped models
    state_dict = model.state_dict()
    if hasattr(model, "_orig_mod"):
        state_dict = model._orig_mod.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if hasattr(scheduler, "state_dict") else None
            ),
            "metric": metric,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[nn.Module, int]:
    """Load checkpoint into *model*. Returns ``(model, epoch)``."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, ckpt.get("epoch", 0)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
