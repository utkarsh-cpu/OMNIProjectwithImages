"""Evaluation metrics, confusion matrix, calibration curve, and scatter plots.

All metrics operate in log10-space and physical space.  Storm-event metrics
use a configurable threshold (``peak_flux > 1e-6`` *or* ``Dst < -50 nT``).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from .config import Config
from .model import SolarStormModel, combined_loss
from .utils import CSVLogger, flare_class_from_log, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-epoch validation (called from train.py)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> Dict[str, float]:
    """Run the validation set and return a dict of scalar metrics.

    Returned keys: ``loss``, ``MAE_log``, ``RMSE_log``, ``MAE_flare``,
    ``RMSE_flare``, ``PICP_90``, ``MPIW_90``.
    """
    model.eval()
    all_pred_flux: List[float] = []
    all_true_flux: List[float] = []
    all_pred_std: List[float] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch_dev = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(
                images=batch_dev["images"],
                omni=batch_dev["omni"],
                image_mask=batch_dev["image_mask"],
            )
            loss = combined_loss(
                outputs,
                target_log_flux=batch_dev["target_log_flux"],
                target_log_dst=batch_dev["target_log_dst"],
            )
        total_loss += loss.item()
        n_batches += 1

        pred = outputs["flux_pred"].cpu().numpy()
        log_std = outputs["flux_log_std"].cpu().numpy()
        true = batch["target_log_flux"].numpy()

        all_pred_flux.extend(pred.tolist())
        all_true_flux.extend(true.tolist())
        all_pred_std.extend(np.exp(log_std).tolist())

    pred_arr = np.array(all_pred_flux)
    true_arr = np.array(all_true_flux)
    std_arr = np.array(all_pred_std)

    metrics = compute_metrics(pred_arr, true_arr, std_arr, cfg)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


# ---------------------------------------------------------------------------
# Full evaluation (standalone)
# ---------------------------------------------------------------------------


def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Run full evaluation, save plots and metrics CSV."""
    if output_dir is None:
        output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_pred: List[float] = []
    all_true: List[float] = []
    all_std: List[float] = []
    all_ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(
                    images=batch_dev["images"],
                    omni=batch_dev["omni"],
                    image_mask=batch_dev["image_mask"],
                )
            all_pred.extend(outputs["flux_pred"].cpu().tolist())
            all_true.extend(batch["target_log_flux"].tolist())
            all_std.extend(np.exp(outputs["flux_log_std"].cpu().numpy()).tolist())
            all_ids.extend(batch["sample_id"])

    pred_arr = np.array(all_pred)
    true_arr = np.array(all_true)
    std_arr = np.array(all_std)

    metrics = compute_metrics(pred_arr, true_arr, std_arr, cfg)

    # Save metrics CSV
    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_logger = CSVLogger(csv_path, list(metrics.keys()))
    csv_logger.log({k: f"{v:.6f}" for k, v in metrics.items()})
    logger.info("Metrics saved to %s", csv_path)

    # Plots
    _plot_scatter(pred_arr, true_arr, output_dir)
    _plot_confusion_matrix(pred_arr, true_arr, cfg, output_dir)
    _plot_calibration(pred_arr, true_arr, std_arr, output_dir)

    for k, v in sorted(metrics.items()):
        logger.info("  %s = %.4f", k, v)

    return metrics


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    pred_log: np.ndarray,
    true_log: np.ndarray,
    pred_std: np.ndarray,
    cfg: Config,
) -> Dict[str, float]:
    """Compute all specified metrics and return as a dict."""
    # ── Regression (log-space) ─────────────────────────────────────────
    residual_log = pred_log - true_log
    mae_log = float(np.mean(np.abs(residual_log)))
    rmse_log = float(np.sqrt(np.mean(residual_log ** 2)))

    # ── Regression (physical space) ───────────────────────────────────
    pred_phys = 10.0 ** pred_log
    true_phys = 10.0 ** true_log
    mae_physical = float(np.mean(np.abs(pred_phys - true_phys)))
    rmse_physical = float(np.sqrt(np.mean((pred_phys - true_phys) ** 2)))

    # ── Storm-event metrics (flare-only) ──────────────────────────────
    flare_mask = true_phys > cfg.flare_threshold
    if flare_mask.sum() > 0:
        mae_flare = float(np.mean(np.abs(residual_log[flare_mask])))
        rmse_flare = float(np.sqrt(np.mean(residual_log[flare_mask] ** 2)))
    else:
        mae_flare = mae_log
        rmse_flare = rmse_log

    # ── Binary classification (storm / no-storm) ─────────────────────
    log_threshold = np.log10(cfg.flare_threshold)
    y_true_bin = (true_log >= log_threshold).astype(int)
    y_pred_bin = (pred_log >= log_threshold).astype(int)

    tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
    fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
    fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())
    tn = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    tss = tpr - fpr
    total = tp + fp + fn + tn
    expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / max(total, 1)
    hss = (tp + tn - expected) / max(total - expected, 1) if total > expected else 0.0

    # ── Uncertainty calibration ───────────────────────────────────────
    z = 1.6449  # 90% CI z-score
    lower = pred_log - z * pred_std
    upper = pred_log + z * pred_std
    inside = (true_log >= lower) & (true_log <= upper)
    picp_90 = float(inside.mean()) if len(inside) > 0 else 0.0
    mpiw_90 = float((upper - lower).mean()) if len(upper) > 0 else 0.0

    return {
        "MAE_log": mae_log,
        "RMSE_log": rmse_log,
        "MAE_physical": mae_physical,
        "RMSE_physical": rmse_physical,
        "MAE_flare": mae_flare,
        "RMSE_flare": rmse_flare,
        "TSS": tss,
        "HSS": hss,
        "Hit_Rate": tpr,
        "False_Alarm_Rate": fpr,
        "PICP_90": picp_90,
        "MPIW_90": mpiw_90,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_scatter(
    pred_log: np.ndarray,
    true_log: np.ndarray,
    output_dir: str,
) -> None:
    """Save predicted-vs-actual scatter plot in log space."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true_log, pred_log, alpha=0.3, s=10, edgecolors="none")
    lims = [
        min(true_log.min(), pred_log.min()) - 0.5,
        max(true_log.max(), pred_log.max()) + 0.5,
    ]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True log10(peak_flux)")
    ax.set_ylabel("Predicted log10(peak_flux)")
    ax.set_title("Predicted vs Actual (log scale)")
    ax.set_aspect("equal")
    fig.tight_layout()
    path = os.path.join(output_dir, "scatter_log.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Scatter plot saved to %s", path)


def _plot_confusion_matrix(
    pred_log: np.ndarray,
    true_log: np.ndarray,
    cfg: Config,
    output_dir: str,
) -> None:
    """Save binary confusion matrix (storm / no-storm)."""
    log_th = np.log10(cfg.flare_threshold)
    y_true = (true_log >= log_th).astype(int)
    y_pred = (pred_log >= log_th).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Storm", "Storm"])
    ax.set_yticklabels(["No Storm", "Storm"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", path)


def _plot_calibration(
    pred_log: np.ndarray,
    true_log: np.ndarray,
    pred_std: np.ndarray,
    output_dir: str,
) -> None:
    """Save uncertainty calibration curve (expected vs observed coverage)."""
    expected = np.linspace(0.05, 0.95, 19)
    observed = []
    for p in expected:
        z = _norm_ppf((1 + p) / 2)
        lower = pred_log - z * pred_std
        upper = pred_log + z * pred_std
        coverage = ((true_log >= lower) & (true_log <= upper)).mean()
        observed.append(float(coverage))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Ideal")
    ax.plot(expected, observed, "b-o", markersize=4, label="Model")
    ax.set_xlabel("Expected Coverage")
    ax.set_ylabel("Observed Coverage")
    ax.set_title("Uncertainty Calibration")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    fig.tight_layout()
    path = os.path.join(output_dir, "calibration_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Calibration plot saved to %s", path)


def _norm_ppf(p: float) -> float:
    """Approximate inverse CDF of standard normal (Beasley-Springer-Moro)."""
    from scipy.stats import norm
    return float(norm.ppf(p))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import DataLoader as DL

    from .config import Config
    from .dataset import SolarDataset, get_eval_transform, load_omni2
    from .model import SolarStormModel
    from .train import load_checkpoint
    from .utils import RobustScaler, get_device, load_channel_stats

    cfg = Config()
    device = get_device()

    omni_df = load_omni2(cfg)
    omni_scaler = RobustScaler().load(
        os.path.join(cfg.output_dir, cfg.omni_scaler_file)
    )
    channel_stats = load_channel_stats(
        os.path.join(cfg.output_dir, cfg.channel_stats_file)
    )

    test_ds = SolarDataset(
        cfg, split="test",
        omni_df=omni_df,
        channel_stats=channel_stats,
        omni_scaler=omni_scaler,
        augmentation=get_eval_transform(),
    )
    test_loader = DL(test_ds, batch_size=cfg.batch_size, shuffle=False,
                     num_workers=cfg.num_workers)

    model = SolarStormModel(cfg).to(device)
    ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
    model, _ = load_checkpoint(model, ckpt_path, device)

    full_evaluation(model, test_loader, device, cfg)
