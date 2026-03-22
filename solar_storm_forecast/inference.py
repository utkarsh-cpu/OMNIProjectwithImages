"""Inference API with MC-Dropout uncertainty estimation and Grad-CAM visualisation."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .config import Config
from .model import SolarStormModel
from .train import load_checkpoint
from .utils import (
    RobustScaler,
    flare_class,
    get_device,
    get_logger,
    load_channel_stats,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------


class GradCAM:
    """Gradient-weighted Class Activation Mapping on the last conv layer.

    Registers forward/backward hooks on *target_layer* and produces a
    heatmap for a given scalar output.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _mod: nn.Module, _inp: Any, out: Any) -> None:
        if isinstance(out, torch.Tensor):
            self._activations = out.detach()
        elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
            self._activations = out[0].detach()

    def _save_gradient(self, _mod: nn.Module, _grad_in: Any, grad_out: Any) -> None:
        if isinstance(grad_out, tuple) and isinstance(grad_out[0], torch.Tensor):
            self._gradients = grad_out[0].detach()
        elif isinstance(grad_out, torch.Tensor):
            self._gradients = grad_out.detach()

    def __call__(self, output_scalar: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM heatmap for a scalar model output.

        Returns
        -------
        heatmap : np.ndarray
            Shape ``(H, W)`` in ``[0, 1]``.
        """
        self.model.zero_grad()
        output_scalar.backward(retain_graph=True)

        if self._activations is None or self._gradients is None:
            return np.zeros((1, 1), dtype=np.float32)

        # Pool gradients over spatial dims
        weights = self._gradients.mean(dim=(-2, -1), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.ndim == 0:
            cam = cam.reshape(1, 1)
        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def _save_gradcam_overlay(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    path: str,
) -> None:
    """Overlay a Grad-CAM heatmap on a grayscale image and save to disk."""
    from PIL import Image as PILImage

    h, w = original_image.shape[:2]
    heatmap_resized = np.array(
        PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (w, h), PILImage.BILINEAR
        )
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(original_image, cmap="gray")
    ax.imshow(heatmap_resized, cmap="jet", alpha=0.4)
    ax.axis("off")
    fig.tight_layout(pad=0)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MC-Dropout helpers
# ---------------------------------------------------------------------------


def _enable_dropout(model: nn.Module) -> None:
    """Set all Dropout layers to training mode for MC-Dropout inference."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


# ---------------------------------------------------------------------------
# Image loading helper
# ---------------------------------------------------------------------------


def _load_images_from_dir(
    image_dir: str,
    cfg: Config,
    channel_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> tuple[torch.Tensor, torch.Tensor, list[Optional[np.ndarray]]]:
    """Load images from *image_dir* following the SDO folder convention.

    Expected layout::

        image_dir/
            t1/   (or files prefixed with t1 timestamps)
            t2/
            t3/
            t4/

    Returns ``(images, mask, raw_images_for_gradcam)``.
    """
    images = torch.zeros(1, cfg.image_timesteps, cfg.image_channels,
                         cfg.image_size, cfg.image_size, dtype=torch.float32)
    mask = torch.zeros(1, cfg.image_timesteps, cfg.image_channels, dtype=torch.bool)
    raw_imgs: list[Optional[np.ndarray]] = []

    # Try subfolder-per-timestep first, then flat layout
    timestep_dirs = sorted(glob.glob(os.path.join(image_dir, "t[1-4]")))
    if len(timestep_dirs) < cfg.image_timesteps:
        # Flat layout — just try to find 4 sets by globbing channel names
        timestep_dirs = [image_dir] * cfg.image_timesteps

    for t_idx in range(cfg.image_timesteps):
        ts_dir = timestep_dirs[t_idx] if t_idx < len(timestep_dirs) else image_dir
        first_raw = None
        for ch_idx, ch_name in enumerate(cfg.selected_channels):
            pattern = os.path.join(ts_dir, f"*{ch_name}*.jpg")
            matches = sorted(glob.glob(pattern))
            if matches:
                img = Image.open(matches[0]).convert("L")
                img = img.resize((cfg.image_size, cfg.image_size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32)
                if first_raw is None:
                    first_raw = arr.copy()
                # Normalise
                if channel_stats and ch_name in channel_stats:
                    mean = channel_stats[ch_name]["mean"]
                    std = channel_stats[ch_name]["std"]
                    arr = (arr - mean) / std
                else:
                    arr = arr / 255.0
                images[0, t_idx, ch_idx] = torch.from_numpy(arr)
                mask[0, t_idx, ch_idx] = True
        raw_imgs.append(first_raw)

    return images, mask, raw_imgs


# ---------------------------------------------------------------------------
# Main predict() function
# ---------------------------------------------------------------------------


def predict(
    image_dir: str,
    omni_recent: pd.DataFrame,
    model_checkpoint: str,
    n_mc_samples: int = 20,
    save_gradcam: bool = True,
    cfg: Optional[Config] = None,
) -> Dict[str, Any]:
    """Run inference on a single sample and return a rich result dictionary.

    Parameters
    ----------
    image_dir : str
        Path to folder with 4 time-step subfolders of JPEGs.
    omni_recent : pd.DataFrame
        ``look_back_x`` rows, columns = 8 selected OMNI2 features.
    model_checkpoint : str
        Path to a ``.pt`` checkpoint file.
    n_mc_samples : int
        Number of MC-Dropout forward passes for uncertainty.
    save_gradcam : bool
        Whether to save Grad-CAM overlay images to disk.
    cfg : Config, optional
        Configuration; built with defaults if ``None``.

    Returns
    -------
    dict
        Prediction results including mean, uncertainty bounds, flare class,
        storm probability, Dst forecast, and Grad-CAM paths.
    """
    if cfg is None:
        cfg = Config()

    device = get_device()

    # ── Load model ─────────────────────────────────────────────────────
    model = SolarStormModel(cfg).to(device)
    model, _ = load_checkpoint(model, model_checkpoint, device)
    model.eval()

    # ── Load scaler / stats ────────────────────────────────────────────
    stats_path = os.path.join(cfg.output_dir, cfg.channel_stats_file)
    channel_stats = load_channel_stats(stats_path) if os.path.exists(stats_path) else None

    scaler_path = os.path.join(cfg.output_dir, cfg.omni_scaler_file)
    omni_scaler = RobustScaler().load(scaler_path) if os.path.exists(scaler_path) else None

    # ── Prepare image input ────────────────────────────────────────────
    images, mask, raw_imgs = _load_images_from_dir(image_dir, cfg, channel_stats)
    images = images.to(device)
    mask = mask.to(device)

    # ── Prepare OMNI2 input ────────────────────────────────────────────
    omni_arr = omni_recent[cfg.omni_col_names].values.astype(np.float64)
    if omni_scaler is not None:
        omni_arr = (omni_arr - omni_scaler.median_) / omni_scaler.iqr_
    omni_arr = np.nan_to_num(omni_arr, nan=0.0).astype(np.float32)
    # Pad / trim to look_back_x
    if omni_arr.shape[0] < cfg.look_back_x:
        pad_rows = cfg.look_back_x - omni_arr.shape[0]
        omni_arr = np.vstack([np.zeros((pad_rows, omni_arr.shape[1]), dtype=np.float32),
                              omni_arr])
    omni_arr = omni_arr[-cfg.look_back_x:]
    omni_tensor = torch.from_numpy(omni_arr).unsqueeze(0).to(device)

    # ── MC-Dropout passes ─────────────────────────────────────────────
    flux_samples: List[float] = []
    dst_samples: List[np.ndarray] = []

    for i in range(n_mc_samples):
        _enable_dropout(model)
        with torch.no_grad():
            outputs = model(images=images, omni=omni_tensor, image_mask=mask)
        flux_samples.append(float(outputs["flux_pred"].cpu()))
        dst_samples.append(outputs["dst_pred"].cpu().numpy().squeeze())

    flux_arr = np.array(flux_samples)
    dst_arr = np.stack(dst_samples, axis=0)  # (N, forecast_y)

    log_flux_mean = float(flux_arr.mean())
    log_flux_std = float(flux_arr.std())
    flux_mean = 10.0 ** log_flux_mean

    flux_lower = float(10.0 ** np.percentile(flux_arr, 5))
    flux_upper = float(10.0 ** np.percentile(flux_arr, 95))

    dst_mean = dst_arr.mean(axis=0).tolist()
    dst_lower = np.percentile(dst_arr, 5, axis=0).tolist()
    dst_upper = np.percentile(dst_arr, 95, axis=0).tolist()

    flare_prob = float((flux_arr >= np.log10(cfg.flare_threshold)).mean())
    storm_prob = float((dst_arr.min(axis=1) > np.log10(50)).mean())

    fc = flare_class(flux_mean)

    # ── Grad-CAM ──────────────────────────────────────────────────────
    gradcam_paths: List[str] = []
    if save_gradcam:
        # Need gradient-enabled forward pass
        target_layer = model.image_encoder.last_conv
        if hasattr(model, "_orig_mod"):
            target_layer = model._orig_mod.image_encoder.last_conv
        gc = GradCAM(model, target_layer)
        # Enable grad
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        images_gc = images.clone().requires_grad_(True)
        outputs_gc = model(images=images_gc, omni=omni_tensor, image_mask=mask)
        for t_idx in range(cfg.image_timesteps):
            heatmap = gc(outputs_gc["flux_pred"].sum())
            gc_path = os.path.join(
                cfg.output_dir, "gradcam", f"gradcam_t{t_idx + 1}.png"
            )
            raw = raw_imgs[t_idx] if t_idx < len(raw_imgs) and raw_imgs[t_idx] is not None else np.zeros((cfg.image_size, cfg.image_size))
            _save_gradcam_overlay(heatmap, raw, gc_path)
            gradcam_paths.append(gc_path)
        gc.remove_hooks()

    channel_mask_flat = mask.cpu().squeeze().any(dim=0).tolist()

    return {
        "log10_flux_mean": log_flux_mean,
        "log10_flux_std": log_flux_std,
        "flux_mean": flux_mean,
        "flux_lower_90": flux_lower,
        "flux_upper_90": flux_upper,
        "flare_class": fc,
        "flare_probability": flare_prob,
        "dst_forecast_mean": dst_mean,
        "dst_lower_90": dst_lower,
        "dst_upper_90": dst_upper,
        "storm_probability": storm_prob,
        "gradcam_paths": gradcam_paths,
        "channel_mask_used": channel_mask_flat,
    }
