"""Paired SDOBenchmark + OMNI2 dataset with physics-safe augmentation.

Actual workspace layout::

    /home/OMNIProjectwithImages/
      omni2_full_dataset.csv           # single CSV, named columns + datetime
      SDOBenchmark-data-full.zip       # or extracted directory below
      SDOBenchmark-data-full/
        training/
          meta_data.csv                # columns: id, start, end, peak_flux
          <AR>/<sample_id>/            # e.g. 11390/2012_01_01_19_06_00_0/
            <timestamp>__131.jpg
            <timestamp>__magnetogram.jpg
            ...
        test/
          meta_data.csv
          <AR>/<sample_id>/...
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import zipfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T

from .config import Config
from .utils import (
    RobustScaler,
    get_logger,
    handle_omni_gaps,
    load_channel_stats,
    save_channel_stats,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SDO Benchmark extractor
# ---------------------------------------------------------------------------


def ensure_sdo_extracted(cfg: Config) -> str:
    """Extract the SDO ZIP if the extracted directory does not exist yet.

    Returns the path to the extracted root (e.g.
    ``…/SDOBenchmark-data-full``).
    """
    if os.path.isdir(cfg.sdo_extracted_dir):
        logger.info("SDO data already extracted at %s", cfg.sdo_extracted_dir)
        return cfg.sdo_extracted_dir

    if not os.path.isfile(cfg.sdo_zip_path):
        raise FileNotFoundError(
            f"Neither extracted dir {cfg.sdo_extracted_dir!r} nor "
            f"ZIP {cfg.sdo_zip_path!r} found."
        )

    logger.info("Extracting %s → %s  (this may take a while …)",
                cfg.sdo_zip_path, cfg.workspace_root)
    with zipfile.ZipFile(cfg.sdo_zip_path, "r") as zf:
        zf.extractall(cfg.workspace_root)
    logger.info("Extraction complete.")
    return cfg.sdo_extracted_dir


# ---------------------------------------------------------------------------
# OMNI2 loader  (single named-column CSV)
# ---------------------------------------------------------------------------


def load_omni2(cfg: Config) -> pd.DataFrame:
    """Load the OMNI2 CSV into a DataFrame indexed by datetime.

    Only the 8 selected features are kept; documented fill-values are
    replaced with ``NaN``.
    """
    csv_path = cfg.omni_csv_path
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"OMNI2 CSV not found: {csv_path}")

    # Read only the columns we need + datetime
    csv_cols_needed = list(cfg.omni_col_map.values()) + ["datetime"]
    raw = pd.read_csv(csv_path, usecols=csv_cols_needed, parse_dates=["datetime"])
    raw = raw.set_index("datetime").sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]

    # Replace fill values with NaN
    for csv_col, fill_val in cfg.omni_fill_values.items():
        if csv_col in raw.columns:
            raw.loc[raw[csv_col] >= fill_val * 0.999, csv_col] = np.nan

    # Rename CSV columns → internal short names
    rename_map = {csv_col: short for short, csv_col in cfg.omni_col_map.items()}
    raw = raw.rename(columns=rename_map)
    # Keep only the 8 internal columns in canonical order
    raw = raw[cfg.omni_col_names]

    logger.info("Loaded OMNI2: %d rows  %s → %s",
                len(raw), raw.index.min(), raw.index.max())
    return raw


# ---------------------------------------------------------------------------
# Per-channel image normalization stats
# ---------------------------------------------------------------------------


def precompute_stats(
    cfg: Config,
    max_images_per_channel: int = 200,
) -> Dict[str, Dict[str, float]]:
    """Compute per-channel (mean, std) from training images and cache to JSON.

    We sample up to *max_images_per_channel* images to keep compute
    feasible.  Stats are saved to ``<output_dir>/<channel_stats_file>``.
    """
    sdo_root = ensure_sdo_extracted(cfg)
    train_root = os.path.join(sdo_root, "training")
    meta = pd.read_csv(os.path.join(train_root, "meta_data.csv"))

    stats: Dict[str, Dict[str, float]] = {}

    for ch_name in cfg.selected_channels:
        arrays: List[np.ndarray] = []
        count = 0
        for _, row in meta.iterrows():
            if count >= max_images_per_channel:
                break
            sample_dir = _sample_dir(train_root, str(row["id"]))
            if sample_dir is None:
                continue
            matches = sorted(glob.glob(os.path.join(sample_dir,
                                                     f"*__{ch_name}.jpg")))
            if matches:
                img = np.array(Image.open(matches[0]).convert("L"),
                               dtype=np.float32)
                arrays.append(img.ravel())
                count += 1

        if arrays:
            all_px = np.concatenate(arrays)
            stats[ch_name] = {
                "mean": float(np.mean(all_px)),
                "std": max(float(np.std(all_px)), 1.0),
            }
        else:
            stats[ch_name] = {"mean": 0.0, "std": 1.0}

    out_path = os.path.join(cfg.output_dir, cfg.channel_stats_file)
    save_channel_stats(stats, out_path)
    logger.info("Saved channel stats to %s", out_path)
    return stats


# ---------------------------------------------------------------------------
# Physics-safe augmentation
# ---------------------------------------------------------------------------


def get_train_augmentation(cfg: Config) -> T.Compose:
    """Return torchvision transform for training (physics-safe).

    Allowed: vertical flip, mild rotation ±5°, brightness/contrast jitter.
    NOT allowed: horizontal flip, random crop.
    """
    return T.Compose([
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=cfg.aug_rotation_deg, fill=0),
        T.ColorJitter(
            brightness=cfg.aug_brightness,
            contrast=cfg.aug_contrast,
        ),
    ])


def get_eval_transform() -> T.Compose:
    """No-op transform for validation / test."""
    return T.Compose([])


# ---------------------------------------------------------------------------
# Helper — locate a sample directory
# ---------------------------------------------------------------------------

def _sample_dir(split_root: str, sample_id: str) -> Optional[str]:
    """Resolve the on-disk path for a sample ID.

    The SDOBenchmark stores images under ``<split>/<AR>/<sample_id>/``.
    Metadata IDs are prefixed with the Active Region number, e.g.
    ``11390_2012_01_05_17_06_01_0``, but the on-disk sample directory is
    ``<split>/11390/2012_01_05_17_06_01_0`` without that prefix.
    """
    ar = sample_id.split("_")[0]
    sample_leaf = sample_id[len(ar) + 1:] if sample_id.startswith(f"{ar}_") else sample_id
    path = os.path.join(split_root, ar, sample_leaf)
    if os.path.isdir(path):
        return path
    return None


def _discover_timesteps(sample_dir: str) -> List[str]:
    """Discover the unique timestamps from JPEG filenames within a sample folder.

    Filenames look like ``2012-01-01T070600__131.jpg``.  We extract the
    timestamp prefix (everything before ``__``), deduplicate, sort, and
    return up to 4 timestamps.
    """
    ts_set: set[str] = set()
    for fname in os.listdir(sample_dir):
        if fname.endswith(".jpg") and "__" in fname:
            ts_part = fname.rsplit("__", 1)[0]
            ts_set.add(ts_part)
    timestamps = sorted(ts_set)
    return timestamps[:4]  # take at most 4


def _normalise_meta_name(name: str) -> str:
    """Return a case-insensitive, punctuation-free metadata column key."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _extract_sample_longitude(
    row: pd.Series,
    normalised_columns: Dict[str, str],
) -> Optional[float]:
    """Extract a central-meridian-relative longitude when metadata provides one."""
    lon_candidates = (
        "lon",
        "longitude",
        "hglon",
        "hglongitude",
        "heliolon",
        "heliographiclon",
        "heliographiclongitude",
        "stonyhurstlon",
        "stonyhurstlongitude",
        "hgslon",
    )

    raw_lon: Optional[float] = None
    for candidate in lon_candidates:
        col = normalised_columns.get(candidate)
        if col is None:
            continue
        try:
            raw_lon = float(row[col])
        except (TypeError, ValueError):
            return None
        break

    if raw_lon is None or not np.isfinite(raw_lon):
        return None

    # Normalise longitudes such as [0, 360) into [-180, 180) around central meridian.
    return ((raw_lon + 180.0) % 360.0) - 180.0


def _parse_sdo_ts(ts_str: str) -> Optional[datetime]:
    """Parse an SDO filename timestamp like ``2012-01-01T070600`` into datetime."""
    for fmt in ("%Y-%m-%dT%H%M%S", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# SolarDataset
# ---------------------------------------------------------------------------


class SolarDataset(Dataset):
    """Paired SDOBenchmark image + OMNI2 time-series dataset.

    Each ``__getitem__`` returns a dict with keys:

    - ``images``:  ``Tensor(4, 5, 256, 256)``
    - ``image_mask``:  ``Tensor(4, 5)``  — ``True`` where image exists
    - ``omni``:  ``Tensor(look_back_x, 8)``
    - ``target_log_flux``:  ``float``  — ``log10(peak_flux)``
    - ``target_log_dst``:  ``Tensor(forecast_y)``  — future ``log10(|Dst|)`` curve
    - ``sample_id``:  ``str``
    - ``timestamp_t4``:  ``str``
    """

    def __init__(
        self,
        cfg: Config,
        split: str = "train",
        omni_df: Optional[pd.DataFrame] = None,
        channel_stats: Optional[Dict[str, Dict[str, float]]] = None,
        omni_scaler: Optional[RobustScaler] = None,
        augmentation: Optional[T.Compose] = None,
    ) -> None:
        super().__init__()
        assert split in ("train", "test"), f"Unknown split: {split}"
        self.cfg = cfg
        self.split = split
        self.augmentation = augmentation

        # ── Ensure SDO data is extracted ──────────────────────────────
        sdo_root = ensure_sdo_extracted(cfg)
        # SDO uses "training" for train, "test" for test
        split_folder = "training" if split == "train" else "test"
        self.split_root = os.path.join(sdo_root, split_folder)

        # ── Meta labels — meta_data.csv inside each split folder ──────
        meta_path = os.path.join(self.split_root, "meta_data.csv")
        self.meta = pd.read_csv(meta_path)
        logger.info("Loaded %d samples from %s", len(self.meta), meta_path)

        # ── OMNI2 ─────────────────────────────────────────────────────
        self.omni = omni_df if omni_df is not None else load_omni2(cfg)

        # ── Scalers / stats ────────────────────────────────────────────
        self.channel_stats = channel_stats
        self.omni_scaler = omni_scaler

        # ── Pre-build sample index: resolve dirs + timestamps + OMNI2 ─
        self._valid_samples = self._build_valid_samples()
        logger.info("Valid samples after filtering: %d / %d",
                     len(self._valid_samples), len(self.meta))

    # ------------------------------------------------------------------
    def _build_valid_samples(self) -> List[Dict[str, Any]]:
        """Return list of dicts with pre-resolved paths and timestamps.

        A sample is valid when:
        1. Its on-disk directory exists.
        2. It has at least 1 image timestamp.
        3. Its OMNI2 window (ending at the last timestamp) has no gap > max_gap.
        """
        valid: List[Dict[str, Any]] = []
        normalised_columns = {
            _normalise_meta_name(col): str(col) for col in self.meta.columns
        }
        has_longitude_metadata = any(
            key in normalised_columns
            for key in (
                "lon",
                "longitude",
                "hglon",
                "hglongitude",
                "heliolon",
                "heliographiclon",
                "heliographiclongitude",
                "stonyhurstlon",
                "stonyhurstlongitude",
                "hgslon",
            )
        )
        for idx in range(len(self.meta)):
            row = self.meta.iloc[idx]
            sid = str(row["id"])
            peak_flux = float(row["peak_flux"])
            sdir = _sample_dir(self.split_root, sid)
            if sdir is None:
                continue

            if has_longitude_metadata:
                longitude = _extract_sample_longitude(row, normalised_columns)
                if longitude is None or abs(longitude) > 40.0:
                    continue

            timestamps = _discover_timesteps(sdir)
            if len(timestamps) < self.cfg.image_timesteps:
                continue

            # Use the last available timestamp as t4 for OMNI2 alignment
            t4_str = timestamps[-1]
            t4_dt = _parse_sdo_ts(t4_str)
            if t4_dt is None:
                continue

            # Check OMNI2 window validity
            window = self._get_omni_window(t4_dt)
            if window is None:
                continue

            valid.append({
                "meta_idx": idx,
                "sample_id": sid,
                "sample_dir": sdir,
                "timestamps": timestamps,
                "t4_str": t4_str,
                "t4_dt": t4_dt,
                "peak_flux": peak_flux,
            })
        return valid

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._valid_samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        info = self._valid_samples[index]
        sid = info["sample_id"]
        sdir = info["sample_dir"]
        peak_flux = info["peak_flux"]
        timestamps = info["timestamps"]
        t4_dt = info["t4_dt"]
        log_flux = float(np.log10(max(peak_flux, 1e-12)))

        # ── Images (4 time-steps × 5 channels) ────────────────────────
        images = torch.zeros(
            self.cfg.image_timesteps, self.cfg.image_channels,
            self.cfg.image_size, self.cfg.image_size,
            dtype=torch.float32,
        )
        mask = torch.zeros(
            self.cfg.image_timesteps, self.cfg.image_channels,
            dtype=torch.bool,
        )

        if len(timestamps) < self.cfg.image_timesteps:
            raise ValueError(
                f"Sample {sid} has only {len(timestamps)} unique timestamps; "
                f"expected at least {self.cfg.image_timesteps}."
            )

        for t_idx, ts_str in enumerate(timestamps[: self.cfg.image_timesteps]):
            for ch_idx, ch_name in enumerate(self.cfg.selected_channels):
                img_path = os.path.join(sdir, f"{ts_str}__{ch_name}.jpg")
                if os.path.isfile(img_path):
                    images[t_idx, ch_idx] = self._load_image(img_path, ch_name)
                    mask[t_idx, ch_idx] = True

        # ── Augmentation (applied identically across all timesteps) ────
        if self.augmentation is not None:
            b = images.shape[0] * images.shape[1]
            flat = images.view(b, self.cfg.image_size, self.cfg.image_size)
            seed_aug = torch.randint(0, 2**31, (1,)).item()
            augmented = []
            for i in range(flat.shape[0]):
                torch.manual_seed(seed_aug)
                s = self.augmentation(flat[i].unsqueeze(0))
                augmented.append(s)
            images = torch.cat(augmented, dim=0).view(images.shape)

        # ── OMNI2 window ──────────────────────────────────────────────
        omni_tensor, target_dst = self._prepare_omni(t4_dt)

        return {
            "images": images,
            "image_mask": mask,
            "omni": omni_tensor,
            "target_log_flux": log_flux,
            "target_log_dst": target_dst,
            "sample_id": sid,
            "timestamp_t4": info["t4_str"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, path: str, ch_name: str) -> torch.Tensor:
        """Load a single grayscale JPEG and apply per-channel z-score normalisation."""
        img = Image.open(path).convert("L")
        img = img.resize(
            (self.cfg.image_size, self.cfg.image_size), Image.BILINEAR
        )
        arr = np.array(img, dtype=np.float32)

        if self.channel_stats and ch_name in self.channel_stats:
            mean = self.channel_stats[ch_name]["mean"]
            std = self.channel_stats[ch_name]["std"]
            arr = (arr - mean) / std
        else:
            arr = arr / 255.0

        return torch.from_numpy(arr)  # (H, W)

    def _get_omni_window(
        self, t4: Optional[datetime]
    ) -> Optional[np.ndarray]:
        """Slice OMNI2 for look_back_x hours ending at t4.

        Returns ``None`` if any feature has a contiguous NaN run > max_gap.
        """
        if t4 is None:
            return None
        start = t4 - timedelta(hours=self.cfg.look_back_x)
        idx = pd.date_range(start=start, periods=self.cfg.look_back_x, freq="h")
        try:
            window = self.omni.reindex(idx).values.astype(np.float64)
        except Exception:
            return None
        _, should_drop = handle_omni_gaps(window, self.cfg.omni_max_gap_hours)
        if should_drop:
            return None
        return window

    def _prepare_omni(
        self, t4: datetime
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(omni_tensor, target_log_dst)``."""
        n_feat = len(self.cfg.omni_col_names)
        omni_tensor = torch.zeros(self.cfg.look_back_x, n_feat,
                                  dtype=torch.float32)
        target_dst = torch.zeros(self.cfg.forecast_y, dtype=torch.float32)

        # --- input window ---
        start = t4 - timedelta(hours=self.cfg.look_back_x)
        idx = pd.date_range(start=start, periods=self.cfg.look_back_x, freq="h")
        window = self.omni.reindex(idx).values.astype(np.float64)
        window, _ = handle_omni_gaps(window, self.cfg.omni_max_gap_hours)

        if self.omni_scaler is not None:
            window = (window - self.omni_scaler.median_) / self.omni_scaler.iqr_

        window = np.nan_to_num(window, nan=0.0).astype(np.float32)
        omni_tensor = torch.from_numpy(window)

        if omni_tensor.shape[0] < self.cfg.look_back_x:
            pad = torch.zeros(self.cfg.look_back_x - omni_tensor.shape[0],
                              n_feat)
            omni_tensor = torch.cat([pad, omni_tensor], dim=0)

        # --- target Dst curve (future forecast_y hours) ---
        future_idx = pd.date_range(
            start=t4 + timedelta(hours=self.cfg.propagation_delay_hours),
            periods=self.cfg.forecast_y,
            freq="h",
        )
        future_dst = self.omni.reindex(future_idx)["Dst"].values.astype(np.float64)
        future_dst = np.nan_to_num(future_dst, nan=0.0)
        abs_dst = np.maximum(np.abs(future_dst), 1.0)
        target_dst = torch.from_numpy(np.log10(abs_dst).astype(np.float32))
        if target_dst.shape[0] < self.cfg.forecast_y:
            pad = torch.zeros(self.cfg.forecast_y - target_dst.shape[0])
            target_dst = torch.cat([target_dst, pad])
        target_dst = target_dst[: self.cfg.forecast_y]

        return omni_tensor, target_dst


# ---------------------------------------------------------------------------
# Oversampling sampler for class imbalance
# ---------------------------------------------------------------------------


def build_flare_sampler(
    dataset: SolarDataset,
    cfg: Config,
) -> WeightedRandomSampler:
    """Weighted random sampler that oversamples flare events.

    Samples with ``peak_flux > cfg.flare_threshold`` receive
    ``cfg.flare_oversample_factor`` × the weight of quiet samples.
    """
    weights: List[float] = []
    for info in dataset._valid_samples:
        pf = info["peak_flux"]
        w = float(cfg.flare_oversample_factor) if pf > cfg.flare_threshold else 1.0
        weights.append(w)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
