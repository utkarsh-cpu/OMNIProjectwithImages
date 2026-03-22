"""Shared utilities: seeding, scaling, logging, flare classification, gap handling."""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create a console (+optional file) logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Flare class mapper
# ---------------------------------------------------------------------------

def flare_class(flux: float) -> str:
    """Map peak X-ray flux (W/m²) to GOES flare class string.

    - flux < 1e-6  → ``"quiet/C"``
    - 1e-6 ≤ flux < 1e-5  → ``"M"``
    - flux ≥ 1e-5  → ``"X"``
    """
    if flux < 1e-6:
        return "quiet/C"
    elif flux < 1e-5:
        return "M"
    else:
        return "X"


def flare_class_from_log(log_flux: float) -> str:
    """Same as :func:`flare_class` but accepts log10(peak_flux)."""
    return flare_class(10.0 ** log_flux)


# ---------------------------------------------------------------------------
# RobustScaler (median / IQR)
# ---------------------------------------------------------------------------

class RobustScaler:
    """Robust scaling using median and inter-quartile range.

    Fitted on training data only, serialised to / from JSON for re-use at
    inference time.
    """

    def __init__(self) -> None:
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "RobustScaler":
        """Compute median and IQR per column (NaN-safe)."""
        self.feature_names_ = list(df.columns)
        vals = df.values.astype(np.float64)
        self.median_ = np.nanmedian(vals, axis=0)
        q25 = np.nanpercentile(vals, 25, axis=0)
        q75 = np.nanpercentile(vals, 75, axis=0)
        self.iqr_ = q75 - q25
        # Prevent division by zero
        self.iqr_[self.iqr_ < 1e-12] = 1.0
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply scaling. Returns numpy array."""
        assert self.median_ is not None, "Scaler not fitted yet."
        vals = df.values.astype(np.float64)
        return (vals - self.median_) / self.iqr_

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, path: str) -> None:
        """Persist scaler parameters to JSON."""
        assert self.median_ is not None, "Scaler not fitted yet."
        obj = {
            "median": self.median_.tolist(),
            "iqr": self.iqr_.tolist(),
            "feature_names": self.feature_names_,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    def load(self, path: str) -> "RobustScaler":
        """Load fitted parameters from JSON."""
        with open(path, "r") as f:
            obj = json.load(f)
        self.median_ = np.array(obj["median"], dtype=np.float64)
        self.iqr_ = np.array(obj["iqr"], dtype=np.float64)
        self.feature_names_ = obj.get("feature_names")
        return self


# ---------------------------------------------------------------------------
# OMNI2 gap handler
# ---------------------------------------------------------------------------

def handle_omni_gaps(
    series: np.ndarray,
    max_gap: int = 3,
) -> Tuple[np.ndarray, bool]:
    """Linearly interpolate NaN gaps ≤ *max_gap* hours; flag if any gap exceeds limit.

    Parameters
    ----------
    series : np.ndarray
        Shape ``(T, F)`` — a window of OMNI2 features.
    max_gap : int
        Maximum contiguous NaN run (along axis 0 for *any* feature) before
        the sample is flagged for dropping.

    Returns
    -------
    filled : np.ndarray
        Same shape, with short gaps interpolated along axis 0.
    should_drop : bool
        ``True`` if any feature has a contiguous NaN run > *max_gap*.
    """
    filled = series.copy()
    should_drop = False
    n_rows, n_feat = filled.shape

    for col in range(n_feat):
        vec = filled[:, col]
        nan_mask = np.isnan(vec)
        if not nan_mask.any():
            continue

        # Detect contiguous NaN runs
        max_run = _max_nan_run(nan_mask)
        if max_run > max_gap:
            should_drop = True
            # Still interpolate what we can for partial use
        # Linear interpolation
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) >= 2:
            filled[:, col] = np.interp(
                np.arange(n_rows), valid_idx, vec[valid_idx]
            )
        elif len(valid_idx) == 1:
            filled[:, col] = vec[valid_idx[0]]
        else:
            filled[:, col] = 0.0  # entire column missing

    return filled, should_drop


def _max_nan_run(mask: np.ndarray) -> int:
    """Return length of the longest contiguous ``True`` run."""
    if not mask.any():
        return 0
    runs = np.diff(np.where(np.concatenate(([False], mask, [False])))[0])
    return int(runs.max()) if len(runs) else 0


# ---------------------------------------------------------------------------
# Channel stats I/O
# ---------------------------------------------------------------------------

def save_channel_stats(stats: Dict[str, Dict[str, float]], path: str) -> None:
    """Save per-channel mean/std dict to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_channel_stats(path: str) -> Dict[str, Dict[str, float]]:
    """Load per-channel mean/std dict from JSON."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

class CSVLogger:
    """Append rows of metrics to a CSV file."""

    def __init__(self, path: str, columns: List[str]) -> None:
        self.path = path
        self.columns = columns
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(",".join(columns) + "\n")

    def log(self, row: Dict[str, object]) -> None:
        with open(self.path, "a") as f:
            vals = [str(row.get(c, "")) for c in self.columns]
            f.write(",".join(vals) + "\n")
