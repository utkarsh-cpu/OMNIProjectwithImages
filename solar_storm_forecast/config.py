"""Central configuration dataclass for the solar storm forecasting pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:
    """All hyperparameters and paths for training / evaluation / inference.

    Default paths assume the workspace root contains:

    - ``omni2_full_dataset.csv``  — single CSV with named columns + ``datetime``
    - ``SDOBenchmark-data-full.zip``  — or an already-extracted directory
      ``SDOBenchmark-data-full/``

    After extraction the SDO layout is::

        SDOBenchmark-data-full/
          training/
            meta_data.csv            # columns: id, start, end, peak_flux
            <AR>/<sample_id>/        # e.g. 11390/2012_01_01_19_06_00_0/
              <timestamp>__131.jpg
              <timestamp>__magnetogram.jpg
              ...
          test/
            meta_data.csv
            <AR>/<sample_id>/...
    """

    # ── Paths ───────────────────────────────────────────────────────────
    workspace_root: str = "/home/OMNIProjectwithImages"
    omni_csv_path: str = ""        # set in __post_init__
    sdo_zip_path: str = ""         # set in __post_init__
    sdo_extracted_dir: str = ""    # set in __post_init__
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"

    # ── Data ────────────────────────────────────────────────────────────
    look_back_x: int = 24          # OMNI2 hours of history
    forecast_y: int = 6            # hours ahead to forecast
    image_timesteps: int = 4       # fixed by SDOBenchmark structure
    image_channels: int = 5        # AIA 131,171,193,211 + HMI mag
    image_size: int = 256          # fixed by dataset
    selected_channels: List[str] = field(
        default_factory=lambda: ["131", "171", "193", "211", "magnetogram"]
    )

    # ── OMNI2 CSV column mapping ────────────────────────────────────────
    # Keys = our internal short names → Values = actual CSV column headers
    omni_col_map: Dict[str, str] = field(
        default_factory=lambda: {
            "Dst": "DST",
            "Bz_GSM": "Bz_GSM",
            "Vsw": "plasma_speed",
            "Np": "proton_density",
            "Kp": "Kp",
            "AE": "AE",
            "E_field": "electric_field",
            "Pdyn": "flow_pressure",
        }
    )
    # Fill values per *CSV column name*
    omni_fill_values: Dict[str, float] = field(
        default_factory=lambda: {
            "DST": 99999,
            "Bz_GSM": 999.9,
            "plasma_speed": 9999.0,
            "proton_density": 999.9,
            "Kp": 99,
            "AE": 9999,
            "electric_field": 999.99,
            "flow_pressure": 99.99,
        }
    )
    omni_max_gap_hours: int = 3

    # ── Training ────────────────────────────────────────────────────────
    batch_size: int = 16
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-5
    dropout: float = 0.1
    early_stopping_patience: int = 10
    grad_clip_norm: float = 1.0
    num_workers: int = 4

    # ── LR schedule ────────────────────────────────────────────────────
    lr_t0: int = 10
    lr_t_mult: int = 2

    # ── Augmentation ────────────────────────────────────────────────────
    flare_oversample_factor: int = 3
    flare_threshold: float = 1e-6
    aug_brightness: float = 0.15
    aug_contrast: float = 0.10
    aug_rotation_deg: float = 5.0

    # ── Checkpointing ──────────────────────────────────────────────────
    checkpoint_metric: str = "val_mae_flare"

    # ── Model ──────────────────────────────────────────────────────────
    efficientnet_variant: str = "efficientnet_b3"
    image_feature_dim: int = 512
    lstm_hidden: int = 128
    lstm_layers: int = 2
    fusion_heads: int = 8
    fusion_dim: int = 512
    decoder_hidden: int = 256
    decoder_out: int = 128

    # ── Inference ──────────────────────────────────────────────────────
    n_mc_samples: int = 20

    # ── Reproducibility ────────────────────────────────────────────────
    seed: int = 42

    # ── Normalisation stats cache ──────────────────────────────────────
    channel_stats_file: str = "channel_stats.json"
    omni_scaler_file: str = "omni_scaler.json"

    @property
    def omni_col_names(self) -> List[str]:
        """Internal short names for the 8 OMNI2 features."""
        return list(self.omni_col_map.keys())

    def __post_init__(self) -> None:
        self.omni_csv_path = os.path.join(self.workspace_root,
                                          "omni2_full_dataset.csv")
        self.sdo_zip_path = os.path.join(self.workspace_root,
                                         "SDOBenchmark-data-full.zip")
        self.sdo_extracted_dir = os.path.join(self.workspace_root,
                                              "SDOBenchmark-data-full")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
