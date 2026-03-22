"""Unified CLI entry point for training and evaluation workflows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import fields
from typing import Any, Iterable, get_args, get_origin, get_type_hints

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from solar_storm_forecast.config import Config


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

_DERIVED_PATH_FIELDS = {"omni_csv_path", "sdo_zip_path", "sdo_extracted_dir"}


def _str_to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    return annotation


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    type_hints = get_type_hints(Config)

    for field in fields(Config):
        annotation = _unwrap_optional(type_hints.get(field.name, field.type))
        arg_name = f"--{field.name.replace('_', '-')}"
        kwargs: dict[str, Any] = {
            "dest": field.name,
            "default": None,
            "help": f"Override Config.{field.name}",
        }
        origin = get_origin(annotation)

        if annotation is bool:
            kwargs["type"] = _str_to_bool
            kwargs["metavar"] = "BOOL"
        elif annotation in {int, float, str}:
            kwargs["type"] = annotation
        elif origin in {list, tuple}:
            inner = get_args(annotation)[0] if get_args(annotation) else str
            kwargs["type"] = inner
            kwargs["nargs"] = "+"
            kwargs["metavar"] = "VALUE"
        elif origin is dict:
            kwargs["type"] = json.loads
            kwargs["metavar"] = "JSON"
        else:
            kwargs["type"] = str

        parser.add_argument(arg_name, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the solar storm forecasting pipeline with a single entry point."
        )
    )
    parser.add_argument(
        "command",
        choices=["train", "test", "evaluate", "all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path for test/evaluate. Defaults to checkpoints/best_model.pt.",
    )
    parser.add_argument(
        "--eval-output-dir",
        default=None,
        help="Directory for evaluation outputs. Defaults to <output_dir>/evaluation.",
    )
    _add_config_arguments(parser)
    return parser


def _build_config(args: argparse.Namespace) -> Config:
    values = vars(args)
    field_names = {field.name for field in fields(Config)}
    explicit_overrides = {
        name: values[name]
        for name in field_names
        if name in values and values[name] is not None and name not in _DERIVED_PATH_FIELDS
    }
    cfg = Config(**explicit_overrides)

    for derived_field in _DERIVED_PATH_FIELDS:
        override = values.get(derived_field)
        if override is not None:
            setattr(cfg, derived_field, override)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    return cfg


def _resolve_stats(cfg: Config) -> dict[str, dict[str, float]]:
    from solar_storm_forecast.dataset import precompute_stats
    from solar_storm_forecast.utils import load_channel_stats

    stats_path = os.path.join(cfg.output_dir, cfg.channel_stats_file)
    if os.path.exists(stats_path):
        logger.info("Loading channel stats from %s", stats_path)
        return load_channel_stats(stats_path)
    logger.info("Channel stats missing; computing them from training data")
    return precompute_stats(cfg)


def _load_omni_scaler(cfg: Config) -> Any:
    from solar_storm_forecast.utils import RobustScaler

    scaler_path = os.path.join(cfg.output_dir, cfg.omni_scaler_file)
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"OMNI scaler not found at {scaler_path}. Run training first or provide matching output_dir."
        )
    logger.info("Loading OMNI scaler from %s", scaler_path)
    return RobustScaler().load(scaler_path)


def _default_checkpoint_path(cfg: Config) -> str:
    return os.path.join(cfg.checkpoint_dir, "best_model.pt")


def run_training(cfg: Config) -> str:
    from solar_storm_forecast.train import train

    return train(cfg)


def run_evaluation(
    cfg: Config,
    checkpoint_path: str,
    eval_output_dir: str | None = None,
) -> dict[str, float]:
    from torch.utils.data import DataLoader

    from solar_storm_forecast.dataset import SolarDataset, get_eval_transform, load_omni2
    from solar_storm_forecast.evaluate import full_evaluation
    from solar_storm_forecast.model import SolarStormModel
    from solar_storm_forecast.train import load_checkpoint
    from solar_storm_forecast.utils import get_device, seed_everything

    seed_everything(cfg.seed)
    device = get_device()
    logger.info("Device: %s", device)

    omni_df = load_omni2(cfg)
    channel_stats = _resolve_stats(cfg)
    omni_scaler = _load_omni_scaler(cfg)

    test_ds = SolarDataset(
        cfg,
        split="test",
        omni_df=omni_df,
        channel_stats=channel_stats,
        omni_scaler=omni_scaler,
        augmentation=get_eval_transform(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SolarStormModel(cfg).to(device)
    model, epoch = load_checkpoint(model, checkpoint_path, device)
    logger.info("Loaded checkpoint from epoch %s", epoch)

    output_dir = eval_output_dir or os.path.join(cfg.output_dir, "evaluation")
    metrics = full_evaluation(model, test_loader, device, cfg, output_dir=output_dir)
    return metrics


def _format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = _build_config(args)

    checkpoint_path = args.checkpoint or _default_checkpoint_path(cfg)

    if args.command == "train":
        best_checkpoint = run_training(cfg)
        logger.info("Training finished. Best checkpoint: %s", best_checkpoint)
        return 0

    if args.command in {"test", "evaluate"}:
        if not os.path.exists(checkpoint_path):
            parser.error(f"Checkpoint not found: {checkpoint_path}")
        metrics = run_evaluation(cfg, checkpoint_path, args.eval_output_dir)
        logger.info("Evaluation complete: %s", _format_metrics(metrics))
        return 0

    best_checkpoint = run_training(cfg)
    metrics = run_evaluation(cfg, best_checkpoint, args.eval_output_dir)
    logger.info("Full pipeline complete: %s", _format_metrics(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())