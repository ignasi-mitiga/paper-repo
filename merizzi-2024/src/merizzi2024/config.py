"""Typed configuration loaded from YAML.

Every number that the authors hard-code in ``setup.py`` lives here instead, including the
normalization maxima that the data preparation commands write back into the YAML file.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SplitFiles:
    hr: str
    lr: str


@dataclass
class ChannelStats:
    """Per-field mean/variance of the max-normalized data, used by the standardization step."""

    lr_mean: float
    lr_var: float
    hr_mean: float
    hr_var: float


@dataclass
class DataConfig:
    root: str
    splits: dict[str, SplitFiles]
    train_split: str
    test_splits: list[str]
    hr_shape: tuple[int, int]
    lr_shape: tuple[int, int]
    frame_offsets: list[int] = field(default_factory=lambda: [-2, -1, 0, 1])
    hr_max: float | None = None
    lr_max: float | None = None
    standardize: bool = True
    stats: ChannelStats | None = None

    @property
    def num_frames(self) -> int:
        return len(self.frame_offsets)

    def split_paths(self, split: str) -> tuple[Path, Path]:
        files = self.splits[split]
        root = Path(self.root)
        return root / files.hr, root / files.lr


@dataclass
class SyntheticConfig:
    num_train: int = 768
    num_test: int = 192
    blur_sigma: float = 1.5
    seed: int = 0


@dataclass
class Era5Config:
    bucket_root: str = "gs://weatherbench2/datasets/era5"
    hr_store: str = "1959-2022-6h-512x256_equiangular_conservative.zarr"
    lr_store: str = "1959-2022-6h-128x64_equiangular_conservative.zarr"
    variable: str = "10m_wind_speed"
    years: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class ModelConfig:
    widths: list[int] = field(default_factory=lambda: [64, 128, 256, 384])
    block_depth: int = 3
    embedding_dims: int = 64
    embedding_max_frequency: float = 1000.0
    bottleneck_attention: bool = True
    schedule: str = "cosine"
    min_signal_rate: float = 0.015
    max_signal_rate: float = 0.95
    ema_decay: float = 0.999


@dataclass
class TrainConfig:
    batch_size: int = 8
    max_steps: int = 110_000
    lr: float = 1e-4
    lr_final: float = 1e-5
    weight_decay: float = 1e-5
    weight_decay_final: float = 1e-6
    precision: str = "16-mixed"
    val_every_n_steps: int = 1000
    val_batches: int = 2
    seed: int = 0
    num_workers: int = 4
    out_dir: str = "runs/run"


@dataclass
class SampleConfig:
    diffusion_steps: int = 5
    ensemble_size: int = 15
    batch_size: int = 32


@dataclass
class Config:
    name: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    era5: Era5Config = field(default_factory=Era5Config)
    path: Path | None = None  # where the YAML came from, for write-back


def _build(cls: type, raw: Any) -> Any:
    """Recursively build a dataclass from a plain dict (None stays None)."""
    if raw is None:
        return None
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in raw:
            kwargs[f.name] = _coerce(hints[f.name], raw[f.name])
    return cls(**kwargs)


def _coerce(tp: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (types.UnionType, typing.Union):
        inner = [a for a in args if a is not type(None)]
        return _coerce(inner[0], value) if len(inner) == 1 else value
    if origin is dict:
        return {k: _coerce(args[1], v) for k, v in value.items()}
    if origin is list:
        return [_coerce(args[0], v) for v in value]
    if origin is tuple:
        return tuple(_coerce(a, v) for a, v in zip(args, value, strict=True))
    if dataclasses.is_dataclass(tp):
        return _build(tp, value)
    if tp in (int, float, str, bool):
        return tp(value)
    return value


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    cfg = _build(Config, raw)
    cfg.path = path
    return cfg


def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)
                if f.name != "path"}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_config(cfg: Config, path: str | Path | None = None) -> Path:
    """Write the configuration back to YAML (used to persist normalization statistics)."""
    path = Path(path) if path is not None else cfg.path
    if path is None:
        raise ValueError("no path to save the config to")
    with open(path, "w") as fh:
        yaml.safe_dump(_to_plain(cfg), fh, sort_keys=False)
    return path
