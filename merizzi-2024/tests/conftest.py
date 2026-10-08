from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from merizzi2024.config import Config, load_config
from merizzi2024.data.synthetic import make_synthetic_dataset

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """The small config with a freshly generated synthetic dataset in a temp dir."""
    cfg = load_config(CONFIGS / "small.yaml")
    root = tmp_path_factory.mktemp("synthetic-small")
    cfg.data = dataclasses.replace(cfg.data, root=str(root))
    cfg.synthetic = dataclasses.replace(cfg.synthetic, num_train=96, num_test=40)
    cfg.path = None
    make_synthetic_dataset(cfg, write_config=False)
    return cfg


@pytest.fixture(scope="session")
def paper_cfg() -> Config:
    return load_config(CONFIGS / "paper.yaml")
