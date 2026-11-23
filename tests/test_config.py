"""Test configuration loading, seeds, and logging."""

from __future__ import annotations

from pathlib import Path

from aqi.config import load_config
from aqi.logging_utils import get_logger
from aqi.seeds import set_all_seeds


def test_load_base_config() -> None:
    config_path = Path("configs/base.yaml")
    assert config_path.exists(), "base.yaml must exist"
    cfg = load_config(config_path)
    assert cfg.cities == ["Ahmedabad", "Surat", "Chennai"]
    assert cfg.metrics.mae is True
    assert cfg.split.horizons == [1, 3, 7]


def test_set_seeds() -> None:
    set_all_seeds(42, 42, 42)


def test_logger() -> None:
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
