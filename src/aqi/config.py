"""Pydantic models and YAML loader for the AQI study configuration."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class MetricsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mae: bool = True
    rmse: bool = True
    r2: bool = True
    skill: bool = True
    band_accuracy: bool = True
    spike_recall: bool = True
    bootstrap_resamples: int = 1000
    bootstrap_block_days: int = 7
    dm_hac_lags: int = 1
    dm_small_sample_correction: bool = True


class SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    random_test_fraction: float = Field(default=0.2, ge=0.05, le=0.5)
    random_seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    chronological_test_fraction: float = Field(default=0.2, ge=0.05, le=0.5)
    rolling_test_fraction: float = Field(default=0.2, ge=0.05, le=0.5)
    horizons: list[int] = Field(default=[1, 3, 7])
    refit_cadence_days: int = Field(default=30)


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    params: dict = Field(default_factory=dict)


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_raw_dir: str = "data/raw"
    data_interim_dir: str = "data/interim"
    data_processed_dir: str = "data/processed"

    results_dir: str = "results"
    results_runs_csv: str = "results/runs.csv"
    results_preds_dir: str = "results/preds"
    results_logs_dir: str = "results/logs"

    figures_dir: str = "results/figures"
    tables_dir: str = "results/tables"

    logs_dir: str = "logs"

    cities_file: str = "configs/cities.yaml"
    base_file: str = "configs/base.yaml"
    features_file: str = "configs/features.yaml"
    protocol_file: str = "configs/protocol.yaml"
    models_dir: str = "configs/models"

    cities: list[str] = Field(default_factory=list)
    seeds: dict = Field(default_factory=dict)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = "configs/base.yaml") -> "Config":
        cfg_path = Path(path)
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(**data)


def load_config(path: str | Path = "configs/base.yaml") -> Config:
    return Config.load(path)
