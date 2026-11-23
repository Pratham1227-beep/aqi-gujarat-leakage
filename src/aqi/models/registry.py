"""Model registry mapping string names to forecaster classes."""

from __future__ import annotations

from typing import Any

from aqi.models.base import Forecaster
from aqi.models.baselines import (
    MovingAverageForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from aqi.models.lstm import LSTMForecaster
from aqi.models.sarimax import SARIMAXForecaster
from aqi.models.sklearn_models import RandomForestForecaster, RidgeForecaster
from aqi.models.xgb import XGBoostForecaster

MODEL_REGISTRY: dict[str, type[Forecaster]] = {
    "persistence": PersistenceForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "moving_average": MovingAverageForecaster,
    "sarimax": SARIMAXForecaster,
    "ridge": RidgeForecaster,
    "random_forest": RandomForestForecaster,
    "xgboost": XGBoostForecaster,
    "lstm": LSTMForecaster,
}


def get_model(name: str, **kwargs: Any) -> Forecaster:
    """Instantiate a registered model by canonical name."""
    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown model name '{name}'. Registered: {list(MODEL_REGISTRY.keys())}"
        )
    return cls(**kwargs)
