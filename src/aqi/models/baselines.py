"""Baseline models: Persistence, Seasonal Naive, and Moving Average."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from aqi.models.base import Forecaster


class PersistenceForecaster(Forecaster):
    """Persistence baseline: predicts the most recently observed AQI."""

    def __init__(self, target_lag_col: str = "aqi_official_lag_1"):
        self.target_lag_col = target_lag_col

    @property
    def name(self) -> str:
        return "persistence"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PersistenceForecaster:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.target_lag_col in X.columns:
            return np.asarray(X[self.target_lag_col].to_numpy())
        return np.full(len(X), np.nan)

    def get_params(self) -> dict[str, Any]:
        return {"target_lag_col": self.target_lag_col}


class SeasonalNaiveForecaster(Forecaster):
    """Seasonal naive baseline: predicts value from same weekday 7 days ago."""

    def __init__(self, target_lag_col: str = "aqi_official_lag_7"):
        self.target_lag_col = target_lag_col

    @property
    def name(self) -> str:
        return "seasonal_naive"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaiveForecaster:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.target_lag_col in X.columns:
            return np.asarray(X[self.target_lag_col].to_numpy())
        return np.full(len(X), np.nan)

    def get_params(self) -> dict[str, Any]:
        return {"target_lag_col": self.target_lag_col}


class MovingAverageForecaster(Forecaster):
    """Moving average baseline: predicts rolling mean over previous window."""

    def __init__(self, roll_mean_col: str = "aqi_official_roll_mean_7"):
        self.roll_mean_col = roll_mean_col

    @property
    def name(self) -> str:
        return "moving_average"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MovingAverageForecaster:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.roll_mean_col in X.columns:
            return np.asarray(X[self.roll_mean_col].to_numpy())
        return np.full(len(X), np.nan)

    def get_params(self) -> dict[str, Any]:
        return {"roll_mean_col": self.roll_mean_col}
