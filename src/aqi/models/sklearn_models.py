"""Scikit-learn models: Ridge and Random Forest."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aqi.models.base import Forecaster


class RidgeForecaster(Forecaster):
    """Ridge regression with internal feature scaling."""

    def __init__(self, alpha: float = 1.0, seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=self.alpha, random_state=self.seed)),
        ])

    @property
    def name(self) -> str:
        return "ridge"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeForecaster:
        self._pipe.fit(X.fillna(0.0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._pipe.predict(X.fillna(0.0)))

    def get_params(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "seed": self.seed}


class RandomForestForecaster(Forecaster):
    """Random Forest regressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=-1,
        )

    @property
    def name(self) -> str:
        return "random_forest"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RandomForestForecaster:
        self._model.fit(X.fillna(0.0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(X.fillna(0.0)))

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "seed": self.seed,
        }
