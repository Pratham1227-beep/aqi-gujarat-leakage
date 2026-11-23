"""Forecaster abstract base class.

Every model implements fit(X, y) and predict(X), name, and get_params.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class Forecaster(ABC):
    """Abstract base class for all forecasting models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return canonical name of the forecaster."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster:
        """Fit model to training features X and targets y."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point forecasts for feature matrix X."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters as a dictionary."""
