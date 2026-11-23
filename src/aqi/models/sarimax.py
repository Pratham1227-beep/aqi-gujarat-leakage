"""SARIMAX forecasting model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from aqi.models.base import Forecaster


class SARIMAXForecaster(Forecaster):
    """SARIMAX wrapper implementing Forecaster ABC."""

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7),
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self._model_res: Any = None

    @property
    def name(self) -> str:
        return "sarimax"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SARIMAXForecaster:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(
            endog=y,
            exog=X if not X.empty else None,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._model_res = model.fit(disp=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model_res is None:
            return np.full(len(X), np.nan)
        preds = self._model_res.forecast(
            steps=len(X), exog=X if not X.empty else None
        )
        return np.asarray(preds)

    def get_params(self) -> dict[str, Any]:
        return {"order": self.order, "seasonal_order": self.seasonal_order}
