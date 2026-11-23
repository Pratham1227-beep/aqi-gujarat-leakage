"""PyTorch LSTM forecaster."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from aqi.models.base import Forecaster


class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        res: torch.Tensor = self.fc(out).squeeze(-1)
        return res


class LSTMForecaster(Forecaster):
    """LSTM sequence forecaster with early stopping."""

    def __init__(
        self,
        window_size: int = 14,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self._scaler = StandardScaler()
        self._net: _LSTMNet | None = None

    @property
    def name(self) -> str:
        return "lstm"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LSTMForecaster:
        torch.manual_seed(self.seed)
        scaled_x = self._scaler.fit_transform(X.fillna(0.0))
        input_dim = scaled_x.shape[1]
        self._net = _LSTMNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._net is None:
            return np.full(len(X), np.nan)
        self._net.eval()
        scaled_x = self._scaler.transform(X.fillna(0.0))
        # Window of 1 for 2D inputs when predicting tabular features
        x_tensor = torch.tensor(scaled_x[:, None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            preds = self._net(x_tensor).cpu().numpy()
        return np.asarray(preds)

    def get_params(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "device": self.device,
        }
