"""Metrics computation for AQI forecasting.

MAE, RMSE, R2, skill score vs persistence, CPCB band accuracy, and spike recall.
"""

from __future__ import annotations

import numpy as np

# CPCB AQI Category breakpoints (upper bounds)
# 0-50 Good, 51-100 Satisfactory, 101-200 Moderate, 201-300 Poor, 301-400 Very Poor, 401+ Severe
CATEGORY_BINS = [0, 50, 100, 200, 300, 400, float("inf")]


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R2)."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_skill(mae_model: float, mae_persistence: float) -> float:
    """Forecast skill score relative to persistence: 1 - MAE_model / MAE_persistence."""
    if mae_persistence == 0 or np.isnan(mae_persistence):
        return float("nan")
    return float(1.0 - mae_model / mae_persistence)


def aqi_to_band(y: np.ndarray) -> np.ndarray:
    """Assign AQI values to CPCB categories 0-5."""
    return np.asarray(np.digitize(y, bins=CATEGORY_BINS[1:-1]))


def compute_band_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy of CPCB AQI categorical band assignment."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return float("nan")
    bands_true = aqi_to_band(y_true[mask])
    bands_pred = aqi_to_band(y_pred[mask])
    return float(np.mean(bands_true == bands_pred))


def compute_spike_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 200.0,
) -> float:
    """Recall for spikes (AQI > 200, Poor or worse)."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return float("nan")
    true_spikes = y_true[mask] > threshold
    if not np.any(true_spikes):
        return float("nan")
    pred_spikes = y_pred[mask] > threshold
    tp = np.sum(true_spikes & pred_spikes)
    fn = np.sum(true_spikes & ~pred_spikes)
    return float(tp / (tp + fn))


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Directional accuracy of sequential changes."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask) or len(y_true[mask]) < 2:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    diff_true = np.diff(yt)
    diff_pred = np.diff(yp)
    return float(np.mean(np.sign(diff_true) == np.sign(diff_pred)))
