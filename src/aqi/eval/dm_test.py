"""Diebold-Mariano test with Harvey small-sample correction."""

from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss: str = "absolute",
) -> tuple[float, float]:
    """Calculate Diebold-Mariano test statistic and two-sided p-value.

    Parameters
    ----------
    e1: Forecast errors of model 1 (y_true - y_pred1).
    e2: Forecast errors of model 2 (y_true - y_pred2).
    h: Forecast horizon (lags for autocovariance truncation = h - 1).
    loss: Loss function ('absolute' or 'squared').

    Returns
    -------
    (dm_stat, p_value)
    """
    mask = ~np.isnan(e1) & ~np.isnan(e2)
    e1_clean = e1[mask]
    e2_clean = e2[mask]
    n = len(e1_clean)
    if n <= 1:
        return float("nan"), float("nan")

    d = e1_clean**2 - e2_clean**2 if loss == "squared" else np.abs(e1_clean) - np.abs(e2_clean)

    d_mean = np.mean(d)

    # Autocovariance terms up to lag h - 1
    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        cov = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
        gamma_sum += 2.0 * cov

    lr_var = (gamma0 + gamma_sum) / n
    if lr_var <= 0:
        return 0.0, 1.0

    # Harvey, Leybourne, Newbold (1997) small-sample correction
    harvey_factor = np.sqrt(
        (n + 1 - 2 * h + h * (h - 1) / n) / n
    )
    dm_stat = float((d_mean / np.sqrt(lr_var)) * harvey_factor)
    p_value = float(2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat), df=n - 1)))

    return dm_stat, p_value
