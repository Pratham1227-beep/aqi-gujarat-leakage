"""Moving block bootstrap for time-series confidence intervals."""

from __future__ import annotations

import numpy as np


def block_bootstrap_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_length: int = 7,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate moving block bootstrap confidence interval for MAE.

    Returns (ci_lower, ci_upper).
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    n = len(yt)
    if n < block_length:
        mae = float(np.mean(np.abs(yt - yp)))
        return mae, mae

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    possible_starts = n - block_length + 1

    boot_maes = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, possible_starts, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]
        boot_maes[i] = np.mean(np.abs(yt[indices] - yp[indices]))

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_maes, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_maes, 100.0 * (1.0 - alpha)))
    return ci_lower, ci_upper
