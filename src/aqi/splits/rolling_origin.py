"""Rolling-origin split generator (Setup C).

Expanding window with periodic refit (e.g. monthly / every 30 days).
Initial training window equals the first 80% span; test covers the last 20%.
Yields (train_idx, test_idx, refit_flag).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def rolling_origin_splits(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    refit_cadence: int = 30,
) -> Iterator[tuple[np.ndarray, np.ndarray, bool]]:
    """Yield (train_idx, test_idx, refit_flag) for rolling origin evaluation."""
    n = len(df)
    split_point = int(np.round(n * (1.0 - test_fraction)))
    train_len = split_point

    for step, test_i in enumerate(range(split_point, n)):
        refit = (step % refit_cadence) == 0
        train_idx = np.arange(0, train_len + step)
        test_idx = np.array([test_i])
        yield train_idx, test_idx, refit
