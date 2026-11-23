"""Chronological train/test split (Setup B).

First 80% train, last 20% test. Strictly time-ordered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def chrono_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train and test integer indices for a chronological split."""
    n = len(df)
    split_point = int(np.round(n * (1.0 - test_fraction)))
    train_idx = np.arange(0, split_point)
    test_idx = np.arange(split_point, n)
    return train_idx, test_idx
