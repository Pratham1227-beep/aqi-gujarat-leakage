"""Random train/test split (Setup A).

80/20 train/test split repeated across 10 random seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train and test integer indices for a random split."""
    n = len(df)
    n_test = int(np.round(n * test_fraction))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return train_idx, test_idx
