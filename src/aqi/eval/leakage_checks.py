"""Runtime leakage guards used by the evaluation harness.

Ensures strict temporal ordering and absence of planted/unlagged future targets.
"""

from __future__ import annotations

import pandas as pd


class LeakageError(ValueError):
    """Raised when data leakage is detected."""


def check_temporal_separation(
    train_dates: pd.Series | pd.DatetimeIndex,
    test_dates: pd.Series | pd.DatetimeIndex,
    setup: str,
) -> None:
    """Verify that max train date is strictly before min test date for setups B and C."""
    if setup in ("B", "C"):
        max_train = pd.to_datetime(train_dates).max()
        min_test = pd.to_datetime(test_dates).min()
        if max_train >= min_test:
            raise LeakageError(
                f"Temporal leak detected in Setup {setup}: "
                f"max train date ({max_train}) >= min test date ({min_test})."
            )


def check_feature_target_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.9999,
) -> None:
    """Detect planted or unlagged future targets in the feature matrix."""
    for col in X.columns:
        valid = X[col].notna() & y.notna()
        if valid.sum() < 10:
            continue
        corr = float(X.loc[valid, col].corr(y[valid]))
        if abs(corr) >= threshold:
            raise LeakageError(
                f"Potential target leakage in feature '{col}': correlation with target is {corr:.4f}"
            )
