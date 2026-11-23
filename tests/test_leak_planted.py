"""Leakage test 3: Planted-leak test."""

from __future__ import annotations

import pandas as pd
import pytest

from aqi.eval.leakage_checks import LeakageError, check_feature_target_correlation


def test_planted_leak_raises() -> None:
    y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0])
    # Plant a feature identical to the target
    X_leaked = pd.DataFrame({
        "lag_1": y.shift(1),
        "planted_future_target": y,
    })
    with pytest.raises(LeakageError, match="Potential target leakage"):
        check_feature_target_correlation(X_leaked, y)
