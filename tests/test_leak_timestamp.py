"""Leakage test 1: Timestamp test."""

from __future__ import annotations

import pandas as pd
import pytest

from aqi.eval.leakage_checks import LeakageError, check_temporal_separation


def test_temporal_separation_guard() -> None:
    train_dates = pd.date_range("2023-01-01", "2023-06-30")
    test_dates_valid = pd.date_range("2023-07-01", "2023-12-31")
    # Must not raise
    check_temporal_separation(train_dates, test_dates_valid, setup="B")
    check_temporal_separation(train_dates, test_dates_valid, setup="C")

    # Overlapping dates must raise LeakageError
    test_dates_leaked = pd.date_range("2023-06-15", "2023-12-31")
    with pytest.raises(LeakageError):
        check_temporal_separation(train_dates, test_dates_leaked, setup="B")
