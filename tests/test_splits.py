"""Unit tests for split strategies (random, chrono, rolling origin)."""

from __future__ import annotations

import pandas as pd

from aqi.splits.chrono_split import chrono_split
from aqi.splits.random_split import random_split
from aqi.splits.rolling_origin import rolling_origin_splits


def test_random_split_sizes() -> None:
    df = pd.DataFrame({"val": range(100)})
    train_idx, test_idx = random_split(df, test_fraction=0.2, seed=42)
    assert len(train_idx) == 80
    assert len(test_idx) == 20
    assert len(set(train_idx).intersection(set(test_idx))) == 0


def test_chrono_split_temporal_order() -> None:
    df = pd.DataFrame({"val": range(100)})
    train_idx, test_idx = chrono_split(df, test_fraction=0.2)
    assert len(train_idx) == 80
    assert len(test_idx) == 20
    assert max(train_idx) < min(test_idx)


def test_rolling_origin_splits() -> None:
    df = pd.DataFrame({"val": range(100)})
    splits = list(rolling_origin_splits(df, test_fraction=0.2, refit_cadence=10))
    assert len(splits) == 20
    # First test point is index 80
    train_0, test_0, refit_0 = splits[0]
    assert len(train_0) == 80
    assert test_0[0] == 80
    assert refit_0 is True
    assert max(train_0) < test_0[0]
