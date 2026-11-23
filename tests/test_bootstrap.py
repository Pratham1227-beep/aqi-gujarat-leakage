"""Unit tests for block bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np

from aqi.eval.bootstrap import block_bootstrap_mae


def test_block_bootstrap_mae() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.uniform(50, 150, size=100)
    y_pred = y_true + rng.normal(0, 5, size=100)

    ci_lo, ci_hi = block_bootstrap_mae(
        y_true, y_pred, block_length=7, n_resamples=200, seed=42
    )
    mae = float(np.mean(np.abs(y_true - y_pred)))
    assert ci_lo <= mae <= ci_hi
    assert ci_lo > 0.0
