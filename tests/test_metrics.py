"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from aqi.eval.metrics import (
    compute_band_accuracy,
    compute_mae,
    compute_r2,
    compute_rmse,
    compute_skill,
    compute_spike_recall,
)


def test_metrics_computations() -> None:
    y_true = np.array([50.0, 100.0, 150.0, 250.0])
    y_pred = np.array([60.0, 90.0, 140.0, 260.0])

    assert compute_mae(y_true, y_pred) == 10.0
    assert compute_rmse(y_true, y_pred) == 10.0
    assert compute_r2(y_true, y_pred) > 0.95

    # Skill
    assert compute_skill(10.0, 20.0) == 0.5
    assert compute_skill(25.0, 20.0) == -0.25

    # Band accuracy (50: Good, 100: Satisfactory, 150: Moderate, 250: Poor)
    # y_pred: 60 is Satisfactory (diff), 90 is Satisfactory (match), 140 is Moderate (match), 260 is Poor (match)
    assert compute_band_accuracy(y_true, y_pred) == 0.75

    # Spike recall (> 200) -> 1 true spike (250), correctly predicted (260)
    assert compute_spike_recall(y_true, y_pred, threshold=200.0) == 1.0
