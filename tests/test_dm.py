"""Unit tests for Diebold-Mariano test."""

from __future__ import annotations

import numpy as np

from aqi.eval.dm_test import diebold_mariano_test


def test_diebold_mariano_superior_model() -> None:
    # Model 1 has much smaller errors than Model 2
    e1 = np.array([1.0, 0.5, -0.5, 1.2, -0.8, 0.4, 0.6, -0.3, 0.5, 0.2] * 5)
    e2 = np.array([5.0, 4.0, -6.0, 5.5, -4.8, 3.9, 6.1, -5.2, 4.8, 3.5] * 5)

    dm_stat, p_val = diebold_mariano_test(e1, e2, h=1, loss="absolute")
    # dm_stat should be strongly negative (e1 has lower loss than e2)
    assert dm_stat < -2.0
    assert p_val < 0.05
