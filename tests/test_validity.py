"""Unit tests for physical range validity and instrument flags."""

from __future__ import annotations

import pandas as pd

from aqi.data.validity import flag_invalid_values


def test_flag_invalid_values() -> None:
    df = pd.DataFrame({
        "pm25": [-5.0, 50.0, 1500.0],
        "co": [0.5, 120.0, 1.2],
    })
    cleaned, counts = flag_invalid_values(df)
    assert counts["pm25"] == 2
    assert counts["co"] == 1
    assert pd.isna(cleaned.loc[0, "pm25"])
    assert cleaned.loc[1, "pm25"] == 50.0
    assert pd.isna(cleaned.loc[2, "pm25"])
    assert pd.isna(cleaned.loc[1, "co"])
