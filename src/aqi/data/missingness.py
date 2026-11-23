"""Missingness reporting: per city and pollutant share, longest gap, missing by month.

No imputation here; imputation happens later, only inside training windows.
Outputs a CSV and a heatmap image.
"""

from __future__ import annotations

import pandas as pd


def report_missingness(
    df: pd.DataFrame,
    city: str,
    pollutant_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return a per-pollutant missingness summary.

    Parameters
    ----------
    df:
        Daily DataFrame with a ``date`` column (or DatetimeIndex).
    city:
        City name for the report.
    pollutant_cols:
        Pollutant column names to report on.

    Returns
    -------
    DataFrame with columns: city, pollutant, n_total, n_missing, share_missing,
    longest_gap_days, missing_by_month.
    """
    if pollutant_cols is None:
        exclude = {"date", "city", "source", "aqi_official", "aqi_recomputed"}
        pollutant_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [float, int]]

    date_series = df["date"] if "date" in df.columns else pd.to_datetime(df.index)
    rows = []
    for col in pollutant_cols:
        if col not in df.columns:
            continue
        missing = df[col].isna()
        n_total = len(missing)
        n_missing = int(missing.sum())
        share = n_missing / n_total if n_total > 0 else float("nan")

        # Longest consecutive run of missing values
        longest_gap = _longest_gap_days(date_series, missing)

        # Missing by month
        months = date_series.dt.to_period("M")
        by_month = missing.groupby(months).mean()

        rows.append(
            {
                "city": city,
                "pollutant": col,
                "n_total": n_total,
                "n_missing": n_missing,
                "share_missing": round(share, 4),
                "longest_gap_days": longest_gap,
                "missing_by_month": by_month.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def _longest_gap_days(dates: pd.Series, missing: pd.Series) -> int:
    """Return the longest consecutive run of missing days."""
    if len(missing) == 0 or not missing.any():
        return 0
    # Sort by date
    df = pd.DataFrame({"date": pd.to_datetime(dates), "missing": missing.astype(bool)}).sort_values("date")
    # Non-missing values increment the group counter so consecutive missing values share the same group
    df["gap_group"] = (~df["missing"]).cumsum()
    gaps = df[df["missing"]].groupby("gap_group").size()
    return int(gaps.max()) if len(gaps) > 0 else 0