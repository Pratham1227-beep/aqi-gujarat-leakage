"""Generate EDA figures and build notebooks/eda_aqi.ipynb."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf

PROCESSED = Path("data/processed")
FIGS = Path("reports/figures")
FIGS.mkdir(parents=True, exist_ok=True)
NOTEBOOKS = Path("notebooks")
NOTEBOOKS.mkdir(parents=True, exist_ok=True)

# Set plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 150

CITIES = ["ahmedabad", "chennai", "gandhinagar", "ankleshwar", "vapi", "surat"]
CITY_LABELS = {
    "ahmedabad": "Ahmedabad",
    "chennai": "Chennai",
    "gandhinagar": "Gandhinagar",
    "ankleshwar": "Ankleshwar",
    "vapi": "Vapi",
    "surat": "Surat",
}

# 1. Time Series of AQI
fig, axes = plt.subplots(len(CITIES), 1, figsize=(12, 14), sharex=True)
for ax, city in zip(axes, CITIES, strict=False):
    df = pd.read_parquet(PROCESSED / f"{city}_daily.parquet")
    df["date"] = pd.to_datetime(df["date"])
    ax.plot(df["date"], df["aqi_official"], label=CITY_LABELS[city], color="#1f77b4", lw=1)
    ax.axhline(100, color="green", ls="--", alpha=0.5, label="Satisfactory (100)" if ax == axes[0] else "")
    ax.axhline(200, color="orange", ls="--", alpha=0.5, label="Moderate (200)" if ax == axes[0] else "")
    ax.axhline(300, color="red", ls="--", alpha=0.5, label="Poor (300)" if ax == axes[0] else "")
    ax.set_ylabel(CITY_LABELS[city], fontsize=9)
    ax.set_ylim(0, 505)
axes[0].set_title("Official CPCB AQI Daily Time Series (2015-2023)", fontsize=13, fontweight="bold")
axes[0].legend(loc="upper right", frameon=True, fontsize=8)
axes[-1].set_xlabel("Date", fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "eda_time_series.png", bbox_inches="tight")
plt.close(fig)

# 2. Missingness Heatmap
missing_records = []
for city in CITIES:
    m_path = PROCESSED / f"{city}_missingness.csv"
    if m_path.exists():
        m_df = pd.read_csv(m_path)
        for _, row in m_df.iterrows():
            missing_records.append({
                "City": CITY_LABELS[city],
                "Pollutant": row["pollutant"],
                "MissingShare": row["share_missing"] * 100,
            })
miss_df = pd.DataFrame(missing_records)
if not miss_df.empty:
    pivot_miss = miss_df.pivot(index="City", columns="Pollutant", values="MissingShare")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot_miss, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "% Missing"}, ax=ax)
    ax.set_title("Missing Data Percentage by City & Feature", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "eda_missingness_heatmap.png", bbox_inches="tight")
    plt.close(fig)

# 3. AQI Category Distributions
def get_cpcb_category(aqi):
    if pd.isna(aqi):
        return np.nan
    if aqi <= 50:
        return "Good (0-50)"
    if aqi <= 100:
        return "Satisfactory (51-100)"
    if aqi <= 200:
        return "Moderate (101-200)"
    if aqi <= 300:
        return "Poor (201-300)"
    if aqi <= 400:
        return "Very Poor (301-400)"
    return "Severe (401-500)"

cat_order = ["Good (0-50)", "Satisfactory (51-100)", "Moderate (101-200)", "Poor (201-300)", "Very Poor (301-400)", "Severe (401-500)"]
cat_colors = ["#2ca02c", "#8c564b", "#ff7f0e", "#d62728", "#9467bd", "#7f7f7f"]

dist_records = []
for city in CITIES:
    df = pd.read_parquet(PROCESSED / f"{city}_daily.parquet")
    cats = df["aqi_official"].apply(get_cpcb_category).dropna()
    counts = cats.value_counts(normalize=True) * 100
    for cat in cat_order:
        dist_records.append({
            "City": CITY_LABELS[city],
            "Category": cat,
            "Percentage": counts.get(cat, 0.0),
        })
dist_df = pd.DataFrame(dist_records)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=dist_df, x="City", y="Percentage", hue="Category", hue_order=cat_order, palette="viridis", ax=ax)
ax.set_title("CPCB AQI Category Distribution Across Cities (%)", fontsize=12, fontweight="bold")
ax.set_ylabel("Percentage of Valid Days")
ax.legend(title="CPCB AQI Band", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(FIGS / "eda_aqi_categories.png", bbox_inches="tight")
plt.close(fig)

# 4. Monthly Seasonality
season_records = []
for city in CITIES:
    df = pd.read_parquet(PROCESSED / f"{city}_daily.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    for m in range(1, 13):
        m_vals = df[df["month"] == m]["aqi_official"].dropna()
        season_records.append({
            "City": CITY_LABELS[city],
            "Month": m,
            "MeanAQI": m_vals.mean(),
            "MedianAQI": m_vals.median(),
        })
season_df = pd.DataFrame(season_records)

fig, ax = plt.subplots(figsize=(10, 5))
for city in CITIES:
    c_df = season_df[season_df["City"] == CITY_LABELS[city]]
    ax.plot(c_df["Month"], c_df["MeanAQI"], marker="o", label=CITY_LABELS[city], lw=2)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_ylabel("Mean AQI")
ax.set_title("Monthly Seasonality of AQI Across Cities", fontsize=12, fontweight="bold")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIGS / "eda_monthly_seasonality.png", bbox_inches="tight")
plt.close(fig)

# 5. Autocorrelation & PACF (Ahmedabad & Chennai)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, city in enumerate(["ahmedabad", "chennai"]):
    df = pd.read_parquet(PROCESSED / f"{city}_daily.parquet")
    aqi_s = df["aqi_official"].dropna()
    acfs = acf(aqi_s, nlags=30)
    pacfs = pacf(aqi_s, nlags=30)
    
    axes[i, 0].bar(range(len(acfs)), acfs, color="#1f77b4", width=0.4)
    axes[i, 0].axhline(0, color="gray", lw=0.8)
    axes[i, 0].set_title(f"{CITY_LABELS[city]} - Autocorrelation (ACF)", fontsize=11)
    axes[i, 0].set_ylabel("ACF")
    axes[i, 0].set_xlabel("Lag (Days)")
    
    axes[i, 1].bar(range(len(pacfs)), pacfs, color="#ff7f0e", width=0.4)
    axes[i, 1].axhline(0, color="gray", lw=0.8)
    axes[i, 1].set_title(f"{CITY_LABELS[city]} - Partial Autocorrelation (PACF)", fontsize=11)
    axes[i, 1].set_ylabel("PACF")
    axes[i, 1].set_xlabel("Lag (Days)")

fig.tight_layout()
fig.savefig(FIGS / "eda_acf_pacf.png", bbox_inches="tight")
plt.close(fig)

# 6. Recomputed vs Official Scatter (Ahmedabad and Chennai)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, city in zip(axes, ["ahmedabad", "chennai"], strict=False):
    df = pd.read_parquet(PROCESSED / f"{city}_daily.parquet")
    valid = df.dropna(subset=["aqi_official", "aqi_recomputed"])
    corr = valid["aqi_official"].corr(valid["aqi_recomputed"])
    r2 = corr ** 2
    mae = (valid["aqi_official"] - valid["aqi_recomputed"]).abs().mean()
    
    ax.scatter(valid["aqi_official"], valid["aqi_recomputed"], alpha=0.3, color="#2ca02c", s=15)
    ax.plot([0, 500], [0, 500], "r--", lw=1.5, label="1:1 Equality")
    ax.set_title(f"{CITY_LABELS[city]}: Recomputed vs Official AQI\n(N={len(valid)}, MAE={mae:.1f}, R²={r2:.3f})", fontsize=11)
    ax.set_xlabel("Official CPCB Bulletin AQI")
    ax.set_ylabel("Recomputed AQI from Pollutants")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIGS / "eda_reconstruction_scatter.png", bbox_inches="tight")
plt.close(fig)

# 7. Build Jupyter Notebook eda_aqi.ipynb
nb_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Exploratory Data Analysis: Gujarat Air Quality Index (2015–2023)\n",
            "This notebook presents the data validation, missingness analysis, seasonality, and reconstruction quality checks for Gujarat cities and the Chennai benchmark."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from IPython.display import Image, display\n",
            "\n",
            "summary_df = pd.read_csv('../reports/data_summary.csv')\n",
            "summary_df"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Official CPCB AQI Daily Time Series (2015-2023)"]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": ["display(Image('../reports/figures/eda_time_series.png'))"]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Missingness Heatmap by City and Feature"]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": ["display(Image('../reports/figures/eda_missingness_heatmap.png'))"]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. CPCB AQI Category Distributions"]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [],
        "source": ["display(Image('../reports/figures/eda_aqi_categories.png'))"]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 4. Monthly Seasonality"]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [],
        "source": ["display(Image('../reports/figures/eda_monthly_seasonality.png'))"]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 5. Autocorrelation & Partial Autocorrelation (ACF/PACF)"]
    },
    {
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [],
        "source": ["display(Image('../reports/figures/eda_acf_pacf.png'))"]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 6. Recomputed AQI vs Official Bulletin"]
    },
    {
        "cell_type": "code",
        "execution_count": 7,
        "metadata": {},
        "outputs": [],
        "source": ["display(Image('../reports/figures/eda_reconstruction_scatter.png'))"]
    }
]

notebook_content = {
    "cells": nb_cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.13.17"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(NOTEBOOKS / "eda_aqi.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print("Successfully generated all EDA figures in reports/figures/ and notebooks/eda_aqi.ipynb")
