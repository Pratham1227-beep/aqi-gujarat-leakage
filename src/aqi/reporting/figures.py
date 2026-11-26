"""Generate camera-ready publication figures from results/runs.csv and results/preds/."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)

RUNS_CSV = Path("results/runs.csv")
PREDS_DIR = Path("results/preds")
FIGS_DIR = Path("results/figures")

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300


def plot_fig1_leakage_inflation(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 1: Visual proof of leakage inflation across Setups A, B, and C."""
    sub = df[(df["city"] == "ahmedabad") & (df["model"].isin(["ridge", "random_forest", "xgboost"]))].copy()
    
    # Setup A is mean across seeds
    a_df = sub[sub["setup"] == "A"].groupby("model").agg({"r2": "mean", "mae": "mean"}).reset_index()
    a_df["setup"] = "Setup A (Random)"
    
    b_df = sub[sub["setup"] == "B"][["model", "r2", "mae"]].copy()
    b_df["setup"] = "Setup B (Chrono)"

    c_df = sub[(sub["setup"] == "C") & (sub["horizon"] == 1)][["model", "r2", "mae"]].copy()
    c_df["setup"] = "Setup C (Rolling)"

    combined = pd.concat([a_df, b_df, c_df], ignore_index=True)
    combined["model"] = combined["model"].str.replace("_", " ").str.title().str.replace("Xgboost", "XGBoost")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    sns.barplot(data=combined, x="model", y="r2", hue="setup", palette="Set1", ax=ax1)
    ax1.set_title("Variance Explained ($R^2$ Collapse)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("$R^2$ Score")
    ax1.set_xlabel("")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(title="Validation Setup", loc="upper right", framealpha=0.9)

    sns.barplot(data=combined, x="model", y="mae", hue="setup", palette="Set1", ax=ax2)
    ax2.set_title("Mean Absolute Error (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("MAE (AQI Points)")
    ax2.set_xlabel("")
    ax2.set_ylim(0, 38)
    ax2.legend(title="Validation Setup", loc="upper left", framealpha=0.9)

    fig.suptitle("Impact of Methodological Leakage in Ahmedabad AQI Modeling", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_leakage_inflation.png", bbox_inches="tight")
    fig.savefig(out_dir / "fig1_leakage_inflation.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote fig1_leakage_inflation")


def plot_fig2_model_comparison_h1(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 2: Setup C (h=1) model comparison vs Persistence baseline across cities."""
    sub = df[(df["setup"] == "C") & (df["horizon"] == 1)].copy()
    sub["city"] = sub["city"].str.capitalize()
    sub["model"] = sub["model"].str.replace("_", " ").str.title().str.replace("Xgboost", "XGBoost")

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.set_facecolor("white")
    palette = sns.color_palette("tab10", n_colors=sub["model"].nunique())
    
    sns.barplot(data=sub, x="city", y="mae", hue="model", palette=palette, ax=ax)
    ax.set_title("Next-Day ($t+1$) MAE Across Gujarat Cities and Chennai (Setup C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (AQI Points)")
    ax.set_xlabel("City")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_model_comparison_h1.png", bbox_inches="tight")
    fig.savefig(out_dir / "fig2_model_comparison_h1.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote fig2_model_comparison_h1")


def plot_fig3_forecast_trajectory(preds_dir: Path, out_dir: Path) -> None:
    """Figure 3: Out-of-sample actual vs predicted trajectory for Ahmedabad."""
    p_pers = preds_dir / "ahmedabad_C_persistence_h1_s42.csv"
    p_rf = preds_dir / "ahmedabad_C_random_forest_h1_s42.csv"
    p_xgb = preds_dir / "ahmedabad_C_xgboost_h1_s42.csv"

    if not p_pers.exists() or not p_rf.exists() or not p_xgb.exists():
        log.warning("Prediction files missing for fig3 trajectory")
        return

    df_pers = pd.read_csv(p_pers)
    df_rf = pd.read_csv(p_rf)
    df_xgb = pd.read_csv(p_xgb)

    # Plot a representative 90-day window
    n = min(len(df_pers), 90)
    idx = range(n)

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(idx, df_pers["y_true"].iloc[:n], label="Official CPCB AQI", color="black", lw=2)
    ax.plot(idx, df_pers["y_pred"].iloc[:n], label="Persistence Forecaster", color="#1f77b4", ls="--", lw=1.5)
    ax.plot(idx, df_rf["y_pred"].iloc[:n], label="Random Forest", color="#2ca02c", alpha=0.8, lw=1.5)
    ax.plot(idx, df_xgb["y_pred"].iloc[:n], label="XGBoost", color="#ff7f0e", alpha=0.8, lw=1.5)

    ax.set_title("Ahmedabad Next-Day Forecast Trajectory (90-Day Operational Test Window)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Consecutive Forecast Days")
    ax.set_ylabel("AQI Value")
    # Position legend in upper left to avoid overlapping peaks (per expert review)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_forecast_trajectory.png", bbox_inches="tight")
    fig.savefig(out_dir / "fig3_forecast_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote fig3_forecast_trajectory")


def plot_fig4_diebold_mariano(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 4: Diebold-Mariano test p-values comparing models to Persistence."""
    sub = df[(df["setup"] == "C") & (df["horizon"] == 1) & (df["model"] != "persistence")].copy()
    sub["city"] = sub["city"].str.capitalize()
    sub["model"] = sub["model"].str.replace("_", " ").str.title().str.replace("Xgboost", "XGBoost")
    pivot = sub.pivot(index="city", columns="model", values="dm_p")

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.set_facecolor("white")
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu_r", cbar_kws={"label": "DM p-value"}, ax=ax, vmin=0, vmax=1)
    ax.set_title("Diebold-Mariano Test p-values vs Persistence (Setup C, $h=1$)\nValues < 0.05 indicate statistical superiority", fontsize=11, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("City")

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_diebold_mariano.png", bbox_inches="tight")
    fig.savefig(out_dir / "fig4_diebold_mariano.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote fig4_diebold_mariano")


def plot_fig5_horizon_decay(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 5: Forecast decay from h=1 to h=3 across cities."""
    sub = df[df["setup"] == "C"].copy()
    sub["city"] = sub["city"].str.capitalize()
    sub["model"] = sub["model"].str.replace("_", " ").str.title().str.replace("Xgboost", "XGBoost")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, facecolor="white")
    axes[0].set_facecolor("white")
    axes[1].set_facecolor("white")
    for ax, h in zip(axes, [1, 3], strict=False):
        h_df = sub[sub["horizon"] == h]
        sns.barplot(data=h_df, x="city", y="mae", hue="model", ax=ax)
        ax.set_title(f"Horizon $h={h}$ ({'Next-Day' if h==1 else '3-Day Ahead'})", fontsize=11, fontweight="bold")
        ax.set_ylabel("MAE (AQI Points)" if ax == axes[0] else "")
        ax.set_xlabel("City")
        if ax != axes[0]:
            ax.legend().remove()
        else:
            ax.legend(title="Model", loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle("Degradation of Forecast Skill Across Lead Times ($h=1$ vs $h=3$)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_horizon_decay.png", bbox_inches="tight")
    fig.savefig(out_dir / "fig5_horizon_decay.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote fig5_horizon_decay")


def generate_figures(
    runs_csv: str | Path = "results/runs.csv",
    output_dir: str | Path = "results/figures",
    preds_dir: str | Path = "results/preds",
) -> None:
    """Generate all publication figures."""
    p_runs = Path(runs_csv)
    if not p_runs.exists():
        log.error("Runs CSV not found: %s", p_runs)
        return
    df = pd.read_csv(p_runs)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_preds = Path(preds_dir)

    plot_fig1_leakage_inflation(df, out_dir)
    plot_fig2_model_comparison_h1(df, out_dir)
    plot_fig3_forecast_trajectory(p_preds, out_dir)
    plot_fig4_diebold_mariano(df, out_dir)
    plot_fig5_horizon_decay(df, out_dir)
    log.info("All 5 publication figures generated in %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_figures()
