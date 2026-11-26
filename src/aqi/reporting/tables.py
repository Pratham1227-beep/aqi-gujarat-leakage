"""Generate camera-ready LaTeX and CSV tables from results/runs.csv."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

RUNS_CSV = Path("results/runs.csv")
TABLES_DIR = Path("results/tables")


def generate_table1_qual(out_dir: Path) -> None:
    """Table 1: City qualification, date ranges, record counts, and qualification status."""
    rows = [
        {"City": "Ahmedabad", "Source": "CPCB Bulletin + Kaggle", "Period": "2015--2023", "Days": 3167, "Valid AQI": 2538, "Valid PM2.5": 1149, "Status": "Qualified (A, B, C)"},
        {"City": "Chennai", "Source": "CPCB Bulletin + Kaggle", "Period": "2015--2023", "Days": 3167, "Valid AQI": 3075, "Valid PM2.5": 1792, "Status": "Qualified (A, B, C Anchor)"},
        {"City": "Gandhinagar", "Source": "CPCB Bulletin", "Period": "2019--2023", "Days": 1791, "Valid AQI": 1704, "Valid PM2.5": "N/A", "Status": "Qualified (Setup C only)"},
        {"City": "Ankleshwar", "Source": "CPCB Bulletin", "Period": "2019--2023", "Days": 1791, "Valid AQI": 1502, "Valid PM2.5": "N/A", "Status": "Qualified (Setup C only)"},
        {"City": "Vapi", "Source": "CPCB Bulletin", "Period": "2019--2023", "Days": 1782, "Valid AQI": 1349, "Valid PM2.5": "N/A", "Status": "Qualified (Setup C only)"},
        {"City": "Surat", "Source": "CPCB Bulletin", "Period": "2022--2023", "Days": 423, "Valid AQI": 327, "Valid PM2.5": "N/A", "Status": "Qualified (Setup C Fallback)"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table1_qual.csv", index=False)
    
    tex = df.to_latex(
        index=False,
        caption="City Qualification, Data Sources, and Evaluation Setups",
        label="tab:city_qualification",
        column_format="lcccccc",
    )
    # Fit cleanly to page width
    tex = tex.replace("\\begin{table}\n", "\\begin{table}[htbp]\n\\centering\n")
    tex = tex.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}")
    tex = tex.replace("\\end{tabular}", "\\end{tabular}}")
    (out_dir / "table1_qual.tex").write_text(tex, encoding="utf-8")
    log.info("Wrote table1_qual.tex and .csv")


def generate_table2_main(df: pd.DataFrame, out_dir: Path) -> None:
    """Table 2: Main benchmark table for Ahmedabad and Chennai under Setup C (h=1)."""
    sub = df[(df["setup"] == "C") & (df["horizon"] == 1) & (df["city"].isin(["ahmedabad", "chennai"]))].copy()
    
    model_order = ["persistence", "seasonal_naive", "moving_average", "ridge", "random_forest", "xgboost"]
    sub["model_rank"] = sub["model"].map(lambda x: model_order.index(x) if x in model_order else 99)
    sub = sub.sort_values(["city", "model_rank"])

    table_rows = []
    for _, r in sub.iterrows():
        dm_str = f"{r['dm_p']:.3f}" if pd.notna(r["dm_p"]) and r["model"] != "persistence" else "--"
        if r["dm_p"] < 0.05 and r["model"] != "persistence":
            dm_str += "$^*$"
        table_rows.append({
            "City": r["city"].capitalize(),
            "Model": r["model"].replace("_", " ").title().replace("Xgboost", "XGBoost"),
            "MAE": f"{r['mae']:.2f}",
            "RMSE": f"{r['rmse']:.2f}",
            "R2": f"{r['r2']:.3f}",
            "Skill": f"{r['skill']:.3f}",
            "Band Acc (\\%)": f"{r['band_acc']*100:.1f}",
            "DM p-value": dm_str,
        })
    res_df = pd.DataFrame(table_rows)
    res_df.to_csv(out_dir / "table2_main.csv", index=False)
    
    tex = res_df.to_latex(
        index=False,
        caption="Setup C Next-Day ($t+1$) Forecasting Benchmark vs Persistence Baseline. \\textit{Note:} Skill score denotes normalized MAE improvement over Persistence ($1 - \\text{MAE}_{\\text{model}}/\\text{MAE}_{\\text{persistence}}$); DM $p$-value evaluates significance vs Persistence.",
        label="tab:main_benchmark_h1",
        column_format="llcccccc",
        escape=False,
    )
    tex = tex.replace("\\begin{table}\n", "\\begin{table}[htbp]\n\\centering\n")
    tex = tex.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}")
    tex = tex.replace("\\end{tabular}", "\\end{tabular}}")
    (out_dir / "table2_main.tex").write_text(tex, encoding="utf-8")
    log.info("Wrote table2_main.tex and .csv")


def generate_table3_inflation(df: pd.DataFrame, out_dir: Path) -> None:
    """Table 3: Leakage Inflation Comparison across Setups A, B, and C."""
    # Ahmedabad
    ahm_a = df[(df["city"] == "ahmedabad") & (df["setup"] == "A")].groupby("model").agg({"mae": "mean", "r2": "mean"}).reset_index()
    ahm_b = df[(df["city"] == "ahmedabad") & (df["setup"] == "B")].groupby("model").agg({"mae": "mean", "r2": "mean"}).reset_index()
    ahm_c = df[(df["city"] == "ahmedabad") & (df["setup"] == "C") & (df["horizon"] == 1)].groupby("model").agg({"mae": "mean", "r2": "mean"}).reset_index()

    models = ["ridge", "random_forest", "xgboost"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{l cccccc c}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Setup A (Random)}} & \multicolumn{2}{c}{\textbf{Setup B (Chrono)}} & \multicolumn{2}{c}{\textbf{Setup C (Rolling)}} & \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"\textbf{Model} & \textbf{MAE} & $\mathbf{R^2}$ & \textbf{MAE} & $\mathbf{R^2}$ & \textbf{MAE} & $\mathbf{R^2}$ & $\Delta R^2 \text{ (A$-$B)}$ \\",
        r"\midrule",
    ]
    for m in models:
        r_a = ahm_a[ahm_a["model"] == m].iloc[0] if len(ahm_a[ahm_a["model"] == m]) else None
        r_b = ahm_b[ahm_b["model"] == m].iloc[0] if len(ahm_b[ahm_b["model"] == m]) else None
        r_c = ahm_c[ahm_c["model"] == m].iloc[0] if len(ahm_c[ahm_c["model"] == m]) else None
        
        name = m.replace("_", " ").title().replace("Xgboost", "XGBoost")
        mae_a = f"{r_a['mae']:.2f}" if r_a is not None else "--"
        r2_a = f"{r_a['r2']:.3f}" if r_a is not None else "--"
        mae_b = f"{r_b['mae']:.2f}" if r_b is not None else "--"
        r2_b = f"{r_b['r2']:.3f}" if r_b is not None else "--"
        mae_c = f"{r_c['mae']:.2f}" if r_c is not None else "--"
        r2_c = f"{r_c['r2']:.3f}" if r_c is not None else "--"
        delta = f"{(r_a['r2'] - r_b['r2']):+.3f}" if (r_a is not None and r_b is not None) else "--"
        lines.append(f"{name} & {mae_a} & {r2_a} & {mae_b} & {r2_b} & {mae_c} & {r2_c} & {delta} \\\\")
        
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\caption{Empirical Leakage Inflation in Ahmedabad: Setup A (Random) vs Setup B (Chrono) vs Setup C (Rolling-Origin)}",
        r"\label{tab:leakage_inflation}",
        r"\end{table}",
    ]
    tex = "\n".join(lines) + "\n"
    (out_dir / "table3_inflation.tex").write_text(tex, encoding="utf-8")
    log.info("Wrote table3_inflation.tex and .csv")


def generate_table4_multicity(df: pd.DataFrame, out_dir: Path) -> None:
    """Table 4: Cross-city performance across all Gujarat cities and Chennai for Setup C."""
    sub = df[(df["setup"] == "C") & (df["horizon"] == 1)].copy()
    sub = sub.drop_duplicates(subset=["city", "model"], keep="last")
    
    summary = sub.groupby(["city", "model"])[["mae", "rmse", "r2", "skill", "dm_p"]].first().reset_index()
    summary["city"] = summary["city"].str.capitalize()
    summary["model"] = summary["model"].str.replace("_", " ").str.title().str.replace("Xgboost", "XGBoost")
    summary.to_csv(out_dir / "table4_multicity.csv", index=False)

    # Pivot table: MAE per city and model
    pivot_mae = sub.pivot(index="city", columns="model", values="mae").round(2)
    pivot_mae.index = pivot_mae.index.str.capitalize()
    pivot_mae.index.name = "City"
    pivot_mae.columns = [c.replace("_", " ").title().replace("Xgboost", "XGBoost") for c in pivot_mae.columns]
    pivot_mae.columns.name = None
    
    tex = pivot_mae.to_latex(
        caption="Cross-City Comparison of MAE under Setup C ($h=1$ Next-Day Forecast)",
        label="tab:cross_city_mae",
        float_format="%.2f",
        escape=False,
    )
    tex = tex.replace("\\begin{table}\n", "\\begin{table}[htbp]\n\\centering\n")
    tex = tex.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}")
    tex = tex.replace("\\end{tabular}", "\\end{tabular}}")
    (out_dir / "table4_multicity.tex").write_text(tex, encoding="utf-8")
    log.info("Wrote table4_multicity.tex and .csv")


def generate_table5_horizons(df: pd.DataFrame, out_dir: Path) -> None:
    """Table 5: Forecast decay from h=1 to h=3 across cities."""
    sub = df[df["setup"] == "C"].copy()
    sub = sub.drop_duplicates(subset=["city", "model", "horizon"], keep="last")
    pivot = sub.pivot(index=["city", "model"], columns="horizon", values="mae").round(2).reset_index()
    pivot["city"] = pivot["city"].str.capitalize()
    pivot["model"] = pivot["model"].str.replace("_", " ").str.title().str.replace("Xgboost", "XGBoost")
    pivot = pivot.rename(columns={1: "MAE (h=1)", 3: "MAE (h=3)"})
    pivot["Degradation (\\%)"] = ((pivot["MAE (h=3)"] - pivot["MAE (h=1)"]) / pivot["MAE (h=1)"] * 100).round(1)
    
    pivot.to_csv(out_dir / "table5_horizons.csv", index=False)
    tex = pivot.to_latex(
        index=False,
        caption="Forecast Horizon Degradation: Error Growth from Next-Day ($h=1$) to 3-Day ($h=3$) Lead Times",
        label="tab:horizon_degradation",
        column_format="llccc",
        float_format="%.2f",
        escape=False,
    )
    tex = tex.replace("\\begin{table}\n", "\\begin{table}[htbp]\n\\centering\n")
    tex = tex.replace("\\begin{tabular}", "\\resizebox{0.75\\linewidth}{!}{\\begin{tabular}")
    tex = tex.replace("\\end{tabular}", "\\end{tabular}}")
    (out_dir / "table5_horizons.tex").write_text(tex, encoding="utf-8")
    log.info("Wrote table5_horizons.tex and .csv")


def generate_tables(
    runs_csv: str | Path = "results/runs.csv",
    output_dir: str | Path = "results/tables",
) -> None:
    """Generate all LaTeX and CSV tables from experimental runs."""
    path = Path(runs_csv)
    if not path.exists():
        log.error("Runs CSV not found: %s", path)
        return
    df = pd.read_csv(path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_table1_qual(out_dir)
    generate_table2_main(df, out_dir)
    generate_table3_inflation(df, out_dir)
    generate_table4_multicity(df, out_dir)
    generate_table5_horizons(df, out_dir)
    log.info("All 5 publication tables generated in %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_tables()
