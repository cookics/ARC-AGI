from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "analysis-python-complexity"
CURVE_DIR = ROOT / "analysis-efficiency" / "arc_training_difficulty"
JOIN_PATH = ANALYSIS_DIR / "arc1_dsl_gptoss_gemini_join.csv"
METRIC_CATALOG_PATH = ANALYSIS_DIR / "arc1_dsl_metric_catalog.csv"
CURVE_PATH = CURVE_DIR / "arc_training_difficulty.csv"
CURVE_SUMMARY_PATH = CURVE_DIR / "arc_training_summary.json"

JOIN_OUT_PATH = ANALYSIS_DIR / "compressarc_curve_difficulty_join.csv"
ALL_CORR_OUT_PATH = ANALYSIS_DIR / "compressarc_curve_difficulty_all_metric_correlations.csv"
LOCKED_OUT_PATH = ANALYSIS_DIR / "compressarc_curve_difficulty_locked_correlations.csv"
BEST_BROAD_OUT_PATH = ANALYSIS_DIR / "compressarc_curve_difficulty_best_broad_metrics.csv"
SUMMARY_OUT_PATH = ANALYSIS_DIR / "compressarc_curve_difficulty_summary.json"
REPORT_OUT_PATH = ANALYSIS_DIR / "compressarc_curve_difficulty_report.md"


EXTRA_METRICS = {
    "complexity_pc1_score": "expanded_composite",
    "ast_node_count": "expanded_python_static",
    "function_count": "expanded_python_static",
    "nonblank_lines": "expanded_python_static",
    "token_count": "expanded_python_static",
    "call_count_static": "expanded_python_static",
    "log1p_branch_opcode_count_dynamic": "expanded_dynamic_execution",
    "log1p_cyclomatic_complexity": "expanded_python_static",
}

LOCKED_METRICS = [
    "complexity_pc1_score",
    "dsl_complexity_pc1",
    "ast_node_count",
    "log1p_branch_opcode_count_dynamic",
    "log1p_cyclomatic_complexity",
    "geometry_op_count",
    "solver_opcode_count_dynamic",
    "function_count",
]


def metric_categories(joined: pd.DataFrame) -> dict[str, str]:
    catalog = pd.read_csv(METRIC_CATALOG_PATH)
    categories = {row["metric"]: row["category"] for _, row in catalog.iterrows()}
    categories.update(EXTRA_METRICS)
    return {metric: category for metric, category in categories.items() if metric in joined.columns}


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    if x.nunique() < 2 or y.nunique() < 2:
        return (np.nan, np.nan)
    if method == "pearson":
        result = pearsonr(x, y)
    else:
        result = spearmanr(x, y)
    return (float(result.statistic), float(result.pvalue))


def compute_all_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, category in metric_categories(joined).items():
        subset = joined[[metric, "difficulty_score", "first_hit_step"]].dropna()
        if subset.empty:
            continue
        pearson_r, pearson_p = safe_corr(subset[metric], subset["difficulty_score"], "pearson")
        spearman_rho, spearman_p = safe_corr(subset[metric], subset["difficulty_score"], "spearman")
        if np.isnan(pearson_r):
            continue
        rows.append(
            {
                "metric": metric,
                "category": category,
                "n": int(len(subset)),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False).reset_index(drop=True)


def write_summary(joined_all: pd.DataFrame, joined_solved: pd.DataFrame, all_corr: pd.DataFrame) -> None:
    def metric_row(metric: str) -> dict[str, float]:
        row = all_corr.loc[all_corr["metric"] == metric].iloc[0]
        return {
            "pearson_r": float(row["pearson_r"]),
            "pearson_p": float(row["pearson_p"]),
            "spearman_rho": float(row["spearman_rho"]),
            "spearman_p": float(row["spearman_p"]),
        }

    best_broad = all_corr.loc[all_corr["category"] != "dsl_primitive_usage"].head(1).to_dict(orient="records")[0]
    summary = {
        "curve_source": str(CURVE_PATH),
        "curve_summary_source": str(CURVE_SUMMARY_PATH),
        "task_count_overlap_all": int(len(joined_all)),
        "task_count_overlap_solved": int(len(joined_solved)),
        "solved_overlap_rate": float(len(joined_solved) / len(joined_all)),
        "difficulty_score_note": "Solved-only continuous difficulty. Lower is earlier first-hit; higher is later first-hit.",
        "complexity_pc1_score": metric_row("complexity_pc1_score"),
        "dsl_complexity_pc1": metric_row("dsl_complexity_pc1"),
        "ast_node_count": metric_row("ast_node_count"),
        "solver_opcode_count_dynamic": metric_row("solver_opcode_count_dynamic"),
        "best_broad_metric": best_broad,
    }
    SUMMARY_OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_report(joined_all: pd.DataFrame, joined_solved: pd.DataFrame, all_corr: pd.DataFrame) -> None:
    top5 = all_corr.loc[all_corr["category"] != "dsl_primitive_usage"].head(5)
    pc1 = all_corr.loc[all_corr["metric"] == "complexity_pc1_score"].iloc[0]
    dsl_pc1 = all_corr.loc[all_corr["metric"] == "dsl_complexity_pc1"].iloc[0]

    lines = [
        "# CompressARC Curve-Based Difficulty Correlations",
        "",
        f"- Input difficulty table: {CURVE_PATH}",
        f"- Overlap with validated ARC-1 complexity table: {len(joined_all)} tasks.",
        f"- Solved-by-curve overlap used for the continuous analysis: {len(joined_solved)} tasks ({len(joined_solved)/len(joined_all):.1%}).",
        "- `difficulty_score` is the normalized first step where the true solution first appears in the top-2 oracle picks.",
        "- On the solved-only subset, `difficulty_score` and `first_hit_step` are perfectly rank-equivalent, so they give the same ordering.",
        "",
        "## Locked Results",
        "",
        f"- `complexity_pc1_score`: r = {pc1['pearson_r']:.3f}, p = {pc1['pearson_p']:.3g}, rho = {pc1['spearman_rho']:.3f}.",
        f"- `dsl_complexity_pc1`: r = {dsl_pc1['pearson_r']:.3f}, p = {dsl_pc1['pearson_p']:.3g}, rho = {dsl_pc1['spearman_rho']:.3f}.",
        "",
        "## Best Broad Metrics",
        "",
        top5.to_csv(index=False),
        "",
        "Interpretation: the solved-only continuous CompressARC difficulty signal is more complexity-linked than the final binary solved/not-solved label, but it is still only modest in size. The strongest broad single metrics are structural or dynamic counts rather than the global expanded complexity PC1.",
        "",
    ]
    REPORT_OUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    curve = pd.read_csv(CURVE_PATH)
    complexity = pd.read_csv(JOIN_PATH)

    joined_all = complexity.merge(
        curve[["task_id", "first_hit_step", "solved_by_end", "difficulty_score"]],
        on="task_id",
        how="inner",
    )
    joined_solved = joined_all.loc[joined_all["solved_by_end"]].copy()

    all_corr = compute_all_correlations(joined_solved)
    locked = (
        all_corr.loc[all_corr["metric"].isin(LOCKED_METRICS)]
        .set_index("metric")
        .loc[[metric for metric in LOCKED_METRICS if metric in all_corr["metric"].values]]
        .reset_index()
    )
    best_broad = all_corr.loc[all_corr["category"] != "dsl_primitive_usage"].head(12).reset_index(drop=True)

    joined_solved.to_csv(JOIN_OUT_PATH, index=False)
    all_corr.to_csv(ALL_CORR_OUT_PATH, index=False)
    locked.to_csv(LOCKED_OUT_PATH, index=False)
    best_broad.to_csv(BEST_BROAD_OUT_PATH, index=False)
    write_summary(joined_all, joined_solved, all_corr)
    write_report(joined_all, joined_solved, all_corr)
    print(f"Done. Outputs saved in {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
