import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent
HUMAN_TABLE_PATH = ROOT_DIR / "analysis-human" / "analysis" / "tables" / "public_eval_human_vs_models.csv"
APPROVED_LLM_PATH = BASE_DIR / "approved_llm_complexity_join.csv"

COMPLEXITY_METRICS = [
    "nonblank_lines",
    "token_count",
    "ast_node_count",
    "function_count",
    "call_count_static",
    "branch_node_count",
    "cyclomatic_complexity",
    "max_nesting_depth",
    "gzip_bytes",
    "halstead_volume",
    "halstead_effort",
    "input_cells_total",
    "output_cells_total",
    "elapsed_ms_total",
    "elapsed_ms_per_test",
    "opcode_count_dynamic",
    "branch_opcode_count_dynamic",
    "peak_memory_bytes",
    "complexity_pc1_score",
    "log1p_ast_node_count",
    "log1p_cyclomatic_complexity",
    "log1p_elapsed_ms_total",
    "log1p_elapsed_ms_per_test",
    "log1p_peak_memory_bytes",
]

SELECTED_PLOT_METRICS = [
    "ast_node_count",
    "token_count",
    "cyclomatic_complexity",
    "complexity_pc1_score",
    "elapsed_ms_total",
    "peak_memory_bytes",
    "input_cells_total",
    "max_nesting_depth",
]

HUMAN_WEIGHTED_COLS = [
    "solve_rate",
    "mean_duration_seconds",
    "mean_submissions",
    "mean_pred_prob",
    "outfit",
    "difficulty",
    "point_biserial",
    "ability_gap",
    "lm_mean",
    "lm_best_across_models",
    "lm_best_single_model",
    "gap_vs_lm_mean",
    "gap_vs_best_single_model",
    "gap_vs_oracle",
]

sns.set_theme(style="whitegrid", context="talk")


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x, y):
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return safe_corr(x_rank, y_rank)


def rsquared(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) == 0:
        return np.nan
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if sst == 0:
        return np.nan
    return 1.0 - (sse / sst)


def weighted_average(series, weights):
    mask = series.notna() & weights.notna()
    if not mask.any():
        return np.nan
    weight_sum = float(weights[mask].sum())
    if weight_sum <= 0:
        return np.nan
    return float(np.average(series[mask], weights=weights[mask]))


def aggregate_human_task_table(human_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_id, group in human_pairs.groupby("task_ID"):
        weights = group["attempts"].astype(float)
        row = {
            "task_id": task_id,
            "human_pair_count": int(len(group)),
            "human_attempts_total": int(group["attempts"].sum()),
            "human_solve_count_total": int(group["solve_count"].sum()),
            "human_solve_rate_weighted": float(group["solve_count"].sum() / group["attempts"].sum()),
            "human_solve_rate_mean": float(group["solve_rate"].mean()),
            "human_max_test_index": int(group["test_index"].max()),
        }
        for column in HUMAN_WEIGHTED_COLS:
            row[f"{column}_weighted"] = weighted_average(group[column], weights)
            if group[column].notna().any():
                row[f"{column}_mean"] = float(group[column].mean())
            else:
                row[f"{column}_mean"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)


def bootstrap_delta_corr(df: pd.DataFrame, metric: str, y_a: str, y_b: str, n_boot: int = 5000, seed: int = 0):
    subset = df[[metric, y_a, y_b]].dropna().reset_index(drop=True)
    if len(subset) < 6:
        return {
            "n": len(subset),
            "r_a": np.nan,
            "r_b": np.nan,
            "delta_r": np.nan,
            "delta_ci_low": np.nan,
            "delta_ci_high": np.nan,
        }

    r_a = safe_corr(subset[metric], subset[y_a])
    r_b = safe_corr(subset[metric], subset[y_b])
    delta_r = r_a - r_b

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(subset), size=len(subset))
        sample = subset.iloc[idx]
        sample_r_a = safe_corr(sample[metric], sample[y_a])
        sample_r_b = safe_corr(sample[metric], sample[y_b])
        if np.isfinite(sample_r_a) and np.isfinite(sample_r_b):
            deltas.append(sample_r_a - sample_r_b)
    if deltas:
        ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    else:
        ci_low, ci_high = np.nan, np.nan
    return {
        "n": len(subset),
        "r_a": r_a,
        "r_b": r_b,
        "delta_r": delta_r,
        "delta_ci_low": float(ci_low),
        "delta_ci_high": float(ci_high),
    }


def fit_linear_and_quadratic(df: pd.DataFrame, x_col: str, y_col: str):
    subset = df[[x_col, y_col]].dropna().copy()
    x = subset[x_col].to_numpy(dtype=float)
    y = subset[y_col].to_numpy(dtype=float)
    linear_coef = np.polyfit(x, y, deg=1)
    linear_pred = np.polyval(linear_coef, x)
    quad_coef = np.polyfit(x, y, deg=2)
    quad_pred = np.polyval(quad_coef, x)
    return {
        "n": len(subset),
        "linear_coef": linear_coef.tolist(),
        "quadratic_coef": quad_coef.tolist(),
        "linear_r2": rsquared(y, linear_pred),
        "quadratic_r2": rsquared(y, quad_pred),
    }


def build_correlation_table(df: pd.DataFrame, human_col: str, llm_col: str) -> pd.DataFrame:
    rows = []
    for metric in COMPLEXITY_METRICS:
        delta = bootstrap_delta_corr(df, metric, llm_col, human_col)
        human_subset = df[[metric, human_col]].dropna()
        llm_subset = df[[metric, llm_col]].dropna()
        rows.append(
            {
                "metric": metric,
                "n_human": len(human_subset),
                "pearson_human": safe_corr(human_subset[metric], human_subset[human_col]),
                "spearman_human": spearman_corr(human_subset[metric], human_subset[human_col]),
                "n_llm": len(llm_subset),
                "pearson_llm": safe_corr(llm_subset[metric], llm_subset[llm_col]),
                "spearman_llm": spearman_corr(llm_subset[metric], llm_subset[llm_col]),
                "delta_llm_minus_human": delta["delta_r"],
                "delta_ci_low": delta["delta_ci_low"],
                "delta_ci_high": delta["delta_ci_high"],
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("delta_llm_minus_human", ascending=False).reset_index(drop=True)


def build_residual_table(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[["task_id", "difficulty_weighted", "logit_difficulty_all"]].dropna().copy()
    x = subset["logit_difficulty_all"].to_numpy(dtype=float)
    y = subset["difficulty_weighted"].to_numpy(dtype=float)
    human_linear = np.polyfit(x, y, deg=1)
    subset["human_residual_after_llm"] = y - np.polyval(human_linear, x)

    x_rev = subset["difficulty_weighted"].to_numpy(dtype=float)
    y_rev = subset["logit_difficulty_all"].to_numpy(dtype=float)
    llm_linear = np.polyfit(x_rev, y_rev, deg=1)
    subset["llm_residual_after_human"] = y_rev - np.polyval(llm_linear, x_rev)

    merged = df.merge(subset[["task_id", "human_residual_after_llm", "llm_residual_after_human"]], on="task_id", how="left")

    probe_cols = [
        "gap_vs_lm_mean_weighted",
        "thinking_advantage",
        "mean_duration_seconds_weighted",
        "ast_node_count",
        "cyclomatic_complexity",
        "complexity_pc1_score",
        "elapsed_ms_total",
        "log1p_elapsed_ms_total",
        "peak_memory_bytes",
        "input_cells_total",
        "output_cells_total",
    ]

    rows = []
    for residual_col in ["human_residual_after_llm", "llm_residual_after_human"]:
        for probe in probe_cols:
            sub = merged[[probe, residual_col]].dropna()
            rows.append(
                {
                    "residual_target": residual_col,
                    "probe_metric": probe,
                    "n": len(sub),
                    "pearson": safe_corr(sub[probe], sub[residual_col]),
                    "spearman": spearman_corr(sub[probe], sub[residual_col]),
                }
            )
    return pd.DataFrame(rows).sort_values(["residual_target", "pearson"], ascending=[True, False]).reset_index(drop=True)


def plot_thinking_advantage_curve(df: pd.DataFrame, output_path: Path):
    subset = df[["logit_difficulty_all", "thinking_advantage", "dataset_key"]].dropna().copy()
    fit = fit_linear_and_quadratic(subset, "logit_difficulty_all", "thinking_advantage")

    x = subset["logit_difficulty_all"].to_numpy(dtype=float)
    y = subset["thinking_advantage"].to_numpy(dtype=float)
    x_grid = np.linspace(x.min(), x.max(), 200)
    linear_y = np.polyval(np.array(fit["linear_coef"]), x_grid)
    quad_y = np.polyval(np.array(fit["quadratic_coef"]), x_grid)

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(
        data=subset,
        x="logit_difficulty_all",
        y="thinking_advantage",
        hue="dataset_key",
        palette={"arc_agi_1_eval": "#1f77b4", "arc_agi_2_eval": "#d62728"},
        s=90,
        ax=ax,
    )
    ax.plot(x_grid, linear_y, color="#444444", linewidth=2, linestyle="--", label=f"Linear fit ($R^2$={fit['linear_r2']:.2f})")
    ax.plot(x_grid, quad_y, color="#111111", linewidth=3, label=f"Quadratic fit ($R^2$={fit['quadratic_r2']:.2f})")
    ax.set_title("Thinking Advantage Shrinks on the Hardest Items")
    ax.set_xlabel("LLM Logit Difficulty")
    ax.set_ylabel("Thinking - Standard Pass Rate")
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_human_vs_llm_bars(corr_table: pd.DataFrame, output_path: Path):
    subset = corr_table.set_index("metric").loc[SELECTED_PLOT_METRICS].reset_index()
    plot_df = pd.DataFrame(
        {
            "metric": np.repeat(subset["metric"].to_numpy(), 2),
            "target": ["Human difficulty", "LLM logit difficulty"] * len(subset),
            "correlation": np.ravel(np.column_stack([subset["pearson_human"], subset["pearson_llm"]])),
        }
    )
    label_map = {
        "ast_node_count": "AST nodes",
        "token_count": "Tokens",
        "cyclomatic_complexity": "Cyclomatic",
        "complexity_pc1_score": "Complexity PC1",
        "elapsed_ms_total": "Runtime ms",
        "peak_memory_bytes": "Peak memory",
        "input_cells_total": "Input cells",
        "max_nesting_depth": "Max nesting",
    }
    plot_df["metric_label"] = plot_df["metric"].map(label_map)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=plot_df,
        y="metric_label",
        x="correlation",
        hue="target",
        palette=["#2ca02c", "#9467bd"],
        orient="h",
        ax=ax,
    )
    ax.axvline(0.0, color="#555555", linewidth=1)
    ax.set_title("Structural Solver Complexity Aligns More with LLM than Human Difficulty")
    ax.set_xlabel("Pearson r on ARC-2 Eval Overlap Tasks")
    ax.set_ylabel("")
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_human_vs_llm_scatter(df: pd.DataFrame, output_path: Path):
    subset = df[
        [
            "task_id",
            "difficulty_weighted",
            "logit_difficulty_all",
            "mean_duration_seconds_weighted",
            "complexity_pc1_score",
        ]
    ].dropna()

    fig, ax = plt.subplots(figsize=(11, 8))
    scatter = ax.scatter(
        subset["difficulty_weighted"],
        subset["logit_difficulty_all"],
        s=30 + (subset["mean_duration_seconds_weighted"] / subset["mean_duration_seconds_weighted"].max()) * 300,
        c=subset["complexity_pc1_score"],
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidth=0.5,
    )
    diagonal_min = min(subset["difficulty_weighted"].min(), subset["logit_difficulty_all"].min())
    diagonal_max = max(subset["difficulty_weighted"].max(), subset["logit_difficulty_all"].max())
    ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], color="#444444", linestyle="--", linewidth=1.5)
    for _, row in subset.sort_values("logit_difficulty_all", ascending=False).head(4).iterrows():
        ax.annotate(row["task_id"], (row["difficulty_weighted"], row["logit_difficulty_all"]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_title("ARC-2 Eval Overlap: Human and LLM Difficulty Partly Shared")
    ax.set_xlabel("Human Item Difficulty (Weighted Task Aggregate)")
    ax.set_ylabel("LLM Logit Difficulty")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Solver Complexity PC1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    llm = pd.read_csv(APPROVED_LLM_PATH)
    human_pairs = pd.read_csv(HUMAN_TABLE_PATH)

    arc2_overlap_source = llm[llm["dataset_key"] == "arc_agi_2_eval"].copy()
    human_task = aggregate_human_task_table(human_pairs)
    overlap = arc2_overlap_source.merge(human_task, on="task_id", how="inner").copy()
    overlap["human_minus_llm_pass_all"] = overlap["human_solve_rate_weighted"] - overlap["pass_rate_all"]
    overlap["human_minus_llm_pass_thinking"] = overlap["human_solve_rate_weighted"] - overlap["pass_rate_thinking"]
    overlap["human_minus_llm_logit_difficulty"] = overlap["difficulty_weighted"] - overlap["logit_difficulty_all"]

    overlap_path = BASE_DIR / "human_llm_overlap_tasks.csv"
    overlap.sort_values("task_id").to_csv(overlap_path, index=False)

    well_sampled = overlap[overlap["human_attempts_total"] >= 8].copy()
    well_sampled.to_csv(BASE_DIR / "human_llm_overlap_tasks_min8.csv", index=False)

    corr_table = build_correlation_table(overlap, human_col="difficulty_weighted", llm_col="logit_difficulty_all")
    corr_table.to_csv(BASE_DIR / "human_llm_complexity_correlation_comparison.csv", index=False)

    corr_table_min8 = build_correlation_table(well_sampled, human_col="difficulty_weighted", llm_col="logit_difficulty_all")
    corr_table_min8.to_csv(BASE_DIR / "human_llm_complexity_correlation_comparison_min8.csv", index=False)

    outcome_pairs = [
        ("difficulty_weighted", "logit_difficulty_all"),
        ("difficulty_weighted", "rasch_difficulty_all_models_pooled"),
        ("human_solve_rate_weighted", "pass_rate_all"),
        ("human_solve_rate_weighted", "pass_rate_thinking"),
        ("point_biserial_weighted", "item_total_corr"),
        ("outfit_weighted", "rasch_outfit"),
        ("outfit_weighted", "rasch_rmsea_x2"),
        ("mean_duration_seconds_weighted", "elapsed_ms_total"),
        ("mean_duration_seconds_weighted", "logit_difficulty_all"),
        ("mean_duration_seconds_weighted", "difficulty_weighted"),
        ("gap_vs_lm_mean_weighted", "thinking_advantage"),
        ("gap_vs_lm_mean_weighted", "pass_rate_all"),
        ("gap_vs_lm_mean_weighted", "logit_difficulty_all"),
    ]
    outcome_rows = []
    for human_col, llm_col in outcome_pairs:
        subset = overlap[[human_col, llm_col]].dropna()
        outcome_rows.append(
            {
                "left_metric": human_col,
                "right_metric": llm_col,
                "n": len(subset),
                "pearson": safe_corr(subset[human_col], subset[llm_col]),
                "spearman": spearman_corr(subset[human_col], subset[llm_col]),
            }
        )
    outcome_table = pd.DataFrame(outcome_rows).sort_values("pearson", ascending=False)
    outcome_table.to_csv(BASE_DIR / "human_llm_outcome_alignment.csv", index=False)

    residual_table = build_residual_table(overlap)
    residual_table.to_csv(BASE_DIR / "human_llm_residual_correlations.csv", index=False)

    thinking_all = llm[["dataset_key", "task_id", "logit_difficulty_all", "thinking_advantage", "thinking_logit_advantage", "rasch_difficulty_all_models_pooled"]].dropna().copy()
    thinking_fit = fit_linear_and_quadratic(thinking_all, "logit_difficulty_all", "thinking_advantage")
    thinking_fit_rasch = fit_linear_and_quadratic(thinking_all, "rasch_difficulty_all_models_pooled", "thinking_advantage")

    tertile = thinking_all.copy()
    tertile["difficulty_tertile"] = pd.qcut(tertile["logit_difficulty_all"], 3, duplicates="drop")
    tertile_summary = (
        tertile.groupby("difficulty_tertile", observed=False)[["thinking_advantage", "thinking_logit_advantage"]]
        .mean()
        .reset_index()
    )
    tertile_summary.to_csv(BASE_DIR / "thinking_advantage_tertiles.csv", index=False)

    plot_thinking_advantage_curve(thinking_all, BASE_DIR / "chart_thinking_advantage_vs_difficulty.png")
    plot_human_vs_llm_bars(corr_table, BASE_DIR / "chart_human_vs_llm_complexity_bars.png")
    plot_human_vs_llm_scatter(overlap, BASE_DIR / "chart_human_vs_llm_difficulty_overlap.png")

    headline = {
        "overlap_tasks_arc2_eval": int(len(overlap)),
        "well_sampled_overlap_tasks_min8_attempts": int(len(well_sampled)),
        "human_vs_llm_difficulty_pearson": safe_corr(overlap["difficulty_weighted"], overlap["logit_difficulty_all"]),
        "human_vs_llm_difficulty_rasch_pearson": safe_corr(overlap["difficulty_weighted"], overlap["rasch_difficulty_all_models_pooled"]),
        "human_vs_llm_solve_rate_pearson": safe_corr(overlap["human_solve_rate_weighted"], overlap["pass_rate_all"]),
        "human_difficulty_vs_human_duration_pearson": safe_corr(overlap["difficulty_weighted"], overlap["mean_duration_seconds_weighted"]),
        "llm_logit_difficulty_vs_human_duration_pearson": safe_corr(overlap["logit_difficulty_all"], overlap["mean_duration_seconds_weighted"]),
        "thinking_advantage_vs_llm_logit_difficulty_pearson": safe_corr(thinking_all["thinking_advantage"], thinking_all["logit_difficulty_all"]),
        "thinking_advantage_vs_llm_logit_difficulty_linear_r2": thinking_fit["linear_r2"],
        "thinking_advantage_vs_llm_logit_difficulty_quadratic_r2": thinking_fit["quadratic_r2"],
        "thinking_advantage_vs_llm_rasch_difficulty_linear_r2": thinking_fit_rasch["linear_r2"],
        "thinking_advantage_vs_llm_rasch_difficulty_quadratic_r2": thinking_fit_rasch["quadratic_r2"],
    }

    corr_lookup = corr_table.set_index("metric")
    summary = {
        "headline": headline,
        "selected_metric_comparison": {},
        "top_positive_llm_minus_human_delta": corr_table.head(8).to_dict(orient="records"),
        "top_negative_llm_minus_human_delta": corr_table.tail(8).to_dict(orient="records"),
    }
    for metric in ["ast_node_count", "token_count", "cyclomatic_complexity", "complexity_pc1_score", "elapsed_ms_total", "peak_memory_bytes"]:
        row = corr_lookup.loc[metric]
        summary["selected_metric_comparison"][metric] = {
            "pearson_human": float(row["pearson_human"]),
            "pearson_llm": float(row["pearson_llm"]),
            "delta_llm_minus_human": float(row["delta_llm_minus_human"]),
            "delta_ci_low": float(row["delta_ci_low"]),
            "delta_ci_high": float(row["delta_ci_high"]),
        }

    summary_path = BASE_DIR / "human_llm_difference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Human vs LLM Solver-Complexity Comparison",
        "",
        "## Scope",
        "",
        "- Human source: `analysis-human/analysis/tables/public_eval_human_vs_models.csv`",
        "- LLM source: `analysis-python-complexity/approved_llm_complexity_join.csv`",
        "- Comparison scope: approved ARC-AGI-2 eval tasks that also appear in the public human testing table",
        f"- Overlap: {len(overlap)} tasks total, {len(well_sampled)} with at least 8 total human attempts",
        "",
        "## Headline Findings",
        "",
        f"- Human and LLM difficulty are only moderately aligned on the overlap: `r = {headline['human_vs_llm_difficulty_pearson']:.3f}` for human difficulty vs LLM logit difficulty, and `r = {headline['human_vs_llm_difficulty_rasch_pearson']:.3f}` vs pooled Rasch difficulty.",
        f"- Human solve rate and LLM pass rate are similarly only moderately aligned: `r = {headline['human_vs_llm_solve_rate_pearson']:.3f}`.",
        f"- Human item difficulty is tied to human time cost: `r = {headline['human_difficulty_vs_human_duration_pearson']:.3f}` with weighted mean human duration.",
        f"- LLM difficulty is almost unrelated to human time cost on the same tasks: `r = {headline['llm_logit_difficulty_vs_human_duration_pearson']:.3f}`.",
        "",
        "## Thinking-Advantage Pattern",
        "",
        f"- Across all approved eval overlap items, `thinking_advantage` falls as LLM difficulty rises: Pearson `r = {headline['thinking_advantage_vs_llm_logit_difficulty_pearson']:.3f}` against logit difficulty.",
        f"- A quadratic fit is clearly better than a straight line here: linear `R^2 = {headline['thinking_advantage_vs_llm_logit_difficulty_linear_r2']:.3f}` vs quadratic `R^2 = {headline['thinking_advantage_vs_llm_logit_difficulty_quadratic_r2']:.3f}`.",
        "- Interpretation: thinking models gain the most on medium-hard items, but the gap compresses on the hardest items where both standard and thinking models often fail.",
        "",
        "## Complexity Contrast",
        "",
        f"- `ast_node_count`: human difficulty `r = {summary['selected_metric_comparison']['ast_node_count']['pearson_human']:.3f}`, LLM difficulty `r = {summary['selected_metric_comparison']['ast_node_count']['pearson_llm']:.3f}`.",
        f"- `token_count`: human difficulty `r = {summary['selected_metric_comparison']['token_count']['pearson_human']:.3f}`, LLM difficulty `r = {summary['selected_metric_comparison']['token_count']['pearson_llm']:.3f}`.",
        f"- `cyclomatic_complexity`: human difficulty `r = {summary['selected_metric_comparison']['cyclomatic_complexity']['pearson_human']:.3f}`, LLM difficulty `r = {summary['selected_metric_comparison']['cyclomatic_complexity']['pearson_llm']:.3f}`.",
        f"- `complexity_pc1_score`: human difficulty `r = {summary['selected_metric_comparison']['complexity_pc1_score']['pearson_human']:.3f}`, LLM difficulty `r = {summary['selected_metric_comparison']['complexity_pc1_score']['pearson_llm']:.3f}`.",
        f"- Runtime burden does not explain human difficulty especially well either, but it matters more for human-vs-model gap and residual differences than structural size does.",
        "",
        "## Working Hypotheses Supported by Current Data",
        "",
        "- Structural solver complexity looks more like an LLM difficulty signal than a human difficulty signal in this overlap slice.",
        "- Human difficulty appears to be closer to time-on-task and interactive search burden than to the amount of code needed in a final solver.",
        "- The hardest model items are not the items with the biggest thinking-model gain; instead, thinking advantage seems to collapse on the hardest tasks.",
        "- Human-vs-LLM differences are real but only moderately estimated here because the overlap is small and the human table is task-pair level before aggregation.",
    ]
    (BASE_DIR / "human_llm_difference_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
