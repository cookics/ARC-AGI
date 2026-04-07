import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression


ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent
HUMAN_PUBLIC_EVAL_PATH = ROOT_DIR / "analysis-human" / "analysis" / "tables" / "public_eval_human_vs_models.csv"
OVERLAP_PATH = BASE_DIR / "human_llm_overlap_tasks.csv"

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


def build_pair_metadata_correlations(df: pd.DataFrame, sample_label: str) -> pd.DataFrame:
    features = [
        "input_cells",
        "output_cells",
        "input_colors",
        "output_colors",
        "n_train_pairs",
        "n_test_pairs",
        "size_change_ratio",
        "mean_duration_seconds",
        "mean_submissions",
    ]
    outcomes = [
        "difficulty",
        "solve_rate",
        "gap_vs_lm_mean",
        "gap_vs_best_single_model",
        "point_biserial",
        "outfit",
    ]
    rows = []
    for outcome in outcomes:
        for feature in features:
            subset = df[[feature, outcome]].dropna()
            rows.append(
                {
                    "sample": sample_label,
                    "outcome": outcome,
                    "feature": feature,
                    "n": len(subset),
                    "pearson": safe_corr(subset[feature], subset[outcome]),
                    "spearman": spearman_corr(subset[feature], subset[outcome]),
                }
            )
    result = pd.DataFrame(rows)
    return result.sort_values(["sample", "outcome", "pearson"], ascending=[True, True, False]).reset_index(drop=True)


def build_within_task_heterogeneity(df: pd.DataFrame) -> pd.DataFrame:
    multi = df.groupby("task_ID").filter(lambda g: len(g) > 1).copy()
    table = (
        multi.groupby("task_ID")
        .agg(
            pair_count=("task_pair_id", "size"),
            attempts_total=("attempts", "sum"),
            difficulty_mean=("difficulty", "mean"),
            difficulty_sd=("difficulty", "std"),
            difficulty_range=("difficulty", lambda s: float(s.max() - s.min())),
            solve_rate_mean=("solve_rate", "mean"),
            solve_rate_range=("solve_rate", lambda s: float(s.max() - s.min())),
            gap_mean=("gap_vs_lm_mean", "mean"),
            gap_range=("gap_vs_lm_mean", lambda s: float(s.max() - s.min())),
            duration_mean=("mean_duration_seconds", "mean"),
            duration_range=("mean_duration_seconds", lambda s: float(s.max() - s.min())),
        )
        .sort_values("difficulty_range", ascending=False)
        .reset_index()
    )
    return table


def build_overlap_residual_table(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[
        [
            "task_id",
            "difficulty_weighted",
            "logit_difficulty_all",
            "rasch_difficulty_all_models_pooled",
            "human_solve_rate_weighted",
            "pass_rate_all",
            "pass_rate_thinking",
            "gap_vs_lm_mean_weighted",
            "thinking_advantage",
            "mean_duration_seconds_weighted",
            "cyclomatic_complexity",
            "complexity_pc1_score",
            "elapsed_ms_total",
            "log1p_elapsed_ms_total",
        ]
    ].dropna()

    x_llm = subset[["logit_difficulty_all"]].to_numpy(dtype=float)
    y_human = subset["difficulty_weighted"].to_numpy(dtype=float)
    human_model = LinearRegression().fit(x_llm, y_human)
    subset = subset.copy()
    subset["human_difficulty_residual_after_llm"] = y_human - human_model.predict(x_llm)

    x_human = subset[["difficulty_weighted"]].to_numpy(dtype=float)
    y_llm = subset["logit_difficulty_all"].to_numpy(dtype=float)
    llm_model = LinearRegression().fit(x_human, y_llm)
    subset["llm_difficulty_residual_after_human"] = y_llm - llm_model.predict(x_human)

    x_structure = subset[["complexity_pc1_score"]].to_numpy(dtype=float)
    y_runtime = subset["log1p_elapsed_ms_total"].to_numpy(dtype=float)
    runtime_model = LinearRegression().fit(x_structure, y_runtime)
    subset["runtime_residual_given_structure"] = y_runtime - runtime_model.predict(x_structure)

    probes = [
        "mean_duration_seconds_weighted",
        "gap_vs_lm_mean_weighted",
        "thinking_advantage",
        "cyclomatic_complexity",
        "complexity_pc1_score",
        "elapsed_ms_total",
        "log1p_elapsed_ms_total",
        "runtime_residual_given_structure",
    ]
    residuals = [
        "human_difficulty_residual_after_llm",
        "llm_difficulty_residual_after_human",
    ]
    rows = []
    for residual in residuals:
        for probe in probes:
            tmp = subset[[probe, residual]].dropna()
            rows.append(
                {
                    "residual_target": residual,
                    "probe": probe,
                    "n": len(tmp),
                    "pearson": safe_corr(tmp[probe], tmp[residual]),
                    "spearman": spearman_corr(tmp[probe], tmp[residual]),
                }
            )
    return pd.DataFrame(rows).sort_values(["residual_target", "pearson"], ascending=[True, False]).reset_index(drop=True)


def plot_within_task_heterogeneity(within_table: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.histplot(within_table["difficulty_range"], bins=16, color="#1f77b4", edgecolor="white", ax=ax)
    ax.axvline(within_table["difficulty_range"].mean(), color="#111111", linewidth=2, linestyle="--", label=f"Mean = {within_table['difficulty_range'].mean():.2f}")
    ax.set_title("Public-Eval Human Difficulty Varies Meaningfully Across Test Pairs")
    ax.set_xlabel("Within-Task Difficulty Range Across Test Pairs")
    ax.set_ylabel("Number of Tasks")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_human_gap_vs_runtime(overlap: pd.DataFrame, output_path: Path):
    subset = overlap[["task_id", "complexity_pc1_score", "log1p_elapsed_ms_total", "human_minus_llm_pass_all"]].dropna().copy()
    reg = LinearRegression().fit(subset[["complexity_pc1_score"]], subset["log1p_elapsed_ms_total"])
    subset["runtime_residual_given_structure"] = subset["log1p_elapsed_ms_total"] - reg.predict(subset[["complexity_pc1_score"]])

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.regplot(
        data=subset,
        x="runtime_residual_given_structure",
        y="human_minus_llm_pass_all",
        scatter_kws={"s": 90, "color": "#d62728", "alpha": 0.85},
        line_kws={"color": "#111111", "linewidth": 2},
        ax=ax,
    )
    for _, row in subset.sort_values("human_minus_llm_pass_all", ascending=False).head(5).iterrows():
        ax.annotate(row["task_id"], (row["runtime_residual_given_structure"], row["human_minus_llm_pass_all"]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_title("Execution-Heavy Tasks Show a Modest Human-Over-LLM Edge")
    ax.set_xlabel("Runtime Residual Given Solver Structure")
    ax.set_ylabel("Human Solve Rate - LLM Pass Rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    human = pd.read_csv(HUMAN_PUBLIC_EVAL_PATH)
    overlap = pd.read_csv(OVERLAP_PATH)

    full_corr = build_pair_metadata_correlations(human, sample_label="all_public_eval_pairs")
    well_sampled = human[human["attempts"] >= 8].copy()
    min8_corr = build_pair_metadata_correlations(well_sampled, sample_label="public_eval_pairs_attempts_ge_8")
    metadata_corr = pd.concat([full_corr, min8_corr], ignore_index=True)
    metadata_corr.to_csv(BASE_DIR / "human_public_eval_metadata_correlations.csv", index=False)

    within_table = build_within_task_heterogeneity(human)
    within_table.to_csv(BASE_DIR / "human_public_eval_within_task_heterogeneity.csv", index=False)

    residual_table = build_overlap_residual_table(overlap)
    residual_table.to_csv(BASE_DIR / "human_overlap_residual_findings.csv", index=False)

    plot_within_task_heterogeneity(within_table, BASE_DIR / "chart_human_within_task_heterogeneity.png")
    plot_human_gap_vs_runtime(overlap, BASE_DIR / "chart_human_gap_vs_runtime_residual.png")

    # Small headline summary for quick reading.
    multi = human.groupby("task_ID").filter(lambda g: len(g) > 1).copy()
    task_dummies = pd.get_dummies(multi["task_ID"], drop_first=True)
    task_fe_model = LinearRegression().fit(task_dummies.to_numpy(dtype=float), multi["difficulty"].to_numpy(dtype=float))

    all_gap = metadata_corr[
        (metadata_corr["sample"] == "all_public_eval_pairs") & (metadata_corr["outcome"] == "gap_vs_lm_mean")
    ].sort_values("pearson", ascending=False)
    min8_gap = metadata_corr[
        (metadata_corr["sample"] == "public_eval_pairs_attempts_ge_8") & (metadata_corr["outcome"] == "gap_vs_lm_mean")
    ].sort_values("pearson", ascending=False)
    all_diff = metadata_corr[
        (metadata_corr["sample"] == "all_public_eval_pairs") & (metadata_corr["outcome"] == "difficulty")
    ].sort_values("pearson", ascending=False)
    min8_diff = metadata_corr[
        (metadata_corr["sample"] == "public_eval_pairs_attempts_ge_8") & (metadata_corr["outcome"] == "difficulty")
    ].sort_values("pearson", ascending=False)

    overlap_r = overlap[["human_minus_llm_pass_all", "gap_vs_lm_mean_weighted"]].dropna()
    summary = {
        "pair_level_counts": {
            "public_eval_pairs": int(len(human)),
            "public_eval_tasks": int(human["task_ID"].nunique()),
            "multi_pair_tasks": int(within_table["task_ID"].nunique()),
        },
        "within_task_heterogeneity": {
            "mean_difficulty_range": float(within_table["difficulty_range"].mean()),
            "median_difficulty_range": float(within_table["difficulty_range"].median()),
            "max_difficulty_range": float(within_table["difficulty_range"].max()),
            "mean_solve_rate_range": float(within_table["solve_rate_range"].mean()),
            "task_fixed_effect_r2_on_multi_pair_subset": float(task_fe_model.score(task_dummies.to_numpy(dtype=float), multi["difficulty"].to_numpy(dtype=float))),
        },
        "top_gap_predictors_all_pairs": all_gap.head(4).to_dict(orient="records"),
        "top_gap_predictors_min8_pairs": min8_gap.head(4).to_dict(orient="records"),
        "top_difficulty_predictors_all_pairs": all_diff.head(4).to_dict(orient="records"),
        "top_difficulty_predictors_min8_pairs": min8_diff.head(4).to_dict(orient="records"),
    }
    (BASE_DIR / "human_additional_findings_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Additional Human-Side Findings",
        "",
        "## Human Difficulty Is Not Just Grid Size",
        "",
        f"- On well-sampled public-eval pairs (`attempts >= 8`), human difficulty correlates with mean human duration at `r = {min8_diff[min8_diff['feature'] == 'mean_duration_seconds']['pearson'].iloc[0]:.3f}`.",
        f"- The same human difficulty signal is much weaker for raw input size (`input_cells`): `r = {min8_diff[min8_diff['feature'] == 'input_cells']['pearson'].iloc[0]:.3f}`.",
        f"- More train/test examples slightly raise human difficulty in this public-eval slice: `n_train_pairs r = {min8_diff[min8_diff['feature'] == 'n_train_pairs']['pearson'].iloc[0]:.3f}`, `n_test_pairs r = {min8_diff[min8_diff['feature'] == 'n_test_pairs']['pearson'].iloc[0]:.3f}`.",
        "",
        "## Human Advantage Over Models Has Its Own Signature",
        "",
        f"- On well-sampled public-eval pairs, human-vs-average-model gap is larger on bigger boards (`input_cells r = {min8_gap[min8_gap['feature'] == 'input_cells']['pearson'].iloc[0]:.3f}`).",
        f"- The same gap is smaller when tasks expose more train/test examples (`n_train_pairs r = {min8_gap[min8_gap['feature'] == 'n_train_pairs']['pearson'].iloc[0]:.3f}`, `n_test_pairs r = {min8_gap[min8_gap['feature'] == 'n_test_pairs']['pearson'].iloc[0]:.3f}`).",
        "- That pattern suggests extra examples may help models close the gap more than they help humans, while larger spatial layouts still favor humans.",
        "",
        "## Pair-Level Heterogeneity Matters",
        "",
        f"- Among {summary['pair_level_counts']['multi_pair_tasks']} public-eval tasks with multiple test pairs, the mean within-task difficulty range is `{summary['within_task_heterogeneity']['mean_difficulty_range']:.3f}` logits and the max is `{summary['within_task_heterogeneity']['max_difficulty_range']:.3f}`.",
        f"- Mean within-task solve-rate range is `{summary['within_task_heterogeneity']['mean_solve_rate_range']:.3f}`.",
        f"- Task fixed effects explain about `{summary['within_task_heterogeneity']['task_fixed_effect_r2_on_multi_pair_subset']:.3f}` of pair-level human difficulty variance on the multi-pair subset.",
        "- So task identity matters a lot, but test-pair choice still contributes substantial human-specific variance that task-level solver complexity cannot see.",
        "",
        "## Residual Human vs Residual LLM Difficulty",
        "",
        "- After controlling for LLM difficulty, residual human difficulty is more connected to human duration than to solver structure.",
        "- After controlling for human difficulty, residual LLM difficulty still aligns with solver structure (`cyclomatic_complexity`, `complexity_pc1_score`) and runtime burden.",
        "- That supports the idea that the two systems are not merely noisy versions of the same latent variable.",
    ]
    (BASE_DIR / "human_additional_findings.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
