import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression


ROOT_DIR = Path(r"C:\Users\cooki\Desktop\ARC-AGI")
BASE_DIR = ROOT_DIR / "Python solutions" / "approved_only"

OVERLAP_PATH = BASE_DIR / "human_llm_overlap_tasks.csv"
KEY_TESTS_PATH = BASE_DIR / "statistical_hypothesis_key_tests.csv"
THINKING_SENS_PATH = BASE_DIR / "thinking_advantage_sensitivity.csv"
THINKING_SCHEMA_PATH = BASE_DIR / "thinking_schema_task_metrics.csv"
WITHIN_TASK_PATH = BASE_DIR / "human_public_eval_within_task_heterogeneity.csv"
HUMAN_PUBLIC_EVAL_PATH = ROOT_DIR / "Human data" / "analysis" / "tables" / "public_eval_human_vs_models.csv"
LABEL_AUDIT_PATH = BASE_DIR / "thinking_label_audit.csv"
LLM_APPROVED_JOIN_PATH = BASE_DIR / "approved_llm_complexity_join.csv"

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


def zscore(series):
    series = pd.Series(series, dtype=float)
    return (series - series.mean()) / series.std(ddof=0)


def label_points(ax, df, x_col, y_col, label_col, top_n=None, fontsize=8):
    work = df.copy()
    if top_n is not None and len(work) > top_n:
        score = (work[x_col] - work[x_col].median()).abs() + (work[y_col] - work[y_col].median()).abs()
        work = work.assign(_score=score).sort_values("_score", ascending=False).head(top_n)
    offsets = [(4, 4), (5, -10), (-20, 4), (-20, -10), (8, 10), (-24, 10)]
    for i, (_, row) in enumerate(work.iterrows()):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            str(row[label_col]),
            (row[x_col], row[y_col]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            alpha=0.9,
        )


def fit_line(ax, x, y, color="#111111", lw=2.0, ls="-"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.unique(x).size < 2:
        return
    coeffs = np.polyfit(x, y, deg=1)
    x_grid = np.linspace(np.min(x), np.max(x), 200)
    ax.plot(x_grid, np.polyval(coeffs, x_grid), color=color, linewidth=lw, linestyle=ls)


def plot_throughline(overlap: pd.DataFrame, key_tests: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Shared human-vs-LLM difficulty
    sub = overlap[["task_id", "difficulty_weighted", "logit_difficulty_all", "human_attempts_total"]].dropna().copy()
    sns.scatterplot(
        data=sub,
        x="difficulty_weighted",
        y="logit_difficulty_all",
        size="human_attempts_total",
        sizes=(70, 240),
        color="#1f77b4",
        edgecolor="black",
        legend=False,
        ax=ax1,
    )
    fit_line(ax1, sub["difficulty_weighted"], sub["logit_difficulty_all"])
    label_points(ax1, sub, "difficulty_weighted", "logit_difficulty_all", "task_id", top_n=17)
    row = key_tests.set_index("claim_id").loc["S1"]
    ax1.set_title(f"Shared Task Difficulty\nr = {row['estimate']:.3f}, p = {row['p_value']:.3g}, q = {row['q_value_bh']:.3g}")
    ax1.set_xlabel("Human task difficulty")
    ax1.set_ylabel("LLM logit difficulty")

    # Structural complexity aligns more with LLM than human difficulty
    sub = overlap[["task_id", "cyclomatic_complexity", "difficulty_weighted", "logit_difficulty_all"]].dropna().copy()
    plot_df = pd.DataFrame(
        {
            "task_id": np.repeat(sub["task_id"].to_numpy(), 2),
            "cyclomatic_complexity": np.repeat(sub["cyclomatic_complexity"].to_numpy(), 2),
            "difficulty_z": np.concatenate([zscore(sub["difficulty_weighted"]), zscore(sub["logit_difficulty_all"])]),
            "target": ["Human difficulty"] * len(sub) + ["LLM difficulty"] * len(sub),
        }
    )
    sns.scatterplot(
        data=plot_df,
        x="cyclomatic_complexity",
        y="difficulty_z",
        hue="target",
        palette=["#2ca02c", "#9467bd"],
        s=75,
        edgecolor="black",
        ax=ax2,
    )
    for target, color in [("Human difficulty", "#2ca02c"), ("LLM difficulty", "#9467bd")]:
        tmp = plot_df[plot_df["target"] == target]
        fit_line(ax2, tmp["cyclomatic_complexity"], tmp["difficulty_z"], color=color)
    d1 = key_tests.set_index("claim_id").loc["D1"]
    ax2.set_title(f"Structural Solver Complexity is More LLM-Like\ndelta-r = {d1['estimate']:.3f}, p = {d1['p_value']:.3g}")
    ax2.set_xlabel("Cyclomatic complexity")
    ax2.set_ylabel("Standardized difficulty")
    ax2.legend(frameon=True, loc="upper left")

    # Time/search burden aligns more with human difficulty than LLM difficulty
    sub = overlap[["task_id", "mean_duration_seconds_weighted", "difficulty_weighted", "logit_difficulty_all"]].dropna().copy()
    plot_df = pd.DataFrame(
        {
            "task_id": np.repeat(sub["task_id"].to_numpy(), 2),
            "mean_duration_seconds_weighted": np.repeat(sub["mean_duration_seconds_weighted"].to_numpy(), 2),
            "difficulty_z": np.concatenate([zscore(sub["difficulty_weighted"]), zscore(sub["logit_difficulty_all"])]),
            "target": ["Human difficulty"] * len(sub) + ["LLM difficulty"] * len(sub),
        }
    )
    sns.scatterplot(
        data=plot_df,
        x="mean_duration_seconds_weighted",
        y="difficulty_z",
        hue="target",
        palette=["#2ca02c", "#9467bd"],
        s=75,
        edgecolor="black",
        ax=ax3,
    )
    for target, color in [("Human difficulty", "#2ca02c"), ("LLM difficulty", "#9467bd")]:
        tmp = plot_df[plot_df["target"] == target]
        fit_line(ax3, tmp["mean_duration_seconds_weighted"], tmp["difficulty_z"], color=color)
    d2 = key_tests.set_index("claim_id").loc["D2"]
    ax3.set_title(f"Human Time Cost is More Human-Like\ndelta-r = {d2['estimate']:.3f}, p = {d2['p_value']:.3g}")
    ax3.set_xlabel("Mean human duration (s)")
    ax3.set_ylabel("Standardized difficulty")
    ax3.legend(frameon=True, loc="upper left")

    # Residuals panel
    sub = overlap[["task_id", "difficulty_weighted", "logit_difficulty_all", "mean_duration_seconds_weighted", "cyclomatic_complexity"]].dropna().copy()
    human_model = LinearRegression().fit(sub[["logit_difficulty_all"]], sub["difficulty_weighted"])
    sub["human_residual_after_llm"] = sub["difficulty_weighted"] - human_model.predict(sub[["logit_difficulty_all"]])
    llm_model = LinearRegression().fit(sub[["difficulty_weighted"]], sub["logit_difficulty_all"])
    sub["llm_residual_after_human"] = sub["logit_difficulty_all"] - llm_model.predict(sub[["difficulty_weighted"]])
    ax4.scatter(sub["mean_duration_seconds_weighted"], sub["human_residual_after_llm"], color="#d62728", s=80, edgecolors="black", label="Human residual vs duration")
    ax4.scatter(sub["cyclomatic_complexity"], sub["llm_residual_after_human"], color="#1f77b4", s=80, edgecolors="black", label="LLM residual vs cyclomatic")
    fit_line(ax4, sub["mean_duration_seconds_weighted"], sub["human_residual_after_llm"], color="#d62728")
    fit_line(ax4, sub["cyclomatic_complexity"], sub["llm_residual_after_human"], color="#1f77b4")
    label_points(ax4, sub.sort_values("llm_residual_after_human", ascending=False).head(6), "cyclomatic_complexity", "llm_residual_after_human", "task_id")
    ax4.axhline(0.0, color="#666666", linewidth=1)
    d3 = key_tests.set_index("claim_id").loc["D3"]
    d4 = key_tests.set_index("claim_id").loc["D4"]
    ax4.set_title(
        "Residual Split\n"
        f"human residual vs duration: r = {d3['estimate']:.3f}, p = {d3['p_value']:.3g}\n"
        f"LLM residual vs cyclomatic: r = {d4['estimate']:.3f}, p = {d4['p_value']:.3g}"
    )
    ax4.set_xlabel("Mean human duration / cyclomatic complexity")
    ax4.set_ylabel("Residual difficulty")
    ax4.legend(frameon=True, loc="upper left")

    fig.suptitle("Synthesis: One Shared Axis, Then a Human-vs-LLM Split", fontsize=22, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pair_level_human(human_pairs: pd.DataFrame, within_task: pd.DataFrame, key_tests: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.ravel()

    sampled = human_pairs[human_pairs["attempts"] >= 8].copy()

    sns.scatterplot(data=sampled, x="mean_duration_seconds", y="difficulty", color="#2ca02c", s=75, edgecolor="black", ax=ax1)
    fit_line(ax1, sampled["mean_duration_seconds"], sampled["difficulty"])
    label_points(ax1, sampled.sort_values("difficulty", ascending=False).head(8), "mean_duration_seconds", "difficulty", "task_pair_id")
    h1 = key_tests.set_index("claim_id").loc["H1"]
    ax1.set_title(f"Human Pair Difficulty Tracks Time Cost\nr = {h1['estimate']:.3f}, p = {h1['p_value']:.3g}")
    ax1.set_xlabel("Mean human duration (s)")
    ax1.set_ylabel("Human pair difficulty")

    sns.scatterplot(data=sampled, x="n_test_pairs", y="gap_vs_lm_mean", color="#d62728", s=75, edgecolor="black", ax=ax2)
    fit_line(ax2, sampled["n_test_pairs"], sampled["gap_vs_lm_mean"])
    label_points(ax2, sampled.sort_values("gap_vs_lm_mean", ascending=False).head(8), "n_test_pairs", "gap_vs_lm_mean", "task_pair_id")
    h4 = key_tests.set_index("claim_id").loc["H4"]
    ax2.set_title(f"More Test Pairs Shrink Human-over-LLM Gap\nr = {h4['estimate']:.3f}, p = {h4['p_value']:.3g}")
    ax2.set_xlabel("Number of test pairs")
    ax2.set_ylabel("Human solve rate - LLM mean pass rate")

    top = within_task.sort_values("difficulty_range", ascending=False).head(12).iloc[::-1]
    sns.barplot(data=top, x="difficulty_range", y="task_ID", color="#17becf", ax=ax3)
    ax3.set_title("Within-Task Human Difficulty Heterogeneity")
    ax3.set_xlabel("Difficulty range across test pairs")
    ax3.set_ylabel("Task ID")
    for i, (_, row) in enumerate(top.iterrows()):
        ax3.text(row["difficulty_range"] + 0.03, i, f"{row['difficulty_range']:.2f}", va="center", fontsize=10)

    features = ["mean_duration_seconds", "input_cells", "n_train_pairs", "n_test_pairs"]
    bars = []
    for feature in features:
        bars.append({"feature": feature, "outcome": "Difficulty", "r": safe_corr(sampled[feature], sampled["difficulty"])})
        bars.append({"feature": feature, "outcome": "Human - LLM gap", "r": safe_corr(sampled[feature], sampled["gap_vs_lm_mean"])})
    bar_df = pd.DataFrame(bars)
    label_map = {
        "mean_duration_seconds": "Human duration",
        "input_cells": "Input cells",
        "n_train_pairs": "Train pairs",
        "n_test_pairs": "Test pairs",
    }
    bar_df["feature_label"] = bar_df["feature"].map(label_map)
    sns.barplot(data=bar_df, x="feature_label", y="r", hue="outcome", palette=["#2ca02c", "#d62728"], ax=ax4)
    ax4.axhline(0.0, color="#666666", linewidth=1)
    ax4.set_title("Different Predictors for Human Difficulty vs Human Advantage")
    ax4.set_xlabel("")
    ax4.set_ylabel("Pearson r")
    for container in ax4.containers:
        ax4.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    ax4.legend(frameon=True, loc="upper right")

    fig.suptitle("Synthesis: Human-Side Pair-Level Structure", fontsize=22, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_thinking_audit(schema_df: pd.DataFrame, sens: pd.DataFrame, label_audit: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.ravel()

    approved_keys = pd.read_csv(LLM_APPROVED_JOIN_PATH)[["dataset_key", "task_id", "logit_difficulty_all"]].drop_duplicates()
    verified = (
        schema_df[schema_df["schema"] == "maximal"]
        .merge(approved_keys, on=["dataset_key", "task_id"], how="inner")
        .copy()
    )

    sns.barplot(data=sens, x="schema", y="thinking_advantage_r", color="#1f77b4", ax=ax1)
    ax1.axhline(0.0, color="#666666", linewidth=1)
    for i, row in sens.reset_index(drop=True).iterrows():
        ax1.errorbar(i, row["thinking_advantage_r"], yerr=[[row["thinking_advantage_r"] - row["thinking_advantage_ci_low"]], [row["thinking_advantage_ci_high"] - row["thinking_advantage_r"]]], color="black", capsize=4)
        ax1.text(i, row["thinking_advantage_r"] + 0.04 * np.sign(row["thinking_advantage_r"] if row["thinking_advantage_r"] != 0 else 1), f"p={row['thinking_advantage_perm_p']:.3g}", ha="center", fontsize=10)
    ax1.set_title("Raw Thinking-Advantage Correlation Depends on Label Schema")
    ax1.set_xlabel("Schema")
    ax1.set_ylabel("r with LLM difficulty")

    verified_plot = verified.copy()
    verified_plot["floor_case"] = np.where(verified_plot["standard_zero_successes"] == 1, "Standard=0", "Standard>0")
    sns.scatterplot(
        data=verified_plot,
        x="logit_difficulty_all",
        y="thinking_advantage",
        hue="floor_case",
        palette={"Standard=0": "#d62728", "Standard>0": "#2ca02c"},
        s=80,
        edgecolor="black",
        ax=ax2,
    )
    fit_line(ax2, verified_plot["logit_difficulty_all"], verified_plot["thinking_advantage"])
    label_points(ax2, verified_plot.sort_values("logit_difficulty_all", ascending=False).head(10), "logit_difficulty_all", "thinking_advantage", "task_id")
    ax2.set_title("Verified Grouping: Raw Gap is Dominated by Standard-Group Zeros")
    ax2.set_xlabel("LLM logit difficulty")
    ax2.set_ylabel("Thinking advantage")
    ax2.legend(frameon=True, loc="upper right")

    subset_rows = []
    conditions = {
        "All verified rows": np.ones(len(verified_plot), dtype=bool),
        "Standard nonzero only": verified_plot["standard_zero_successes"] == 0,
        "Both groups interior": (verified_plot["pass_rate_thinking"] > 0) & (verified_plot["pass_rate_thinking"] < 1) & (verified_plot["pass_rate_standard"] > 0) & (verified_plot["pass_rate_standard"] < 1),
    }
    for label, mask in conditions.items():
        tmp = verified_plot.loc[mask]
        subset_rows.append({"subset": label, "metric": "Raw gap", "r": safe_corr(tmp["thinking_advantage"], tmp["logit_difficulty_all"])})
        subset_rows.append({"subset": label, "metric": "Logit gap", "r": safe_corr(tmp["thinking_logit_advantage"], tmp["logit_difficulty_all"])})
    subset_df = pd.DataFrame(subset_rows)
    sns.barplot(data=subset_df, x="subset", y="r", hue="metric", palette=["#1f77b4", "#ff7f0e"], ax=ax3)
    ax3.axhline(0.0, color="#666666", linewidth=1)
    ax3.set_title("Verified Grouping: Signal Collapses After Removing Floor Cases")
    ax3.set_xlabel("")
    ax3.set_ylabel("r with LLM difficulty")
    for container in ax3.containers:
        ax3.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    ax3.legend(frameon=True, loc="upper right")

    audit = label_audit[label_audit["certainty"] == "low"].copy()
    y_pos = np.arange(len(audit))
    ax4.scatter(np.zeros(len(audit)), y_pos, color="#999999", s=100, label="Strict = Standard")
    ax4.scatter(np.ones(len(audit)), y_pos, color="#1f77b4", s=100, label="Verified = Thinking")
    for i, (_, row) in enumerate(audit.iterrows()):
        ax4.plot([0, 1], [i, i], color="#bbbbbb", linewidth=1.5)
        ax4.text(-0.05, i, row["model_name"], ha="right", va="center", fontsize=10)
        ax4.text(1.05, i, row["evidence"], ha="left", va="center", fontsize=10)
    ax4.set_xlim(-0.2, 1.6)
    ax4.set_yticks([])
    ax4.set_xticks([0, 1], ["Strict", "Verified"])
    ax4.set_title("Ambiguous Models that Matter for the Thinking Audit")
    ax4.legend(frameon=True, loc="lower right")

    fig.suptitle("Synthesis: Thinking-Advantage Audit and Interpretation", fontsize=22, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stats_forest(key_tests: pd.DataFrame, output_path: Path):
    select = ["S1", "S2", "D1", "D2", "D3", "D4", "H1", "H3", "H4", "T1", "T2"]
    df = key_tests[key_tests["claim_id"].isin(select)].copy()
    order = ["S1", "S2", "D1", "D2", "D3", "D4", "H1", "H3", "H4", "T1", "T2"]
    df["claim_id"] = pd.Categorical(df["claim_id"], categories=order, ordered=True)
    df = df.sort_values("claim_id").reset_index(drop=True)
    label_map = {
        "S1": "Shared task difficulty",
        "S2": "Shared solve rate",
        "D1": "Cyclomatic: LLM > human",
        "D2": "Duration: human > LLM",
        "D3": "Residual human ~ duration",
        "D4": "Residual LLM ~ cyclomatic",
        "H1": "Human pair difficulty ~ duration",
        "H3": "Duration > board size",
        "H4": "More test pairs reduce human gap",
        "T1": "Legacy thinking gap ~ difficulty",
        "T2": "Legacy GLM interaction",
    }

    fig, ax = plt.subplots(figsize=(12, 10))
    y_pos = np.arange(len(df))
    colors = ["#2ca02c" if q <= 0.05 else "#999999" for q in df["q_value_bh"]]
    ax.scatter(df["estimate"], y_pos, color=colors, s=80, zorder=3)
    for i, row in df.iterrows():
        if np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"]):
            ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=colors[i], linewidth=3)
        ax.text(row["estimate"], i + 0.22, f"p={row['p_value']:.3g}, q={row['q_value_bh']:.3g}", fontsize=9, ha="center")
    ax.axvline(0.0, color="#666666", linewidth=1)
    ax.set_yticks(y_pos, [label_map[c] for c in df["claim_id"]])
    ax.set_xlabel("Estimate")
    ax.set_ylabel("")
    ax.set_title("Key Hypothesis Tests with 95% CIs and FDR q-values")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    overlap = pd.read_csv(OVERLAP_PATH)
    key_tests = pd.read_csv(KEY_TESTS_PATH)
    sens = pd.read_csv(THINKING_SENS_PATH)
    schema_df = pd.read_csv(THINKING_SCHEMA_PATH)
    within_task = pd.read_csv(WITHIN_TASK_PATH)
    human_pairs = pd.read_csv(HUMAN_PUBLIC_EVAL_PATH)
    label_audit = pd.read_csv(LABEL_AUDIT_PATH)

    plot_throughline(overlap, key_tests, BASE_DIR / "chart_synthesis_throughline.png")
    plot_pair_level_human(human_pairs, within_task, key_tests, BASE_DIR / "chart_synthesis_human_pair_level.png")
    plot_thinking_audit(schema_df, sens, label_audit, BASE_DIR / "chart_synthesis_thinking_audit.png")
    plot_stats_forest(key_tests, BASE_DIR / "chart_synthesis_stats_forest.png")

    manifest = {
        "throughline": str(BASE_DIR / "chart_synthesis_throughline.png"),
        "human_pair_level": str(BASE_DIR / "chart_synthesis_human_pair_level.png"),
        "thinking_audit": str(BASE_DIR / "chart_synthesis_thinking_audit.png"),
        "stats_forest": str(BASE_DIR / "chart_synthesis_stats_forest.png"),
    }
    (BASE_DIR / "chart_synthesis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
