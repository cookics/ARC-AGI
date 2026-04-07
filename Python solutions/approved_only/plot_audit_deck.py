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
APPROVED_LLM_PATH = BASE_DIR / "approved_llm_complexity_join.csv"
COMPLEXITY_COMPARE_PATH = BASE_DIR / "human_llm_complexity_correlation_comparison.csv"
WITHIN_TASK_PATH = BASE_DIR / "human_public_eval_within_task_heterogeneity.csv"
HUMAN_META_CORR_PATH = BASE_DIR / "human_public_eval_metadata_correlations.csv"

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


def label_points(ax, x, y, labels, fontsize=8):
    offsets = [(4, 4), (4, -10), (-18, 4), (-18, -10), (6, 10), (-24, 10)]
    for i, (xi, yi, label) in enumerate(zip(x, y, labels)):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            str(label),
            (xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            alpha=0.9,
        )


def regline(ax, x, y, color="#111111", lw=2.0, ls="-"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return
    coeffs = np.polyfit(x, y, deg=1)
    x_grid = np.linspace(np.min(x), np.max(x), 200)
    ax.plot(x_grid, np.polyval(coeffs, x_grid), color=color, linewidth=lw, linestyle=ls)


def plot_shared_signals(overlap: pd.DataFrame, llm: pd.DataFrame, complexity_compare: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # A. Human difficulty vs LLM difficulty
    sub = overlap[["task_id", "difficulty_weighted", "logit_difficulty_all", "human_attempts_total"]].dropna().copy()
    sns.scatterplot(
        data=sub,
        x="difficulty_weighted",
        y="logit_difficulty_all",
        size="human_attempts_total",
        sizes=(60, 240),
        color="#1f77b4",
        edgecolor="black",
        legend=False,
        ax=ax1,
    )
    regline(ax1, sub["difficulty_weighted"], sub["logit_difficulty_all"], color="#111111")
    label_points(ax1, sub["difficulty_weighted"], sub["logit_difficulty_all"], sub["task_id"])
    ax1.set_title(f"Human vs LLM Difficulty\nr = {safe_corr(sub['difficulty_weighted'], sub['logit_difficulty_all']):.3f}")
    ax1.set_xlabel("Human Task Difficulty")
    ax1.set_ylabel("LLM Logit Difficulty")

    # B. Human solve rate vs average-model pass rate
    sub = overlap[["task_id", "human_solve_rate_weighted", "pass_rate_all", "pass_rate_thinking", "human_attempts_total"]].dropna().copy()
    sns.scatterplot(
        data=sub,
        x="human_solve_rate_weighted",
        y="pass_rate_all",
        size="human_attempts_total",
        sizes=(60, 240),
        color="#2ca02c",
        edgecolor="black",
        legend=False,
        ax=ax2,
    )
    regline(ax2, sub["human_solve_rate_weighted"], sub["pass_rate_all"], color="#111111")
    label_points(ax2, sub["human_solve_rate_weighted"], sub["pass_rate_all"], sub["task_id"])
    r_all = safe_corr(sub["human_solve_rate_weighted"], sub["pass_rate_all"])
    r_think = safe_corr(sub["human_solve_rate_weighted"], sub["pass_rate_thinking"])
    ax2.set_title(f"Human Solve Rate vs LLM Pass Rate\nall-model r = {r_all:.3f}; thinking-only r = {r_think:.3f}")
    ax2.set_xlabel("Human Solve Rate")
    ax2.set_ylabel("LLM Pass Rate (All Models)")

    # C. Complexity metrics: human vs LLM correlations
    metric_order = ["ast_node_count", "token_count", "cyclomatic_complexity", "complexity_pc1_score"]
    label_map = {
        "ast_node_count": "AST nodes",
        "token_count": "Tokens",
        "cyclomatic_complexity": "Cyclomatic",
        "complexity_pc1_score": "Complexity PC1",
    }
    sub = complexity_compare.set_index("metric").loc[metric_order].reset_index()
    plot_df = pd.DataFrame(
        {
            "metric": np.repeat(sub["metric"].to_numpy(), 2),
            "target": ["Human difficulty", "LLM difficulty"] * len(sub),
            "correlation": np.ravel(np.column_stack([sub["pearson_human"], sub["pearson_llm"]])),
        }
    )
    plot_df["metric_label"] = plot_df["metric"].map(label_map)
    sns.barplot(
        data=plot_df,
        x="metric_label",
        y="correlation",
        hue="target",
        palette=["#2ca02c", "#9467bd"],
        ax=ax3,
    )
    ax3.axhline(0.0, color="#666666", linewidth=1)
    ax3.set_title("Same Structural Metrics, Different Strength")
    ax3.set_xlabel("")
    ax3.set_ylabel("Pearson r")
    for container in ax3.containers:
        ax3.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    ax3.legend(frameon=True, loc="upper left")

    # D. LLM difficulty agreement checks
    pairs = [
        ("prev latent vs pooled Rasch", "latent_difficulty_prev_intersection22", "rasch_difficulty_all_models_pooled"),
        ("pooled Rasch vs 2PL", "rasch_difficulty_all_models_pooled", "two_pl_difficulty_all_models"),
        ("pooled Rasch vs logit", "rasch_difficulty_all_models_pooled", "logit_difficulty_all"),
        ("pooled Rasch vs fail rate", "rasch_difficulty_all_models_pooled", "fail_rate_all"),
    ]
    rows = []
    for label, a, b in pairs:
        tmp = llm[[a, b]].dropna()
        rows.append({"comparison": label, "pearson": safe_corr(tmp[a], tmp[b]), "n": len(tmp)})
    agree = pd.DataFrame(rows).sort_values("pearson", ascending=True)
    sns.barplot(data=agree, x="pearson", y="comparison", color="#8c564b", ax=ax4)
    ax4.set_xlim(0.0, 1.02)
    ax4.set_title("LLM Difficulty Definitions Agree Strongly")
    ax4.set_xlabel("Pearson r")
    ax4.set_ylabel("")
    for i, row in agree.reset_index(drop=True).iterrows():
        ax4.text(row["pearson"] + 0.01, i, f"r={row['pearson']:.3f}, n={int(row['n'])}", va="center", fontsize=10)

    fig.suptitle("Audit Deck: Strongest Shared Signals", fontsize=22, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_difference_signals(overlap: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.ravel()

    sub = overlap[
        [
            "task_id",
            "difficulty_weighted",
            "logit_difficulty_all",
            "mean_duration_seconds_weighted",
            "cyclomatic_complexity",
        ]
    ].dropna().copy()

    # residuals
    human_model = LinearRegression().fit(sub[["logit_difficulty_all"]], sub["difficulty_weighted"])
    sub["human_resid_after_llm"] = sub["difficulty_weighted"] - human_model.predict(sub[["logit_difficulty_all"]])
    llm_model = LinearRegression().fit(sub[["difficulty_weighted"]], sub["logit_difficulty_all"])
    sub["llm_resid_after_human"] = sub["logit_difficulty_all"] - llm_model.predict(sub[["difficulty_weighted"]])

    # A. human duration vs human difficulty
    sns.scatterplot(
        data=sub,
        x="mean_duration_seconds_weighted",
        y="difficulty_weighted",
        color="#2ca02c",
        edgecolor="black",
        s=90,
        ax=ax1,
    )
    regline(ax1, sub["mean_duration_seconds_weighted"], sub["difficulty_weighted"], color="#111111")
    label_points(ax1, sub["mean_duration_seconds_weighted"], sub["difficulty_weighted"], sub["task_id"])
    ax1.set_title(f"Human Time Cost Tracks Human Difficulty\nr = {safe_corr(sub['mean_duration_seconds_weighted'], sub['difficulty_weighted']):.3f}")
    ax1.set_xlabel("Mean Human Duration (s)")
    ax1.set_ylabel("Human Task Difficulty")

    # B. human duration vs LLM difficulty
    sns.scatterplot(
        data=sub,
        x="mean_duration_seconds_weighted",
        y="logit_difficulty_all",
        color="#9467bd",
        edgecolor="black",
        s=90,
        ax=ax2,
    )
    regline(ax2, sub["mean_duration_seconds_weighted"], sub["logit_difficulty_all"], color="#111111")
    label_points(ax2, sub["mean_duration_seconds_weighted"], sub["logit_difficulty_all"], sub["task_id"])
    ax2.set_title(f"Human Time Cost Barely Tracks LLM Difficulty\nr = {safe_corr(sub['mean_duration_seconds_weighted'], sub['logit_difficulty_all']):.3f}")
    ax2.set_xlabel("Mean Human Duration (s)")
    ax2.set_ylabel("LLM Logit Difficulty")

    # C. residual human difficulty after LLM
    sns.scatterplot(
        data=sub,
        x="mean_duration_seconds_weighted",
        y="human_resid_after_llm",
        color="#d62728",
        edgecolor="black",
        s=90,
        ax=ax3,
    )
    regline(ax3, sub["mean_duration_seconds_weighted"], sub["human_resid_after_llm"], color="#111111")
    label_points(ax3, sub["mean_duration_seconds_weighted"], sub["human_resid_after_llm"], sub["task_id"])
    ax3.axhline(0.0, color="#666666", linewidth=1)
    ax3.set_title(f"Human-Specific Residual Tracks Duration\nr = {safe_corr(sub['mean_duration_seconds_weighted'], sub['human_resid_after_llm']):.3f}")
    ax3.set_xlabel("Mean Human Duration (s)")
    ax3.set_ylabel("Residual Human Difficulty\n(after removing LLM difficulty)")

    # D. residual LLM difficulty after human
    sns.scatterplot(
        data=sub,
        x="cyclomatic_complexity",
        y="llm_resid_after_human",
        color="#1f77b4",
        edgecolor="black",
        s=90,
        ax=ax4,
    )
    regline(ax4, sub["cyclomatic_complexity"], sub["llm_resid_after_human"], color="#111111")
    label_points(ax4, sub["cyclomatic_complexity"], sub["llm_resid_after_human"], sub["task_id"])
    ax4.axhline(0.0, color="#666666", linewidth=1)
    ax4.set_title(f"LLM-Specific Residual Tracks Structure\nr = {safe_corr(sub['cyclomatic_complexity'], sub['llm_resid_after_human']):.3f}")
    ax4.set_xlabel("Cyclomatic Complexity")
    ax4.set_ylabel("Residual LLM Difficulty\n(after removing human difficulty)")

    fig.suptitle("Audit Deck: Strongest Divergence Signals", fontsize=22, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_human_specific_signals(overlap: pd.DataFrame, llm: pd.DataFrame, complexity_compare: pd.DataFrame, within_task: pd.DataFrame, human_meta_corr: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # A. thinking advantage curve
    think = llm[
        [
            "task_id",
            "dataset_key",
            "thinking_advantage",
            "thinking_logit_advantage",
            "logit_difficulty_all",
            "pass_rate_thinking",
            "pass_rate_standard",
        ]
    ].dropna().copy()
    sns.scatterplot(
        data=think,
        x="logit_difficulty_all",
        y="thinking_advantage",
        hue="dataset_key",
        palette={"arc_agi_1_eval": "#1f77b4", "arc_agi_2_eval": "#d62728"},
        edgecolor="black",
        s=80,
        ax=ax1,
    )
    x = think["logit_difficulty_all"].to_numpy(dtype=float)
    y = think["thinking_advantage"].to_numpy(dtype=float)
    quad = np.polyfit(x, y, deg=2)
    x_grid = np.linspace(np.min(x), np.max(x), 300)
    ax1.plot(x_grid, np.polyval(quad, x_grid), color="#111111", linewidth=2.5)
    # label the hardest and biggest-gain cases
    label_df = pd.concat(
        [
            think.sort_values("logit_difficulty_all", ascending=False).head(8),
            think.sort_values("thinking_advantage", ascending=False).head(6),
            think.sort_values("thinking_advantage", ascending=True).head(6),
        ]
    ).drop_duplicates("task_id")
    label_points(ax1, label_df["logit_difficulty_all"], label_df["thinking_advantage"], label_df["task_id"], fontsize=8)
    ax1.set_title(f"Thinking Advantage Falls on Hard Items\nr = {safe_corr(think['logit_difficulty_all'], think['thinking_advantage']):.3f}; no both-zero items")
    ax1.set_xlabel("LLM Logit Difficulty")
    ax1.set_ylabel("Thinking - Standard Pass Rate")
    ax1.legend(frameon=True, loc="upper right")

    # B. complexity delta bars
    metric_order = [
        "cyclomatic_complexity",
        "complexity_pc1_score",
        "peak_memory_bytes",
        "ast_node_count",
        "token_count",
        "log1p_elapsed_ms_total",
    ]
    label_map = {
        "cyclomatic_complexity": "Cyclomatic",
        "complexity_pc1_score": "Complexity PC1",
        "peak_memory_bytes": "Peak memory",
        "ast_node_count": "AST nodes",
        "token_count": "Tokens",
        "log1p_elapsed_ms_total": "log runtime",
    }
    delta = complexity_compare.set_index("metric").loc[metric_order].reset_index()
    y_pos = np.arange(len(delta))
    ax2.barh(y_pos, delta["delta_llm_minus_human"], color="#ff7f0e")
    for i, row in delta.iterrows():
        ax2.plot([row["delta_ci_low"], row["delta_ci_high"]], [i, i], color="#111111", linewidth=2)
        ax2.scatter(row["delta_llm_minus_human"], i, color="#111111", s=35)
        ax2.text(row["delta_llm_minus_human"] + 0.015, i, f"{row['delta_llm_minus_human']:.2f}", va="center", fontsize=10)
    ax2.axvline(0.0, color="#666666", linewidth=1)
    ax2.set_yticks(y_pos, [label_map[m] for m in delta["metric"]])
    ax2.set_title("LLM Minus Human Correlation Gap\n(selected solver metrics)")
    ax2.set_xlabel("Delta r")
    ax2.set_ylabel("")

    # C. within-task heterogeneity top tasks
    top_within = within_task.sort_values("difficulty_range", ascending=False).head(12).iloc[::-1]
    sns.barplot(data=top_within, x="difficulty_range", y="task_ID", color="#17becf", ax=ax3)
    ax3.set_title("Human Pair Difficulty Varies Within the Same Task")
    ax3.set_xlabel("Within-Task Difficulty Range")
    ax3.set_ylabel("Task ID")
    for i, (_, row) in enumerate(top_within.iterrows()):
        ax3.text(row["difficulty_range"] + 0.03, i, f"{row['difficulty_range']:.2f}", va="center", fontsize=10)

    # D. pair-level feature correlation bars
    sub = human_meta_corr[
        human_meta_corr["sample"] == "public_eval_pairs_attempts_ge_8"
    ].copy()
    features = ["mean_duration_seconds", "input_cells", "n_train_pairs", "n_test_pairs"]
    outcomes = ["difficulty", "gap_vs_lm_mean"]
    label_map_features = {
        "mean_duration_seconds": "Human duration",
        "input_cells": "Input cells",
        "n_train_pairs": "Train pairs",
        "n_test_pairs": "Test pairs",
    }
    rows = []
    for feature in features:
        for outcome in outcomes:
            row = sub[(sub["feature"] == feature) & (sub["outcome"] == outcome)].iloc[0]
            rows.append(
                {
                    "feature": label_map_features[feature],
                    "outcome": "Human difficulty" if outcome == "difficulty" else "Human - LLM gap",
                    "pearson": row["pearson"],
                }
            )
    plot_df = pd.DataFrame(rows)
    sns.barplot(
        data=plot_df,
        x="feature",
        y="pearson",
        hue="outcome",
        palette=["#2ca02c", "#d62728"],
        ax=ax4,
    )
    ax4.axhline(0.0, color="#666666", linewidth=1)
    ax4.set_title("Different Predictors for Human Difficulty vs Human Advantage")
    ax4.set_xlabel("")
    ax4.set_ylabel("Pearson r")
    for container in ax4.containers:
        ax4.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    ax4.legend(frameon=True, loc="upper right")

    fig.suptitle("Audit Deck: Human-Specific Patterns and Checks", fontsize=22, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    overlap = pd.read_csv(OVERLAP_PATH)
    llm = pd.read_csv(APPROVED_LLM_PATH)
    complexity_compare = pd.read_csv(COMPLEXITY_COMPARE_PATH)
    within_task = pd.read_csv(WITHIN_TASK_PATH)
    human_meta_corr = pd.read_csv(HUMAN_META_CORR_PATH)

    plot_shared_signals(overlap, llm, complexity_compare, BASE_DIR / "chart_audit_shared_signals.png")
    plot_difference_signals(overlap, BASE_DIR / "chart_audit_divergence_signals.png")
    plot_human_specific_signals(overlap, llm, complexity_compare, within_task, human_meta_corr, BASE_DIR / "chart_audit_human_specific_signals.png")

    # build summary lists
    similarity_items = [
        {
            "signal": "Human difficulty vs LLM logit difficulty",
            "pearson": safe_corr(overlap["difficulty_weighted"], overlap["logit_difficulty_all"]),
            "n": int(overlap[["difficulty_weighted", "logit_difficulty_all"]].dropna().shape[0]),
        },
        {
            "signal": "Human solve rate vs all-model pass rate",
            "pearson": safe_corr(overlap["human_solve_rate_weighted"], overlap["pass_rate_all"]),
            "n": int(overlap[["human_solve_rate_weighted", "pass_rate_all"]].dropna().shape[0]),
        },
        {
            "signal": "Human solve rate vs thinking-model pass rate",
            "pearson": safe_corr(overlap["human_solve_rate_weighted"], overlap["pass_rate_thinking"]),
            "n": int(overlap[["human_solve_rate_weighted", "pass_rate_thinking"]].dropna().shape[0]),
        },
        {
            "signal": "LLM pooled Rasch vs LLM logit difficulty",
            "pearson": safe_corr(llm["rasch_difficulty_all_models_pooled"], llm["logit_difficulty_all"]),
            "n": int(llm[["rasch_difficulty_all_models_pooled", "logit_difficulty_all"]].dropna().shape[0]),
        },
        {
            "signal": "LLM previous latent vs pooled Rasch",
            "pearson": safe_corr(llm["latent_difficulty_prev_intersection22"], llm["rasch_difficulty_all_models_pooled"]),
            "n": int(llm[["latent_difficulty_prev_intersection22", "rasch_difficulty_all_models_pooled"]].dropna().shape[0]),
        },
    ]

    diff_lookup = complexity_compare.set_index("metric")
    difference_items = [
        {
            "signal": "Cyclomatic complexity predicts LLM difficulty much more than human difficulty",
            "human_r": float(diff_lookup.loc["cyclomatic_complexity", "pearson_human"]),
            "llm_r": float(diff_lookup.loc["cyclomatic_complexity", "pearson_llm"]),
            "delta_r": float(diff_lookup.loc["cyclomatic_complexity", "delta_llm_minus_human"]),
        },
        {
            "signal": "Human duration tracks human difficulty but not LLM difficulty",
            "human_r": safe_corr(overlap["mean_duration_seconds_weighted"], overlap["difficulty_weighted"]),
            "llm_r": safe_corr(overlap["mean_duration_seconds_weighted"], overlap["logit_difficulty_all"]),
            "delta_r": safe_corr(overlap["mean_duration_seconds_weighted"], overlap["difficulty_weighted"]) - safe_corr(overlap["mean_duration_seconds_weighted"], overlap["logit_difficulty_all"]),
        },
        {
            "signal": "Thinking advantage falls as LLM difficulty rises",
            "pearson": safe_corr(llm["thinking_advantage"], llm["logit_difficulty_all"]),
            "n": int(llm[["thinking_advantage", "logit_difficulty_all"]].dropna().shape[0]),
        },
        {
            "signal": "Residual human difficulty after LLM aligns with human duration",
            "pearson": 0.46988156541813136,
            "n": 17,
        },
        {
            "signal": "Residual LLM difficulty after human aligns with cyclomatic complexity",
            "pearson": 0.6031976748169638,
            "n": 17,
        },
        {
            "signal": "Human pair difficulty varies substantially within a task",
            "mean_range": float(within_task["difficulty_range"].mean()),
            "max_range": float(within_task["difficulty_range"].max()),
            "n_tasks": int(within_task["task_ID"].nunique()),
        },
    ]

    summary = {
        "strongest_similarity_items": similarity_items,
        "strongest_difference_items": difference_items,
        "charts": {
            "shared_signals": str(BASE_DIR / "chart_audit_shared_signals.png"),
            "divergence_signals": str(BASE_DIR / "chart_audit_divergence_signals.png"),
            "human_specific_signals": str(BASE_DIR / "chart_audit_human_specific_signals.png"),
        },
    }
    (BASE_DIR / "audit_deck_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Audit Deck Summary",
        "",
        "## Strongest Similarity Signals",
        "",
    ]
    for item in similarity_items:
        if "n" in item:
            lines.append(f"- {item['signal']}: `r = {item['pearson']:.3f}` on `n = {item['n']}`.")
        else:
            lines.append(f"- {item['signal']}: `r = {item['pearson']:.3f}`.")
    lines.extend(
        [
            "",
            "## Strongest Difference Signals",
            "",
            f"- Cyclomatic complexity: human difficulty `r = {difference_items[0]['human_r']:.3f}` vs LLM difficulty `r = {difference_items[0]['llm_r']:.3f}`.",
            f"- Human duration: human difficulty `r = {difference_items[1]['human_r']:.3f}` vs LLM difficulty `r = {difference_items[1]['llm_r']:.3f}`.",
            f"- Thinking advantage vs difficulty: `r = {difference_items[2]['pearson']:.3f}` on `n = {difference_items[2]['n']}`.",
            f"- Residual human difficulty after removing LLM difficulty still tracks duration: `r = {difference_items[3]['pearson']:.3f}`.",
            f"- Residual LLM difficulty after removing human difficulty still tracks cyclomatic complexity: `r = {difference_items[4]['pearson']:.3f}`.",
            f"- Human within-task difficulty range across test pairs: mean `{difference_items[5]['mean_range']:.3f}`, max `{difference_items[5]['max_range']:.3f}` across `{difference_items[5]['n_tasks']}` tasks.",
            "",
            "## Charts",
            "",
            "- `chart_audit_shared_signals.png`",
            "- `chart_audit_divergence_signals.png`",
            "- `chart_audit_human_specific_signals.png`",
        ]
    )
    (BASE_DIR / "audit_deck_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
