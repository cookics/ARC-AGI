from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent

OLD_JOIN_PATH = ANALYSIS_DIR / "approved_llm_complexity_join.csv"
OLD_HEADLINE_PATH = ANALYSIS_DIR / "approved_llm_headline_by_dataset.csv"
NEW_JOIN_PATH = ANALYSIS_DIR / "arc1_dsl_expanded_oldstyle_join.csv"
NEW_SUMMARY_PATH = ANALYSIS_DIR / "arc1_dsl_human_llm_summary.json"

COMPARISON_PATH = ANALYSIS_DIR / "old_new_complexity_regime_comparison.csv"
SUMMARY_PATH = ANALYSIS_DIR / "old_new_complexity_regime_summary.json"
REPORT_PATH = ANALYSIS_DIR / "old_new_complexity_regime_report.md"
COMPLEXITY_PLOT_PATH = ANALYSIS_DIR / "chart_old_vs_new_complexity_scatter.png"
ALIGNMENT_PLOT_PATH = ANALYSIS_DIR / "chart_old_vs_new_human_llm_alignment.png"


def fisher_compare_independent(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    if any(math.isnan(v) for v in (r1, r2)) or n1 <= 3 or n2 <= 3:
        return float("nan"), float("nan")
    z1 = np.arctanh(np.clip(r1, -0.999999, 0.999999))
    z2 = np.arctanh(np.clip(r2, -0.999999, 0.999999))
    se = math.sqrt((1.0 / (n1 - 3)) + (1.0 / (n2 - 3)))
    if se == 0:
        return float("nan"), float("nan")
    z = (z1 - z2) / se
    p = 2 * stats.norm.sf(abs(z))
    return float(z), float(p)


def corr(df: pd.DataFrame, x: str, y: str) -> tuple[int, float, float]:
    pair = df[[x, y]].dropna()
    if len(pair) < 3:
        return len(pair), float("nan"), float("nan")
    pearson = float(pair[x].corr(pair[y], method="pearson"))
    spearman = float(pair[x].corr(pair[y], method="spearman"))
    return len(pair), pearson, spearman


def williams_test(r_xy: float, r_xz: float, r_yz: float, n: int) -> tuple[float, float]:
    if n <= 3 or any(math.isnan(v) for v in (r_xy, r_xz, r_yz)):
        return float("nan"), float("nan")
    k = 1 - r_xy**2 - r_xz**2 - r_yz**2 + 2 * r_xy * r_xz * r_yz
    if k <= 0:
        return float("nan"), float("nan")
    numerator = (r_xy - r_xz) * math.sqrt((n - 1) * (1 + r_yz))
    denominator = math.sqrt(
        (2 * (n - 1) / (n - 3)) * k + (((r_xy + r_xz) ** 2) / 4.0) * ((1 - r_yz) ** 3)
    )
    if denominator == 0:
        return float("nan"), float("nan")
    t_value = numerator / denominator
    p_value = 2 * stats.t.sf(abs(t_value), df=n - 3)
    return float(t_value), float(p_value)


def add_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if len(x) < 2:
        return
    order = np.argsort(x)
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(x[order], intercept + slope * x[order], color=color, linewidth=2.0)


def plot_scatter(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, title: str, color: str) -> tuple[int, float]:
    pair = df[[x, y]].dropna()
    n = len(pair)
    r = float(pair[x].corr(pair[y], method="pearson")) if n >= 3 else float("nan")
    ax.scatter(pair[x], pair[y], s=18, alpha=0.75, color=color, edgecolors="none")
    add_fit(ax, pair[x].to_numpy(dtype=float), pair[y].to_numpy(dtype=float), color)
    ax.set_title(f"{title}\n(n={n}, r={r:.3f})", fontsize=10)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.2)
    return n, r


def main() -> None:
    old = pd.read_csv(OLD_JOIN_PATH)
    old_headline = pd.read_csv(OLD_HEADLINE_PATH)
    new = pd.read_csv(NEW_JOIN_PATH)
    new_summary = json.loads(NEW_SUMMARY_PATH.read_text())

    old_arc1 = old.loc[old["dataset_key"] == "arc_agi_1_eval"].copy()
    new_gap30 = new.loc[new["human_llm_pair_gap"] <= 0.30].copy()

    comparisons: list[dict[str, object]] = []

    metric_specs = [
        ("log1p_cyclomatic_complexity", "old_logit_difficulty_all", "pooled_pair_difficulty"),
        ("ast_node_count", "old_logit_difficulty_all", "pooled_pair_difficulty"),
        ("complexity_pc1_score", "old_logit_difficulty_all", "pooled_pair_difficulty"),
    ]

    old_frames = [
        ("old_overlap56_many_model", old, "logit_difficulty_all"),
        ("old_arc1_eval_many_model", old_arc1, "logit_difficulty_all"),
    ]
    new_frames = [
        ("new_arc1_train_full_two_model", new, "pooled_pair_difficulty"),
        ("new_arc1_train_gap30_two_model", new_gap30, "pooled_pair_difficulty"),
        ("new_arc1_train_full_human", new, "human_difficulty_complete"),
        ("new_arc1_train_gap30_human", new_gap30, "human_difficulty_complete"),
    ]

    dataset_rows: dict[tuple[str, str], tuple[int, float, float]] = {}
    for metric, _, _ in metric_specs:
        for label, frame, outcome in old_frames + new_frames:
            n, pearson, spearman = corr(frame, metric, outcome)
            dataset_rows[(label, metric)] = (n, pearson, spearman)
            comparisons.append(
                {
                    "dataset_label": label,
                    "complexity_metric": metric,
                    "outcome_metric": outcome,
                    "n": n,
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                }
            )

    fisher_rows: list[dict[str, object]] = []
    for metric, _, _ in metric_specs:
        old_n56, old_r56, _ = dataset_rows[("old_overlap56_many_model", metric)]
        old_n38, old_r38, _ = dataset_rows[("old_arc1_eval_many_model", metric)]
        for new_label in (
            "new_arc1_train_full_two_model",
            "new_arc1_train_gap30_two_model",
            "new_arc1_train_full_human",
            "new_arc1_train_gap30_human",
        ):
            new_n, new_r, _ = dataset_rows[(new_label, metric)]
            z56, p56 = fisher_compare_independent(old_r56, old_n56, new_r, new_n)
            z38, p38 = fisher_compare_independent(old_r38, old_n38, new_r, new_n)
            fisher_rows.append(
                {
                    "complexity_metric": metric,
                    "new_dataset_label": new_label,
                    "old_overlap56_r": old_r56,
                    "old_overlap56_n": old_n56,
                    "new_r": new_r,
                    "new_n": new_n,
                    "fisher_z_vs_old_overlap56": z56,
                    "fisher_p_vs_old_overlap56": p56,
                    "old_arc1_eval_r": old_r38,
                    "old_arc1_eval_n": old_n38,
                    "fisher_z_vs_old_arc1_eval": z38,
                    "fisher_p_vs_old_arc1_eval": p38,
                }
            )

    comparison_df = pd.DataFrame(comparisons)
    fisher_df = pd.DataFrame(fisher_rows)
    output_df = comparison_df.merge(
        fisher_df,
        how="left",
        left_on=["complexity_metric", "dataset_label"],
        right_on=["complexity_metric", "new_dataset_label"],
    )
    output_df.to_csv(COMPARISON_PATH, index=False)

    # Human vs pooled within the new dataset, same metric.
    within_rows: list[dict[str, object]] = []
    for subset_label, frame in [("full", new), ("gap30", new_gap30)]:
        for metric in ("log1p_cyclomatic_complexity", "ast_node_count", "complexity_pc1_score"):
            pair = frame[[metric, "human_difficulty_complete", "pooled_pair_difficulty"]].dropna()
            n = len(pair)
            r_h = float(pair[metric].corr(pair["human_difficulty_complete"], method="pearson"))
            r_p = float(pair[metric].corr(pair["pooled_pair_difficulty"], method="pearson"))
            r_hp = float(pair["human_difficulty_complete"].corr(pair["pooled_pair_difficulty"], method="pearson"))
            t_value, p_value = williams_test(r_h, r_p, r_hp, n)
            within_rows.append(
                {
                    "subset_label": subset_label,
                    "complexity_metric": metric,
                    "n": n,
                    "human_r": r_h,
                    "pooled_pair_r": r_p,
                    "human_vs_pooled_r": r_hp,
                    "williams_t": t_value,
                    "williams_p": p_value,
                }
            )
    within_df = pd.DataFrame(within_rows)

    # Scatterplots.
    plt.style.use("default")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    plot_scatter(
        axes[0],
        old,
        "log1p_cyclomatic_complexity",
        "logit_difficulty_all",
        "Old many-model overlap",
        "#1f77b4",
    )
    plot_scatter(
        axes[1],
        new,
        "log1p_cyclomatic_complexity",
        "pooled_pair_difficulty",
        "New ARC-1 train full",
        "#d62728",
    )
    plot_scatter(
        axes[2],
        new_gap30,
        "log1p_cyclomatic_complexity",
        "pooled_pair_difficulty",
        "New ARC-1 train matched gap<=0.30",
        "#2ca02c",
    )
    fig.suptitle("Complexity vs LLM difficulty under old and new regimes", fontsize=13)
    fig.savefig(COMPLEXITY_PLOT_PATH, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    plot_scatter(
        axes[0],
        new,
        "human_difficulty_complete",
        "pooled_pair_difficulty",
        "Human vs pooled LLM difficulty (full set)",
        "#9467bd",
    )
    plot_scatter(
        axes[1],
        new_gap30,
        "human_difficulty_complete",
        "pooled_pair_difficulty",
        "Human vs pooled LLM difficulty (gap<=0.30)",
        "#8c564b",
    )
    fig.suptitle("Human-LLM difficulty alignment changes after regime matching", fontsize=13)
    fig.savefig(ALIGNMENT_PLOT_PATH, dpi=180)
    plt.close(fig)

    summary = {
        "old_dataset_summary": old_headline.to_dict(orient="records"),
        "new_headline_differences": new_summary["headline_differences"],
        "within_metric_human_vs_pooled": within_rows,
        "fisher_comparisons": fisher_rows,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    report_lines = [
        "# Old vs New Complexity Regimes",
        "",
        "## Main read",
        "",
        "- The old high correlation is real, but it came from a many-model item-difficulty signal on a small overlap sample.",
        "- The new ARC-1 training GPT+Claude signal is much coarser, and the same complexity metric is significantly weaker on the full set.",
        "- Matching the solve-rate regime helps a lot, but it still does not fully recover the old many-model effect.",
        "",
        "## Same-metric comparison: `log1p_cyclomatic_complexity`",
        "",
    ]
    for row in fisher_rows:
        if row["complexity_metric"] != "log1p_cyclomatic_complexity":
            continue
        report_lines.append(
            "- "
            f"{row['new_dataset_label']}: new r = {row['new_r']:.3f} "
            f"(n={row['new_n']}), vs old overlap56 r = {row['old_overlap56_r']:.3f} "
            f"(p = {row['fisher_p_vs_old_overlap56']:.4g}), "
            f"vs old arc1_eval r = {row['old_arc1_eval_r']:.3f} "
            f"(p = {row['fisher_p_vs_old_arc1_eval']:.4g})."
        )
    report_lines.extend(
        [
            "",
            "## Within current ARC-1 training data",
            "",
        ]
    )
    for row in within_rows:
        report_lines.append(
            "- "
            f"{row['subset_label']} / {row['complexity_metric']}: "
            f"human r = {row['human_r']:.3f}, pooled pair r = {row['pooled_pair_r']:.3f}, "
            f"human-vs-pooled alignment = {row['human_vs_pooled_r']:.3f}, "
            f"Williams p = {row['williams_p']:.4g}."
        )
    REPORT_PATH.write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
