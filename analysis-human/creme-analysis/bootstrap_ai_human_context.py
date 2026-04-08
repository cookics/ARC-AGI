from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
HUMAN_DATA_DIR = REPO_ROOT / "data-human"
MAIN_ANALYSIS_DIR = BASE_DIR.parent / "analysis"

RAW_HUMAN_CSV = HUMAN_DATA_DIR / "test_pair_attempts.csv"
COMPARISON_CSV = MAIN_ANALYSIS_DIR / "tables" / "public_eval_human_vs_models.csv"

FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def frame_to_text_table(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    human = pd.read_csv(RAW_HUMAN_CSV)
    human = human[human["task_set"] == "Public Eval"].copy()
    human["solved"] = (human["correct_submissions"] > 0).astype(int)
    human["task_pair_id"] = human["task_ID"] + "__" + human["test_index"].astype(str)

    comparison = pd.read_csv(COMPARISON_CSV)
    comparison = comparison[comparison["attempts"] >= 8].copy()
    robust_pairs = set(comparison["task_pair_id"])
    human = human[human["task_pair_id"].isin(robust_pairs)].copy()
    return human, comparison


def split_half_correlations(human: pd.DataFrame, n_sims: int = 5000, seed: int = 0) -> pd.DataFrame:
    sessions = np.array(sorted(human["session_ID"].unique()))
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for _ in range(n_sims):
        perm = rng.permutation(sessions)
        half_a = set(perm[: len(perm) // 2])

        part_a = human[human["session_ID"].isin(half_a)].groupby("task_pair_id").agg(rate=("solved", "mean"), n=("solved", "size"))
        part_b = human[~human["session_ID"].isin(half_a)].groupby("task_pair_id").agg(rate=("solved", "mean"), n=("solved", "size"))
        merged = part_a.join(part_b, lsuffix="_a", rsuffix="_b", how="inner")
        merged = merged[(merged["n_a"] >= 2) & (merged["n_b"] >= 2)]

        if len(merged) < 20:
            continue

        rows.append(
            {
                "pearson": merged["rate_a"].corr(merged["rate_b"]),
                "spearman": merged["rate_a"].corr(merged["rate_b"], method="spearman"),
                "n_items": len(merged),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_corr(y: np.ndarray, x: np.ndarray, n_boot: int = 10000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        draws[i] = np.corrcoef(y[idx], x[idx])[0, 1]
    return draws


def build_ai_bootstrap_summary(comparison: pd.DataFrame, split_halves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = comparison["solve_rate"].to_numpy()
    rows = []
    draw_rows = []

    for seed, (label, col, color) in enumerate(
        [
            ("Average model", "lm_mean", "#F58518"),
            ("Best single model", "lm_best_single_model", "#E45756"),
            ("Per-pair oracle", "lm_best_across_models", "#72B7B2"),
        ]
    ):
        x = comparison[col].to_numpy()
        obs_pearson = float(np.corrcoef(y, x)[0, 1])
        obs_spearman = float(comparison["solve_rate"].corr(comparison[col], method="spearman"))
        draws = bootstrap_corr(y, x, n_boot=8000, seed=seed)
        draw_rows.extend({"series": label, "draw": float(draw)} for draw in draws)
        rows.append(
            {
                "series": label,
                "color": color,
                "observed_pearson": obs_pearson,
                "observed_spearman": obs_spearman,
                "bootstrap_median": float(np.median(draws)),
                "bootstrap_ci_lo": float(np.quantile(draws, 0.025)),
                "bootstrap_ci_hi": float(np.quantile(draws, 0.975)),
                "percentile_vs_human_split": float((split_halves["pearson"] <= obs_pearson).mean()),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(draw_rows)


def plot_bootstrap_context(split_halves: pd.DataFrame, ai_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(split_halves["pearson"], bins=35, color="#4C78A8", alpha=0.75, ax=ax)

    median = split_halves["pearson"].median()
    ci_lo = split_halves["pearson"].quantile(0.025)
    ci_hi = split_halves["pearson"].quantile(0.975)
    ax.axvline(median, color="#1F3552", linewidth=2.5, label="Human split-half median")
    ax.axvspan(ci_lo, ci_hi, color="#4C78A8", alpha=0.10, label="Human split-half 95% interval")

    for _, row in ai_summary.iterrows():
        ax.axvspan(row["bootstrap_ci_lo"], row["bootstrap_ci_hi"], color=row["color"], alpha=0.12)
        ax.axvline(row["observed_pearson"], color=row["color"], linestyle="--", linewidth=2.5, label=f"{row['series']} observed")

    ax.set_title("AI-vs-human correlation in context of human split-half correlations")
    ax.set_xlabel("Pearson correlation across robust Public Eval task pairs")
    ax.set_ylabel("Random split-half simulations")
    ax.legend(frameon=False, ncol=2)
    fig.text(
        0.01,
        0.01,
        "Human reference uses Public Eval task pairs with >=8 human attempts overall and requires >=2 attempts in each split-half.",
        fontsize=10,
    )
    fig.savefig(FIGURES_DIR / "fig03_bootstrap_ai_vs_human_context.png")
    plt.close(fig)


def write_note(split_halves: pd.DataFrame, ai_summary: pd.DataFrame) -> None:
    median = split_halves["pearson"].median()
    ci_lo = split_halves["pearson"].quantile(0.025)
    ci_hi = split_halves["pearson"].quantile(0.975)

    lines = [
        "# Bootstrap AI vs Human Context",
        "",
        "This note puts the AI-vs-human item correlation in context by comparing it to a large reference distribution of human-vs-human split-half correlations.",
        "",
        "## Setup",
        "",
        "- Restrict to Public Eval task pairs with at least 8 human attempts overall.",
        "- Randomly split human sessions into two halves 5,000 times and compute the item-level correlation each time.",
        "- Bootstrap the AI-vs-human item correlation 8,000 times by resampling task pairs with replacement.",
        "",
        "## Human reference",
        "",
        f"- Human split-half median Pearson correlation: {median:.3f}",
        f"- Human split-half 95% interval: [{ci_lo:.3f}, {ci_hi:.3f}]",
        "",
        "## AI in context",
        "",
        frame_to_text_table(
            ai_summary[
                [
                    "series",
                    "observed_pearson",
                    "bootstrap_median",
                    "bootstrap_ci_lo",
                    "bootstrap_ci_hi",
                    "percentile_vs_human_split",
                ]
            ].round(3)
        ),
        "",
        "## Takeaway",
        "",
        "- The average-model correlation sits right in the middle of the human split-half distribution, so that aggregate AI profile is genuinely tracking human difficulty structure on this ARC subset.",
        "- The best single model sits much lower in the human split-half distribution, so a single frontier model is still not as human-like as one human subsample is to another.",
        "- This is the cleanest context for the AI correlation: not just `is it above zero?`, but `is it as strong as a noisy human-vs-human benchmark?`",
        "",
    ]

    (BASE_DIR / "bootstrap_ai_human_context.md").write_text("\n".join(lines), encoding="utf-8")

    main_lines = [
        "# Bootstrap Context Note",
        "",
        f"Human split-half median correlation on the robust Public Eval set is {median:.3f} with a 95% interval of [{ci_lo:.3f}, {ci_hi:.3f}].",
        "The average-model human correlation lands right around the middle of that distribution, while the best single model lands much lower.",
        "So the AI correlation is real, but only the aggregate model profile looks fully competitive with the noisy human-vs-human benchmark on this subset.",
        "",
        "Full note and plot live in `analysis-human/creme-analysis/bootstrap_ai_human_context.md` and `fig03_bootstrap_ai_vs_human_context.png`.",
        "",
    ]
    (MAIN_ANALYSIS_DIR / "bootstrap_context_note.md").write_text("\n".join(main_lines), encoding="utf-8")


def main() -> None:
    configure_style()
    ensure_dirs()

    human, comparison = load_inputs()
    split_halves = split_half_correlations(human, n_sims=5000, seed=0)
    ai_summary, ai_draws = build_ai_bootstrap_summary(comparison, split_halves)

    split_halves.to_csv(TABLES_DIR / "bootstrap_split_half_correlations.csv", index=False)
    ai_summary.to_csv(TABLES_DIR / "bootstrap_ai_context_summary.csv", index=False)
    ai_draws.to_csv(TABLES_DIR / "bootstrap_ai_context_draws.csv", index=False)

    plot_bootstrap_context(split_halves, ai_summary)
    write_note(split_halves, ai_summary)

    print(f"Done. Outputs written to {BASE_DIR}")


if __name__ == "__main__":
    main()
