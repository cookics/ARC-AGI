from __future__ import annotations

import json
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
COMPARISON_CSV = MAIN_ANALYSIS_DIR / "tables" / "public_eval_human_vs_models.csv"
SPLIT_HALF_CSV = BASE_DIR / "tables" / "bootstrap_split_half_correlations.csv"

TRUTH_DIR = REPO_ROOT / "data-llm" / "ARC-AGI-2" / "data" / "evaluation"
PREDS_DIR = REPO_ROOT / "data-llm" / "arc_agi_v2_public_eval"

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


def normalize_grid(grid: object) -> str:
    if not isinstance(grid, list) or not grid:
        return "EMPTY"
    return ",".join(str(cell) for row in grid for cell in row)


def family_for_model(model: str) -> str:
    if model.startswith(("gpt-5", "gpt-4")):
        return "GPT"
    if model.startswith("claude-opus"):
        return "Claude Opus"
    if model.startswith("claude-sonnet"):
        return "Claude Sonnet"
    if model.startswith("claude-haiku"):
        return "Claude Haiku"
    if model.startswith("gemini"):
        return "Gemini"
    if model.startswith("grok"):
        return "Grok"
    if model.startswith("qwen"):
        return "Qwen"
    return "Other"


def short_label(model: str) -> str:
    replacements = {
        "gpt-5-1-2025-11-13-thinking-low": "5.1 low",
        "gpt-5-1-2025-11-13-thinking-medium": "5.1 med",
        "gpt-5-1-2025-11-13-thinking-high": "5.1 high",
        "gpt-5-2-2025-12-11-thinking-low": "5.2 low",
        "gpt-5-2-2025-12-11-thinking-medium": "5.2 med",
        "gpt-5-2-2025-12-11-thinking-high": "5.2 high",
        "gpt-5-2-2025-12-11-thinking-xhigh": "5.2 xhigh",
        "gpt-5-2-pro-2025-12-11-medium": "5.2 pro med",
        "gpt-5-2-pro-2025-12-11-high": "5.2 pro high",
        "gpt-5-pro-2025-10-06": "5 pro",
        "claude-opus-4-5-20251101-thinking-8k": "Opus 8k",
        "claude-opus-4-5-20251101-thinking-16k": "Opus 16k",
        "claude-opus-4-5-20251101-thinking-32k": "Opus 32k",
        "claude-opus-4-5-20251101-thinking-64k": "Opus 64k",
        "claude-opus-4-5-20251101-thinking-none": "Opus none",
        "gemini-3-flash-preview-thinking-minimal": "Flash min",
        "gemini-3-flash-preview-thinking-low": "Flash low",
        "gemini-3-flash-preview-thinking-medium": "Flash med",
        "gemini-3-flash-preview-thinking-high": "Flash high",
    }
    return replacements.get(model, model)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison = pd.read_csv(COMPARISON_CSV)
    comparison = comparison[comparison["attempts"] >= 8].copy()
    split_halves = pd.read_csv(SPLIT_HALF_CSV)
    return comparison, split_halves


def build_model_matrix(pair_ids: list[str]) -> pd.DataFrame:
    truth_cache = {path.name: json.loads(path.read_text()) for path in TRUTH_DIR.glob("*.json")}
    pair_set = set(pair_ids)
    truth_outputs: dict[str, str] = {}

    for path in TRUTH_DIR.glob("*.json"):
        obj = truth_cache[path.name]
        for idx, pair in enumerate(obj.get("test", [])):
            pair_id = f"{path.stem}__{idx}"
            if pair_id in pair_set:
                truth_outputs[pair_id] = normalize_grid(pair["output"])

    rows: dict[str, dict[str, int]] = {}
    for model_dir in sorted(PREDS_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        row = {pair_id: 0 for pair_id in pair_ids}
        for pred_path in model_dir.glob("*.json"):
            truth_obj = truth_cache.get(pred_path.name)
            if truth_obj is None:
                continue
            pred_obj = json.loads(pred_path.read_text())
            for idx, _pair in enumerate(truth_obj.get("test", [])):
                pair_id = f"{pred_path.stem}__{idx}"
                if pair_id not in row:
                    continue
                pred_entry = None
                for candidate in pred_obj:
                    if candidate.get("metadata", {}).get("pair_index") == idx:
                        pred_entry = candidate
                        break
                if pred_entry is None and idx < len(pred_obj):
                    pred_entry = pred_obj[idx]
                answer = None
                if pred_entry:
                    answer = (pred_entry.get("attempt_1") or {}).get("answer")
                    if not answer:
                        answer = (pred_entry.get("attempt_2") or {}).get("answer")
                row[pair_id] = int(normalize_grid(answer) == truth_outputs[pair_id])
        rows[model_dir.name] = row

    return pd.DataFrame.from_dict(rows, orient="index")[pair_ids]


def corr_or_nan(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    if np.unique(arr_a).size < 2 or np.unique(arr_b).size < 2:
        return np.nan
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def build_single_model_table(
    comparison: pd.DataFrame, model_matrix: pd.DataFrame, split_halves: pd.DataFrame
) -> pd.DataFrame:
    human = comparison.set_index("task_pair_id")["solve_rate"]
    rows = []
    for model in model_matrix.index:
        profile = model_matrix.loc[model]
        pearson = corr_or_nan(profile, human)
        if pd.isna(pearson):
            continue
        rows.append(
            {
                "model": model,
                "label": short_label(model),
                "family": family_for_model(model),
                "pair_accuracy": float(profile.mean()),
                "human_pearson": pearson,
                "human_spearman": float(profile.corr(human, method="spearman")),
                "percentile_vs_human_split": float((split_halves["pearson"] <= pearson).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["human_pearson", "pair_accuracy"], ascending=[False, False])


def simulate_ensemble_sizes(
    comparison: pd.DataFrame,
    model_matrix: pd.DataFrame,
    split_halves: pd.DataFrame,
    sizes: list[int] | None = None,
    n_sims: int = 3000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    human = comparison.set_index("task_pair_id")["solve_rate"]
    valid_models = [model for model in model_matrix.index if model_matrix.loc[model].nunique() >= 2]
    rng = np.random.default_rng(seed)

    if sizes is None:
        sizes = [1, 2, 3, 5, 8, 12, 20, 30]
    sizes = [size for size in sizes if size <= len(valid_models)]

    draw_rows: list[dict] = []
    summary_rows: list[dict] = []
    for size in sizes:
        values: list[float] = []
        for sim_idx in range(n_sims):
            chosen = rng.choice(valid_models, size=size, replace=False)
            profile = model_matrix.loc[list(chosen)].mean(axis=0)
            pearson = corr_or_nan(profile, human)
            if pd.isna(pearson):
                continue
            values.append(pearson)
            draw_rows.append(
                {
                    "ensemble_size": size,
                    "simulation": sim_idx,
                    "human_pearson": pearson,
                }
            )

        vals = np.asarray(values, dtype=float)
        summary_rows.append(
            {
                "ensemble_size": size,
                "n_draws": int(len(vals)),
                "median_pearson": float(np.median(vals)),
                "ci_lo": float(np.quantile(vals, 0.025)),
                "ci_hi": float(np.quantile(vals, 0.975)),
                "median_percentile_vs_human_split": float((split_halves["pearson"] <= np.median(vals)).mean()),
            }
        )

    draws = pd.DataFrame(draw_rows)
    summary = pd.DataFrame(summary_rows)
    return summary, draws


def build_family_consensus_tables(comparison: pd.DataFrame, model_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    human = comparison.set_index("task_pair_id")["solve_rate"]
    family_members: dict[str, list[str]] = {}
    for model in model_matrix.index:
        family = family_for_model(model)
        family_members.setdefault(family, []).append(model)

    profiles: dict[str, pd.Series] = {"Human": human}
    summary_rows = []
    for family, models in sorted(family_members.items()):
        if family in {"Other", "Qwen"}:
            continue
        profile = model_matrix.loc[models].mean(axis=0)
        if profile.nunique() < 2:
            continue
        profiles[family] = profile
        summary_rows.append(
            {
                "family": family,
                "n_models": len(models),
                "human_pearson": corr_or_nan(profile, human),
                "human_spearman": float(profile.corr(human, method="spearman")),
                "mean_pair_accuracy": float(profile.mean()),
            }
        )

    corr_df = pd.DataFrame(profiles).corr()
    summary = pd.DataFrame(summary_rows).sort_values("human_pearson", ascending=False)
    return corr_df, summary


def plot_ensemble_sizes(
    ensemble_summary: pd.DataFrame,
    ensemble_draws: pd.DataFrame,
    split_halves: pd.DataFrame,
    average_model_corr: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=ensemble_draws,
        x="ensemble_size",
        y="human_pearson",
        color="#A7C5EB",
        showfliers=False,
        width=0.6,
        ax=ax,
    )
    ax.axhspan(
        float(split_halves["pearson"].quantile(0.025)),
        float(split_halves["pearson"].quantile(0.975)),
        color="#9FD0CB",
        alpha=0.25,
        label="Human split-half 95% interval",
    )
    ax.axhline(float(split_halves["pearson"].median()), color="#1F3552", linewidth=2, label="Human split-half median")
    ax.axhline(average_model_corr, color="#F58518", linewidth=2, linestyle="--", label="Observed average model")

    medians = ensemble_summary["median_pearson"].to_numpy()
    xpos = np.arange(len(ensemble_summary))
    ax.plot(xpos, medians, color="#4C78A8", marker="o", linewidth=2)

    ax.set_title("Human alignment rises as we average over more models")
    ax.set_xlabel("Random ensemble size")
    ax.set_ylabel("Pearson correlation with human item solve rates")
    ax.legend(frameon=False)
    fig.text(0.01, 0.01, "Each box summarizes random model ensembles on Public Eval task pairs with at least 8 human attempts.", fontsize=10)
    fig.savefig(FIGURES_DIR / "fig04_ensemble_size_alignment.png")
    plt.close(fig)


def plot_family_heatmap(corr_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Consensus item profiles: humans vs model families")
    fig.savefig(FIGURES_DIR / "fig05_family_consensus_heatmap.png")
    plt.close(fig)


def write_report(
    split_halves: pd.DataFrame,
    comparison: pd.DataFrame,
    single_models: pd.DataFrame,
    ensemble_summary: pd.DataFrame,
    family_corr: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> None:
    avg_corr = corr_or_nan(comparison["lm_mean"], comparison["solve_rate"])
    best_single_by_score = single_models.sort_values("pair_accuracy", ascending=False).iloc[0]
    best_single_by_alignment = single_models.iloc[0]
    median_single = single_models["human_pearson"].median()
    ensemble_30 = ensemble_summary.loc[ensemble_summary["ensemble_size"] == ensemble_summary["ensemble_size"].max()].iloc[0]
    family_human_mean = float(family_summary["human_pearson"].mean())

    family_only = family_corr.drop(index="Human", columns="Human")
    family_pair_mean = float(family_only.where(~np.eye(len(family_only), dtype=bool)).stack().mean())

    lines = [
        "# Stronger Construct Checks",
        "",
        "This note pushes the ARC human-vs-model comparison one step further. It still cannot prove shared reasoning, but it can test whether the average-model result is mostly an ensemble phenomenon and whether model families cluster together more tightly than they cluster with humans.",
        "",
        "## What we can and cannot test",
        "",
        "- We can test item-level alignment, ensemble effects, and cross-family convergence because we have pair-level human correctness and pair-level model correctness on the same Public Eval items.",
        "- We cannot directly test solution-path similarity or wrong-answer similarity, because the human testing file does not contain human grid outputs or action traces.",
        "",
        "## Stronger results we do have",
        "",
        f"- The existing analyses already showed one important difference: the best single model by score (`{best_single_by_score['label']}`) is not especially human-aligned ({best_single_by_score['human_pearson']:.3f}, {100 * best_single_by_score['percentile_vs_human_split']:.1f}th percentile vs the human split-half distribution).",
        f"- But if we cherry-pick for alignment instead of score, the most human-aligned single model is `{best_single_by_alignment['label']}` at {best_single_by_alignment['human_pearson']:.3f}, which lands at the {100 * best_single_by_alignment['percentile_vs_human_split']:.1f}th percentile of the human split-half distribution. That is interesting, but it is not the same as saying frontier models are generally human-like.",
        f"- Across all non-degenerate single models, the median human-correlation is only {median_single:.3f}, well below the human split-half median of {split_halves['pearson'].median():.3f}.",
        "",
        "## Ensemble effect",
        "",
        f"- The observed average-model correlation is {avg_corr:.3f}.",
        f"- Random one-model draws are usually much lower than that. As ensemble size grows, the median human-correlation rises steadily.",
        f"- By the largest tested ensemble size ({int(ensemble_30['ensemble_size'])} models), the median random-ensemble correlation reaches {ensemble_30['median_pearson']:.3f}, which is almost exactly at the human split-half median.",
        "- That means the strong aggregate result can be reproduced by averaging over many imperfect, partially idiosyncratic models. This is a concrete non-`general intelligence` explanation for why the average-model profile looks so human-aligned.",
        "",
        "## Cross-family convergence",
        "",
        f"- Model-family consensus profiles are fairly correlated with humans, but they are typically even more correlated with each other. The average human-to-family correlation is {family_human_mean:.3f}, whereas the average family-to-family correlation is {family_pair_mean:.3f}.",
        "- That pattern is what you would expect if there is a shared machine consensus about which items are broadly easy or hard, without needing to assume that the machines and humans are using the same cognitive process.",
        "",
        "## Interpretation",
        "",
        "- These checks strengthen the skeptical interpretation more than the `same construct` interpretation.",
        "- The average-model benchmark looks real, but much of it is explainable as consensus smoothing across many models rather than a single model cleanly matching human cognition.",
        "- The remaining open possibility is a shared latent difficulty axis that is richer than trivial grid-size cues but still much weaker than psychometric equivalence.",
        "",
        "## Top single-model alignments",
        "",
        frame_to_text_table(
            single_models[
                ["label", "family", "pair_accuracy", "human_pearson", "percentile_vs_human_split"]
            ]
            .head(12)
            .round(3)
        ),
        "",
        "## Ensemble size summary",
        "",
        frame_to_text_table(ensemble_summary.round(3)),
        "",
        "## Family summary",
        "",
        frame_to_text_table(family_summary.round(3)),
        "",
    ]

    (BASE_DIR / "stronger_construct_checks.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_style()
    ensure_dirs()

    comparison, split_halves = load_inputs()
    model_matrix = build_model_matrix(comparison["task_pair_id"].tolist())
    single_models = build_single_model_table(comparison, model_matrix, split_halves)
    ensemble_summary, ensemble_draws = simulate_ensemble_sizes(comparison, model_matrix, split_halves)
    family_corr, family_summary = build_family_consensus_tables(comparison, model_matrix)

    single_models.to_csv(TABLES_DIR / "single_model_split_context.csv", index=False)
    ensemble_summary.to_csv(TABLES_DIR / "ensemble_size_alignment.csv", index=False)
    ensemble_draws.to_csv(TABLES_DIR / "ensemble_size_draws.csv", index=False)
    family_corr.to_csv(TABLES_DIR / "family_consensus_correlations.csv")
    family_summary.to_csv(TABLES_DIR / "family_consensus_summary.csv", index=False)

    plot_ensemble_sizes(
        ensemble_summary=ensemble_summary,
        ensemble_draws=ensemble_draws,
        split_halves=split_halves,
        average_model_corr=corr_or_nan(comparison["lm_mean"], comparison["solve_rate"]),
    )
    plot_family_heatmap(family_corr)
    write_report(split_halves, comparison, single_models, ensemble_summary, family_corr, family_summary)

    print(f"Done. Outputs written to {BASE_DIR}")


if __name__ == "__main__":
    main()
