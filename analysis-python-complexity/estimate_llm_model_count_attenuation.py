from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent

RESPONSE_MATRIX_PATH = ANALYSIS_DIR / "llm_response_matrix_arc_agi_1_eval.csv"
OLD_JOIN_PATH = ANALYSIS_DIR / "approved_llm_complexity_join.csv"

RESULTS_PATH = ANALYSIS_DIR / "llm_model_count_attenuation.csv"
SUMMARY_PATH = ANALYSIS_DIR / "llm_model_count_attenuation_summary.json"
PLOT_PATH = ANALYSIS_DIR / "chart_llm_model_count_attenuation.png"


def logit_fail(pass_rate: pd.Series, eps: float = 1e-4) -> pd.Series:
    fail_rate = 1.0 - pass_rate
    clipped = fail_rate.clip(eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def sample_subsets(model_names: list[str], k: int, rng: np.random.Generator, draws: int) -> list[tuple[str, ...]]:
    if k == len(model_names):
        return [tuple(model_names)]
    if k <= 3 and math.comb(len(model_names), k) <= draws:
        return list(combinations(model_names, k))
    seen: set[tuple[str, ...]] = set()
    while len(seen) < draws:
        subset = tuple(sorted(rng.choice(model_names, size=k, replace=False).tolist()))
        seen.add(subset)
    return list(seen)


def main() -> None:
    rng = np.random.default_rng(0)

    responses_wide = pd.read_csv(RESPONSE_MATRIX_PATH, index_col=0)
    old = pd.read_csv(OLD_JOIN_PATH)
    old_arc1 = old.loc[old["dataset_key"] == "arc_agi_1_eval"].copy()
    task_ids = old_arc1["task_id"].tolist()

    responses = responses_wide[task_ids].astype(float)
    model_names = responses.index.tolist()
    full_pass_rate = responses.mean(axis=0)
    full_logit = logit_fail(full_pass_rate)
    old_arc1["full_sample_logit"] = old_arc1["task_id"].map(full_logit.to_dict())

    pair_corrs = responses.T.corr()
    mask = np.triu(np.ones(pair_corrs.shape, dtype=bool), 1)
    mean_pair_corr = float(pair_corrs.where(mask).stack().mean())

    metric_cols = ["log1p_cyclomatic_complexity", "ast_node_count", "complexity_pc1_score"]
    subset_sizes = [1, 2, 3, 5, 8, 12, 24]
    draws_by_k = {1: 24, 2: 276, 3: 2024, 5: 3000, 8: 3000, 12: 3000, 24: 1}

    rows: list[dict[str, object]] = []
    for k in subset_sizes:
        subsets = sample_subsets(model_names, k, rng, draws_by_k[k])
        for subset in subsets:
            subset_pass_rate = responses.loc[list(subset)].mean(axis=0)
            subset_logit = logit_fail(subset_pass_rate)
            joined = old_arc1.copy()
            joined["subset_logit_difficulty"] = joined["task_id"].map(subset_logit.to_dict())
            diff_corr = float(joined["subset_logit_difficulty"].corr(joined["full_sample_logit"], method="pearson"))
            for metric in metric_cols:
                pearson_r = float(joined[metric].corr(joined["subset_logit_difficulty"], method="pearson"))
                rows.append(
                    {
                        "k_models": k,
                        "subset_size": k,
                        "draw_id": "|".join(subset),
                        "complexity_metric": metric,
                        "pearson_r": pearson_r,
                        "difficulty_vs_full_r": diff_corr,
                    }
                )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)

    grouped = (
        results.groupby(["k_models", "complexity_metric"], as_index=False)
        .agg(
            mean_pearson_r=("pearson_r", "mean"),
            sd_pearson_r=("pearson_r", "std"),
            q10_pearson_r=("pearson_r", lambda s: s.quantile(0.10)),
            q50_pearson_r=("pearson_r", "median"),
            q90_pearson_r=("pearson_r", lambda s: s.quantile(0.90)),
            mean_difficulty_vs_full_r=("difficulty_vs_full_r", "mean"),
        )
    )

    grouped["attenuation_vs_k24"] = np.nan
    for metric in metric_cols:
        full_val = grouped.loc[
            (grouped["k_models"] == 24) & (grouped["complexity_metric"] == metric), "mean_pearson_r"
        ].iloc[0]
        mask_metric = grouped["complexity_metric"] == metric
        grouped.loc[mask_metric, "attenuation_vs_k24"] = grouped.loc[mask_metric, "mean_pearson_r"] / full_val

    summary = {
        "num_models": len(model_names),
        "num_tasks": len(task_ids),
        "mean_pairwise_model_corr": mean_pair_corr,
        "grouped_results": grouped.to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for metric, color in zip(metric_cols, ["#1f77b4", "#d62728", "#2ca02c"], strict=False):
        sub = grouped[grouped["complexity_metric"] == metric]
        axes[0].plot(sub["k_models"], sub["mean_pearson_r"], marker="o", color=color, label=metric)
        axes[0].fill_between(
            sub["k_models"],
            sub["q10_pearson_r"],
            sub["q90_pearson_r"],
            color=color,
            alpha=0.15,
        )
        axes[1].plot(
            sub["k_models"],
            sub["mean_difficulty_vs_full_r"],
            marker="o",
            color=color,
            label=metric,
        )

    axes[0].set_title("Complexity correlation by number of models")
    axes[0].set_xlabel("Number of models in sampled difficulty score")
    axes[0].set_ylabel("Pearson r with sampled difficulty")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Sampled difficulty vs full 24-model difficulty")
    axes[1].set_xlabel("Number of models in sampled difficulty score")
    axes[1].set_ylabel("Pearson r")
    axes[1].grid(alpha=0.2)

    fig.suptitle("Attenuation from using fewer LLMs on ARC-1 eval", fontsize=13)
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
