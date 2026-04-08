from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from split_half_latent_reliability import (
    ARC1_LLM_CACHE,
    ARC1_LLM_DIR,
    ARC1_TRUTH_DIR,
    ARC2_LLM_CACHE,
    ARC2_LLM_DIR,
    ARC2_TRUTH_DIR,
    SUMMARY_CSV,
    build_human_arc1_sidecar_subset,
    build_human_arc2_subset,
    ensure_dirs,
    fit_latent_item_difficulty,
    load_human_attempts,
    load_or_build_llm_matrix,
    load_truth_outputs,
)


BASE_DIR = Path(__file__).resolve().parent
TABLES_DIR = BASE_DIR / "tables"
OUT_SUMMARY_CSV = TABLES_DIR / "llm_sparsity_stress_summary.csv"
OUT_DRAWS_CSV = TABLES_DIR / "llm_sparsity_stress_draws.csv"
OUT_REPORT_MD = BASE_DIR / "llm_sparsity_stress_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress-test LLM latent split-half reliability under human-like sparsity.")
    parser.add_argument("--n-sims", type=int, default=1000, help="Number of random masked split-half simulations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--min-human-attempts",
        type=int,
        default=8,
        help="Minimum total human attempts for an item to enter the benchmark subset.",
    )
    parser.add_argument("--min-items", type=int, default=20, help="Minimum usable items required per split.")
    parser.add_argument(
        "--min-item-obs-per-half",
        type=int,
        default=2,
        help="Minimum observed responses per item in each split half after masking.",
    )
    parser.add_argument(
        "--rebuild-matrices",
        action="store_true",
        help="Rebuild ARC1/ARC2 LLM matrices instead of using cached CSVs.",
    )
    return parser.parse_args()


def matrix_to_long(matrix: pd.DataFrame) -> pd.DataFrame:
    long_df = matrix.reset_index(names="respondent_id").melt(
        id_vars="respondent_id",
        var_name="task_pair_id",
        value_name="solved",
    )
    long_df = long_df.dropna(subset=["solved"]).copy()
    long_df["solved"] = long_df["solved"].astype(int)
    return long_df


def mask_uniform(matrix: pd.DataFrame, keep_probability: float, rng: np.random.Generator) -> pd.DataFrame:
    mask = rng.random(matrix.shape) < keep_probability
    masked = matrix.astype(float).where(mask)
    return masked


def mask_item_counts(
    matrix: pd.DataFrame,
    item_counts: pd.Series,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    n_models = matrix.shape[0]
    masked = pd.DataFrame(np.nan, index=matrix.index, columns=matrix.columns, dtype=float)
    capped_items = 0

    for item in matrix.columns:
        target = int(item_counts.loc[item])
        if target > n_models:
            target = n_models
            capped_items += 1
        chosen = rng.choice(matrix.index.to_numpy(), size=target, replace=False)
        masked.loc[chosen, item] = matrix.loc[chosen, item].astype(float)

    return masked, capped_items


def session_pattern_matched_correlations(
    matrix: pd.DataFrame,
    human_attempts: pd.DataFrame,
    n_sims: int,
    seed: int,
    min_items: int,
    min_item_obs_per_half: int,
) -> pd.DataFrame:
    sessions = np.array(sorted(human_attempts["session_ID"].unique()))
    models = np.array(sorted(matrix.index))
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for sim in range(n_sims):
        assignment = pd.Series(rng.choice(models, size=len(sessions), replace=True), index=sessions)
        perm = rng.permutation(sessions)
        cut = len(perm) // 2
        half_a_sessions = set(perm[:cut])

        pseudo = human_attempts[["session_ID", "task_pair_id"]].copy()
        pseudo["respondent_id"] = pseudo["session_ID"]
        pseudo["assigned_model"] = pseudo["session_ID"].map(assignment)
        pseudo["solved"] = [
            int(matrix.loc[model_name, item_id])
            for model_name, item_id in zip(pseudo["assigned_model"], pseudo["task_pair_id"])
        ]

        half_a = pseudo.loc[pseudo["session_ID"].isin(half_a_sessions), ["respondent_id", "task_pair_id", "solved"]].copy()
        half_b = pseudo.loc[~pseudo["session_ID"].isin(half_a_sessions), ["respondent_id", "task_pair_id", "solved"]].copy()

        counts_a = half_a.groupby("task_pair_id")["solved"].size().rename("n_a")
        counts_b = half_b.groupby("task_pair_id")["solved"].size().rename("n_b")
        eligible = pd.concat([counts_a, counts_b], axis=1, join="inner").dropna()
        eligible = eligible.loc[(eligible["n_a"] >= min_item_obs_per_half) & (eligible["n_b"] >= min_item_obs_per_half)]
        if len(eligible) < min_items:
            continue

        half_a = half_a.loc[half_a["task_pair_id"].isin(eligible.index)]
        half_b = half_b.loc[half_b["task_pair_id"].isin(eligible.index)]

        diff_a = fit_latent_item_difficulty(half_a)
        diff_b = fit_latent_item_difficulty(half_b)
        merged = pd.concat([diff_a.rename("difficulty_a"), diff_b.rename("difficulty_b")], axis=1, join="inner").dropna()
        if len(merged) < min_items or merged["difficulty_a"].nunique() < 2 or merged["difficulty_b"].nunique() < 2:
            continue

        rows.append(
            {
                "simulation": sim,
                "mask_kind": "session_pattern_matched",
                "pearson": float(merged["difficulty_a"].corr(merged["difficulty_b"])),
                "spearman": float(merged["difficulty_a"].corr(merged["difficulty_b"], method="spearman")),
                "n_items": int(len(merged)),
                "observed_cells": int(len(half_a) + len(half_b)),
                "capped_items": 0,
            }
        )

    return pd.DataFrame(rows)


def masked_split_latent_correlations(
    matrix: pd.DataFrame,
    n_sims: int,
    seed: int,
    mask_kind: str,
    human_item_counts: pd.Series,
    total_keep_probability: float,
    min_items: int,
    min_item_obs_per_half: int,
) -> tuple[pd.DataFrame, int]:
    respondents = np.array(sorted(matrix.index))
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    capped_items_max = 0

    for sim in range(n_sims):
        if mask_kind == "uniform_budget":
            masked = mask_uniform(matrix, keep_probability=total_keep_probability, rng=rng)
            capped_items = 0
        elif mask_kind == "item_count_matched":
            masked, capped_items = mask_item_counts(matrix, item_counts=human_item_counts, rng=rng)
        else:
            raise ValueError(f"Unknown mask_kind: {mask_kind}")

        capped_items_max = max(capped_items_max, capped_items)

        perm = rng.permutation(respondents)
        cut = len(perm) // 2
        half_a = masked.loc[perm[:cut]]
        half_b = masked.loc[perm[cut:]]

        long_a = matrix_to_long(half_a)
        long_b = matrix_to_long(half_b)
        if long_a.empty or long_b.empty:
            continue

        counts_a = long_a.groupby("task_pair_id")["solved"].size().rename("n_a")
        counts_b = long_b.groupby("task_pair_id")["solved"].size().rename("n_b")
        eligible = pd.concat([counts_a, counts_b], axis=1, join="inner").dropna()
        eligible = eligible.loc[(eligible["n_a"] >= min_item_obs_per_half) & (eligible["n_b"] >= min_item_obs_per_half)]
        if len(eligible) < min_items:
            continue

        long_a = long_a.loc[long_a["task_pair_id"].isin(eligible.index)]
        long_b = long_b.loc[long_b["task_pair_id"].isin(eligible.index)]

        diff_a = fit_latent_item_difficulty(long_a)
        diff_b = fit_latent_item_difficulty(long_b)
        merged = pd.concat([diff_a.rename("difficulty_a"), diff_b.rename("difficulty_b")], axis=1, join="inner").dropna()
        if len(merged) < min_items or merged["difficulty_a"].nunique() < 2 or merged["difficulty_b"].nunique() < 2:
            continue

        rows.append(
            {
                "simulation": sim,
                "mask_kind": mask_kind,
                "pearson": float(merged["difficulty_a"].corr(merged["difficulty_b"])),
                "spearman": float(merged["difficulty_a"].corr(merged["difficulty_b"], method="spearman")),
                "n_items": int(len(merged)),
                "observed_cells": int(len(long_a) + len(long_b)),
                "capped_items": int(capped_items),
            }
        )

    return pd.DataFrame(rows), capped_items_max


def summarize_draws(
    draws: pd.DataFrame,
    benchmark: str,
    mask_kind: str,
    n_models: int,
    n_items: int,
    human_total_observations: int,
    keep_probability: float,
    capped_items_max: int,
) -> dict[str, object]:
    median_pearson = float(draws["pearson"].median())
    return {
        "benchmark": benchmark,
        "mask_kind": mask_kind,
        "n_models": n_models,
        "n_items": n_items,
        "completed_sims": int(len(draws)),
        "human_total_observations": human_total_observations,
        "mean_observed_cells": float(draws["observed_cells"].mean()),
        "keep_probability": keep_probability,
        "capped_items_max": capped_items_max,
        "pearson_mean": float(draws["pearson"].mean()),
        "pearson_median": median_pearson,
        "pearson_ci_lo": float(draws["pearson"].quantile(0.025)),
        "pearson_ci_hi": float(draws["pearson"].quantile(0.975)),
        "spearman_mean": float(draws["spearman"].mean()),
        "spearman_median": float(draws["spearman"].median()),
        "median_items_per_split": float(draws["n_items"].median()),
        "spearman_brown_from_median_pearson": float((2 * median_pearson) / (1 + median_pearson)),
    }


def load_dense_baseline_summary() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_CSV)
    return summary


def write_report(summary: pd.DataFrame, baselines: pd.DataFrame) -> None:
    display = summary[
        [
            "benchmark",
            "mask_kind",
            "completed_sims",
            "human_total_observations",
            "mean_observed_cells",
            "pearson_median",
            "pearson_ci_lo",
            "pearson_ci_hi",
            "spearman_brown_from_median_pearson",
        ]
    ].copy()
    baseline_display = baselines[
        ["benchmark", "population", "pearson_median", "pearson_ci_lo", "pearson_ci_hi"]
    ].copy()

    lines = [
        "# LLM Sparsity Stress Test",
        "",
        "This note asks whether the lower human latent split-half reliability could mostly be a sparse-observation artifact.",
        "",
        "## Masking Designs",
        "",
        "- `uniform_budget`: randomly keep LLM model-item cells with probability chosen to match the human total observation budget.",
        "- `item_count_matched`: for each task pair, randomly keep exactly as many LLM observations as humans had on that item, capped at the number of available LLM models.",
        "- `session_pattern_matched`: assign LLM models to human sessions at random and only reveal the model responses on the exact items humans attempted, preserving human session lengths and item exposure counts.",
        "- In each case, the same latent split-half recovery pipeline is rerun after masking.",
        "",
        "## Dense Baselines",
        "",
        "```text",
        baseline_display.round(3).to_string(index=False),
        "```",
        "",
        "## Masked LLM Results",
        "",
        "```text",
        display.round(3).to_string(index=False),
        "```",
        "",
        "## Readout",
        "",
        "- If masked LLM reliability stays far above the human baseline, sparse observation counts alone are not enough to explain the human-vs-LLM gap.",
        "- The cleanest comparison is ARC-2, where the human item counts are all below the number of available LLM models, so the `item_count_matched` mask is exact.",
        "",
    ]
    OUT_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    human_raw = load_human_attempts()
    human_arc1, arc1_pair_ids = build_human_arc1_sidecar_subset(human_raw, args.min_human_attempts)
    human_arc2, arc2_pair_ids = build_human_arc2_subset(human_raw, args.min_human_attempts)

    arc1_truth = load_truth_outputs(ARC1_TRUTH_DIR, single_pair_only=True)
    arc2_truth = load_truth_outputs(ARC2_TRUTH_DIR)
    llm_arc1 = load_or_build_llm_matrix(
        ARC1_LLM_CACHE,
        ARC1_LLM_DIR,
        arc1_truth,
        arc1_pair_ids,
        rebuild=args.rebuild_matrices,
    )
    llm_arc2 = load_or_build_llm_matrix(
        ARC2_LLM_CACHE,
        ARC2_LLM_DIR,
        arc2_truth,
        arc2_pair_ids,
        rebuild=args.rebuild_matrices,
    )

    benchmark_specs = [
        {
            "benchmark": "ARC1",
            "human": human_arc1,
            "matrix": llm_arc1,
        },
        {
            "benchmark": "ARC2",
            "human": human_arc2,
            "matrix": llm_arc2,
        },
    ]

    draw_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for bench_idx, spec in enumerate(benchmark_specs):
        human_item_counts = spec["human"].groupby("task_pair_id")["solved"].size().reindex(spec["matrix"].columns)
        human_total_observations = int(human_item_counts.sum())
        keep_probability = human_total_observations / float(spec["matrix"].shape[0] * spec["matrix"].shape[1])

        for offset, mask_kind in enumerate(["uniform_budget", "item_count_matched"]):
            draws, capped_items_max = masked_split_latent_correlations(
                spec["matrix"],
                n_sims=args.n_sims,
                seed=args.seed + bench_idx * 100 + offset * 1000,
                mask_kind=mask_kind,
                human_item_counts=human_item_counts,
                total_keep_probability=keep_probability,
                min_items=args.min_items,
                min_item_obs_per_half=args.min_item_obs_per_half,
            )
            draws["benchmark"] = spec["benchmark"]
            draw_frames.append(draws)
            summary_rows.append(
                summarize_draws(
                    draws,
                    benchmark=spec["benchmark"],
                    mask_kind=mask_kind,
                    n_models=int(spec["matrix"].shape[0]),
                    n_items=int(spec["matrix"].shape[1]),
                    human_total_observations=human_total_observations,
                    keep_probability=keep_probability,
                    capped_items_max=capped_items_max,
                )
            )

        session_draws = session_pattern_matched_correlations(
            spec["matrix"],
            human_attempts=spec["human"],
            n_sims=args.n_sims,
            seed=args.seed + bench_idx * 100 + 5000,
            min_items=args.min_items,
            min_item_obs_per_half=args.min_item_obs_per_half,
        )
        session_draws["benchmark"] = spec["benchmark"]
        draw_frames.append(session_draws)
        summary_rows.append(
            summarize_draws(
                session_draws,
                benchmark=spec["benchmark"],
                mask_kind="session_pattern_matched",
                n_models=int(spec["matrix"].shape[0]),
                n_items=int(spec["matrix"].shape[1]),
                human_total_observations=human_total_observations,
                keep_probability=keep_probability,
                capped_items_max=0,
            )
        )

    all_draws = pd.concat(draw_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(["benchmark", "mask_kind"]).reset_index(drop=True)
    baselines = load_dense_baseline_summary()

    all_draws.to_csv(OUT_DRAWS_CSV, index=False)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    write_report(summary, baselines)

    print(summary.round(4).to_string(index=False))
    print(f"\nSaved draws to: {OUT_DRAWS_CSV}")
    print(f"Saved summary to: {OUT_SUMMARY_CSV}")
    print(f"Saved report to: {OUT_REPORT_MD}")


if __name__ == "__main__":
    main()
