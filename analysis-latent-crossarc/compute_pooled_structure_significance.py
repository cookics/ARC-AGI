from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


BASE_DIR = Path(__file__).resolve().parent
TABLES_DIR = BASE_DIR / "tables"

RNG_SEED = 0
N_PERM = 20_000
N_BOOT = 5_000


def within_group_z(df: pd.DataFrame, col: str, group_col: str = "benchmark_label") -> pd.Series:
    return df.groupby(group_col)[col].transform(lambda s: (s - s.mean()) / s.std(ddof=0))


def grouped_permutation_corr_p(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str = "benchmark_label",
    n_perm: int = N_PERM,
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    observed = float(pearsonr(df[x_col], df[y_col]).statistic)
    exceed = 0
    for _ in range(n_perm):
        permuted_parts: list[pd.Series] = []
        for _, sub in df.groupby(group_col):
            vals = sub[y_col].to_numpy().copy()
            rng.shuffle(vals)
            permuted_parts.append(pd.Series(vals, index=sub.index))
        permuted = pd.concat(permuted_parts).sort_index()
        stat = float(pearsonr(df[x_col], permuted).statistic)
        if abs(stat) >= abs(observed):
            exceed += 1
    p_value = (exceed + 1) / (n_perm + 1)
    return observed, float(p_value)


def grouped_permutation_delta_p(
    df: pd.DataFrame,
    x_col: str,
    y_human_col: str,
    y_llm_col: str,
    group_col: str = "benchmark_label",
    n_perm: int = N_PERM,
    seed: int = RNG_SEED,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    human_r = float(pearsonr(df[x_col], df[y_human_col]).statistic)
    llm_r = float(pearsonr(df[x_col], df[y_llm_col]).statistic)
    observed_delta = llm_r - human_r
    exceed = 0
    for _ in range(n_perm):
        permuted_parts: list[pd.Series] = []
        for _, sub in df.groupby(group_col):
            vals = sub[x_col].to_numpy().copy()
            rng.shuffle(vals)
            permuted_parts.append(pd.Series(vals, index=sub.index))
        permuted = pd.concat(permuted_parts).sort_index()
        perm_h = float(pearsonr(permuted, df[y_human_col]).statistic)
        perm_l = float(pearsonr(permuted, df[y_llm_col]).statistic)
        if abs((perm_l - perm_h)) >= abs(observed_delta):
            exceed += 1
    p_value = (exceed + 1) / (n_perm + 1)
    return human_r, llm_r, observed_delta, float(p_value)


def grouped_bootstrap_delta_ci(
    df: pd.DataFrame,
    x_col: str,
    y_human_col: str,
    y_llm_col: str,
    group_col: str = "benchmark_label",
    n_boot: int = N_BOOT,
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        parts: list[pd.DataFrame] = []
        for _, sub in df.groupby(group_col):
            idx = rng.integers(0, len(sub), size=len(sub))
            parts.append(sub.iloc[idx])
        sample = pd.concat(parts, ignore_index=True)
        human_r = float(pearsonr(sample[x_col], sample[y_human_col]).statistic)
        llm_r = float(pearsonr(sample[x_col], sample[y_llm_col]).statistic)
        draws.append(llm_r - human_r)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    human = pd.read_csv(TABLES_DIR / "human_task_latent_summary.csv")
    llm = pd.read_csv(TABLES_DIR / "llm_common_task_summary.csv")
    complexity = pd.read_csv(TABLES_DIR / "complexity_expanded.csv")

    pooled_human = human.loc[human["benchmark_label"].isin(["arc1_sidecar", "arc2_eval"])].copy()
    pooled_human = pooled_human.merge(
        complexity,
        left_on=["task_ID", "dataset_key"],
        right_on=["task_id", "dataset_key"],
        how="inner",
    )
    for col in ["latent_human_difficulty", "cyclomatic_complexity", "structure_pc1"]:
        pooled_human[f"{col}_z"] = within_group_z(pooled_human, col)

    llm_frames: list[pd.DataFrame] = []
    for benchmark_label, rate_col, dataset_key in [
        ("arc1_eval", "arc1_pass_rate", "arc_agi_1_eval"),
        ("arc2_eval", "arc2_pass_rate", "arc_agi_2_eval"),
    ]:
        frame = llm[["task_id", "llm_latent_difficulty", rate_col]].copy()
        frame = frame.loc[frame[rate_col].notna()].rename(columns={rate_col: "llm_pass_rate"})
        frame["benchmark_label"] = benchmark_label
        frame["dataset_key"] = dataset_key
        llm_frames.append(frame)
    pooled_llm = pd.concat(llm_frames, ignore_index=True).merge(
        complexity,
        on=["task_id", "dataset_key"],
        how="inner",
    )
    for col in ["llm_latent_difficulty", "cyclomatic_complexity", "structure_pc1"]:
        pooled_llm[f"{col}_z"] = within_group_z(pooled_llm, col)

    shared = pooled_human.merge(
        llm[["task_id", "llm_latent_difficulty"]],
        left_on="task_ID",
        right_on="task_id",
        how="inner",
    )
    for col in ["llm_latent_difficulty", "latent_human_difficulty", "cyclomatic_complexity", "structure_pc1"]:
        shared[f"{col}_z"] = within_group_z(shared, col)

    rows: list[dict[str, object]] = []
    for predictor in ["cyclomatic_complexity", "structure_pc1"]:
        human_r, human_p = grouped_permutation_corr_p(
            pooled_human[[f"{predictor}_z", "latent_human_difficulty_z", "benchmark_label"]].dropna(),
            f"{predictor}_z",
            "latent_human_difficulty_z",
        )
        rows.append(
            {
                "analysis": "human_pooled_benchmark_z",
                "predictor": predictor,
                "n": int(len(pooled_human[[f"{predictor}_z", "latent_human_difficulty_z", "benchmark_label"]].dropna())),
                "estimate": human_r,
                "p_value": human_p,
                "ci_lo": "",
                "ci_hi": "",
                "notes": "latent_human_difficulty vs predictor on pooled ARC1 sidecar plus ARC2 eval human tasks",
            }
        )

        llm_r, llm_p = grouped_permutation_corr_p(
            pooled_llm[[f"{predictor}_z", "llm_latent_difficulty_z", "benchmark_label"]].dropna(),
            f"{predictor}_z",
            "llm_latent_difficulty_z",
        )
        rows.append(
            {
                "analysis": "llm_pooled_benchmark_z",
                "predictor": predictor,
                "n": int(len(pooled_llm[[f"{predictor}_z", "llm_latent_difficulty_z", "benchmark_label"]].dropna())),
                "estimate": llm_r,
                "p_value": llm_p,
                "ci_lo": "",
                "ci_hi": "",
                "notes": "llm_latent_difficulty vs predictor on pooled ARC1 eval plus ARC2 eval LLM tasks",
            }
        )

        sub = shared[[f"{predictor}_z", "latent_human_difficulty_z", "llm_latent_difficulty_z", "benchmark_label"]].dropna()
        shared_human_r, shared_human_p = grouped_permutation_corr_p(
            sub,
            f"{predictor}_z",
            "latent_human_difficulty_z",
        )
        rows.append(
            {
                "analysis": "shared_human_pooled_benchmark_z",
                "predictor": predictor,
                "n": int(len(sub)),
                "estimate": shared_human_r,
                "p_value": shared_human_p,
                "ci_lo": "",
                "ci_hi": "",
                "notes": "latent_human_difficulty vs predictor on the shared pooled ARC1 sidecar plus ARC2 eval tasks",
            }
        )

        shared_llm_r, shared_llm_p = grouped_permutation_corr_p(
            sub,
            f"{predictor}_z",
            "llm_latent_difficulty_z",
        )
        rows.append(
            {
                "analysis": "shared_llm_pooled_benchmark_z",
                "predictor": predictor,
                "n": int(len(sub)),
                "estimate": shared_llm_r,
                "p_value": shared_llm_p,
                "ci_lo": "",
                "ci_hi": "",
                "notes": "llm_latent_difficulty vs predictor on the shared pooled ARC1 sidecar plus ARC2 eval tasks",
            }
        )

        human_shared_r, llm_shared_r, delta_r, delta_p = grouped_permutation_delta_p(
            sub,
            f"{predictor}_z",
            "latent_human_difficulty_z",
            "llm_latent_difficulty_z",
        )
        ci_lo, ci_hi = grouped_bootstrap_delta_ci(
            sub,
            f"{predictor}_z",
            "latent_human_difficulty_z",
            "llm_latent_difficulty_z",
        )
        rows.append(
            {
                "analysis": "shared_delta_pooled_benchmark_z",
                "predictor": predictor,
                "n": int(len(sub)),
                "estimate": delta_r,
                "p_value": delta_p,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "notes": (
                    f"shared pooled delta with human_r={human_shared_r:.6f} "
                    f"and llm_r={llm_shared_r:.6f}"
                ),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "pooled_structure_significance_summary.csv", index=False)


if __name__ == "__main__":
    main()
