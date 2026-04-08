from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t


BASE_DIR = Path(__file__).resolve().parent
TABLES_DIR = BASE_DIR / "tables"

RNG_SEED = 0
N_BOOT = 1000


def within_group_z(df: pd.DataFrame, col: str, group_col: str = "benchmark_label") -> pd.Series:
    return df.groupby(group_col)[col].transform(lambda s: (s - s.mean()) / s.std(ddof=0))


def corr_with_pvalue(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan"), float("nan")
    r = float(pearsonr(pair["x"], pair["y"]).statistic)
    t_stat = abs(r) * np.sqrt((len(pair) - 2) / max(1e-15, 1.0 - r * r))
    p = 2.0 * float(t.sf(t_stat, df=len(pair) - 2))
    return r, p


def bootstrap_delta_p(
    df: pd.DataFrame,
    x_col: str,
    y_human_col: str,
    y_llm_col: str,
    group_col: str = "benchmark_label",
    n_boot: int = N_BOOT,
    seed: int = RNG_SEED,
) -> tuple[float, float, float, float, float]:
    rng = np.random.default_rng(seed)
    human_r, _ = corr_with_pvalue(df[x_col], df[y_human_col])
    llm_r, _ = corr_with_pvalue(df[x_col], df[y_llm_col])
    draws: list[float] = []
    for _ in range(n_boot):
        parts: list[pd.DataFrame] = []
        for _, sub in df.groupby(group_col):
            idx = rng.integers(0, len(sub), size=len(sub))
            parts.append(sub.iloc[idx])
        sample = pd.concat(parts, ignore_index=True)
        h_r, _ = corr_with_pvalue(sample[x_col], sample[y_human_col])
        l_r, _ = corr_with_pvalue(sample[x_col], sample[y_llm_col])
        if np.isfinite(h_r) and np.isfinite(l_r):
            draws.append(l_r - h_r)
    draws_arr = np.asarray(draws, dtype=float)
    if draws_arr.size == 0:
        return human_r, llm_r, float("nan"), float("nan"), float("nan")
    delta = llm_r - human_r
    p_delta = 2.0 * min(float(np.mean(draws_arr <= 0.0)), float(np.mean(draws_arr >= 0.0)))
    ci_lo, ci_hi = np.quantile(draws_arr, [0.025, 0.975])
    return human_r, llm_r, delta, float(p_delta), float(ci_lo), float(ci_hi)


def main() -> None:
    human = pd.read_csv(TABLES_DIR / "human_task_latent_summary.csv")
    llm = pd.read_csv(TABLES_DIR / "llm_common_task_summary.csv")
    complexity = pd.read_csv(TABLES_DIR / "complexity_expanded.csv")

    predictor_cols = [col for col in complexity.columns if pd.api.types.is_numeric_dtype(complexity[col])]
    predictor_cols = [col for col in predictor_cols if col != "task_id"]
    # Preserve file order, while keeping structure_pc1 in the panel.
    seen: set[str] = set()
    predictor_cols = [col for col in predictor_cols if not (col in seen or seen.add(col))]

    human_pool = human.loc[human["benchmark_label"].isin(["arc1_sidecar", "arc2_eval"])].copy()
    human_pool = human_pool.merge(
        complexity,
        left_on=["task_ID", "dataset_key"],
        right_on=["task_id", "dataset_key"],
        how="inner",
    )

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
    llm_pool = pd.concat(llm_frames, ignore_index=True).merge(
        complexity,
        on=["task_id", "dataset_key"],
        how="inner",
    )

    shared_pool = human_pool.merge(
        llm[["task_id", "llm_latent_difficulty"]],
        left_on="task_ID",
        right_on="task_id",
        how="inner",
        suffixes=("", "_llm"),
    )

    # Benchmark-standardized versions of every metric we test.
    for df in [human_pool, llm_pool, shared_pool]:
        for col in predictor_cols:
            df[f"{col}_z"] = within_group_z(df, col)

    human_pool["latent_human_difficulty_z"] = within_group_z(human_pool, "latent_human_difficulty")
    human_pool["human_difficulty_pc1_z"] = within_group_z(human_pool, "human_difficulty_pc1")
    human_pool["latent_duration_log_z"] = within_group_z(human_pool, "latent_duration_log")

    llm_pool["llm_latent_difficulty_z"] = within_group_z(llm_pool, "llm_latent_difficulty")
    llm_pool["llm_pass_rate_z"] = within_group_z(llm_pool, "llm_pass_rate")

    shared_pool["latent_human_difficulty_z"] = within_group_z(shared_pool, "latent_human_difficulty")
    shared_pool["llm_latent_difficulty_z"] = within_group_z(shared_pool, "llm_latent_difficulty")

    rows: list[dict[str, object]] = []
    for predictor in predictor_cols:
        human_x = f"{predictor}_z"
        if human_x not in human_pool.columns or llm_pool[human_x].nunique(dropna=True) < 2 or human_pool[human_x].nunique(dropna=True) < 2:
            continue
        human_r, human_p = corr_with_pvalue(human_pool[human_x], human_pool["latent_human_difficulty_z"])
        llm_r, llm_p = corr_with_pvalue(llm_pool[human_x], llm_pool["llm_latent_difficulty_z"])
        shared_human_r, shared_human_p = corr_with_pvalue(shared_pool[human_x], shared_pool["latent_human_difficulty_z"])
        shared_llm_r, shared_llm_p = corr_with_pvalue(shared_pool[human_x], shared_pool["llm_latent_difficulty_z"])
        delta_human_r, delta_llm_r, delta_r, delta_p, ci_lo, ci_hi = bootstrap_delta_p(
            shared_pool[[human_x, "latent_human_difficulty_z", "llm_latent_difficulty_z", "benchmark_label"]].dropna(),
            human_x,
            "latent_human_difficulty_z",
            "llm_latent_difficulty_z",
        )

        rows.extend(
            [
                {
                    "analysis": "human_pooled_benchmark_z",
                    "predictor": predictor,
                    "n": int(len(human_pool[[human_x, "latent_human_difficulty_z", "benchmark_label"]].dropna())),
                    "estimate": human_r,
                    "p_value": human_p,
                    "ci_lo": "",
                    "ci_hi": "",
                    "notes": "human pooled benchmark-z correlation",
                },
                {
                    "analysis": "llm_pooled_benchmark_z",
                    "predictor": predictor,
                    "n": int(len(llm_pool[[human_x, "llm_latent_difficulty_z", "benchmark_label"]].dropna())),
                    "estimate": llm_r,
                    "p_value": llm_p,
                    "ci_lo": "",
                    "ci_hi": "",
                    "notes": "llm pooled benchmark-z correlation",
                },
                {
                    "analysis": "shared_human_pooled_benchmark_z",
                    "predictor": predictor,
                    "n": int(len(shared_pool[[human_x, "latent_human_difficulty_z", "benchmark_label"]].dropna())),
                    "estimate": shared_human_r,
                    "p_value": shared_human_p,
                    "ci_lo": "",
                    "ci_hi": "",
                    "notes": "shared pooled human benchmark-z correlation",
                },
                {
                    "analysis": "shared_llm_pooled_benchmark_z",
                    "predictor": predictor,
                    "n": int(len(shared_pool[[human_x, "llm_latent_difficulty_z", "benchmark_label"]].dropna())),
                    "estimate": shared_llm_r,
                    "p_value": shared_llm_p,
                    "ci_lo": "",
                    "ci_hi": "",
                    "notes": "shared pooled llm benchmark-z correlation",
                },
                {
                    "analysis": "shared_delta_pooled_benchmark_z",
                    "predictor": predictor,
                    "n": int(len(shared_pool[[human_x, "latent_human_difficulty_z", "llm_latent_difficulty_z", "benchmark_label"]].dropna())),
                    "estimate": delta_r,
                    "p_value": delta_p,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "notes": (
                        f"shared pooled delta with human_r={delta_human_r:.6f} "
                        f"and llm_r={delta_llm_r:.6f}"
                    ),
                },
            ]
        )

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "pooled_structure_significance_all.csv", index=False)

    summary = {
        "predictor_count": int(len(predictor_cols)),
        "human_p_lt_0_05": int((out.query("analysis == 'human_pooled_benchmark_z'")["p_value"] < 0.05).sum()),
        "llm_p_lt_0_05": int((out.query("analysis == 'llm_pooled_benchmark_z'")["p_value"] < 0.05).sum()),
        "delta_p_lt_0_05": int((out.query("analysis == 'shared_delta_pooled_benchmark_z'")["p_value"] < 0.05).sum()),
        "delta_positive_count": int((out.query("analysis == 'shared_delta_pooled_benchmark_z'")["estimate"] > 0).sum()),
        "top_delta_predictors": (
            out.query("analysis == 'shared_delta_pooled_benchmark_z'")
            .sort_values("estimate", ascending=False)
            .head(10)[["predictor", "estimate", "p_value"]]
            .to_dict(orient="records")
        ),
    }
    (TABLES_DIR / "pooled_structure_significance_all_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
