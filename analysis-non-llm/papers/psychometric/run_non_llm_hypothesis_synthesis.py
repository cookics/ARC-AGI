from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from run_non_llm_psychometric_analysis import (
    ANALYSIS_DIR,
    FEATURE_COLUMNS,
    PRIMARY_THRESHOLD,
    ROOT_DIR,
    TABLES_DIR,
    build_human_subset,
    build_llm_matrix,
    build_trm_matrix,
    build_varc_matrix,
    corr_or_nan,
    load_arc2_truth,
    load_human_inputs,
    split_half_correlations,
)


ARC1_TRUTH_DIR = ROOT_DIR / "data-llm" / "ARC-AGI" / "data" / "evaluation"
ARC1_LLM_DIR = ROOT_DIR / "data-llm" / "arc_agi_v1_public_eval"
COMPRESS_ARC_SUMMARY = ROOT_DIR / "data-non-llm" / "processed" / "compress_arc_predictions_evaluation_summary.json"
TECHNICAL_MD = ANALYSIS_DIR / "hypothesis_synthesis.md"
RESULTS_MD = ANALYSIS_DIR / "paper_results_section.md"


def frame_to_text_table(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def normalize_grid(grid: object) -> str:
    if not isinstance(grid, list) or not grid:
        return "EMPTY"
    return ",".join(str(cell) for row in grid for cell in row)


def empirical_pvalue(draws: np.ndarray | pd.Series, obs: float, alternative: str = "two-sided") -> float:
    arr = np.asarray(draws, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0 or not np.isfinite(obs):
        return float("nan")
    if alternative == "greater":
        return float((np.sum(arr >= obs) + 1) / (len(arr) + 1))
    if alternative == "less":
        return float((np.sum(arr <= obs) + 1) / (len(arr) + 1))
    p_lo = (np.sum(arr <= obs) + 1) / (len(arr) + 1)
    p_hi = (np.sum(arr >= obs) + 1) / (len(arr) + 1)
    return float(min(1.0, 2 * min(p_lo, p_hi)))


def bh_adjust(pvalues: pd.Series | list[float]) -> np.ndarray:
    arr = np.asarray(pvalues, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(arr))
    if len(finite_idx) == 0:
        return out
    finite = arr[finite_idx]
    order = np.argsort(finite)
    ranked = finite[order]
    m = len(ranked)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * m / rank)
        adjusted[i] = running
    restored = np.empty(m, dtype=float)
    restored[order] = adjusted
    out[finite_idx] = np.clip(restored, 0.0, 1.0)
    return out


def hypergeom_tail(k: int, population: int, successes: int, draws: int) -> float:
    if k <= 0:
        return 1.0
    denom = math.comb(population, draws)
    total = 0.0
    for hits in range(k, min(successes, draws) + 1):
        total += math.comb(successes, hits) * math.comb(population - successes, draws - hits) / denom
    return total


def bootstrap_corr_stats(
    y: pd.Series,
    x: pd.Series,
    n_boot: int = 5000,
    seed: int = 0,
    alternative: str = "two-sided",
) -> tuple[float, float, float, float]:
    obs = corr_or_nan(x, y)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    y_arr = y.to_numpy()
    x_arr = x.to_numpy()
    n_items = len(y_arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        s_y = pd.Series(y_arr[idx])
        s_x = pd.Series(x_arr[idx])
        if s_y.nunique() < 2 or s_x.nunique() < 2:
            continue
        draws.append(float(s_x.corr(s_y)))
    draw_arr = np.asarray(draws, dtype=float)
    if alternative == "greater":
        p_value = empirical_pvalue(draw_arr, 0.0, alternative="less")
    elif alternative == "less":
        p_value = empirical_pvalue(draw_arr, 0.0, alternative="greater")
    else:
        p_value = empirical_pvalue(draw_arr, 0.0, alternative="two-sided")
    return (
        obs,
        float(np.quantile(draw_arr, 0.025)),
        float(np.quantile(draw_arr, 0.975)),
        p_value,
    )


def bootstrap_corr_difference_stats(
    y: pd.Series,
    x1: pd.Series,
    x2: pd.Series,
    n_boot: int = 8000,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    obs = corr_or_nan(x1, y) - corr_or_nan(x2, y)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    y_arr = y.to_numpy()
    x1_arr = x1.to_numpy()
    x2_arr = x2.to_numpy()
    n_items = len(y_arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        s_y = pd.Series(y_arr[idx])
        s_x1 = pd.Series(x1_arr[idx])
        s_x2 = pd.Series(x2_arr[idx])
        if s_y.nunique() < 2 or s_x1.nunique() < 2 or s_x2.nunique() < 2:
            continue
        draws.append(float(s_x1.corr(s_y) - s_x2.corr(s_y)))
    draw_arr = np.asarray(draws, dtype=float)
    return (
        obs,
        float(np.quantile(draw_arr, 0.025)),
        float(np.quantile(draw_arr, 0.975)),
        empirical_pvalue(draw_arr, 0.0, alternative="two-sided"),
    )


def bootstrap_partial_corr_stats(
    y: pd.Series,
    x: pd.Series,
    controls: pd.DataFrame | pd.Series,
    n_boot: int = 4000,
    seed: int = 0,
    alternative: str = "greater",
) -> tuple[float, float, float, float]:
    controls_df = pd.DataFrame(controls).fillna(0.0)
    lr = LinearRegression().fit(controls_df, y)
    y_res = y - lr.predict(controls_df)
    lr = LinearRegression().fit(controls_df, x)
    x_res = x - lr.predict(controls_df)
    obs = corr_or_nan(y_res, x_res)

    rng = np.random.default_rng(seed)
    y_arr = y.to_numpy()
    x_arr = x.to_numpy()
    c_arr = controls_df.to_numpy()
    n_items = len(y_arr)
    draws: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        y_boot = pd.Series(y_arr[idx])
        x_boot = pd.Series(x_arr[idx])
        c_boot = c_arr[idx]
        lr = LinearRegression().fit(c_boot, y_boot)
        y_r = y_boot - lr.predict(c_boot)
        lr = LinearRegression().fit(c_boot, x_boot)
        x_r = x_boot - lr.predict(c_boot)
        if y_r.nunique() < 2 or x_r.nunique() < 2:
            continue
        draws.append(float(pd.Series(y_r).corr(pd.Series(x_r))))
    draw_arr = np.asarray(draws, dtype=float)
    if alternative == "greater":
        p_value = empirical_pvalue(draw_arr, 0.0, alternative="less")
    elif alternative == "less":
        p_value = empirical_pvalue(draw_arr, 0.0, alternative="greater")
    else:
        p_value = empirical_pvalue(draw_arr, 0.0, alternative="two-sided")
    return (
        obs,
        float(np.quantile(draw_arr, 0.025)),
        float(np.quantile(draw_arr, 0.975)),
        p_value,
    )


def fixed_accuracy_random_null(
    profile: pd.Series,
    human: pd.Series,
    n_sim: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    aligned = profile.reindex(human.index).fillna(0).astype(int)
    obs = corr_or_nan(aligned, human)
    k = int(aligned.sum())
    n_items = len(aligned)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_sim):
        sample = np.zeros(n_items, dtype=int)
        if k:
            sample[rng.choice(n_items, size=k, replace=False)] = 1
        draws.append(float(pd.Series(sample, index=human.index).corr(human)))
    draw_arr = np.asarray(draws, dtype=float)
    return {
        "n_items": n_items,
        "successes": k,
        "pair_accuracy": float(aligned.mean()),
        "observed_corr": obs,
        "null_mean": float(np.mean(draw_arr)),
        "null_ci_lo": float(np.quantile(draw_arr, 0.025)),
        "null_ci_hi": float(np.quantile(draw_arr, 0.975)),
        "p_value": empirical_pvalue(draw_arr, obs, alternative="greater"),
    }


def permutation_corr_test(
    x: pd.Series,
    y: pd.Series,
    method: str = "spearman",
    n_perm: int = 20000,
    seed: int = 0,
) -> tuple[float, float]:
    obs = corr_or_nan(x, y, method=method)
    rng = np.random.default_rng(seed)
    x_arr = pd.Series(x).to_numpy()
    y_arr = pd.Series(y).to_numpy()
    draws: list[float] = []
    for _ in range(n_perm):
        perm = rng.permutation(y_arr)
        draws.append(float(pd.Series(x_arr).corr(pd.Series(perm), method=method)))
    draw_arr = np.asarray(draws, dtype=float)
    return obs, empirical_pvalue(draw_arr, obs, alternative="two-sided")


def find_accuracy_matched_llms(
    system_name: str,
    profile: pd.Series,
    llm_matrix: pd.DataFrame,
    human: pd.Series,
    tolerance: float = 0.03,
) -> dict:
    accuracy = float(profile.mean())
    llm_acc = llm_matrix.mean(axis=1)
    matched = llm_matrix.loc[(llm_acc >= accuracy - tolerance) & (llm_acc <= accuracy + tolerance)]
    matched_corrs = matched.apply(lambda row: corr_or_nan(row, human), axis=1).sort_values(ascending=False)
    return {
        "system": system_name,
        "system_accuracy": accuracy,
        "system_human_corr": corr_or_nan(profile, human),
        "matched_llm_count": int(len(matched_corrs)),
        "matched_llm_corr_median": float(matched_corrs.median()) if len(matched_corrs) else float("nan"),
        "matched_llm_corr_min": float(matched_corrs.min()) if len(matched_corrs) else float("nan"),
        "matched_llm_corr_max": float(matched_corrs.max()) if len(matched_corrs) else float("nan"),
        "matched_llm_best": str(matched_corrs.index[0]) if len(matched_corrs) else "",
    }


def build_arc2_profiles() -> tuple[pd.Series, dict[str, pd.Series], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, truth_outputs = load_arc2_truth()
    human_raw, human_meta = load_human_inputs()
    subset = build_human_subset(human_meta, human_raw, PRIMARY_THRESHOLD)
    llm_matrix = build_llm_matrix(subset.pairs, truth_outputs)
    trm_matrix = build_trm_matrix(subset.pairs, truth_outputs)
    varc_matrix = build_varc_matrix(subset.pairs, truth_outputs)

    profiles = {
        "LLM average": llm_matrix.mean(axis=0),
        "Best-score LLM": llm_matrix.loc["gpt-5-2-2025-12-11-thinking-xhigh"].astype(float),
        "Best-aligned LLM": llm_matrix.loc["claude-opus-4-5-20251101-thinking-16k"].astype(float),
        "TRM 651522 pass@2": trm_matrix.loc["TRM 651522 pass@2"].astype(float),
        "TRM 361957 pass@2": trm_matrix.loc["TRM 361957 pass@2"].astype(float),
        "VARC ARC-2_ViT pass@2": varc_matrix.loc["VARC ARC-2_ViT pass@2"].astype(float),
    }
    return subset.human, profiles, subset.frame.set_index("task_pair_id"), llm_matrix, trm_matrix


def build_arc1_sidecar() -> tuple[pd.Series, dict[str, pd.Series], pd.DataFrame]:
    human_raw = pd.read_csv(ROOT_DIR / "data-human" / "test_pair_attempts.csv")
    human_raw["task_pair_id"] = human_raw["task_ID"] + "__" + human_raw["test_index"].astype(str)
    human_raw["solved"] = (human_raw["correct_submissions"] > 0).astype(int)

    truth_outputs: dict[str, str] = {}
    single_pair_ids: list[str] = []
    for path in sorted(ARC1_TRUTH_DIR.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if len(obj.get("test", [])) != 1:
            continue
        pair_id = f"{path.stem}__0"
        truth_outputs[pair_id] = normalize_grid(obj["test"][0]["output"])
        single_pair_ids.append(pair_id)

    human_arc1 = human_raw.loc[
        (human_raw["task_set"] == "Public Train") & (human_raw["task_pair_id"].isin(single_pair_ids))
    ].copy()
    item = (
        human_arc1.groupby("task_pair_id")
        .agg(human_attempts=("solved", "size"), human_solve_rate=("solved", "mean"))
        .reset_index()
    )
    item = item.loc[item["human_attempts"] >= 8].sort_values("task_pair_id")
    pairs = item["task_pair_id"].tolist()
    human = item.set_index("task_pair_id")["human_solve_rate"]

    llm_rows: dict[str, dict[str, int]] = {}
    for model_dir in sorted(ARC1_LLM_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        row = {pair_id: 0 for pair_id in pairs}
        for pred_path in model_dir.glob("*.json"):
            pred_obj = json.loads(pred_path.read_text(encoding="utf-8"))
            for idx, pred in enumerate(pred_obj):
                pair_id = f"{pred_path.stem}__{idx}"
                if pair_id not in row:
                    continue
                answer = None
                if pred:
                    answer = (pred.get("attempt_1") or {}).get("answer")
                    if not answer:
                        answer = (pred.get("attempt_2") or {}).get("answer")
                row[pair_id] = int(normalize_grid(answer) == truth_outputs[pair_id])
        llm_rows[model_dir.name] = row
    llm_matrix = pd.DataFrame.from_dict(llm_rows, orient="index")[pairs]

    compress = json.loads(COMPRESS_ARC_SUMMARY.read_text(encoding="utf-8"))
    pass2_ids = set(compress["task_ids"]["final_pick_pass2"])
    best_aligned_llm = llm_matrix.apply(lambda row: corr_or_nan(row, human), axis=1).sort_values(ascending=False).index[0]
    profiles = {
        "ARC1 LLM average": llm_matrix.mean(axis=0),
        "ARC1 best-aligned LLM": llm_matrix.loc[best_aligned_llm].astype(float),
        "CompressARC top2": pd.Series({pair_id: int(pair_id.split("__")[0] in pass2_ids) for pair_id in pairs}),
    }
    return human, profiles, llm_matrix


def write_results_section(
    data_scope: pd.DataFrame,
    arc2_hypotheses: pd.DataFrame,
    diff_tests: pd.DataFrame,
    fixed_accuracy: pd.DataFrame,
    matched_arc2: pd.DataFrame,
    residual_tests: pd.DataFrame,
    complementarity: pd.DataFrame,
    trajectory_tests: pd.DataFrame,
    arc1_summary: pd.DataFrame,
    matched_arc1: pd.DataFrame,
) -> None:
    arc2_scope = data_scope.loc[data_scope["dataset"] == "ARC-2 primary"].iloc[0]
    arc1_scope = data_scope.loc[data_scope["dataset"] == "ARC-1 sidecar"].iloc[0]
    h1 = arc2_hypotheses.loc[arc2_hypotheses["null_id"] == "H1"].copy()
    h2 = arc2_hypotheses.loc[arc2_hypotheses["null_id"] == "H2"].copy()

    llm_avg_h1 = h1.loc[h1["system"] == "LLM average"].iloc[0]
    best_align_h1 = h1.loc[h1["system"] == "Best-aligned LLM"].iloc[0]
    best_score_h1 = h1.loc[h1["system"] == "Best-score LLM"].iloc[0]
    trm361_h1 = h1.loc[h1["system"] == "TRM 361957 pass@2"].iloc[0]
    trm651_h1 = h1.loc[h1["system"] == "TRM 651522 pass@2"].iloc[0]
    varc_h1 = h1.loc[h1["system"] == "VARC ARC-2_ViT pass@2"].iloc[0]

    fixed_subset = fixed_accuracy.loc[fixed_accuracy["dataset"] == "ARC-2 primary"].copy()
    trm361_fixed = fixed_subset.loc[fixed_subset["system"] == "TRM 361957 pass@2"].iloc[0]
    trm651_fixed = fixed_subset.loc[fixed_subset["system"] == "TRM 651522 pass@2"].iloc[0]
    varc_fixed = fixed_subset.loc[fixed_subset["system"] == "VARC ARC-2_ViT pass@2"].iloc[0]

    diff_main = diff_tests.loc[diff_tests["contrast"] == "LLM average - TRM 651522 pass@2"].iloc[0]
    diff_varc = diff_tests.loc[diff_tests["contrast"] == "LLM average - VARC ARC-2_ViT pass@2"].iloc[0]
    diff_mid = diff_tests.loc[diff_tests["contrast"] == "LLM average - TRM 361957 pass@2"].iloc[0]

    trm361_res = residual_tests.loc[residual_tests["system"] == "TRM 361957 pass@2"].iloc[0]
    trm651_res = residual_tests.loc[residual_tests["system"] == "TRM 651522 pass@2"].iloc[0]
    varc_res = residual_tests.loc[residual_tests["system"] == "VARC ARC-2_ViT pass@2"].iloc[0]

    trm_union = complementarity.loc[complementarity["system"] == "TRM+VARC union"].iloc[0]
    arc1_comp = arc1_summary.loc[arc1_summary["system"] == "CompressARC top2"].iloc[0]

    trajectory_acc = trajectory_tests.loc[trajectory_tests["test"] == "TRM pass@2 step vs accuracy"].iloc[0]
    trajectory_human = trajectory_tests.loc[trajectory_tests["test"] == "TRM pass@2 step vs human_corr"].iloc[0]
    trajectory_tradeoff = trajectory_tests.loc[trajectory_tests["test"] == "TRM pass@2 accuracy vs human_corr"].iloc[0]

    lines = [
        "# Results",
        "",
        "## Reliability Context and Expected Magnitudes",
        "",
        f"Our primary ARC-2 analysis uses `{int(arc2_scope['human_pairs'])}` public-evaluation test pairs with at least `{PRIMARY_THRESHOLD}` human attempts each. Human split-half reliability on this subset is moderate rather than high (median Pearson `{arc2_scope['human_split_median']:.3f}`, 95% interval `{arc2_scope['human_split_ci']}`), which implies a Spearman-Brown full-length reliability of `{arc2_scope['human_full_reliability']:.3f}` and an approximate raw-correlation ceiling of `{arc2_scope['observable_corr_ceiling']:.3f}` for any perfect latent predictor measured against these noisy item solve rates. That matters because raw correlations in the `0.2` to `0.4` range are not automatically trivial on this benchmark.",
        "",
        f"A second source of apparent weakness is score sparsity. For low-accuracy binary systems on ARC-2, the fixed-accuracy random-placement null already spans roughly `[{fixed_subset['null_ci_lo'].min():.3f}, {fixed_subset['null_ci_hi'].max():.3f}]` in raw correlation. In other words, a low raw correlation can still be meaningful if it is larger than what a system with the same number of wins would achieve by solving random items.",
        "",
        "## Primary ARC-2 Null Hypotheses",
        "",
        "We formalized `humans, LLMs, and non-LLMs are the same` as several narrower operational nulls. The table below reports the core tests, with Benjamini-Hochberg q-values computed within each hypothesis family.",
        "",
        frame_to_text_table(
            arc2_hypotheses[
                ["null_id", "system", "estimate", "ci_lo", "ci_hi", "p_value", "q_value", "decision", "note"]
            ].round(3)
        ),
        "",
        "The strongest ARC-2 result is that the human-equivalence null is not rejected for the LLM aggregate (`r = "
        f"{llm_avg_h1['estimate']:.3f}`, `p = {llm_avg_h1['p_value']:.3f}`) or for the most human-aligned single LLM (`r = {best_align_h1['estimate']:.3f}`, `p = {best_align_h1['p_value']:.3f}`), but it is rejected for the best current non-LLM profiles: TRM mid-training (`r = {trm361_h1['estimate']:.3f}`, `p = {trm361_h1['p_value']:.4f}`), TRM best-score (`r = {trm651_h1['estimate']:.3f}`, `p = {trm651_h1['p_value']:.4f}`), and VARC (`r = {varc_h1['estimate']:.3f}`, `p = {varc_h1['p_value']:.4f}`). The best-score LLM sits right on the boundary (`r = {best_score_h1['estimate']:.3f}`, `p = {best_score_h1['p_value']:.3f}`), so we treat that case as borderline rather than decisive.",
        "",
        "The feature-only null is also too strong. After controlling for coarse task features, the LLM average remains aligned with humans (`partial r = "
        f"{h2.loc[h2['system'] == 'LLM average', 'estimate'].iloc[0]:.3f}`, `p < 0.001`), and so do the best-aligned TRM checkpoint and best VARC profile. That means their human-alignment is not reducible to simple size, color-count, or train-pair cues alone. However, the strongest residual null is more mixed: after controlling for the LLM average itself, the TRM best-score checkpoint does not retain clear extra human signal (`partial r = {trm651_res['partial_corr_given_llm_average']:.3f}`, `p = {trm651_res['partial_p_value']:.3f}`), whereas the mid-training TRM checkpoint (`partial r = {trm361_res['partial_corr_given_llm_average']:.3f}`, `p = {trm361_res['partial_p_value']:.3f}`) and VARC (`partial r = {varc_res['partial_corr_given_llm_average']:.3f}`, `p = {varc_res['partial_p_value']:.3f}`) do show weak residual alignment under regression control. We treat that residual evidence as suggestive rather than fully settled, because the more conservative subtraction-based check is less decisive.",
        "",
        "## Low Correlations: What Is Expected, and What Is Not",
        "",
        "The low absolute correlations are partly expected here for three reasons. First, the human benchmark itself is noisy. Second, the ARC-2 primary subset contains only `110` robustly sampled test pairs. Third, several non-LLM systems solve very few items, so their achievable item-profile signal is inherently sparse. The right question is therefore not `is the raw correlation numerically large?` but `is it larger than the two relevant nulls: human measurement noise and random item placement at the same accuracy?`",
        "",
        frame_to_text_table(
            fixed_accuracy[
                ["dataset", "system", "pair_accuracy", "successes", "observed_corr", "null_ci_lo", "null_ci_hi", "p_value", "q_value", "decision"]
            ].round(3)
        ),
        "",
        "That fixed-accuracy null changes the interpretation substantially. The mid-training TRM checkpoint exceeds the same-accuracy random-placement null (`p = "
        f"{trm361_fixed['p_value']:.3f}`), as does VARC, but only narrowly (`p = {varc_fixed['p_value']:.3f}`). The later high-score TRM checkpoint does not (`p = {trm651_fixed['p_value']:.3f}`). So the `TRM 651522` profile is better at raw ARC scoring but not at placing its wins on specifically human-like items. This is exactly the kind of distinction that the raw accuracy table misses.",
        "",
        "## Direct System-to-System Comparisons",
        "",
        frame_to_text_table(diff_tests.round(3)),
        "",
        "Paired bootstrap comparison tests show that the LLM average is more human-aligned than the best-score TRM checkpoint (`delta r = "
        f"{diff_main['estimate']:.3f}`, `p = {diff_main['p_value']:.3f}`) and more human-aligned than VARC (`delta r = {diff_varc['estimate']:.3f}`, `p = {diff_varc['p_value']:.3f}`). The comparison against the most human-aligned mid-training TRM checkpoint points in the same direction but does not clear the `0.05` threshold on this subset (`delta r = {diff_mid['estimate']:.3f}`, `p = {diff_mid['p_value']:.3f}`). This is one reason not to over-index on a simple winner-loser story: there is a weak residual non-LLM signal, but it is not as stable or as large as the LLM aggregate alignment.",
        "",
        "The accuracy-matched LLM check reaches a similar conclusion.",
        "",
        frame_to_text_table(matched_arc2.round(3)),
        "",
        "The best-score TRM checkpoint falls below the median human-alignment of accuracy-matched weak LLMs. The best-aligned TRM checkpoint and VARC sit inside the weak-LLM band, not clearly above it. So current non-LLMs do not look like the LLM consensus, but neither do they look wholly alien to the low-performance end of the LLM distribution.",
        "",
    ]

    lines.extend(
        [
            "## Complementarity and Training Dynamics",
            "",
            frame_to_text_table(complementarity.round(3)),
            "",
            frame_to_text_table(trajectory_tests.round(3)),
            "",
            "Despite their weaker overall human alignment, the non-LLM systems are not redundant. The TRM+VARC union rescues `"
            f"{int(trm_union['rescued_human_easy_llm_hard'])}` human-easy / LLM-hard ARC-2 items, which is more than expected by chance (`p = {trm_union['hypergeom_p_value']:.3f}`). The trajectory analysis clarifies what is happening inside TRM: training step strongly predicts accuracy (`Spearman rho = {trajectory_acc['estimate']:.3f}`, `p = {trajectory_acc['p_value']:.3f}`), but not human alignment (`Spearman rho = {trajectory_human['estimate']:.3f}`, `p = {trajectory_human['p_value']:.3f}`). The accuracy-vs-alignment relation is negative but not significant (`rho = {trajectory_tradeoff['estimate']:.3f}`, `p = {trajectory_tradeoff['p_value']:.3f}`). In plain language, later checkpoints get better at the benchmark without reliably becoming more human-like.",
            "",
            "## ARC-1 Sidecar",
            "",
            f"We do not have a dedicated ARC-1 human benchmark file, but we do have an ARC-1 sidecar through task reuse in the ARC-AGI-2 Public Train human data. That yields `{int(arc1_scope['human_pairs'])}` single-pair ARC-1 evaluation tasks with a human split-half median of `{arc1_scope['human_split_median']:.3f}`.",
            "",
            frame_to_text_table(arc1_summary.round(3)),
            "",
            frame_to_text_table(matched_arc1.round(3)),
            "",
            f"On that ARC-1 overlap, CompressARC is genuinely non-random with respect to human difficulty (`r = {arc1_comp['human_pearson']:.3f}`, fixed-accuracy `p = {fixed_accuracy.loc[fixed_accuracy['system'] == 'CompressARC top2', 'p_value'].iloc[0]:.3f}`), but it still falls well below the human split-half benchmark (`p < 0.001` against human-equivalence). So CompressARC is a useful real prediction artifact, but it does not overturn the broader ARC-2 pattern.",
            "",
            "## Interpretation",
            "",
            "Taken together, these tests support a mixed but fairly clear synthesis. We reject the strong null that the current non-LLM systems are psychometrically indistinguishable from humans on ARC-2. We also reject the claim that all observed alignment is just trivial task-feature matching. At the same time, we do not have grounds to claim that non-LLMs define a wholly separate and robust human-like axis beyond the LLM average. Some evidence for a small residual exists, especially for the mid-training TRM checkpoint and VARC, but it is weak, method-sensitive, and much smaller than the main LLM aggregate effect.",
            "",
            "The most defensible take is therefore: human item difficulty on ARC-2 is better approximated by the LLM consensus than by the currently stored non-LLM systems, yet the non-LLM systems still contribute complementary successes and some nontrivial human-relevant structure. The important distinction is not `same` versus `completely different`; it is `closer to the human difficulty axis`, and on that measure the LLM aggregate is currently ahead.",
            "",
        ]
    )

    text = "\n".join(lines)
    RESULTS_MD.write_text(text, encoding="utf-8")
    TECHNICAL_MD.write_text(text, encoding="utf-8")


def main() -> None:
    _, truth_outputs = load_arc2_truth()
    human_raw, human_meta = load_human_inputs()
    subset = build_human_subset(human_meta, human_raw, PRIMARY_THRESHOLD)
    arc2_human, arc2_profiles, arc2_frame, llm_matrix, _trm_matrix = build_arc2_profiles()

    split_halves = split_half_correlations(subset.raw_attempts, n_sims=5000, seed=0)
    split_median = float(split_halves["pearson"].median())
    split_ci_lo = float(split_halves["pearson"].quantile(0.025))
    split_ci_hi = float(split_halves["pearson"].quantile(0.975))
    split_reliability_full = 2 * split_median / (1 + split_median)
    observable_corr_ceiling = math.sqrt(split_reliability_full)

    human_counts = subset.raw_attempts.groupby("task_pair_id")["solved"].sum().reindex(subset.pairs)
    human_attempts = subset.raw_attempts.groupby("task_pair_id")["solved"].size().reindex(subset.pairs)
    smoothed_human = (human_counts + 0.5) / (human_attempts + 1.0)
    human_minus_llm = arc2_human - arc2_profiles["LLM average"]
    feature_controls = arc2_frame.loc[subset.pairs, FEATURE_COLUMNS].fillna(0.0)

    data_scope_rows = [
        {
            "dataset": "ARC-2 primary",
            "human_pairs": len(subset.pairs),
            "human_split_median": split_median,
            "human_split_ci": f"[{split_ci_lo:.3f}, {split_ci_hi:.3f}]",
            "human_full_reliability": split_reliability_full,
            "observable_corr_ceiling": observable_corr_ceiling,
            "note": "Public Eval pairs with >=8 human attempts",
        },
        {
            "dataset": "ARC-2 eval overlap",
            "human_pairs": int(human_meta["task_pair_id"].nunique()),
            "human_split_median": np.nan,
            "human_split_ci": "",
            "human_full_reliability": np.nan,
            "observable_corr_ceiling": np.nan,
            "note": "Public Eval pairs available in the existing human-vs-model table",
        },
    ]

    arc2_hypothesis_rows: list[dict] = []
    h1_systems = [
        "LLM average",
        "Best-aligned LLM",
        "Best-score LLM",
        "TRM 361957 pass@2",
        "TRM 651522 pass@2",
        "VARC ARC-2_ViT pass@2",
    ]
    for system_name in h1_systems:
        profile = arc2_profiles[system_name]
        human_corr = corr_or_nan(profile, arc2_human)
        percentile = float((split_halves["pearson"] <= human_corr).mean())
        p_value = empirical_pvalue(split_halves["pearson"], human_corr, alternative="two-sided")
        arc2_hypothesis_rows.append(
            {
                "family": "human_equivalence",
                "null_id": "H1",
                "system": system_name,
                "test": "human_equivalence_vs_split_half",
                "estimate": human_corr,
                "ci_lo": split_ci_lo,
                "ci_hi": split_ci_hi,
                "reference": percentile,
                "p_value": p_value,
                "decision": "reject" if p_value < 0.05 else "fail_to_reject",
                "note": "Two-sided empirical p-value against the human split-half distribution",
            }
        )

    for seed, system_name in enumerate(["LLM average", "TRM 651522 pass@2", "TRM 361957 pass@2", "VARC ARC-2_ViT pass@2"]):
        profile = arc2_profiles[system_name]
        obs, lo, hi, p_value = bootstrap_partial_corr_stats(
            arc2_human,
            profile,
            feature_controls,
            n_boot=4000,
            seed=seed,
            alternative="greater",
        )
        arc2_hypothesis_rows.append(
            {
                "family": "feature_controls",
                "null_id": "H2",
                "system": system_name,
                "test": "no_alignment_after_feature_controls",
                "estimate": obs,
                "ci_lo": lo,
                "ci_hi": hi,
                "reference": 0.0,
                "p_value": p_value,
                "decision": "reject" if p_value < 0.05 else "fail_to_reject",
                "note": "One-sided bootstrap test on partial corr after simple feature controls",
            }
        )

    for seed, system_name in enumerate(["TRM 651522 pass@2", "TRM 361957 pass@2", "VARC ARC-2_ViT pass@2"], start=20):
        profile = arc2_profiles[system_name]
        obs, lo, hi, p_value = bootstrap_partial_corr_stats(
            arc2_human,
            profile,
            arc2_profiles["LLM average"],
            n_boot=4000,
            seed=seed,
            alternative="greater",
        )
        arc2_hypothesis_rows.append(
            {
                "family": "llm_control",
                "null_id": "H3",
                "system": system_name,
                "test": "no_extra_human_signal_after_controlling_llm_average",
                "estimate": obs,
                "ci_lo": lo,
                "ci_hi": hi,
                "reference": 0.0,
                "p_value": p_value,
                "decision": "reject" if p_value < 0.05 else "fail_to_reject",
                "note": "One-sided bootstrap test on partial corr controlling the LLM average",
            }
        )

    arc2_hypotheses = pd.DataFrame(arc2_hypothesis_rows)
    arc2_hypotheses["q_value"] = np.nan
    for idx in arc2_hypotheses.groupby("family").groups.values():
        arc2_hypotheses.loc[idx, "q_value"] = bh_adjust(arc2_hypotheses.loc[idx, "p_value"])

    complementarity_population = len(subset.pairs)
    human_easy_llm_hard = arc2_frame.loc[(arc2_human >= 0.7) & (arc2_profiles["LLM average"] <= 0.1)]
    complementarity_successes = len(human_easy_llm_hard)
    union_profile = pd.Series(
        np.maximum(
            arc2_profiles["TRM 651522 pass@2"].to_numpy(),
            arc2_profiles["VARC ARC-2_ViT pass@2"].to_numpy(),
        ),
        index=arc2_profiles["TRM 651522 pass@2"].index,
    )
    complementarity_rows: list[dict] = []
    for system_name, profile in [
        ("TRM 651522 pass@2", arc2_profiles["TRM 651522 pass@2"]),
        ("VARC ARC-2_ViT pass@2", arc2_profiles["VARC ARC-2_ViT pass@2"]),
        ("TRM+VARC union", union_profile),
    ]:
        solved_set = set(profile.index[profile > 0])
        rescue_set = solved_set & set(human_easy_llm_hard.index)
        draws = len(solved_set)
        hits = len(rescue_set)
        complementarity_rows.append(
            {
                "system": system_name,
                "total_solved_items": draws,
                "rescued_human_easy_llm_hard": hits,
                "expected_if_random": draws * complementarity_successes / complementarity_population,
                "hypergeom_p_value": hypergeom_tail(hits, complementarity_population, complementarity_successes, draws),
            }
        )
    complementarity = pd.DataFrame(complementarity_rows)
    complementarity["q_value"] = bh_adjust(complementarity["hypergeom_p_value"])
    complementarity["decision"] = np.where(complementarity["hypergeom_p_value"] < 0.05, "reject", "fail_to_reject")

    diff_rows = []
    diff_specs = [
        ("LLM average - TRM 361957 pass@2", arc2_profiles["LLM average"], arc2_profiles["TRM 361957 pass@2"]),
        ("LLM average - TRM 651522 pass@2", arc2_profiles["LLM average"], arc2_profiles["TRM 651522 pass@2"]),
        ("LLM average - VARC ARC-2_ViT pass@2", arc2_profiles["LLM average"], arc2_profiles["VARC ARC-2_ViT pass@2"]),
        ("Best-aligned LLM - TRM 361957 pass@2", arc2_profiles["Best-aligned LLM"], arc2_profiles["TRM 361957 pass@2"]),
        ("TRM 361957 pass@2 - TRM 651522 pass@2", arc2_profiles["TRM 361957 pass@2"], arc2_profiles["TRM 651522 pass@2"]),
    ]
    for seed, (contrast, left, right) in enumerate(diff_specs):
        estimate, ci_lo, ci_hi, p_value = bootstrap_corr_difference_stats(arc2_human, left, right, n_boot=6000, seed=seed)
        diff_rows.append(
            {
                "contrast": contrast,
                "estimate": estimate,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p_value": p_value,
                "decision": "reject" if p_value < 0.05 else "fail_to_reject",
            }
        )
    diff_tests = pd.DataFrame(diff_rows)
    diff_tests["q_value"] = bh_adjust(diff_tests["p_value"])

    matched_arc2 = pd.DataFrame(
        [
            find_accuracy_matched_llms("TRM 651522 pass@2", arc2_profiles["TRM 651522 pass@2"], llm_matrix, arc2_human),
            find_accuracy_matched_llms("TRM 361957 pass@2", arc2_profiles["TRM 361957 pass@2"], llm_matrix, arc2_human),
            find_accuracy_matched_llms("VARC ARC-2_ViT pass@2", arc2_profiles["VARC ARC-2_ViT pass@2"], llm_matrix, arc2_human),
        ]
    )

    residual_rows = []
    for seed, name in enumerate(
        ["LLM average", "Best-aligned LLM", "Best-score LLM", "TRM 651522 pass@2", "TRM 361957 pass@2", "VARC ARC-2_ViT pass@2"],
        start=100,
    ):
        profile = arc2_profiles[name]
        partial_obs, partial_lo, partial_hi, partial_p = bootstrap_partial_corr_stats(
            arc2_human,
            profile,
            arc2_profiles["LLM average"],
            n_boot=3000,
            seed=seed,
            alternative="greater",
        )
        conservative_obs, conservative_lo, conservative_hi, conservative_p = bootstrap_corr_stats(
            human_minus_llm,
            profile,
            n_boot=3000,
            seed=seed + 50,
            alternative="greater",
        )
        residual_rows.append(
            {
                "system": name,
                "raw_human_corr": corr_or_nan(profile, arc2_human),
                "smoothed_human_corr": corr_or_nan(profile, smoothed_human),
                "disattenuated_human_corr": corr_or_nan(profile, arc2_human) / math.sqrt(split_reliability_full),
                "corr_with_llm_average": corr_or_nan(profile, arc2_profiles["LLM average"]),
                "partial_corr_given_llm_average": partial_obs,
                "partial_ci_lo": partial_lo,
                "partial_ci_hi": partial_hi,
                "partial_p_value": partial_p,
                "conservative_subtraction_corr": conservative_obs,
                "conservative_ci_lo": conservative_lo,
                "conservative_ci_hi": conservative_hi,
                "conservative_p_value": conservative_p,
            }
        )
    residual_tests = pd.DataFrame(residual_rows)

    fixed_rows = []
    for seed, system_name in enumerate(
        ["Best-score LLM", "Best-aligned LLM", "TRM 361957 pass@2", "TRM 651522 pass@2", "VARC ARC-2_ViT pass@2"]
    ):
        row = fixed_accuracy_random_null(arc2_profiles[system_name], arc2_human, n_sim=5000, seed=seed)
        row["dataset"] = "ARC-2 primary"
        row["system"] = system_name
        fixed_rows.append(row)

    arc1_human, arc1_profiles, arc1_llm_matrix = build_arc1_sidecar()
    arc1_raw = pd.read_csv(ROOT_DIR / "data-human" / "test_pair_attempts.csv")
    arc1_raw["task_pair_id"] = arc1_raw["task_ID"] + "__" + arc1_raw["test_index"].astype(str)
    arc1_raw["solved"] = (arc1_raw["correct_submissions"] > 0).astype(int)
    arc1_raw = arc1_raw.loc[(arc1_raw["task_set"] == "Public Train") & (arc1_raw["task_pair_id"].isin(set(arc1_human.index)))]
    arc1_split = split_half_correlations(arc1_raw, n_sims=4000, seed=1)
    arc1_median = float(arc1_split["pearson"].median())
    arc1_ci_lo = float(arc1_split["pearson"].quantile(0.025))
    arc1_ci_hi = float(arc1_split["pearson"].quantile(0.975))
    arc1_reliability_full = 2 * arc1_median / (1 + arc1_median)
    arc1_ceiling = math.sqrt(arc1_reliability_full)

    data_scope_rows.append(
        {
            "dataset": "ARC-1 sidecar",
            "human_pairs": int(len(arc1_human)),
            "human_split_median": arc1_median,
            "human_split_ci": f"[{arc1_ci_lo:.3f}, {arc1_ci_hi:.3f}]",
            "human_full_reliability": arc1_reliability_full,
            "observable_corr_ceiling": arc1_ceiling,
            "note": "Single-pair ARC-1 eval tasks reused inside ARC-AGI-2 Public Train",
        }
    )
    data_scope = pd.DataFrame(data_scope_rows)

    arc1_rows = []
    for system_name, profile in arc1_profiles.items():
        human_corr = corr_or_nan(profile, arc1_human)
        percentile = float((arc1_split["pearson"] <= human_corr).mean())
        p_value = empirical_pvalue(arc1_split["pearson"], human_corr, alternative="two-sided")
        arc1_rows.append(
            {
                "system": system_name,
                "pair_accuracy": float(profile.mean()),
                "human_pearson": human_corr,
                "percentile_vs_human_split": percentile,
                "human_split_median": arc1_median,
                "human_split_ci_lo": arc1_ci_lo,
                "human_split_ci_hi": arc1_ci_hi,
                "p_value": p_value,
                "decision": "reject" if p_value < 0.05 else "fail_to_reject",
            }
        )
    arc1_summary = pd.DataFrame(arc1_rows)
    arc1_summary["q_value"] = bh_adjust(arc1_summary["p_value"])

    arc1_fixed = fixed_accuracy_random_null(arc1_profiles["CompressARC top2"], arc1_human, n_sim=5000, seed=50)
    arc1_fixed["dataset"] = "ARC-1 sidecar"
    arc1_fixed["system"] = "CompressARC top2"
    fixed_rows.append(arc1_fixed)

    fixed_accuracy = pd.DataFrame(fixed_rows)
    fixed_accuracy["q_value"] = bh_adjust(fixed_accuracy["p_value"])
    fixed_accuracy["decision"] = np.where(fixed_accuracy["p_value"] < 0.05, "reject", "fail_to_reject")

    matched_arc1 = pd.DataFrame(
        [find_accuracy_matched_llms("CompressARC top2", arc1_profiles["CompressARC top2"], arc1_llm_matrix, arc1_human)]
    )

    trm_pass2 = pd.read_csv(TABLES_DIR / "trm_trajectory.csv")
    trm_pass2 = trm_pass2.loc[trm_pass2["score_mode"] == "pass@2"].sort_values("step")
    trajectory_rows = []
    for seed, (label, x_col, y_col, method) in enumerate(
        [
            ("TRM pass@2 step vs accuracy", "step", "pair_accuracy", "spearman"),
            ("TRM pass@2 step vs human_corr", "step", "human_pearson", "spearman"),
            ("TRM pass@2 accuracy vs human_corr", "pair_accuracy", "human_pearson", "spearman"),
        ]
    ):
        estimate, p_value = permutation_corr_test(trm_pass2[x_col], trm_pass2[y_col], method=method, n_perm=20000, seed=seed)
        trajectory_rows.append(
            {
                "test": label,
                "statistic": method,
                "estimate": estimate,
                "p_value": p_value,
                "decision": "reject" if p_value < 0.05 else "fail_to_reject",
            }
        )
    trajectory_tests = pd.DataFrame(trajectory_rows)
    trajectory_tests["q_value"] = bh_adjust(trajectory_tests["p_value"])

    data_scope.to_csv(TABLES_DIR / "data_scope_summary.csv", index=False)
    arc2_hypotheses.to_csv(TABLES_DIR / "hypothesis_test_summary.csv", index=False)
    diff_tests.to_csv(TABLES_DIR / "correlation_difference_tests.csv", index=False)
    matched_arc2.to_csv(TABLES_DIR / "accuracy_matched_llm_comparison.csv", index=False)
    residual_tests.to_csv(TABLES_DIR / "residual_alignment_tests.csv", index=False)
    complementarity.to_csv(TABLES_DIR / "complementarity_hypothesis_tests.csv", index=False)
    arc1_summary.to_csv(TABLES_DIR / "arc1_sidecar_summary.csv", index=False)
    matched_arc1.to_csv(TABLES_DIR / "arc1_accuracy_matched_llm.csv", index=False)
    fixed_accuracy.to_csv(TABLES_DIR / "fixed_accuracy_null_tests.csv", index=False)
    trajectory_tests.to_csv(TABLES_DIR / "trajectory_hypothesis_tests.csv", index=False)

    write_results_section(
        data_scope=data_scope,
        arc2_hypotheses=arc2_hypotheses,
        diff_tests=diff_tests,
        fixed_accuracy=fixed_accuracy,
        matched_arc2=matched_arc2,
        residual_tests=residual_tests,
        complementarity=complementarity,
        trajectory_tests=trajectory_tests,
        arc1_summary=arc1_summary,
        matched_arc1=matched_arc1,
    )

    print(f"Done. Outputs written to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
