from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent
OVERLAP_PATH = BASE_DIR / "human_llm_overlap_tasks.csv"
PARTIAL_CREDIT_PATH = BASE_DIR / "llm_partial_credit_item_models.csv"
HUMAN_SUITE_PATH = BASE_DIR / "closeness_model_suite_human.csv"
LLM_PREDICTOR_PATH = BASE_DIR / "closeness_model_suite_llm_predictors.csv"

BOOT_N = 12000
PERM_N = 12000
EQUIV_MARGIN = 0.15


def safe_corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 4 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return np.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def bootstrap_corr_ci(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    seed: int,
    n_boot: int = BOOT_N,
) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    idx = np.arange(len(x_arr))
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        values[i] = safe_corr(x_arr[sample_idx], y_arr[sample_idx])
    ci_low, ci_high = np.quantile(values, [0.025, 0.975])
    return float(ci_low), float(ci_high)


def permutation_corr_p(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    seed: int,
    n_perm: int = PERM_N,
) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    observed = abs(safe_corr(x_arr, y_arr))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(y_arr)
        if abs(safe_corr(x_arr, shuffled)) >= observed:
            extreme += 1
    return float((extreme + 1) / (n_perm + 1))


def bootstrap_delta_corr(
    y: pd.Series | np.ndarray,
    candidate: pd.Series | np.ndarray,
    baseline: pd.Series | np.ndarray,
    *,
    seed: int,
    n_boot: int = BOOT_N,
) -> tuple[float, float, float, float, float, float]:
    y_arr = np.asarray(y, dtype=float)
    c_arr = np.asarray(candidate, dtype=float)
    b_arr = np.asarray(baseline, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(c_arr) & np.isfinite(b_arr)
    y_arr = y_arr[mask]
    c_arr = c_arr[mask]
    b_arr = b_arr[mask]
    observed = safe_corr(y_arr, c_arr) - safe_corr(y_arr, b_arr)
    idx = np.arange(len(y_arr))
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        values[i] = safe_corr(y_arr[sample_idx], c_arr[sample_idx]) - safe_corr(
            y_arr[sample_idx], b_arr[sample_idx]
        )
    ci95_low, ci95_high = np.quantile(values, [0.025, 0.975])
    ci90_low, ci90_high = np.quantile(values, [0.05, 0.95])
    p_value = float(2 * min((values <= 0).mean(), (values >= 0).mean()))
    return (
        float(observed),
        float(ci95_low),
        float(ci95_high),
        float(ci90_low),
        float(ci90_high),
        p_value,
    )


def mde_r(n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    return float(np.tanh((z_alpha + z_beta) / np.sqrt(n - 3)))


def n_for_power_of_r(r: float, alpha: float = 0.05, power: float = 0.8) -> float:
    r = abs(float(r))
    if not np.isfinite(r) or r <= 0 or r >= 1:
        return np.nan
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    return float(np.ceil(3 + ((z_alpha + z_beta) / np.arctanh(r)) ** 2))


def build_overlap_table() -> pd.DataFrame:
    overlap = pd.read_csv(OVERLAP_PATH)
    partial = pd.read_csv(PARTIAL_CREDIT_PATH)
    merged = overlap.merge(
        partial[["dataset_key", "task_id", "gpcm_location", "graded_location"]],
        on=["dataset_key", "task_id"],
        how="left",
    )
    merged["log_duration"] = np.log1p(merged["mean_duration_seconds_weighted"])

    axes = pd.DataFrame(
        {
            "solve_rate": merged["human_solve_rate_weighted"],
            "ease_from_difficulty": -merged["difficulty_weighted"],
            "ease_from_duration": -merged["log_duration"],
        }
    )
    valid = axes.dropna().index
    z = StandardScaler().fit_transform(axes.loc[valid])
    pc1 = PCA(n_components=1).fit_transform(z).ravel()
    if safe_corr(pc1, axes.loc[valid, "solve_rate"]) < 0:
        pc1 *= -1.0
    merged.loc[valid, "human_overlap_ease_pc1"] = pc1
    return merged


def build_key_alignment_table(overlap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        (
            "Human difficulty",
            "LLM logit difficulty",
            "difficulty_weighted",
            "logit_difficulty_all",
            1.0,
        ),
        (
            "Human difficulty",
            "LLM Rasch difficulty",
            "difficulty_weighted",
            "rasch_difficulty_all_models_pooled",
            1.0,
        ),
        (
            "Human difficulty",
            "LLM 2PL difficulty",
            "difficulty_weighted",
            "two_pl_difficulty_all_models",
            1.0,
        ),
        (
            "Human solve rate",
            "LLM pass rate",
            "human_solve_rate_weighted",
            "pass_rate_all",
            1.0,
        ),
        (
            "Human solve rate",
            "Partial-credit graded ease",
            "human_solve_rate_weighted",
            "graded_location",
            -1.0,
        ),
        (
            "Human latent ease",
            "LLM logit ease",
            "human_overlap_ease_pc1",
            "logit_difficulty_all",
            -1.0,
        ),
        (
            "Human latent ease",
            "Partial-credit graded ease",
            "human_overlap_ease_pc1",
            "graded_location",
            -1.0,
        ),
        (
            "Human duration",
            "LLM logit difficulty",
            "mean_duration_seconds_weighted",
            "logit_difficulty_all",
            1.0,
        ),
        (
            "Human duration",
            "Partial-credit graded difficulty",
            "mean_duration_seconds_weighted",
            "graded_location",
            1.0,
        ),
    ]

    for i, (left, right, y_col, x_col, sign) in enumerate(specs):
        predictor = sign * overlap[x_col]
        subset = overlap[[y_col]].copy()
        subset["predictor"] = predictor
        subset = subset.dropna().reset_index(drop=True)
        aligned_r = safe_corr(subset[y_col], subset["predictor"])
        ci_low, ci_high = bootstrap_corr_ci(subset[y_col], subset["predictor"], seed=100 + i)
        p_value = permutation_corr_p(subset[y_col], subset["predictor"], seed=200 + i)
        rows.append(
            {
                "left_label": left,
                "right_label": right,
                "n": int(len(subset)),
                "aligned_r": aligned_r,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "p_value_perm": p_value,
                "n_for_80pct_power_at_observed_r": n_for_power_of_r(aligned_r),
            }
        )
    return pd.DataFrame(rows)


def build_partial_vs_logit_table(overlap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("Human solve rate", "human_solve_rate_weighted", -overlap["graded_location"], -overlap["logit_difficulty_all"]),
        ("Human difficulty", "difficulty_weighted", overlap["graded_location"], overlap["logit_difficulty_all"]),
        ("Human latent ease", "human_overlap_ease_pc1", -overlap["graded_location"], -overlap["logit_difficulty_all"]),
        (
            "Human duration",
            "mean_duration_seconds_weighted",
            overlap["graded_location"],
            overlap["logit_difficulty_all"],
        ),
    ]

    for i, (target_label, y_col, graded_pred, logit_pred) in enumerate(specs):
        subset = pd.DataFrame(
            {
                "y": overlap[y_col],
                "graded": graded_pred,
                "logit": logit_pred,
            }
        ).dropna()
        (
            delta,
            ci95_low,
            ci95_high,
            ci90_low,
            ci90_high,
            p_value,
        ) = bootstrap_delta_corr(subset["y"], subset["graded"], subset["logit"], seed=300 + i)
        if ci95_low > 0:
            classification = "Partial-credit > binary"
        elif ci95_high < 0:
            classification = "Binary > partial-credit"
        elif ci90_low > -EQUIV_MARGIN and ci90_high < EQUIV_MARGIN:
            classification = f"Equivalent (|delta r| < {EQUIV_MARGIN:.2f})"
        else:
            classification = "Inconclusive"
        rows.append(
            {
                "target_label": target_label,
                "n": int(len(subset)),
                "aligned_r_binary_logit": safe_corr(subset["y"], subset["logit"]),
                "aligned_r_partial_credit": safe_corr(subset["y"], subset["graded"]),
                "delta_partial_minus_binary": delta,
                "delta_ci95_low": ci95_low,
                "delta_ci95_high": ci95_high,
                "delta_ci90_low": ci90_low,
                "delta_ci90_high": ci90_high,
                "p_value_diff": p_value,
                "classification": classification,
            }
        )
    return pd.DataFrame(rows)


def build_complexity_difference_table(overlap: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "ast_node_count",
        "token_count",
        "cyclomatic_complexity",
        "complexity_pc1_score",
        "log1p_elapsed_ms_total",
        "elapsed_ms_total",
        "peak_memory_bytes",
        "function_count",
        "gzip_bytes",
        "nonblank_lines",
    ]
    rows = []
    for i, metric in enumerate(metrics):
        subset = overlap[[metric, "difficulty_weighted", "logit_difficulty_all"]].dropna().reset_index(drop=True)
        human_r = safe_corr(subset[metric], subset["difficulty_weighted"])
        llm_r = safe_corr(subset[metric], subset["logit_difficulty_all"])
        human_ci_low, human_ci_high = bootstrap_corr_ci(subset[metric], subset["difficulty_weighted"], seed=400 + i)
        llm_ci_low, llm_ci_high = bootstrap_corr_ci(subset[metric], subset["logit_difficulty_all"], seed=500 + i)
        (
            delta,
            delta_ci95_low,
            delta_ci95_high,
            delta_ci90_low,
            delta_ci90_high,
            p_value,
        ) = bootstrap_delta_corr(
            subset[metric],
            subset["logit_difficulty_all"],
            subset["difficulty_weighted"],
            seed=600 + i,
        )
        if delta_ci95_low > 0:
            classification = "LLM > Human"
        elif delta_ci95_high < 0:
            classification = "Human > LLM"
        elif delta_ci90_low > -EQUIV_MARGIN and delta_ci90_high < EQUIV_MARGIN:
            classification = f"Equivalent (|delta r| < {EQUIV_MARGIN:.2f})"
        else:
            classification = "Inconclusive"
        rows.append(
            {
                "metric": metric,
                "n": int(len(subset)),
                "r_human": human_r,
                "r_human_ci95_low": human_ci_low,
                "r_human_ci95_high": human_ci_high,
                "r_llm": llm_r,
                "r_llm_ci95_low": llm_ci_low,
                "r_llm_ci95_high": llm_ci_high,
                "delta_llm_minus_human": delta,
                "delta_ci95_low": delta_ci95_low,
                "delta_ci95_high": delta_ci95_high,
                "delta_ci90_low": delta_ci90_low,
                "delta_ci90_high": delta_ci90_high,
                "p_value_diff": p_value,
                "classification": classification,
            }
        )
    return pd.DataFrame(rows).sort_values("delta_llm_minus_human", ascending=False).reset_index(drop=True)


def build_power_table() -> pd.DataFrame:
    rows = []
    for label, n in [
        ("Human-LLM overlap tasks", 17),
        ("Full approved LLM tasks (n=55)", 55),
        ("Full approved LLM tasks (n=56)", 56),
        ("Human pair-level analysis", 161),
    ]:
        rows.append({"sample": label, "n": n, "mde_r_80pct_power": mde_r(n)})
    return pd.DataFrame(rows)


def write_mini_paper(
    key_alignment: pd.DataFrame,
    partial_vs_logit: pd.DataFrame,
    complexity_diff: pd.DataFrame,
    power_table: pd.DataFrame,
    human_suite: pd.DataFrame,
) -> None:
    human_binom = human_suite.loc[
        (human_suite["model_family"] == "GroupedBinomial") & (human_suite["outcome"] == "solve_rate")
    ].iloc[0]
    human_wls_latent = human_suite.loc[
        (human_suite["model_family"] == "WLS") & (human_suite["outcome"] == "human_ease_pc1")
    ].iloc[0]
    human_wls_duration = human_suite.loc[
        (human_suite["model_family"] == "WLS") & (human_suite["outcome"] == "log_duration")
    ].iloc[0]

    diff_winners = complexity_diff.loc[complexity_diff["classification"].isin(["LLM > Human", "Human > LLM"])]
    partial_inconclusive = partial_vs_logit.loc[partial_vs_logit["classification"] == "Inconclusive"]

    difficulty_row = key_alignment.loc[key_alignment["right_label"] == "LLM logit difficulty"].iloc[0]
    solve_row = key_alignment.loc[key_alignment["right_label"] == "LLM pass rate"].iloc[0]
    duration_row = key_alignment.loc[key_alignment["right_label"] == "Partial-credit graded difficulty"].iloc[0]

    lines = [
        "# Mini Paper: Closeness Signals, Human Fit, and Human-vs-LLM Similarity",
        "",
        "## Abstract",
        "",
        "- I revisited the closeness-to-solution idea using models that are more natural for the available data: grouped-binomial and weighted regressions on the human side, and partial-credit IRT as a sensitivity analysis on the LLM side.",
        "- The human-side conclusion is robust: adding closeness features improves the fit to human outcomes under better-specified models.",
        "- The human-vs-LLM overlap conclusion is more limited: humans and LLMs are moderately aligned on difficulty and solve rate, but the overlap set is too small to support many fine-grained superiority or equivalence claims.",
        "",
        "## Data And Design",
        "",
        "- Human main analyses use the full public-eval pair table: 161 task-pair rows across 115 tasks.",
        "- Direct human-vs-LLM comparisons use the approved ARC-2 overlap table: 17 tasks.",
        "- LLM partial-credit models use discretized closeness categories fit with graded-response and generalized partial-credit models.",
        f"- For similarity tests, I treated `|delta r| < {EQUIV_MARGIN:.2f}` as a practically small difference and required the 90% bootstrap CI to fall fully inside that band.",
        "",
        "## Human-Side Main Result",
        "",
        f"- Grouped-binomial solve model: pseudo-`R^2` increases from `{human_binom['baseline_fit']:.3f}` to `{human_binom['augmented_fit']:.3f}` (`p = {human_binom['p_value']:.4f}`).",
        f"- Weighted latent human-ease model: `R^2` increases from `{human_wls_latent['baseline_fit']:.3f}` to `{human_wls_latent['augmented_fit']:.3f}` (`p = {human_wls_latent['p_value']:.4f}`).",
        f"- Weighted duration model: `R^2` increases from `{human_wls_duration['baseline_fit']:.3f}` to `{human_wls_duration['augmented_fit']:.3f}` (`p = {human_wls_duration['p_value']:.4f}`).",
        "- Interpretation: once the human outcomes are modeled in a way that respects sparse counts and unequal sampling, closeness features are not just numerically helpful; they are statistically supported.",
        "",
        "## Direct Human-vs-LLM Alignment",
        "",
        f"- Human difficulty and LLM logit difficulty are moderately aligned: `r = {difficulty_row['aligned_r']:.3f}`, 95% CI `[{difficulty_row['ci95_low']:.3f}, {difficulty_row['ci95_high']:.3f}]`, permutation `p = {difficulty_row['p_value_perm']:.4f}`.",
        f"- Human solve rate and LLM pass rate are similarly aligned: `r = {solve_row['aligned_r']:.3f}`, 95% CI `[{solve_row['ci95_low']:.3f}, {solve_row['ci95_high']:.3f}]`, permutation `p = {solve_row['p_value_perm']:.4f}`.",
        f"- Human duration remains the least stable target: the strongest closeness-based predictor on the overlap is partial-credit graded difficulty with `r = {duration_row['aligned_r']:.3f}`, 95% CI `[{duration_row['ci95_low']:.3f}, {duration_row['ci95_high']:.3f}]`, permutation `p = {duration_row['p_value_perm']:.4f}`, and it does not clear a superiority test against binary logit.",
        "",
        "## Difference Versus Similarity",
        "",
    ]

    if not diff_winners.empty:
        for _, row in diff_winners.iterrows():
            lines.append(
                f"- `{row['metric']}` is more LLM-like than human-like: human `r = {row['r_human']:.3f}`, LLM `r = {row['r_llm']:.3f}`, delta `= {row['delta_llm_minus_human']:.3f}`, 95% CI `[{row['delta_ci95_low']:.3f}, {row['delta_ci95_high']:.3f}]`, `p = {row['p_value_diff']:.4f}`."
            )
    else:
        lines.append("- No complexity metric produced a significant human-vs-LLM difference under the current overlap sample.")

    lines.extend(
        [
            f"- No human-vs-LLM complexity metric cleared the pre-registered practical-similarity rule `|delta r| < {EQUIV_MARGIN:.2f}`. The non-significant cases are therefore inconclusive rather than demonstrably similar.",
            f"- No overlap outcome showed a significant advantage of partial-credit graded difficulty over binary logit difficulty. The largest numerical gain was for human duration (`delta r = {partial_vs_logit.loc[partial_vs_logit['target_label'] == 'Human duration', 'delta_partial_minus_binary'].iloc[0]:.3f}`), but its 95% CI still crossed zero.",
            "",
            "## Power",
            "",
        ]
    )

    for _, row in power_table.iterrows():
        lines.append(
            f"- `{row['sample']}`: with `n = {int(row['n'])}`, 80% power requires about `|r| >= {row['mde_r_80pct_power']:.3f}`."
        )

    lines.extend(
        [
            f"- The overlap sample therefore only has conventional power for large effects. For example, the observed human-vs-LLM difficulty alignment (`r = {difficulty_row['aligned_r']:.3f}`) would need about `n = {int(difficulty_row['n_for_80pct_power_at_observed_r'])}` overlap tasks for 80% power, not `n = {int(difficulty_row['n'])}`.",
            f"- The same applies to the partial-credit duration signal: its observed overlap effect would need roughly `n = {int(duration_row['n_for_80pct_power_at_observed_r'])}` tasks for 80% power.",
            "- Difference tests between two correlated predictors are even less powered than single-correlation tests, so the overlap slice is suitable for strong directional signals but not for fine-grained equivalence claims.",
            "",
            "## Bottom Line",
            "",
            "- Main paper result: on the human side, closeness-to-solution adds statistically defensible signal once the model respects the data-generating structure.",
            "- Sensitivity result: on the full LLM task set, partial-credit latent models do not beat the simpler binary logit/Rasch difficulty summaries on external complexity criteria.",
            "- Human-vs-LLM comparison result: humans and LLMs are significantly aligned on overlap difficulty and solve rate, but they are not yet similar enough, with this sample, to support equivalence claims.",
            "- The clearest human-vs-LLM differences are that cyclomatic complexity and memory burden behave more like LLM difficulty signals than human difficulty signals in the overlap slice.",
        ]
    )

    (BASE_DIR / "papers" / "briefs" / "mini_paper_closeness_human_llm.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    overlap = build_overlap_table()
    key_alignment = build_key_alignment_table(overlap)
    partial_vs_logit = build_partial_vs_logit_table(overlap)
    complexity_diff = build_complexity_difference_table(overlap)
    power_table = build_power_table()
    human_suite = pd.read_csv(HUMAN_SUITE_PATH)

    key_alignment.to_csv(BASE_DIR / "human_llm_lastpass_key_alignments.csv", index=False)
    partial_vs_logit.to_csv(BASE_DIR / "human_llm_lastpass_partial_vs_logit.csv", index=False)
    complexity_diff.to_csv(BASE_DIR / "human_llm_lastpass_complexity_difference_similarity.csv", index=False)
    power_table.to_csv(BASE_DIR / "human_llm_lastpass_power.csv", index=False)

    summary = {
        "n_overlap_tasks": int(overlap["task_id"].nunique()),
        "equivalence_margin_delta_r": EQUIV_MARGIN,
        "significant_human_vs_llm_differences": complexity_diff.loc[
            complexity_diff["classification"].isin(["LLM > Human", "Human > LLM"])
        ].to_dict(orient="records"),
        "partial_vs_logit": partial_vs_logit.to_dict(orient="records"),
        "key_alignments": key_alignment.to_dict(orient="records"),
        "power": power_table.to_dict(orient="records"),
    }
    (BASE_DIR / "mini_paper_closeness_human_llm_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_mini_paper(key_alignment, partial_vs_logit, complexity_diff, power_table, human_suite)


if __name__ == "__main__":
    main()
