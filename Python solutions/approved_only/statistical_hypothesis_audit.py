import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import permutation_test
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests


ROOT_DIR = Path(r"C:\Users\cooki\Desktop\ARC-AGI")
BASE_DIR = ROOT_DIR / "Python solutions" / "approved_only"
HUMAN_TABLE_PATH = ROOT_DIR / "Human data" / "analysis" / "tables" / "public_eval_human_vs_models.csv"

APPROVED_LLM_PATH = BASE_DIR / "approved_llm_complexity_join.csv"
MODEL_META_PATH = BASE_DIR / "llm_model_metadata.csv"
RESPONSE_MATRIX_PATHS = {
    "arc_agi_1_eval": BASE_DIR / "llm_response_matrix_arc_agi_1_eval.csv",
    "arc_agi_2_eval": BASE_DIR / "llm_response_matrix_arc_agi_2_eval.csv",
}


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_corr_ci(x, y, n_boot=8000, seed=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = safe_corr(x[idx], y[idx])
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def corr_statistic(x, y, axis=None):
    return safe_corr(x, y)


def permutation_corr_p(x, y, n_resamples=12000, seed=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    result = permutation_test(
        (x, y),
        statistic=corr_statistic,
        permutation_type="pairings",
        alternative="two-sided",
        n_resamples=n_resamples,
        random_state=seed,
        vectorized=False,
    )
    return float(result.pvalue)


def bootstrap_diff_corr(x, y_a, y_b, n_boot=8000, seed=0):
    x = np.asarray(x, dtype=float)
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y_a) & np.isfinite(y_b)
    x = x[mask]
    y_a = y_a[mask]
    y_b = y_b[mask]
    if len(x) < 6:
        return (np.nan, np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r_a = safe_corr(x[idx], y_a[idx])
        r_b = safe_corr(x[idx], y_b[idx])
        if np.isfinite(r_a) and np.isfinite(r_b):
            vals.append(r_a - r_b)
    if not vals:
        return (np.nan, np.nan, np.nan, np.nan)
    vals = np.asarray(vals, dtype=float)
    est = safe_corr(x, y_a) - safe_corr(x, y_b)
    ci_low, ci_high = np.percentile(vals, [2.5, 97.5])
    p_boot = 2.0 * min(np.mean(vals >= 0.0), np.mean(vals <= 0.0))
    p_boot = max(p_boot, 1.0 / len(vals))
    return float(est), float(ci_low), float(ci_high), float(p_boot)


def classify_model(model_name: str):
    if re.search(r"thinking-none", model_name, flags=re.IGNORECASE):
        return {
            "legacy_label": "Standard",
            "strict_label": "Standard",
            "maximal_label": "Standard",
            "verified_label": "Standard",
            "evidence": "explicit_thinking_none",
            "certainty": "high",
        }
    if re.search(r"thinking-(?!none)", model_name, flags=re.IGNORECASE):
        return {
            "legacy_label": "Thinking",
            "strict_label": "Thinking",
            "maximal_label": "Thinking",
            "verified_label": "Thinking",
            "evidence": "explicit_thinking_tag",
            "certainty": "high",
        }
    if re.search(r"deep-think|reasoning", model_name, flags=re.IGNORECASE):
        return {
            "legacy_label": "Thinking",
            "strict_label": "Thinking",
            "maximal_label": "Thinking",
            "verified_label": "Thinking",
            "evidence": "explicit_reasoning_name",
            "certainty": "high",
        }
    if re.search(r"gemini-3-pro-preview", model_name, flags=re.IGNORECASE):
        return {
            "legacy_label": "Thinking",
            "strict_label": "Standard",
            "maximal_label": "Thinking",
            "verified_label": "Thinking",
            "evidence": "ambiguous_gemini_pro",
            "certainty": "low",
        }
    if re.search(r"gpt-5-2-pro-.*-(high|medium)$", model_name, flags=re.IGNORECASE):
        return {
            "legacy_label": "Standard",
            "strict_label": "Standard",
            "maximal_label": "Thinking",
            "verified_label": "Thinking",
            "evidence": "ambiguous_budget_variant",
            "certainty": "low",
        }
    if re.search(r"gpt-5-pro-", model_name, flags=re.IGNORECASE):
        return {
            "legacy_label": "Thinking",
            "strict_label": "Standard",
            "maximal_label": "Thinking",
            "verified_label": "Thinking",
            "evidence": "ambiguous_pro_variant",
            "certainty": "low",
        }
    if re.search(r"QwQ", model_name, flags=re.IGNORECASE):
        return {
            "legacy_label": "Standard",
            "strict_label": "Standard",
            "maximal_label": "Thinking",
            "verified_label": "Thinking",
            "evidence": "ambiguous_reasoning_brand",
            "certainty": "low",
        }
    if re.search(r"gemini", model_name, flags=re.IGNORECASE):
        return {
            "legacy_label": "Thinking",
            "strict_label": "Standard",
            "maximal_label": "Standard",
            "verified_label": "Standard",
            "evidence": "plain_gemini_name",
            "certainty": "high",
        }
    return {
        "legacy_label": "Standard",
        "strict_label": "Standard",
        "maximal_label": "Standard",
        "verified_label": "Standard",
        "evidence": "plain_standard_name",
        "certainty": "high",
    }


def derive_gap_metrics(response: pd.DataFrame, schema: str) -> pd.DataFrame:
    rows = []
    labels = []
    for model_name in response.index:
        cls = classify_model(model_name)
        labels.append(cls[f"{schema}_label"])
    labels = pd.Series(labels, index=response.index)
    think_mask = labels.eq("Thinking").to_numpy()
    standard_mask = labels.eq("Standard").to_numpy()
    n_thinking = int(think_mask.sum())
    n_standard = int(standard_mask.sum())

    for task_id in response.columns:
        y = response[task_id].to_numpy(dtype=float)
        think_successes = int(np.round(y[think_mask].sum()))
        standard_successes = int(np.round(y[standard_mask].sum()))
        pass_thinking = think_successes / n_thinking if n_thinking > 0 else np.nan
        pass_standard = standard_successes / n_standard if n_standard > 0 else np.nan
        p_thinking = (think_successes + 0.5) / (n_thinking + 1.0) if n_thinking > 0 else np.nan
        p_standard = (standard_successes + 0.5) / (n_standard + 1.0) if n_standard > 0 else np.nan
        rows.append(
            {
                "task_id": task_id,
                "num_models_thinking": n_thinking,
                "num_models_standard": n_standard,
                "thinking_successes": think_successes,
                "standard_successes": standard_successes,
                "pass_rate_thinking": pass_thinking,
                "pass_rate_standard": pass_standard,
                "thinking_advantage": pass_thinking - pass_standard,
                "thinking_logit_advantage": math.log(p_thinking / (1.0 - p_thinking)) - math.log(p_standard / (1.0 - p_standard)),
                "both_zero_successes": int(think_successes == 0 and standard_successes == 0),
                "thinking_zero_successes": int(think_successes == 0),
                "standard_zero_successes": int(standard_successes == 0),
            }
        )
    return pd.DataFrame(rows)


def fit_thinking_glm(df: pd.DataFrame, difficulty_col: str):
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "task_id": row["task_id"],
                "group": "Thinking",
                "successes": row["thinking_successes"],
                "failures": row["num_models_thinking"] - row["thinking_successes"],
                "difficulty": row[difficulty_col],
            }
        )
        rows.append(
            {
                "task_id": row["task_id"],
                "group": "Standard",
                "successes": row["standard_successes"],
                "failures": row["num_models_standard"] - row["standard_successes"],
                "difficulty": row[difficulty_col],
            }
        )
    glm_df = pd.DataFrame(rows)
    glm_df["type_thinking"] = glm_df["group"].eq("Thinking").astype(int)
    glm_df["interaction"] = glm_df["type_thinking"] * glm_df["difficulty"]
    endog = glm_df[["successes", "failures"]]
    exog = sm.add_constant(glm_df[["type_thinking", "difficulty", "interaction"]])
    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    result = model.fit()

    exog_null = sm.add_constant(glm_df[["type_thinking", "difficulty"]])
    null_result = sm.GLM(endog, exog_null, family=sm.families.Binomial()).fit()
    lr_stat = 2.0 * (result.llf - null_result.llf)
    from scipy.stats import chi2

    lr_p = float(chi2.sf(lr_stat, 1))
    return glm_df, result, lr_stat, lr_p


def bootstrap_glm_interaction_ci(df: pd.DataFrame, difficulty_col: str, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    task_rows = df.reset_index(drop=True)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(task_rows), size=len(task_rows))
        sample = task_rows.iloc[idx].reset_index(drop=True)
        try:
            _, result, _, _ = fit_thinking_glm(sample, difficulty_col)
            vals.append(float(result.params["interaction"]))
        except Exception:
            continue
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def add_test(rows, **kwargs):
    rows.append(kwargs)


def main():
    llm = pd.read_csv(APPROVED_LLM_PATH)
    human = pd.read_csv(HUMAN_TABLE_PATH)
    model_meta = pd.read_csv(MODEL_META_PATH)

    audit_rows = []
    for model_name in sorted(model_meta["model_name"].unique()):
        cls = classify_model(model_name)
        datasets = sorted(model_meta.loc[model_meta["model_name"] == model_name, "dataset_key"].unique())
        audit_rows.append({"model_name": model_name, "datasets_present": "|".join(datasets), **cls})
    label_audit = pd.DataFrame(audit_rows)
    label_audit.to_csv(BASE_DIR / "thinking_label_audit.csv", index=False)

    response_tables = {key: pd.read_csv(path, index_col=0) for key, path in RESPONSE_MATRIX_PATHS.items()}
    schema_outputs = []
    schema_counts = []
    for schema in ["legacy", "strict", "maximal"]:
        for dataset_key, response in response_tables.items():
            gap_df = derive_gap_metrics(response, schema=schema)
            gap_df["dataset_key"] = dataset_key
            gap_df["schema"] = schema
            schema_outputs.append(gap_df)
            labels = [classify_model(name)[f"{schema}_label"] for name in response.index]
            schema_counts.append(
                {
                    "schema": schema,
                    "dataset_key": dataset_key,
                    "num_models_thinking": int(sum(label == "Thinking" for label in labels)),
                    "num_models_standard": int(sum(label == "Standard" for label in labels)),
                }
            )
    schema_gap_table = pd.concat(schema_outputs, ignore_index=True)
    schema_gap_table.to_csv(BASE_DIR / "thinking_schema_task_metrics.csv", index=False)
    pd.DataFrame(schema_counts).to_csv(BASE_DIR / "thinking_schema_counts.csv", index=False)

    approved_subset = llm[["dataset_key", "task_id", "logit_difficulty_all", "rasch_difficulty_all_models_pooled"]].copy()
    schema_join = approved_subset.merge(schema_gap_table, on=["dataset_key", "task_id"], how="left")

    key_rows = []
    overlap = pd.read_csv(BASE_DIR / "human_llm_overlap_tasks.csv")
    well_sampled_pairs = human[human["attempts"] >= 8].copy()

    x = overlap["difficulty_weighted"]
    y = overlap["logit_difficulty_all"]
    ci_low, ci_high = bootstrap_corr_ci(x, y, seed=1)
    add_test(
        key_rows,
        family="shared_alignment",
        claim_id="S1",
        claim="Human and LLM task difficulty are positively aligned on the approved ARC-2 eval overlap.",
        sample="17 approved ARC-2 eval overlap tasks",
        predictor="human task difficulty",
        outcome="LLM logit difficulty",
        method="Pearson r with bootstrap CI and permutation p-value",
        null_hypothesis="Task ordering is unrelated between human difficulty and LLM difficulty.",
        estimate=safe_corr(x, y),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=permutation_corr_p(x, y, seed=1),
        n=int(overlap[["difficulty_weighted", "logit_difficulty_all"]].dropna().shape[0]),
        notes="",
    )

    x = overlap["human_solve_rate_weighted"]
    y = overlap["pass_rate_all"]
    ci_low, ci_high = bootstrap_corr_ci(x, y, seed=2)
    add_test(
        key_rows,
        family="shared_alignment",
        claim_id="S2",
        claim="Human solve rate aligns with average-model pass rate on the same overlap tasks.",
        sample="17 approved ARC-2 eval overlap tasks",
        predictor="human solve rate",
        outcome="LLM pass rate (all models)",
        method="Pearson r with bootstrap CI and permutation p-value",
        null_hypothesis="Human solve rate and model pass rate are unrelated across tasks.",
        estimate=safe_corr(x, y),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=permutation_corr_p(x, y, seed=2),
        n=int(overlap[["human_solve_rate_weighted", "pass_rate_all"]].dropna().shape[0]),
        notes="",
    )

    x = llm["latent_difficulty_prev_intersection22"]
    y = llm["rasch_difficulty_all_models_pooled"]
    ci_low, ci_high = bootstrap_corr_ci(x, y, seed=3)
    add_test(
        key_rows,
        family="llm_internal_consistency",
        claim_id="S3",
        claim="The older shared-model latent scale and the new pooled Rasch scale are effectively the same LLM difficulty axis.",
        sample="55 approved eval rows",
        predictor="previous LLM latent difficulty",
        outcome="pooled LLM Rasch difficulty",
        method="Pearson r with bootstrap CI and permutation p-value",
        null_hypothesis="The two LLM difficulty estimates are unrelated.",
        estimate=safe_corr(x, y),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=permutation_corr_p(x, y, seed=3),
        n=int(llm[["latent_difficulty_prev_intersection22", "rasch_difficulty_all_models_pooled"]].dropna().shape[0]),
        notes="",
    )

    x = llm["rasch_difficulty_all_models_pooled"]
    y = llm["logit_difficulty_all"]
    ci_low, ci_high = bootstrap_corr_ci(x, y, seed=4)
    add_test(
        key_rows,
        family="llm_internal_consistency",
        claim_id="S4",
        claim="Pooled Rasch difficulty and simple LLM logit difficulty are almost identical on the approved subset.",
        sample="55 approved eval rows",
        predictor="pooled LLM Rasch difficulty",
        outcome="LLM logit difficulty",
        method="Pearson r with bootstrap CI and permutation p-value",
        null_hypothesis="The two LLM difficulty summaries are unrelated.",
        estimate=safe_corr(x, y),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=permutation_corr_p(x, y, seed=4),
        n=int(llm[["rasch_difficulty_all_models_pooled", "logit_difficulty_all"]].dropna().shape[0]),
        notes="",
    )

    x = overlap["cyclomatic_complexity"]
    y_a = overlap["logit_difficulty_all"]
    y_b = overlap["difficulty_weighted"]
    est, ci_low, ci_high, p_boot = bootstrap_diff_corr(x, y_a, y_b, seed=5)
    add_test(
        key_rows,
        family="difference_claims",
        claim_id="D1",
        claim="Cyclomatic complexity is more strongly associated with LLM difficulty than with human difficulty.",
        sample="17 approved ARC-2 eval overlap tasks",
        predictor="cyclomatic complexity",
        outcome="corr(LLM difficulty) - corr(human difficulty)",
        method="Bootstrap difference of correlations",
        null_hypothesis="Cyclomatic complexity is equally associated with human and LLM difficulty.",
        estimate=est,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_boot,
        n=int(overlap[["cyclomatic_complexity", "logit_difficulty_all", "difficulty_weighted"]].dropna().shape[0]),
        notes=f"Raw correlations: human={safe_corr(x, y_b):.3f}, llm={safe_corr(x, y_a):.3f}",
    )

    x = overlap["mean_duration_seconds_weighted"]
    y_a = overlap["difficulty_weighted"]
    y_b = overlap["logit_difficulty_all"]
    est, ci_low, ci_high, p_boot = bootstrap_diff_corr(x, y_a, y_b, seed=6)
    add_test(
        key_rows,
        family="difference_claims",
        claim_id="D2",
        claim="Human duration is more strongly associated with human difficulty than with LLM difficulty.",
        sample="17 approved ARC-2 eval overlap tasks",
        predictor="mean human duration",
        outcome="corr(human difficulty) - corr(LLM difficulty)",
        method="Bootstrap difference of correlations",
        null_hypothesis="Human duration is equally associated with human and LLM difficulty.",
        estimate=est,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_boot,
        n=int(overlap[["mean_duration_seconds_weighted", "difficulty_weighted", "logit_difficulty_all"]].dropna().shape[0]),
        notes=f"Raw correlations: human={safe_corr(x, y_a):.3f}, llm={safe_corr(x, y_b):.3f}",
    )

    residual_df = overlap[["difficulty_weighted", "logit_difficulty_all", "mean_duration_seconds_weighted", "cyclomatic_complexity"]].dropna().copy()
    human_model = LinearRegression().fit(residual_df[["logit_difficulty_all"]], residual_df["difficulty_weighted"])
    residual_df["human_residual_after_llm"] = residual_df["difficulty_weighted"] - human_model.predict(residual_df[["logit_difficulty_all"]])
    llm_model = LinearRegression().fit(residual_df[["difficulty_weighted"]], residual_df["logit_difficulty_all"])
    residual_df["llm_residual_after_human"] = residual_df["logit_difficulty_all"] - llm_model.predict(residual_df[["difficulty_weighted"]])

    x = residual_df["mean_duration_seconds_weighted"]
    y = residual_df["human_residual_after_llm"]
    ci_low, ci_high = bootstrap_corr_ci(x, y, seed=7)
    add_test(
        key_rows,
        family="difference_claims",
        claim_id="D3",
        claim="Human-specific residual difficulty still tracks human duration after removing shared LLM difficulty.",
        sample="17 approved ARC-2 eval overlap tasks",
        predictor="mean human duration",
        outcome="residual human difficulty after LLM",
        method="Pearson r with bootstrap CI and permutation p-value",
        null_hypothesis="Residual human difficulty is unrelated to human duration.",
        estimate=safe_corr(x, y),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=permutation_corr_p(x, y, seed=7),
        n=int(residual_df[["mean_duration_seconds_weighted", "human_residual_after_llm"]].dropna().shape[0]),
        notes="",
    )

    x = residual_df["cyclomatic_complexity"]
    y = residual_df["llm_residual_after_human"]
    ci_low, ci_high = bootstrap_corr_ci(x, y, seed=8)
    add_test(
        key_rows,
        family="difference_claims",
        claim_id="D4",
        claim="LLM-specific residual difficulty still tracks solver structure after removing shared human difficulty.",
        sample="17 approved ARC-2 eval overlap tasks",
        predictor="cyclomatic complexity",
        outcome="residual LLM difficulty after human",
        method="Pearson r with bootstrap CI and permutation p-value",
        null_hypothesis="Residual LLM difficulty is unrelated to cyclomatic complexity.",
        estimate=safe_corr(x, y),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=permutation_corr_p(x, y, seed=8),
        n=int(residual_df[["cyclomatic_complexity", "llm_residual_after_human"]].dropna().shape[0]),
        notes="",
    )

    x = well_sampled_pairs["mean_duration_seconds"]
    y = well_sampled_pairs["difficulty"]
    ci_low, ci_high = bootstrap_corr_ci(x, y, seed=9)
    add_test(
        key_rows,
        family="human_pair_level",
        claim_id="H1",
        claim="On well-sampled public-eval pairs, human difficulty tracks human time cost.",
        sample="110 public-eval task pairs with >=8 attempts",
        predictor="mean human duration",
        outcome="human pair difficulty",
        method="Pearson r with bootstrap CI and permutation p-value",
        null_hypothesis="Human difficulty is unrelated to human time cost.",
        estimate=safe_corr(x, y),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=permutation_corr_p(x, y, seed=9),
        n=int(well_sampled_pairs[["mean_duration_seconds", "difficulty"]].dropna().shape[0]),
        notes="",
    )

    x = well_sampled_pairs["input_cells"]
    y = well_sampled_pairs["difficulty"]
    ci_low, ci_high = bootstrap_corr_ci(x, y, seed=10)
    add_test(
        key_rows,
        family="human_pair_level",
        claim_id="H2",
        claim="On well-sampled public-eval pairs, raw board size alone is weak for human difficulty.",
        sample="110 public-eval task pairs with >=8 attempts",
        predictor="input cells",
        outcome="human pair difficulty",
        method="Pearson r with bootstrap CI and permutation p-value",
        null_hypothesis="Human difficulty is unrelated to input grid size.",
        estimate=safe_corr(x, y),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=permutation_corr_p(x, y, seed=10),
        n=int(well_sampled_pairs[["input_cells", "difficulty"]].dropna().shape[0]),
        notes="",
    )

    est, ci_low, ci_high, p_boot = bootstrap_diff_corr(
        well_sampled_pairs["difficulty"],
        well_sampled_pairs["mean_duration_seconds"],
        well_sampled_pairs["input_cells"],
        seed=11,
    )
    add_test(
        key_rows,
        family="human_pair_level",
        claim_id="H3",
        claim="Human duration is more strongly associated with human difficulty than raw board size is.",
        sample="110 public-eval task pairs with >=8 attempts",
        predictor="human duration vs input cells",
        outcome="difference in correlations with human difficulty",
        method="Bootstrap difference of correlations",
        null_hypothesis="Human difficulty is equally associated with duration and board size.",
        estimate=est,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_boot,
        n=int(well_sampled_pairs[["difficulty", "mean_duration_seconds", "input_cells"]].dropna().shape[0]),
        notes=f"Raw correlations: duration={safe_corr(well_sampled_pairs['difficulty'], well_sampled_pairs['mean_duration_seconds']):.3f}, input_cells={safe_corr(well_sampled_pairs['difficulty'], well_sampled_pairs['input_cells']):.3f}",
    )

    x = well_sampled_pairs["n_test_pairs"]
    y = well_sampled_pairs["gap_vs_lm_mean"]
    ci_low, ci_high = bootstrap_corr_ci(x, y, seed=12)
    add_test(
        key_rows,
        family="human_pair_level",
        claim_id="H4",
        claim="Human-over-LLM advantage shrinks when public-eval tasks expose more test pairs.",
        sample="110 public-eval task pairs with >=8 attempts",
        predictor="number of test pairs",
        outcome="human solve rate minus average-model pass rate",
        method="Pearson r with bootstrap CI and permutation p-value",
        null_hypothesis="Human-over-LLM gap is unrelated to the number of test pairs.",
        estimate=safe_corr(x, y),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=permutation_corr_p(x, y, seed=12),
        n=int(well_sampled_pairs[["n_test_pairs", "gap_vs_lm_mean"]].dropna().shape[0]),
        notes="",
    )

    multi = human.groupby("task_ID").filter(lambda g: len(g) > 1).copy()
    ols_null = ols("difficulty ~ 1", data=multi).fit()
    ols_task = ols("difficulty ~ C(task_ID)", data=multi).fit()
    anova = anova_lm(ols_null, ols_task)
    add_test(
        key_rows,
        family="human_pair_level",
        claim_id="H5",
        claim="Task identity explains a large share of pair-level human difficulty variation on multi-pair tasks.",
        sample=f"{len(multi)} pair rows across {multi['task_ID'].nunique()} multi-pair public-eval tasks",
        predictor="task fixed effects",
        outcome="human pair difficulty",
        method="Nested OLS ANOVA",
        null_hypothesis="Task identity does not improve fit over an intercept-only model.",
        estimate=float(ols_task.rsquared),
        ci_low=np.nan,
        ci_high=np.nan,
        p_value=float(anova['Pr(>F)'].iloc[1]),
        n=int(len(multi)),
        notes="Descriptive within-task ranges are reported separately.",
    )

    thinking_rows = []
    for offset, schema in enumerate(["legacy", "strict", "maximal"]):
        sub = schema_join[schema_join["schema"] == schema].dropna(subset=["thinking_advantage", "thinking_logit_advantage", "logit_difficulty_all"]).copy()
        raw_ci = bootstrap_corr_ci(sub["thinking_advantage"], sub["logit_difficulty_all"], seed=100 + offset)
        logit_ci = bootstrap_corr_ci(sub["thinking_logit_advantage"], sub["logit_difficulty_all"], seed=200 + offset)
        _, glm_result, _, lr_p = fit_thinking_glm(sub, "logit_difficulty_all")
        boot_low, boot_high = bootstrap_glm_interaction_ci(sub, "logit_difficulty_all", seed=300 + offset)
        thinking_rows.append(
            {
                "schema": schema,
                "n_task_rows": int(len(sub)),
                "num_both_zero_items": int(sub["both_zero_successes"].sum()),
                "num_thinking_models": int(sub["num_models_thinking"].iloc[0]),
                "num_standard_models": int(sub["num_models_standard"].iloc[0]),
                "thinking_advantage_r": safe_corr(sub["thinking_advantage"], sub["logit_difficulty_all"]),
                "thinking_advantage_ci_low": raw_ci[0],
                "thinking_advantage_ci_high": raw_ci[1],
                "thinking_advantage_perm_p": permutation_corr_p(sub["thinking_advantage"], sub["logit_difficulty_all"], seed=100 + offset),
                "thinking_logit_advantage_r": safe_corr(sub["thinking_logit_advantage"], sub["logit_difficulty_all"]),
                "thinking_logit_advantage_ci_low": logit_ci[0],
                "thinking_logit_advantage_ci_high": logit_ci[1],
                "thinking_logit_advantage_perm_p": permutation_corr_p(sub["thinking_logit_advantage"], sub["logit_difficulty_all"], seed=200 + offset),
                "glm_interaction_coef": float(glm_result.params["interaction"]),
                "glm_interaction_wald_p": float(glm_result.pvalues["interaction"]),
                "glm_interaction_boot_ci_low": boot_low,
                "glm_interaction_boot_ci_high": boot_high,
                "glm_interaction_lr_p": lr_p,
            }
        )
    thinking_table = pd.DataFrame(thinking_rows)
    thinking_table.to_csv(BASE_DIR / "thinking_advantage_sensitivity.csv", index=False)

    legacy_row = thinking_table[thinking_table["schema"] == "legacy"].iloc[0]
    add_test(
        key_rows,
        family="thinking_advantage",
        claim_id="T1",
        claim="Thinking advantage declines as approved-item LLM difficulty rises under the legacy label schema.",
        sample=f"{int(legacy_row['n_task_rows'])} approved eval rows",
        predictor="LLM logit difficulty",
        outcome="thinking advantage (thinking pass rate - standard pass rate)",
        method="Pearson r with bootstrap CI and permutation p-value",
        null_hypothesis="Thinking advantage is unrelated to LLM difficulty.",
        estimate=float(legacy_row["thinking_advantage_r"]),
        ci_low=float(legacy_row["thinking_advantage_ci_low"]),
        ci_high=float(legacy_row["thinking_advantage_ci_high"]),
        p_value=float(legacy_row["thinking_advantage_perm_p"]),
        n=int(legacy_row["n_task_rows"]),
        notes=f"Both-zero items under this schema: {int(legacy_row['num_both_zero_items'])}.",
    )
    add_test(
        key_rows,
        family="thinking_advantage",
        claim_id="T2",
        claim="Thinking-vs-standard success probability has a negative difficulty interaction under the legacy label schema.",
        sample=f"{int(legacy_row['n_task_rows'])} approved eval rows",
        predictor="item difficulty x model-group interaction",
        outcome="binomial success probability",
        method="Grouped binomial GLM",
        null_hypothesis="Thinking and standard groups have the same difficulty slope (interaction = 0).",
        estimate=float(legacy_row["glm_interaction_coef"]),
        ci_low=float(legacy_row["glm_interaction_boot_ci_low"]),
        ci_high=float(legacy_row["glm_interaction_boot_ci_high"]),
        p_value=float(legacy_row["glm_interaction_wald_p"]),
        n=int(legacy_row["n_task_rows"] * 2),
        notes=f"Likelihood-ratio p-value: {legacy_row['glm_interaction_lr_p']:.4g}.",
    )

    key_table = pd.DataFrame(key_rows)
    reject, qvals, _, _ = multipletests(key_table["p_value"].fillna(1.0), method="fdr_bh")
    key_table["q_value_bh"] = qvals
    key_table["reject_fdr_0_05"] = reject
    key_table.to_csv(BASE_DIR / "statistical_hypothesis_key_tests.csv", index=False)

    ambiguous = label_audit[label_audit["certainty"] == "low"].copy()
    summary = {
        "thinking_advantage_definition": {
            "thinking_advantage": "pass_rate_thinking - pass_rate_standard",
            "thinking_logit_advantage": "smoothed log-odds(pass_rate_thinking) - smoothed log-odds(pass_rate_standard)",
        },
        "label_audit_counts": {
            "high_certainty_models": int((label_audit["certainty"] == "high").sum()),
            "low_certainty_models": int((label_audit["certainty"] == "low").sum()),
            "ambiguous_models": ambiguous["model_name"].tolist(),
        },
        "thinking_sensitivity": thinking_table.to_dict(orient="records"),
    }
    (BASE_DIR / "statistical_hypothesis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Statistical Hypothesis Audit",
        "",
        "## Null-Hypothesis Framing",
        "",
        "- For correlations, the null is exchangeability: shuffling task or pair identities should not yield a stronger association than observed.",
        "- For difference-of-correlation claims, the null is equal association with the two outcomes on the same sampled rows.",
        "- For `thinking_advantage`, the null is that model group does not interact with item difficulty in a grouped binomial-logit model.",
        "",
        "## Thinking-Advantage Derivation",
        "",
        "- `thinking_advantage = pass_rate_thinking - pass_rate_standard`.",
        "- `thinking_logit_advantage` uses the same group counts on a smoothed log-odds scale.",
        "- Labels are audited in [thinking_label_audit.csv](C:/Users/cooki/Desktop/ARC-AGI/Python%20solutions/approved_only/thinking_label_audit.csv).",
        "",
        "## Ambiguous Models",
        "",
        f"- {', '.join(ambiguous['model_name'].tolist()) if len(ambiguous) else 'None'}",
        "",
        "## Key Tests",
        "",
    ]
    for _, row in key_table.iterrows():
        ci_text = "n/a"
        if np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"]):
            ci_text = f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        lines.append(
            f"- `{row['claim_id']}` {row['claim']} Estimate `{row['estimate']:.3f}`, CI `{ci_text}`, p `{row['p_value']:.4g}`, q `{row['q_value_bh']:.4g}`."
        )
    lines.extend(["", "## Thinking Sensitivity", ""])
    for _, row in thinking_table.iterrows():
        lines.append(
            f"- Schema `{row['schema']}`: raw-gap `r = {row['thinking_advantage_r']:.3f}`, p `{row['thinking_advantage_perm_p']:.4g}`; logit-gap `r = {row['thinking_logit_advantage_r']:.3f}`, p `{row['thinking_logit_advantage_perm_p']:.4g}`; GLM interaction `{row['glm_interaction_coef']:.3f}`, Wald p `{row['glm_interaction_wald_p']:.4g}`, both-zero items `{int(row['num_both_zero_items'])}`."
        )
    (BASE_DIR / "statistical_hypothesis_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
