from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent
HUMAN_JOIN_PATH = BASE_DIR / "solution_closeness_human_pair_join.csv"
APPROVED_LLM_PATH = BASE_DIR / "approved_llm_complexity_join.csv"
HUMAN_OVERLAP_PATH = BASE_DIR / "human_llm_overlap_tasks.csv"
TASK_CLOSENESS_PATH = BASE_DIR / "solution_closeness_task_metrics_approved.csv"
SOLUTION_CLOSENESS_SCRIPT = BASE_DIR / "analyze_solution_closeness.py"

HUMAN_FEATURES = [
    "lm_mean",
    "exact_any_mean_all",
    "cell_accuracy_padded_mean_all",
    "shape_iou_mean_all",
    "color_iou_mean_all",
    "component_size_iou_mean_all",
    "adjacency_iou_mean_all",
]

LLM_DIFFICULTY_FEATURES = [
    "logit_difficulty_all",
    "exact_any_difficulty_logit",
    "cell_accuracy_padded_difficulty_logit",
    "shape_iou_difficulty_logit",
    "color_iou_difficulty_logit",
    "component_size_iou_difficulty_logit",
    "adjacency_iou_difficulty_logit",
]

RIDGE_ALPHAS = np.logspace(-3, 3, 30)


def safe_corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return np.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def bootstrap_delta_corr(
    y: pd.Series | np.ndarray,
    candidate: pd.Series | np.ndarray,
    baseline: pd.Series | np.ndarray,
    n_boot: int = 6000,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    y_arr = np.asarray(y, dtype=float)
    c_arr = np.asarray(candidate, dtype=float)
    b_arr = np.asarray(baseline, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(c_arr) & np.isfinite(b_arr)
    y_arr = y_arr[mask]
    c_arr = c_arr[mask]
    b_arr = b_arr[mask]
    observed = safe_corr(y_arr, c_arr) - safe_corr(y_arr, b_arr)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_arr))
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot_idx = rng.choice(idx, size=len(idx), replace=True)
        samples[i] = safe_corr(y_arr[boot_idx], c_arr[boot_idx]) - safe_corr(y_arr[boot_idx], b_arr[boot_idx])
    ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
    p_value = float(2 * min((samples <= 0).mean(), (samples >= 0).mean()))
    return float(observed), float(ci_low), float(ci_high), p_value


def build_human_latent_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_duration"] = np.log1p(out["mean_duration_seconds"])

    human_axes = pd.DataFrame(
        {
            "solve_rate": out["solve_rate"],
            "ease_from_difficulty": -out["difficulty"],
            "ease_from_duration": -out["log_duration"],
        }
    )
    valid = human_axes.dropna().index
    z = StandardScaler().fit_transform(human_axes.loc[valid])
    pc1 = PCA(n_components=1).fit_transform(z).ravel()
    if safe_corr(pc1, human_axes.loc[valid, "solve_rate"]) < 0:
        pc1 *= -1.0
    out.loc[valid, "human_ease_pc1"] = pc1
    return out


def fit_linear_nested(
    subset: pd.DataFrame,
    outcome: str,
    family: str,
    weight_col: str | None = None,
) -> dict[str, float | str]:
    y = subset[outcome].to_numpy(dtype=float)
    baseline_exog = sm.add_constant(subset[["lm_mean"]])
    augmented_exog = sm.add_constant(subset[HUMAN_FEATURES])
    weights = subset[weight_col].to_numpy(dtype=float) if weight_col else None

    if family == "OLS":
        baseline_model = sm.OLS(y, baseline_exog).fit()
        augmented_model = sm.OLS(y, augmented_exog).fit()
    elif family == "WLS":
        baseline_model = sm.WLS(y, baseline_exog, weights=weights).fit()
        augmented_model = sm.WLS(y, augmented_exog, weights=weights).fit()
    else:
        raise ValueError(f"Unknown family: {family}")

    f_stat, p_value, df_diff = augmented_model.compare_f_test(baseline_model)

    loo = LeaveOneOut()
    x0 = subset[["lm_mean"]].to_numpy(dtype=float)
    x1 = subset[HUMAN_FEATURES].to_numpy(dtype=float)
    pred0 = np.zeros(len(subset), dtype=float)
    pred1 = np.zeros(len(subset), dtype=float)
    pred0_ridge = np.zeros(len(subset), dtype=float)
    pred1_ridge = np.zeros(len(subset), dtype=float)
    for train_idx, test_idx in loo.split(x0):
        if weight_col:
            fit_weights = weights[train_idx]
            pred0[test_idx] = LinearRegression().fit(x0[train_idx], y[train_idx], sample_weight=fit_weights).predict(
                x0[test_idx]
            )
            pred1[test_idx] = LinearRegression().fit(x1[train_idx], y[train_idx], sample_weight=fit_weights).predict(
                x1[test_idx]
            )
            pred0_ridge[test_idx] = RidgeCV(alphas=RIDGE_ALPHAS).fit(
                x0[train_idx], y[train_idx], sample_weight=fit_weights
            ).predict(x0[test_idx])
            pred1_ridge[test_idx] = RidgeCV(alphas=RIDGE_ALPHAS).fit(
                x1[train_idx], y[train_idx], sample_weight=fit_weights
            ).predict(x1[test_idx])
        else:
            pred0[test_idx] = LinearRegression().fit(x0[train_idx], y[train_idx]).predict(x0[test_idx])
            pred1[test_idx] = LinearRegression().fit(x1[train_idx], y[train_idx]).predict(x1[test_idx])
            pred0_ridge[test_idx] = RidgeCV(alphas=RIDGE_ALPHAS).fit(x0[train_idx], y[train_idx]).predict(x0[test_idx])
            pred1_ridge[test_idx] = RidgeCV(alphas=RIDGE_ALPHAS).fit(x1[train_idx], y[train_idx]).predict(x1[test_idx])

    return {
        "domain": "human",
        "model_family": family,
        "outcome": outcome,
        "n": int(len(subset)),
        "baseline_metric": "R2",
        "baseline_fit": float(baseline_model.rsquared),
        "augmented_fit": float(augmented_model.rsquared),
        "delta_fit": float(augmented_model.rsquared - baseline_model.rsquared),
        "p_value": float(p_value),
        "test_stat": float(f_stat),
        "df_diff": float(df_diff),
        "loo_baseline_fit": float(r2_score(y, pred0)),
        "loo_augmented_fit": float(r2_score(y, pred1)),
        "loo_delta_fit": float(r2_score(y, pred1) - r2_score(y, pred0)),
        "ridge_loo_baseline_fit": float(r2_score(y, pred0_ridge)),
        "ridge_loo_augmented_fit": float(r2_score(y, pred1_ridge)),
        "ridge_loo_delta_fit": float(r2_score(y, pred1_ridge) - r2_score(y, pred0_ridge)),
    }


def fit_grouped_binomial(subset: pd.DataFrame) -> dict[str, float | str]:
    endog = np.column_stack(
        [
            subset["solve_count"].to_numpy(dtype=float),
            (subset["attempts"] - subset["solve_count"]).to_numpy(dtype=float),
        ]
    )
    baseline_exog = sm.add_constant(subset[["lm_mean"]])
    augmented_exog = sm.add_constant(subset[HUMAN_FEATURES])
    baseline_model = sm.GLM(endog, baseline_exog, family=sm.families.Binomial()).fit()
    augmented_model = sm.GLM(endog, augmented_exog, family=sm.families.Binomial()).fit()
    lr_stat = float(2.0 * (augmented_model.llf - baseline_model.llf))
    df_diff = float(augmented_model.df_model - baseline_model.df_model)
    p_value = float(1.0 - chi2.cdf(lr_stat, df_diff))
    baseline_fit = float(1.0 - baseline_model.deviance / baseline_model.null_deviance)
    augmented_fit = float(1.0 - augmented_model.deviance / augmented_model.null_deviance)
    return {
        "domain": "human",
        "model_family": "GroupedBinomial",
        "outcome": "solve_rate",
        "n": int(len(subset)),
        "baseline_metric": "DeviancePseudoR2",
        "baseline_fit": baseline_fit,
        "augmented_fit": augmented_fit,
        "delta_fit": float(augmented_fit - baseline_fit),
        "p_value": p_value,
        "test_stat": lr_stat,
        "df_diff": df_diff,
        "loo_baseline_fit": np.nan,
        "loo_augmented_fit": np.nan,
        "loo_delta_fit": np.nan,
        "ridge_loo_baseline_fit": np.nan,
        "ridge_loo_augmented_fit": np.nan,
        "ridge_loo_delta_fit": np.nan,
    }


def build_human_suite() -> pd.DataFrame:
    df = build_human_latent_scores(pd.read_csv(HUMAN_JOIN_PATH))
    results: list[dict[str, float | str]] = []

    for family, weight_col in [("OLS", None), ("WLS", "attempts")]:
        for outcome in ["solve_rate", "difficulty", "log_duration", "human_ease_pc1"]:
            subset = df[[outcome, "attempts", *HUMAN_FEATURES]].dropna().reset_index(drop=True)
            results.append(fit_linear_nested(subset, outcome, family=family, weight_col=weight_col))

    solve_subset = df[["solve_count", "attempts", *HUMAN_FEATURES]].dropna().reset_index(drop=True)
    results.append(fit_grouped_binomial(solve_subset))
    return pd.DataFrame(results)


def load_solution_closeness_module():
    spec = importlib.util.spec_from_file_location("solution_closeness", SOLUTION_CLOSENESS_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_llm_task_model_table() -> pd.DataFrame:
    module = load_solution_closeness_module()
    approved_df = pd.read_csv(APPROVED_LLM_PATH)
    task_ids_by_dataset = {
        dataset_key: sorted(group["task_id"].unique()) for dataset_key, group in approved_df.groupby("dataset_key")
    }
    pair_frames = [module.collect_pair_rows(dataset_key, task_ids) for dataset_key, task_ids in task_ids_by_dataset.items()]
    pair_df = pd.concat(pair_frames, ignore_index=True)
    _, task_model_df = module.build_task_metric_table(pair_df)
    return task_model_df


def fit_partial_credit_models(task_model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    non_exact = task_model_df.loc[task_model_df["exact_any"] < 1.0, "soft_composite"]
    q1, q2, q3 = non_exact.quantile([0.25, 0.5, 0.75]).tolist()

    def assign_category(row: pd.Series) -> int:
        if row["exact_any"] >= 1.0:
            return 4
        value = float(row["soft_composite"])
        return int(value >= q1) + int(value >= q2) + int(value >= q3)

    task_model_df = task_model_df.copy()
    task_model_df["partial_credit_category"] = task_model_df.apply(assign_category, axis=1)
    task_model_df["item_id"] = task_model_df["dataset_key"] + "__" + task_model_df["task_id"]
    matrix = task_model_df.pivot(index="model_name", columns="item_id", values="partial_credit_category")
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)

    r_code = textwrap.dedent(
        """
        args <- commandArgs(trailingOnly = TRUE)
        suppressPackageStartupMessages(library(mirt))
        df <- read.csv(args[1], row.names = 1, check.names = FALSE)
        df[] <- lapply(df, function(x) as.numeric(x))
        var_cols <- sapply(df, function(x) length(unique(na.omit(x))) > 1)
        df_var <- df[, var_cols, drop = FALSE]

        mod_gpcm <- mirt(df_var, 1, itemtype = 'gpcm', verbose = FALSE, technical = list(NCYCLES = 1500))
        coef_gpcm <- coef(mod_gpcm, IRTpars = TRUE, simplify = TRUE)$items
        out_gpcm <- data.frame(
            item_id = rownames(coef_gpcm),
            gpcm_a = coef_gpcm[, 'a'],
            gpcm_location = rowMeans(coef_gpcm[, colnames(coef_gpcm) != 'a', drop = FALSE], na.rm = TRUE),
            row.names = NULL
        )
        write.csv(out_gpcm, args[2], row.names = FALSE)

        mod_graded <- mirt(df_var, 1, itemtype = 'graded', verbose = FALSE, technical = list(NCYCLES = 1500))
        coef_graded <- coef(mod_graded, IRTpars = TRUE, simplify = TRUE)$items
        out_graded <- data.frame(
            item_id = rownames(coef_graded),
            graded_a = coef_graded[, 'a'],
            graded_location = rowMeans(coef_graded[, colnames(coef_graded) != 'a', drop = FALSE], na.rm = TRUE),
            row.names = NULL
        )
        write.csv(out_graded, args[3], row.names = FALSE)
        """
    )

    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as handle:
        handle.write(r_code)
        script_path = Path(handle.name)

    matrix_path = BASE_DIR / "llm_partial_credit_response_matrix.csv"
    gpcm_path = BASE_DIR / "llm_partial_credit_gpcm_items.csv"
    graded_path = BASE_DIR / "llm_partial_credit_graded_items.csv"
    matrix.to_csv(matrix_path)
    try:
        subprocess.run(
            ["Rscript", str(script_path), str(matrix_path), str(gpcm_path), str(graded_path)],
            check=True,
            cwd=str(ROOT_DIR),
        )
    finally:
        script_path.unlink(missing_ok=True)

    gpcm = pd.read_csv(gpcm_path)
    graded = pd.read_csv(graded_path)
    for frame in (gpcm, graded):
        frame[["dataset_key", "task_id"]] = frame["item_id"].str.split("__", n=1, expand=True)
    partial_credit = gpcm.merge(
        graded[["item_id", "graded_a", "graded_location", "dataset_key", "task_id"]],
        on=["item_id", "dataset_key", "task_id"],
        how="outer",
    )
    return matrix.reset_index(), partial_credit


def build_llm_complexity_criterion(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    criteria = pd.DataFrame(
        {
            "complexity_pc1_score": out["complexity_pc1_score"],
            "log_cyclomatic": np.log1p(out["cyclomatic_complexity"]),
            "log_runtime": out["log1p_elapsed_ms_total"],
        }
    )
    valid = criteria.dropna().index
    z = StandardScaler().fit_transform(criteria.loc[valid])
    pc1 = PCA(n_components=1).fit_transform(z).ravel()
    if safe_corr(pc1, criteria.loc[valid, "complexity_pc1_score"]) < 0:
        pc1 *= -1.0
    out.loc[valid, "llm_complexity_pc1"] = pc1
    return out


def build_human_overlap_latent(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_duration"] = np.log1p(out["mean_duration_seconds_weighted"])
    criteria = pd.DataFrame(
        {
            "solve_rate": out["human_solve_rate_weighted"],
            "ease_from_difficulty": -out["difficulty_weighted"],
            "ease_from_duration": -out["log_duration"],
        }
    )
    valid = criteria.dropna().index
    z = StandardScaler().fit_transform(criteria.loc[valid])
    pc1 = PCA(n_components=1).fit_transform(z).ravel()
    if safe_corr(pc1, criteria.loc[valid, "solve_rate"]) < 0:
        pc1 *= -1.0
    out.loc[valid, "human_overlap_ease_pc1"] = pc1
    return out


def build_llm_regression_suite(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    baseline_options = ["logit_difficulty_all", "rasch_difficulty_all_models_pooled"]
    outcomes = ["llm_complexity_pc1", "complexity_pc1_score", "log1p_elapsed_ms_total"]

    for baseline in baseline_options:
        feature_cols = [baseline, *LLM_DIFFICULTY_FEATURES[1:]]
        for outcome in outcomes:
            subset = df[[outcome, *feature_cols]].dropna().reset_index(drop=True)
            y = subset[outcome].to_numpy(dtype=float)
            x0 = sm.add_constant(subset[[baseline]])
            x1 = sm.add_constant(subset[feature_cols])
            baseline_model = sm.OLS(y, x0).fit()
            augmented_model = sm.OLS(y, x1).fit()
            f_stat, p_value, df_diff = augmented_model.compare_f_test(baseline_model)

            loo = LeaveOneOut()
            xb = subset[[baseline]].to_numpy(dtype=float)
            xa = subset[feature_cols].to_numpy(dtype=float)
            pred0 = np.zeros(len(subset), dtype=float)
            pred1 = np.zeros(len(subset), dtype=float)
            pred0_ridge = np.zeros(len(subset), dtype=float)
            pred1_ridge = np.zeros(len(subset), dtype=float)
            for train_idx, test_idx in loo.split(xb):
                pred0[test_idx] = LinearRegression().fit(xb[train_idx], y[train_idx]).predict(xb[test_idx])
                pred1[test_idx] = LinearRegression().fit(xa[train_idx], y[train_idx]).predict(xa[test_idx])
                pred0_ridge[test_idx] = RidgeCV(alphas=RIDGE_ALPHAS).fit(xb[train_idx], y[train_idx]).predict(
                    xb[test_idx]
                )
                pred1_ridge[test_idx] = RidgeCV(alphas=RIDGE_ALPHAS).fit(xa[train_idx], y[train_idx]).predict(
                    xa[test_idx]
                )

            rows.append(
                {
                    "domain": "llm",
                    "model_family": f"OLS_nested_from_{baseline}",
                    "outcome": outcome,
                    "n": int(len(subset)),
                    "baseline_metric": "R2",
                    "baseline_fit": float(baseline_model.rsquared),
                    "augmented_fit": float(augmented_model.rsquared),
                    "delta_fit": float(augmented_model.rsquared - baseline_model.rsquared),
                    "p_value": float(p_value),
                    "test_stat": float(f_stat),
                    "df_diff": float(df_diff),
                    "loo_baseline_fit": float(r2_score(y, pred0)),
                    "loo_augmented_fit": float(r2_score(y, pred1)),
                    "loo_delta_fit": float(r2_score(y, pred1) - r2_score(y, pred0)),
                    "ridge_loo_baseline_fit": float(r2_score(y, pred0_ridge)),
                    "ridge_loo_augmented_fit": float(r2_score(y, pred1_ridge)),
                    "ridge_loo_delta_fit": float(r2_score(y, pred1_ridge) - r2_score(y, pred0_ridge)),
                }
            )
    return pd.DataFrame(rows)


def build_llm_predictor_suite(df: pd.DataFrame) -> pd.DataFrame:
    predictor_specs = [
        ("logit_difficulty_all", "Binary logit difficulty", "full"),
        ("rasch_difficulty_all_models_pooled", "Binary Rasch difficulty", "full"),
        ("two_pl_difficulty_all_models", "Binary 2PL difficulty", "full"),
        ("exact_any_difficulty_logit", "Exact-any logit difficulty", "full"),
        ("soft_composite_difficulty_logit", "Soft-composite logit difficulty", "full"),
        ("gpcm_location", "Partial-credit GPCM location", "partial"),
        ("graded_location", "Partial-credit graded location", "partial"),
    ]

    target_specs = [
        ("llm_complexity_pc1", "LLM complexity PC1", "hardness"),
        ("complexity_pc1_score", "Code complexity PC1", "hardness"),
        ("cyclomatic_complexity", "Cyclomatic complexity", "hardness"),
        ("log1p_elapsed_ms_total", "log1p runtime", "hardness"),
        ("human_overlap_ease_pc1", "Human overlap ease PC1", "ease"),
        ("human_solve_rate_weighted", "Human overlap solve rate", "ease"),
        ("difficulty_weighted", "Human overlap difficulty", "hardness"),
        ("mean_duration_seconds_weighted", "Human overlap duration", "hardness"),
    ]

    rows: list[dict[str, float | str]] = []
    for target, target_label, orientation in target_specs:
        for predictor, predictor_label, predictor_kind in predictor_specs:
            selection_cols = list(dict.fromkeys([target, predictor, "logit_difficulty_all"]))
            subset = df[selection_cols].dropna().reset_index(drop=True)
            if len(subset) < 10:
                continue
            signed_target = subset[target] if orientation == "hardness" else -subset[target]
            corr = safe_corr(signed_target, subset[predictor])
            fit_r2 = corr**2 if np.isfinite(corr) else np.nan
            row = {
                "target": target,
                "target_label": target_label,
                "orientation": orientation,
                "predictor": predictor,
                "predictor_label": predictor_label,
                "predictor_kind": predictor_kind,
                "n": int(len(subset)),
                "aligned_corr": corr,
                "fit_r2": fit_r2,
                "delta_vs_logit_corr": np.nan,
                "delta_ci_low": np.nan,
                "delta_ci_high": np.nan,
                "p_value_vs_logit": np.nan,
            }
            if predictor != "logit_difficulty_all":
                delta, ci_low, ci_high, p_value = bootstrap_delta_corr(
                    signed_target,
                    subset[predictor],
                    subset["logit_difficulty_all"],
                    seed=17,
                )
                row["delta_vs_logit_corr"] = delta
                row["delta_ci_low"] = ci_low
                row["delta_ci_high"] = ci_high
                row["p_value_vs_logit"] = p_value
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(human_results: pd.DataFrame, llm_regressions: pd.DataFrame, llm_predictors: pd.DataFrame) -> None:
    best_human_binomial = human_results.loc[
        (human_results["model_family"] == "GroupedBinomial") & (human_results["outcome"] == "solve_rate")
    ].iloc[0]
    best_human_latent = human_results.loc[
        (human_results["model_family"] == "WLS") & (human_results["outcome"] == "human_ease_pc1")
    ].iloc[0]
    best_human_duration = human_results.loc[
        (human_results["model_family"] == "WLS") & (human_results["outcome"] == "log_duration")
    ].iloc[0]

    full_targets = llm_predictors.loc[
        llm_predictors["target"].isin(
            ["llm_complexity_pc1", "complexity_pc1_score", "cyclomatic_complexity", "log1p_elapsed_ms_total"]
        )
    ].copy()
    best_full = full_targets.sort_values("fit_r2", ascending=False).iloc[0]
    best_partial_full = full_targets.loc[full_targets["predictor_kind"] == "partial"].sort_values(
        "fit_r2", ascending=False
    )
    best_overlap_partial = llm_predictors.loc[
        llm_predictors["predictor_kind"] == "partial"
    ].sort_values("aligned_corr", ascending=False)

    report = [
        "# Closeness Model Suite",
        "",
        "## What Counts As Straightforward",
        "",
        "- Human pair outcomes are sparse and heteroskedastic, so the straightforward models are grouped-binomial for solve counts and weighted regression for continuous item summaries.",
        "- LLM response data support true latent-difficulty fits, so the straightforward sensitivity check is a partial-credit IRT model on discretized closeness scores, not just another correlation on averaged soft metrics.",
        "",
        "## Human Side",
        "",
        f"- Grouped-binomial solve model: pseudo-`R^2` rises from `{best_human_binomial['baseline_fit']:.3f}` to `{best_human_binomial['augmented_fit']:.3f}` (`delta = {best_human_binomial['delta_fit']:.3f}`, `p = {best_human_binomial['p_value']:.4f}`).",
        f"- Weighted latent human-ease model: `R^2` rises from `{best_human_latent['baseline_fit']:.3f}` to `{best_human_latent['augmented_fit']:.3f}` (`delta = {best_human_latent['delta_fit']:.3f}`, `p = {best_human_latent['p_value']:.4f}`).",
        f"- Weighted duration model: `R^2` rises from `{best_human_duration['baseline_fit']:.3f}` to `{best_human_duration['augmented_fit']:.3f}` (`delta = {best_human_duration['delta_fit']:.3f}`, `p = {best_human_duration['p_value']:.4f}`).",
        "",
        "## LLM Side",
        "",
        f"- On the full approved task set, the best external-complexity fit is still `{best_full['predictor_label']}` for `{best_full['target_label']}` with aligned `r = {best_full['aligned_corr']:.3f}`.",
        f"- The best partial-credit model on the full task set is `{best_partial_full.iloc[0]['predictor_label']}` for `{best_partial_full.iloc[0]['target_label']}` with aligned `r = {best_partial_full.iloc[0]['aligned_corr']:.3f}`.",
        f"- Partial-credit regression add-ons do not generalize well on the full 55-56 task set: the nested LLM regressions show positive in-sample deltas but little or negative leave-one-out gain.",
        "",
        "## Read",
        "",
        "- Human conclusion: yes, there is a clean and more psychometrically sensible way to do this, and it helps.",
        "- LLM conclusion: the proper partial-credit latent models are worth reporting as a sensitivity analysis, but they do not beat the simpler binary logit/Rasch difficulty on the main full-task complexity criteria.",
        "- Exploratory exception: on the tiny 17-task human-overlap subset, partial-credit latent difficulty can look better, but those gains are not stable enough to treat as a main result.",
        "",
    ]

    if not best_overlap_partial.empty:
        top_overlap = best_overlap_partial.iloc[0]
        report.insert(
            -2,
            f"- Best overlap-only partial-credit result: `{top_overlap['predictor_label']}` reaches aligned `r = {top_overlap['aligned_corr']:.3f}` on `{top_overlap['target_label']}` over `n = {int(top_overlap['n'])}` tasks.",
        )

    (BASE_DIR / "closeness_model_suite_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    human_results = build_human_suite()
    human_results.to_csv(BASE_DIR / "closeness_model_suite_human.csv", index=False)

    llm_task_model_df = build_llm_task_model_table()
    partial_matrix, partial_credit_items = fit_partial_credit_models(llm_task_model_df)
    partial_matrix.to_csv(BASE_DIR / "llm_partial_credit_response_matrix_long.csv", index=False)
    partial_credit_items.to_csv(BASE_DIR / "llm_partial_credit_item_models.csv", index=False)

    llm = pd.read_csv(APPROVED_LLM_PATH).merge(pd.read_csv(TASK_CLOSENESS_PATH), on=["dataset_key", "task_id"], how="left")
    llm = llm.merge(partial_credit_items, on=["dataset_key", "task_id"], how="left")
    llm = build_llm_complexity_criterion(llm)

    overlap = pd.read_csv(HUMAN_OVERLAP_PATH)[
        ["dataset_key", "task_id", "human_solve_rate_weighted", "difficulty_weighted", "mean_duration_seconds_weighted"]
    ]
    llm = llm.merge(overlap, on=["dataset_key", "task_id"], how="left")
    llm = build_human_overlap_latent(llm)

    llm_regressions = build_llm_regression_suite(llm)
    llm_predictors = build_llm_predictor_suite(llm)
    llm_regressions.to_csv(BASE_DIR / "closeness_model_suite_llm_regressions.csv", index=False)
    llm_predictors.to_csv(BASE_DIR / "closeness_model_suite_llm_predictors.csv", index=False)

    summary = {
        "human_rows": int(len(human_results)),
        "llm_regression_rows": int(len(llm_regressions)),
        "llm_predictor_rows": int(len(llm_predictors)),
        "human_best_weighted_latent": human_results.loc[
            (human_results["model_family"] == "WLS") & (human_results["outcome"] == "human_ease_pc1")
        ]
        .iloc[0]
        .to_dict(),
        "human_best_binomial": human_results.loc[
            (human_results["model_family"] == "GroupedBinomial") & (human_results["outcome"] == "solve_rate")
        ]
        .iloc[0]
        .to_dict(),
        "llm_best_predictor_full": llm_predictors.loc[
            llm_predictors["target"].isin(
                ["llm_complexity_pc1", "complexity_pc1_score", "cyclomatic_complexity", "log1p_elapsed_ms_total"]
            )
        ]
        .sort_values("fit_r2", ascending=False)
        .iloc[0]
        .to_dict(),
    }
    (BASE_DIR / "closeness_model_suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(human_results, llm_regressions, llm_predictors)


if __name__ == "__main__":
    main()
