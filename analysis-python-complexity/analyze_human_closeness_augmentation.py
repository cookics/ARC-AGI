from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent
HUMAN_JOIN_PATH = BASE_DIR / "solution_closeness_human_pair_join.csv"

OUTCOME_LABELS = {
    "solve_rate": "Human solve rate",
    "difficulty": "Human difficulty",
    "log_duration": "log1p human duration",
    "human_ease_pc1": "Human latent ease PC1",
}

CLOSENESS_FEATURES = [
    "exact_any_mean_all",
    "cell_accuracy_padded_mean_all",
    "shape_iou_mean_all",
    "color_iou_mean_all",
    "component_size_iou_mean_all",
    "adjacency_iou_mean_all",
]


def build_human_score(df: pd.DataFrame) -> pd.DataFrame:
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
    if np.corrcoef(pc1, human_axes.loc[valid, "solve_rate"])[0, 1] < 0:
        pc1 *= -1.0
    out.loc[valid, "human_ease_pc1"] = pc1
    return out


def fit_nested_models(df: pd.DataFrame, outcome: str) -> dict[str, float]:
    subset = df[[outcome, "lm_mean", *CLOSENESS_FEATURES]].dropna().reset_index(drop=True)
    y = subset[outcome].to_numpy(dtype=float)
    x0 = sm.add_constant(subset[["lm_mean"]])
    x1 = sm.add_constant(subset[["lm_mean", *CLOSENESS_FEATURES]])

    baseline = sm.OLS(y, x0).fit()
    augmented = sm.OLS(y, x1).fit()
    f_stat, p_value, df_diff = augmented.compare_f_test(baseline)

    loo = LeaveOneOut()
    pred0 = np.zeros(len(subset), dtype=float)
    pred1 = np.zeros(len(subset), dtype=float)
    xb0 = subset[["lm_mean"]].to_numpy(dtype=float)
    xb1 = subset[["lm_mean", *CLOSENESS_FEATURES]].to_numpy(dtype=float)
    for train_idx, test_idx in loo.split(xb0):
        pred0[test_idx] = LinearRegression().fit(xb0[train_idx], y[train_idx]).predict(xb0[test_idx])
        pred1[test_idx] = LinearRegression().fit(xb1[train_idx], y[train_idx]).predict(xb1[test_idx])

    return {
        "outcome": outcome,
        "outcome_label": OUTCOME_LABELS[outcome],
        "n": int(len(subset)),
        "baseline_r2": float(baseline.rsquared),
        "augmented_r2": float(augmented.rsquared),
        "delta_r2": float(augmented.rsquared - baseline.rsquared),
        "baseline_adj_r2": float(baseline.rsquared_adj),
        "augmented_adj_r2": float(augmented.rsquared_adj),
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "df_diff": float(df_diff),
        "loo_baseline_r2": float(r2_score(y, pred0)),
        "loo_augmented_r2": float(r2_score(y, pred1)),
        "loo_delta_r2": float(r2_score(y, pred1) - r2_score(y, pred0)),
        "loo_baseline_corr": float(np.corrcoef(y, pred0)[0, 1]),
        "loo_augmented_corr": float(np.corrcoef(y, pred1)[0, 1]),
    }


def fit_augmented_coefficients(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    subset = df[[outcome, "lm_mean", *CLOSENESS_FEATURES]].dropna().reset_index(drop=True)
    y = subset[outcome].to_numpy(dtype=float)
    x = sm.add_constant(subset[["lm_mean", *CLOSENESS_FEATURES]])
    model = sm.OLS(y, x).fit()
    table = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "std_err": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values,
        }
    )
    table.insert(0, "outcome", outcome)
    return table


def write_report(results: pd.DataFrame) -> None:
    latent = results.loc[results["outcome"] == "human_ease_pc1"].iloc[0]
    duration = results.loc[results["outcome"] == "log_duration"].iloc[0]

    lines = [
        "# Human Closeness Augmentation",
        "",
        "## Setup",
        "",
        "- Baseline model: `outcome ~ lm_mean`.",
        "- Augmented model: `outcome ~ lm_mean + exact_any + padded cell accuracy + shape IoU + color IoU + component-size IoU + adjacency IoU`.",
        "- By construction, the augmented model cannot fit worse in-sample because it nests the baseline.",
        "- Human latent ease PC1 is the first principal component of standardized `solve_rate`, `-difficulty`, and `-log1p(duration)`.",
        "- Leave-one-out (LOO) results are included as a basic out-of-sample check.",
        "",
        "## Headline",
        "",
        f"- The latent human score improves from `R^2 = {latent['baseline_r2']:.3f}` to `R^2 = {latent['augmented_r2']:.3f}` (`delta = {latent['delta_r2']:.3f}`, nested-model `p = {latent['p_value']:.4f}`).",
        f"- On LOO validation, the latent human score also improves slightly from `R^2 = {latent['loo_baseline_r2']:.3f}` to `R^2 = {latent['loo_augmented_r2']:.3f}`.",
        f"- Human time cost shows the strongest gain: `R^2 = {duration['baseline_r2']:.3f}` to `R^2 = {duration['augmented_r2']:.3f}` with `p = {duration['p_value']:.4f}`; LOO `R^2` rises from `{duration['loo_baseline_r2']:.3f}` to `{duration['loo_augmented_r2']:.3f}`.",
        "",
        "## Interpretation",
        "",
        "- Replacing pass/fail with a single soft metric was mostly disappointing.",
        "- Letting closeness enter as an additive block works better: it preserves the original pass/fail signal and only uses the softer metrics when they explain residual human variation.",
        "- The extra signal appears to be strongest for how long humans take, and weaker for raw solve rate or psychometric difficulty on their own.",
        "",
    ]
    (BASE_DIR / "human_closeness_augmentation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(HUMAN_JOIN_PATH)
    df = build_human_score(df)

    outcomes = ["solve_rate", "difficulty", "log_duration", "human_ease_pc1"]
    results = pd.DataFrame([fit_nested_models(df, outcome) for outcome in outcomes])
    coeffs = pd.concat([fit_augmented_coefficients(df, outcome) for outcome in outcomes], ignore_index=True)

    results.to_csv(BASE_DIR / "human_closeness_augmentation_results.csv", index=False)
    coeffs.to_csv(BASE_DIR / "human_closeness_augmentation_coefficients.csv", index=False)

    summary = {
        "n_pairs": int(df["task_pair_id"].nunique()),
        "n_tasks": int(df["task_id"].nunique()),
        "closeness_features": CLOSENESS_FEATURES,
        "headline": results.loc[results["outcome"] == "human_ease_pc1"].iloc[0].to_dict(),
        "duration": results.loc[results["outcome"] == "log_duration"].iloc[0].to_dict(),
    }
    (BASE_DIR / "human_closeness_augmentation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_report(results)


if __name__ == "__main__":
    main()
