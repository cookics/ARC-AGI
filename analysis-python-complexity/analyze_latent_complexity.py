import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MERGED_PATH = BASE_DIR / "complexity_with_latent_scale.csv"

PCA_METRICS = [
    "nonblank_lines",
    "token_count",
    "ast_node_count",
    "function_count",
    "call_count_static",
    "branch_node_count",
    "cyclomatic_complexity",
    "max_nesting_depth",
    "gzip_bytes",
    "halstead_volume",
    "input_cells_total",
    "output_cells_total",
    "elapsed_ms_per_test",
    "opcode_count_dynamic",
    "branch_opcode_count_dynamic",
    "python_call_count_dynamic",
    "peak_memory_bytes",
    "opcode_per_input_cell",
    "elapsed_ms_per_input_cell",
]


def safe_corr(x, y):
    if len(x) < 3:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rank_array(values):
    series = pd.Series(values)
    return series.rank(method="average").to_numpy(dtype=float)


def spearman_corr(x, y):
    return safe_corr(rank_array(x), rank_array(y))


def loo_predictions(estimator, x, y):
    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in loo.split(x):
        model = clone(estimator)
        model.fit(x[train_idx], y[train_idx])
        preds[test_idx[0]] = float(model.predict(x[test_idx])[0])
    return preds


def model_summary(name, estimator, x, y):
    estimator.fit(x, y)
    train_pred = estimator.predict(x)
    loo_pred = loo_predictions(estimator, x, y)
    return {
        "model": name,
        "train_pearson_r": safe_corr(train_pred, y),
        "train_spearman_rho": spearman_corr(train_pred, y),
        "train_r2": float(r2_score(y, train_pred)),
        "loo_pearson_r": safe_corr(loo_pred, y),
        "loo_spearman_rho": spearman_corr(loo_pred, y),
        "loo_r2": float(r2_score(y, loo_pred)),
    }, train_pred, loo_pred


def single_metric_summaries(df, metrics, target_col):
    rows = []
    for metric in metrics:
        pearson_r = safe_corr(df[metric], df[target_col])
        spearman_rho = spearman_corr(df[metric], df[target_col])
        rows.append(
            {
                "metric": metric,
                "pearson_r": pearson_r,
                "spearman_rho": spearman_rho,
                "abs_pearson_r": abs(pearson_r) if pearson_r is not None else None,
            }
        )
    out = pd.DataFrame(rows).sort_values("abs_pearson_r", ascending=False)
    return out


def pca_regression_summaries(x, y, max_components):
    summaries = []
    for n_components in range(1, max_components + 1):
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("reg", LinearRegression()),
            ]
        )
        summary, _, _ = model_summary(f"pcr_{n_components}pc", estimator, x, y)
        summary["n_components"] = n_components
        summaries.append(summary)
    return summaries


def pls_summaries(x, y, max_components):
    summaries = []
    for n_components in range(1, max_components + 1):
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pls", PLSRegression(n_components=n_components)),
            ]
        )
        summary, _, _ = model_summary(f"pls_{n_components}comp", estimator, x, y)
        summary["n_components"] = n_components
        summaries.append(summary)
    return summaries


def top_loadings(loadings_df, component, top_n=8):
    series = loadings_df[component].sort_values(key=lambda s: s.abs(), ascending=False)
    out = []
    for metric, value in series.head(top_n).items():
        out.append({"metric": metric, "loading": float(value)})
    return out


def main():
    df = pd.read_csv(MERGED_PATH)

    for metric in PCA_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["latent_difficulty"] = pd.to_numeric(df["latent_difficulty"], errors="coerce")

    missing = df[PCA_METRICS + ["latent_difficulty"]].isna().any(axis=1)
    df = df.loc[~missing].copy()

    # Heavy-tailed metrics are easier to compare on a log scale.
    transformed = df[PCA_METRICS].copy()
    for metric in PCA_METRICS:
        transformed[metric] = np.log1p(transformed[metric].clip(lower=0))

    x = transformed.to_numpy(dtype=float)
    y = df["latent_difficulty"].to_numpy(dtype=float)

    pca = PCA()
    x_scaled = StandardScaler().fit_transform(x)
    pcs = pca.fit_transform(x_scaled)

    component_names = [f"PC{i + 1}" for i in range(pca.components_.shape[0])]
    scores_df = pd.DataFrame(pcs, columns=component_names, index=df.index)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=PCA_METRICS,
        columns=component_names,
    )

    pc_corr_rows = []
    for component in component_names:
        pc_corr_rows.append(
            {
                "component": component,
                "explained_variance_ratio": float(
                    pca.explained_variance_ratio_[int(component[2:]) - 1]
                ),
                "pearson_r_with_latent_difficulty": safe_corr(scores_df[component], y),
                "spearman_rho_with_latent_difficulty": spearman_corr(
                    scores_df[component], y
                ),
            }
        )
    pc_corr_df = pd.DataFrame(pc_corr_rows)

    max_components = min(8, x.shape[1], x.shape[0] - 2)
    pcr_df = pd.DataFrame(pca_regression_summaries(x, y, max_components))
    pls_df = pd.DataFrame(pls_summaries(x, y, max_components))

    single_metric_df = single_metric_summaries(df, PCA_METRICS, "latent_difficulty")

    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "ridge",
                RidgeCV(alphas=np.logspace(-3, 3, 49)),
            ),
        ]
    )
    ridge_summary, ridge_train_pred, ridge_loo_pred = model_summary("ridge", ridge, x, y)
    ridge.fit(x, y)
    ridge_alpha = float(ridge.named_steps["ridge"].alpha_)

    pc1 = pcs[:, 0]
    pc1_summary = {
        "model": "pc1_only_unsupervised",
        "train_pearson_r": safe_corr(pc1, y),
        "train_spearman_rho": spearman_corr(pc1, y),
    }

    best_pcr = pcr_df.sort_values("loo_pearson_r", ascending=False).iloc[0].to_dict()
    best_pls = pls_df.sort_values("loo_pearson_r", ascending=False).iloc[0].to_dict()
    best_single = single_metric_df.iloc[0].to_dict()

    best_pcr_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=int(best_pcr["n_components"]))),
            ("reg", LinearRegression()),
        ]
    )
    _, best_pcr_train_pred, best_pcr_loo_pred = model_summary(
        best_pcr["model"], best_pcr_pipeline, x, y
    )

    best_pls_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=int(best_pls["n_components"]))),
        ]
    )
    _, best_pls_train_pred, best_pls_loo_pred = model_summary(
        best_pls["model"], best_pls_pipeline, x, y
    )

    with_components = df.copy()
    with_components["complexity_pc1_score"] = pc1
    with_components["complexity_ridge_train_pred"] = ridge_train_pred
    with_components["complexity_ridge_loo_pred"] = ridge_loo_pred
    with_components["complexity_best_pcr_train_pred"] = best_pcr_train_pred
    with_components["complexity_best_pcr_loo_pred"] = best_pcr_loo_pred
    with_components["complexity_best_pls_train_pred"] = best_pls_train_pred
    with_components["complexity_best_pls_loo_pred"] = best_pls_loo_pred
    for component in component_names[:5]:
        with_components[component] = scores_df[component]

    explained_df = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )

    transformed_corr = transformed.corr()

    explained_df.to_csv(BASE_DIR / "latent_complexity_pca_explained_variance.csv", index=False)
    loadings.to_csv(BASE_DIR / "latent_complexity_pca_loadings.csv")
    pc_corr_df.to_csv(BASE_DIR / "latent_complexity_pc_correlations.csv", index=False)
    pcr_df.to_csv(BASE_DIR / "latent_complexity_pcr_models.csv", index=False)
    pls_df.to_csv(BASE_DIR / "latent_complexity_pls_models.csv", index=False)
    single_metric_df.to_csv(BASE_DIR / "latent_complexity_single_metric_baseline.csv", index=False)
    transformed_corr.to_csv(BASE_DIR / "latent_complexity_metric_correlations.csv")
    with_components.to_csv(BASE_DIR / "complexity_with_latent_components.csv", index=False)

    summary = {
        "row_count": int(len(df)),
        "metric_count": len(PCA_METRICS),
        "pc1_unsupervised": pc1_summary,
        "best_single_metric": best_single,
        "explained_variance_first_five": explained_df.head(5).to_dict(orient="records"),
        "pc_correlations_first_five": pc_corr_df.head(5).to_dict(orient="records"),
        "best_pcr_by_loo_pearson": best_pcr,
        "best_pls_by_loo_pearson": best_pls,
        "ridge": {
            **ridge_summary,
            "alpha": ridge_alpha,
        },
        "top_pc1_loadings": top_loadings(loadings, "PC1"),
        "top_pc2_loadings": top_loadings(loadings, "PC2"),
        "top_pc3_loadings": top_loadings(loadings, "PC3"),
    }
    with open(BASE_DIR / "latent_complexity_pca_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_lines = [
        "# Latent Difficulty vs Solver Complexity",
        "",
        f"- Rows analyzed: `{len(df)}`",
        f"- Complexity metrics in PCA: `{len(PCA_METRICS)}`",
        "",
        "## Explained Variance",
        "",
    ]
    for row in explained_df.head(5).to_dict(orient="records"):
        report_lines.append(
            f"- {row['component']}: `{row['explained_variance_ratio']:.3f}` "
            f"(cumulative `{row['cumulative_explained_variance_ratio']:.3f}`)"
        )

    report_lines.extend(
        [
            "",
            "## Latent Difficulty Correlations",
            "",
            f"- Best single metric: `{best_single['metric']}` with Pearson `{best_single['pearson_r']:.3f}` "
            f"and Spearman `{best_single['spearman_rho']:.3f}`",
            f"- PC1 score vs latent difficulty: Pearson `{pc1_summary['train_pearson_r']:.3f}`, "
            f"Spearman `{pc1_summary['train_spearman_rho']:.3f}`",
            f"- Best PCR model by LOO Pearson: `{best_pcr['model']}` with "
            f"train Pearson `{best_pcr['train_pearson_r']:.3f}` and LOO Pearson `{best_pcr['loo_pearson_r']:.3f}`",
            f"- Best PLS model by LOO Pearson: `{best_pls['model']}` with "
            f"train Pearson `{best_pls['train_pearson_r']:.3f}` and LOO Pearson `{best_pls['loo_pearson_r']:.3f}`",
            f"- Ridge model: train Pearson `{ridge_summary['train_pearson_r']:.3f}`, "
            f"LOO Pearson `{ridge_summary['loo_pearson_r']:.3f}`, alpha `{ridge_alpha:.3f}`",
            "",
            "## Component Readings",
            "",
            "- PC1 usually reads like overall solver size / structure complexity.",
            "- PC2 usually separates runtime and grid-volume behavior from pure code-size effects.",
            "- PC3 often picks up normalized execution intensity rather than raw solver length.",
            "",
            "## Files",
            "",
            "- `latent_complexity_pca_explained_variance.csv`",
            "- `latent_complexity_pca_loadings.csv`",
            "- `latent_complexity_pc_correlations.csv`",
            "- `latent_complexity_pcr_models.csv`",
            "- `latent_complexity_pls_models.csv`",
            "- `latent_complexity_single_metric_baseline.csv`",
            "- `latent_complexity_metric_correlations.csv`",
            "- `complexity_with_latent_components.csv`",
            "- `latent_complexity_pca_summary.json`",
        ]
    )

    (BASE_DIR / "latent_complexity_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
