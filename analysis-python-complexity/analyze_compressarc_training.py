from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "analysis-python-complexity"
TMP_DIR = ANALYSIS_DIR / "compressarc_tmp"
TRAIN_DIR = ROOT / "data-llm" / "ARC-AGI" / "data" / "training"
JOIN_PATH = ANALYSIS_DIR / "arc1_dsl_gptoss_gemini_join.csv"
METRIC_CATALOG_PATH = ANALYSIS_DIR / "arc1_dsl_metric_catalog.csv"
NPZ_URL = "https://raw.githubusercontent.com/iliao2345/CompressARC/master/results_for_the_blog_post/predictions_training.npz"
NPZ_PATH = TMP_DIR / "predictions_training.npz"

TASK_SCORE_PATH = ANALYSIS_DIR / "compressarc_training_task_scores.csv"
JOIN_OUT_PATH = ANALYSIS_DIR / "compressarc_training_join.csv"
CORR_OUT_PATH = ANALYSIS_DIR / "compressarc_training_all_metric_correlations.csv"
LOCKED_OUT_PATH = ANALYSIS_DIR / "compressarc_training_locked_correlations.csv"
BEST_BROAD_OUT_PATH = ANALYSIS_DIR / "compressarc_training_best_broad_metrics.csv"
SUMMARY_OUT_PATH = ANALYSIS_DIR / "compressarc_training_summary.json"
REPORT_OUT_PATH = ANALYSIS_DIR / "compressarc_training_report.md"


EXTRA_METRICS = {
    "complexity_pc1_score": "expanded_composite",
    "ast_node_count": "expanded_python_static",
    "function_count": "expanded_python_static",
    "nonblank_lines": "expanded_python_static",
    "token_count": "expanded_python_static",
    "call_count_static": "expanded_python_static",
    "log1p_branch_opcode_count_dynamic": "expanded_dynamic_execution",
    "log1p_cyclomatic_complexity": "expanded_python_static",
}

LOCKED_METRICS = [
    "complexity_pc1_score",
    "dsl_complexity_pc1",
    "ast_node_count",
    "log1p_branch_opcode_count_dynamic",
    "log1p_cyclomatic_complexity",
    "geometry_op_count",
    "solver_opcode_count_dynamic",
    "function_count",
]

OUTCOMES = [
    "compressarc_top1_failure",
    "compressarc_top2_failure",
    "compressarc_top1_first_correct_step",
    "compressarc_top2_first_correct_step",
    "compressarc_top1_fraction_correct_steps",
    "compressarc_top2_fraction_correct_steps",
]


def ensure_dirs() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def download_npz_if_needed() -> None:
    if NPZ_PATH.exists():
        return
    response = requests.get(NPZ_URL, timeout=120)
    response.raise_for_status()
    NPZ_PATH.write_bytes(response.content)


def load_true_solution_hashes() -> tuple[list[str], list[int]]:
    task_ids = [path.stem for path in sorted(TRAIN_DIR.glob("*.json"))]
    true_hashes: list[int] = []
    for task_id in task_ids:
        obj = json.loads((TRAIN_DIR / f"{task_id}.json").read_text(encoding="utf-8"))
        solution = tuple(tuple(tuple(row) for row in pair["output"]) for pair in obj.get("test", []))
        true_hashes.append(hash(solution))
    return task_ids, true_hashes


def build_task_scores() -> pd.DataFrame:
    predictions = np.load(NPZ_PATH, allow_pickle=True)
    picks = predictions["solution_picks_histories"]
    task_ids, true_hashes = load_true_solution_hashes()

    rows: list[dict[str, object]] = []
    for task_index, task_id in enumerate(task_ids):
        history = picks[task_index]
        target_hash = true_hashes[task_index]
        top1_correct = np.array([int(step[0] == target_hash) for step in history], dtype=int)
        top2_correct = np.array([int(any(hash_value == target_hash for hash_value in step)) for step in history], dtype=int)

        ever_top1 = int(top1_correct.max())
        ever_top2 = int(top2_correct.max())
        rows.append(
            {
                "task_id": task_id,
                "compressarc_top1_solved": int(top1_correct[-1]),
                "compressarc_top2_solved": int(top2_correct[-1]),
                "compressarc_top1_failure": int(1 - top1_correct[-1]),
                "compressarc_top2_failure": int(1 - top2_correct[-1]),
                "compressarc_top1_ever_solved": ever_top1,
                "compressarc_top2_ever_solved": ever_top2,
                "compressarc_top1_fraction_correct_steps": float(top1_correct.mean()),
                "compressarc_top2_fraction_correct_steps": float(top2_correct.mean()),
                "compressarc_top1_first_correct_step": float(np.argmax(top1_correct)) if ever_top1 else np.nan,
                "compressarc_top2_first_correct_step": float(np.argmax(top2_correct)) if ever_top2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_metric_categories(joined: pd.DataFrame) -> dict[str, str]:
    catalog = pd.read_csv(METRIC_CATALOG_PATH)
    metric_categories = {row["metric"]: row["category"] for _, row in catalog.iterrows()}
    metric_categories.update(EXTRA_METRICS)
    return {metric: category for metric, category in metric_categories.items() if metric in joined.columns}


def safe_corr(series_x: pd.Series, series_y: pd.Series, method: str) -> tuple[float, float]:
    if series_x.nunique() < 2 or series_y.nunique() < 2:
        return (np.nan, np.nan)
    if method == "pearson":
        result = pearsonr(series_x, series_y)
    else:
        result = spearmanr(series_x, series_y)
    return (float(result.statistic), float(result.pvalue))


def compute_correlations(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_categories = build_metric_categories(joined)
    rows: list[dict[str, object]] = []
    for metric, category in metric_categories.items():
        for outcome in OUTCOMES:
            subset = joined[[metric, outcome]].dropna()
            if subset.empty:
                continue
            pearson_r, pearson_p = safe_corr(subset[metric], subset[outcome], "pearson")
            spearman_rho, spearman_p = safe_corr(subset[metric], subset[outcome], "spearman")
            if np.isnan(pearson_r):
                continue
            rows.append(
                {
                    "metric": metric,
                    "category": category,
                    "outcome": outcome,
                    "n": int(len(subset)),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                }
            )

    all_corr = pd.DataFrame(rows).sort_values(["outcome", "pearson_r"], ascending=[True, False]).reset_index(drop=True)

    broad = all_corr.loc[all_corr["category"] != "dsl_primitive_usage"].copy()
    best_broad = (
        broad.sort_values(["outcome", "pearson_r"], ascending=[True, False])
        .groupby("outcome", as_index=False)
        .head(12)
        .reset_index(drop=True)
    )
    return all_corr, best_broad


def build_locked_table(all_corr: pd.DataFrame) -> pd.DataFrame:
    locked = all_corr.loc[all_corr["metric"].isin(LOCKED_METRICS)].copy()
    metric_order = {metric: index for index, metric in enumerate(LOCKED_METRICS)}
    locked["metric_order"] = locked["metric"].map(metric_order)
    locked = locked.sort_values(["outcome", "metric_order"]).drop(columns=["metric_order"]).reset_index(drop=True)
    return locked


def write_summary(
    task_scores: pd.DataFrame,
    joined: pd.DataFrame,
    locked: pd.DataFrame,
    best_broad: pd.DataFrame,
) -> None:
    def lookup(metric: str, outcome: str) -> dict[str, float]:
        row = locked.loc[(locked["metric"] == metric) & (locked["outcome"] == outcome)].iloc[0]
        return {
            "pearson_r": float(row["pearson_r"]),
            "pearson_p": float(row["pearson_p"]),
            "spearman_rho": float(row["spearman_rho"]),
            "spearman_p": float(row["spearman_p"]),
        }

    summary = {
        "source_url": NPZ_URL,
        "task_count_total": int(len(task_scores)),
        "task_count_overlap_with_complexity_table": int(len(joined)),
        "final_top1_accuracy_total": float(task_scores["compressarc_top1_solved"].mean()),
        "final_top2_accuracy_total": float(task_scores["compressarc_top2_solved"].mean()),
        "final_top1_accuracy_overlap": float(joined["compressarc_top1_solved"].mean()),
        "final_top2_accuracy_overlap": float(joined["compressarc_top2_solved"].mean()),
        "complexity_pc1_top1_failure": lookup("complexity_pc1_score", "compressarc_top1_failure"),
        "complexity_pc1_top2_failure": lookup("complexity_pc1_score", "compressarc_top2_failure"),
        "best_broad_top1_failure": best_broad.loc[best_broad["outcome"] == "compressarc_top1_failure"].head(1).to_dict(orient="records")[0],
        "best_broad_top2_failure": best_broad.loc[best_broad["outcome"] == "compressarc_top2_failure"].head(1).to_dict(orient="records")[0],
        "best_broad_top2_first_correct_step": best_broad.loc[best_broad["outcome"] == "compressarc_top2_first_correct_step"].head(1).to_dict(orient="records")[0],
    }
    SUMMARY_OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_report(locked: pd.DataFrame, best_broad: pd.DataFrame, joined: pd.DataFrame) -> None:
    top1_acc = joined["compressarc_top1_solved"].mean()
    top2_acc = joined["compressarc_top2_solved"].mean()

    top1_pc1 = locked.loc[
        (locked["metric"] == "complexity_pc1_score") & (locked["outcome"] == "compressarc_top1_failure")
    ].iloc[0]
    top2_pc1 = locked.loc[
        (locked["metric"] == "complexity_pc1_score") & (locked["outcome"] == "compressarc_top2_failure")
    ].iloc[0]
    best_top2 = best_broad.loc[best_broad["outcome"] == "compressarc_top2_failure"].head(5)
    best_first = best_broad.loc[best_broad["outcome"] == "compressarc_top2_first_correct_step"].head(5)

    lines = [
        "# CompressARC ARC-1 Training Complexity Correlations",
        "",
        f"- Source archive: {NPZ_URL}",
        f"- Overlap with validated ARC-1 DSL complexity table: {len(joined)} tasks.",
        f"- Final strict top-1 accuracy on the overlap: {int(joined['compressarc_top1_solved'].sum())}/{len(joined)} = {top1_acc:.1%}.",
        f"- Final top-2 accuracy on the overlap: {int(joined['compressarc_top2_solved'].sum())}/{len(joined)} = {top2_acc:.1%}.",
        "",
        "## Complexity PC1",
        "",
        f"- Top-1 failure vs `complexity_pc1_score`: r = {top1_pc1['pearson_r']:.3f}, p = {top1_pc1['pearson_p']:.3g}.",
        f"- Top-2 failure vs `complexity_pc1_score`: r = {top2_pc1['pearson_r']:.3f}, p = {top2_pc1['pearson_p']:.3g}.",
        "",
        "## Best Broad Metrics",
        "",
        best_top2.to_csv(index=False),
        "",
        "## First-Correct-Step Signals",
        "",
        best_first.to_csv(index=False),
        "",
        "Interpretation: the final binary CompressARC outcome has only a weak relationship with the existing global complexity PC1, but some broader structural and dynamic metrics show modest correlations around r ≈ 0.18 to 0.21. The archive's richer training-trajectory measure `compressarc_top2_first_correct_step` is somewhat more complexity-sensitive than the final solved/not-solved flag.",
        "",
    ]
    REPORT_OUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    download_npz_if_needed()

    print("Scoring CompressARC training tasks...")
    task_scores = build_task_scores()
    task_scores.to_csv(TASK_SCORE_PATH, index=False)

    print("Joining onto ARC-1 complexity table...")
    joined = pd.read_csv(JOIN_PATH).merge(task_scores, on="task_id", how="inner")
    joined.to_csv(JOIN_OUT_PATH, index=False)

    print("Computing complexity correlations...")
    all_corr, best_broad = compute_correlations(joined)
    locked = build_locked_table(all_corr)

    all_corr.to_csv(CORR_OUT_PATH, index=False)
    locked.to_csv(LOCKED_OUT_PATH, index=False)
    best_broad.to_csv(BEST_BROAD_OUT_PATH, index=False)

    write_summary(task_scores, joined, locked, best_broad)
    write_report(locked, best_broad, joined)
    print(f"Done. Outputs saved in {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
