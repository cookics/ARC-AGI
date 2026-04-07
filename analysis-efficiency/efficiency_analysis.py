from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LLM_DATA_ROOT = REPO_ROOT / "data-llm"
HUMAN_ANALYSIS_ROOT = REPO_ROOT / "analysis-human" / "analysis"
NON_LLM_DATA_ROOT = REPO_ROOT / "data-non-llm"

LLM_V1_ROOT = LLM_DATA_ROOT / "arc_agi_v1_public_eval"
LLM_V2_ROOT = LLM_DATA_ROOT / "arc_agi_v2_public_eval"

HUMAN_TABLES = HUMAN_ANALYSIS_ROOT / "tables"
NON_LLM_PROCESSED = NON_LLM_DATA_ROOT / "processed"

OUT_DIR = SCRIPT_DIR
FIGURES_DIR = OUT_DIR / "figures"
TABLES_DIR = OUT_DIR / "tables"


def ensure_dirs() -> None:
    for path in [OUT_DIR, FIGURES_DIR, TABLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def guess_thinking_bucket(model_name: str) -> tuple[str, float]:
    lower = model_name.lower()
    ordered = [
        ("thinking-none", "none", 0.0),
        ("thinking-minimal", "minimal", 1.0),
        ("thinking-low", "low", 2.0),
        ("thinking-medium", "medium", 3.0),
        ("thinking-high", "high", 4.0),
        ("thinking-xhigh", "xhigh", 5.0),
        ("deep-think", "deep", 6.0),
    ]
    for token, label, rank in ordered:
        if token in lower:
            return label, rank
    suffix_match = re.search(r"-(low|medium|high|xhigh)$", lower)
    if suffix_match:
        label = suffix_match.group(1)
        return label, {"low": 2.0, "medium": 3.0, "high": 4.0, "xhigh": 5.0}[label]
    return "default", np.nan


def guess_model_family(model_name: str) -> str:
    lower = model_name.lower()
    patterns = [
        (r"^gpt-5-2-pro", "gpt-5.2-pro"),
        (r"^gpt-5-2", "gpt-5.2"),
        (r"^gpt-5-1", "gpt-5.1"),
        (r"^gpt-4-1", "gpt-4.1"),
        (r"^gemini-3", "gemini-3"),
        (r"^claude-opus-4-5", "claude-opus-4.5"),
        (r"^claude-sonnet-4-5", "claude-sonnet-4.5"),
        (r"^claude-haiku-4-5", "claude-haiku-4.5"),
        (r"^grok-4", "grok-4"),
        (r"^qwen3", "qwen3"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, lower):
            return label
    return lower.split("-")[0]


def load_human_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session_df = pd.read_csv(HUMAN_TABLES / "session_summary.csv")
    item_df = pd.read_csv(HUMAN_TABLES / "item_summary.csv")
    public_eval_df = pd.read_csv(HUMAN_TABLES / "public_eval_human_vs_models.csv")
    model_pair_df = pd.read_csv(HUMAN_TABLES / "model_pair_summary.csv")
    return session_df, item_df, public_eval_df, model_pair_df


def load_llm_run_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_summary = pd.read_csv(HUMAN_TABLES / "model_pair_summary.csv").rename(
        columns={"model": "model_name", "type": "llm_type", "pair_accuracy": "pair_accuracy"}
    )
    pair_summary["llm_type"] = pair_summary["llm_type"].fillna("Unknown")

    run_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []

    for root in [LLM_V1_ROOT, LLM_V2_ROOT]:
        if not root.exists():
            continue
        split_name = root.name
        for results_path in sorted(root.glob("*/results.json")):
            model_name = results_path.parent.name
            payload = read_json(results_path)
            task_results = payload.get("task_results", {}) or {}
            values = list(task_results.values())

            solved_tasks = float(payload.get("score", sum(float(v.get("score", 0.0)) for v in values)))
            total_tasks = int(payload.get("total_tasks", len(values) or 1))
            total_cost = float(payload.get("total_cost", sum(float(v.get("cost", 0.0)) for v in values)))
            total_attempts = int(payload.get("total_attempts", sum(int(v.get("attempts", 0)) for v in values)))
            total_output_tokens = float(
                payload.get("avg_output_tokens_per_task", np.nan) * total_tasks
                if payload.get("avg_output_tokens_per_task") is not None
                else sum(float(v.get("output_tokens", 0.0)) for v in values)
            )
            total_tokens = float(
                payload.get("avg_total_tokens_per_task", np.nan) * total_tasks
                if payload.get("avg_total_tokens_per_task") is not None
                else sum(float(v.get("total_tokens", 0.0)) for v in values)
            )
            total_duration = float(
                payload.get("avg_duration_per_task", np.nan) * total_tasks
                if payload.get("avg_duration_per_task") is not None
                else sum(float(v.get("duration", 0.0)) for v in values)
            )
            empty_attempts = int(
                payload.get("num_attempts_with_empty_list", sum(int(v.get("num_attempts_with_empty_list", 0)) for v in values))
            )
            paired = pair_summary.loc[pair_summary["model_name"] == model_name]
            pair_accuracy = float(paired["pair_accuracy"].iloc[0]) if not paired.empty else np.nan
            llm_type = str(paired["llm_type"].iloc[0]) if not paired.empty else "Unknown"
            bucket, bucket_rank = guess_thinking_bucket(model_name)
            family = guess_model_family(model_name)

            run_rows.append(
                {
                    "source_family": "llm",
                    "subtype": "llm_model",
                    "model_name": model_name,
                    "model_family": family,
                    "llm_type": llm_type,
                    "thinking_bucket": bucket,
                    "thinking_rank": bucket_rank,
                    "split": split_name,
                    "performance_rate": solved_tasks / max(total_tasks, 1),
                    "task_solve_rate": solved_tasks / max(total_tasks, 1),
                    "pair_accuracy": pair_accuracy,
                    "task_count": total_tasks,
                    "solved_tasks": solved_tasks,
                    "total_cost": total_cost,
                    "total_attempts": total_attempts,
                    "avg_cost_per_task": float(payload.get("avg_cost_per_task", total_cost / max(total_tasks, 1))),
                    "avg_cost_per_attempt": float(payload.get("avg_cost_per_attempt", total_cost / max(total_attempts, 1))),
                    "avg_output_tokens_per_task": float(
                        payload.get("avg_output_tokens_per_task", total_output_tokens / max(total_tasks, 1))
                    ),
                    "avg_total_tokens_per_task": float(payload.get("avg_total_tokens_per_task", total_tokens / max(total_tasks, 1))),
                    "avg_duration_per_task": float(payload.get("avg_duration_per_task", total_duration / max(total_tasks, 1))),
                    "attempts_per_task": total_attempts / max(total_tasks, 1),
                    "cost_per_solved_task": total_cost / max(solved_tasks, 1),
                    "tokens_per_solved_task": total_tokens / max(solved_tasks, 1),
                    "duration_per_solved_task": total_duration / max(solved_tasks, 1),
                    "empty_attempt_rate": empty_attempts / max(total_attempts, 1),
                    "empty_attempts": empty_attempts,
                }
            )

            for task_id, result in task_results.items():
                task_rows.append(
                    {
                        "source_family": "llm",
                        "split": split_name,
                        "model_name": model_name,
                        "task_ID": task_id,
                        "task_score": float(result.get("score", np.nan)),
                        "task_cost": float(result.get("cost", np.nan)),
                        "task_attempts": float(result.get("attempts", np.nan)),
                        "task_output_tokens": float(result.get("output_tokens", np.nan)),
                        "task_total_tokens": float(result.get("total_tokens", np.nan)),
                        "task_duration_seconds": float(result.get("duration", np.nan)),
                        "task_empty_attempts": float(result.get("num_attempts_with_empty_list", 0)),
                    }
                )

    run_df = pd.DataFrame(run_rows)
    task_df = pd.DataFrame(task_rows)
    if not run_df.empty:
        run_df = run_df.merge(pair_summary, on="model_name", how="left", suffixes=("", "_pair"))
    return run_df, task_df


def load_human_sessions() -> pd.DataFrame:
    session_df = pd.read_csv(HUMAN_TABLES / "session_summary.csv").copy()
    session_df["source_family"] = "human"
    session_df["subtype"] = session_df["session_mix"].fillna("Unknown")
    session_df["label"] = session_df["session_ID"]
    session_df["performance_rate"] = session_df["solve_rate"]
    session_df["primary_effort"] = session_df["mean_duration_seconds"]
    session_df["secondary_effort"] = session_df["mean_submissions"]
    session_df["effort_ratio"] = session_df["mean_submissions"] / session_df["mean_duration_seconds"].replace(0, np.nan)
    session_df["effort_per_success_seconds"] = session_df["total_duration_seconds"] / session_df["total_solved"].replace(0, np.nan)
    session_df["log_tasks_per_minute"] = np.log1p(session_df["tasks_per_minute"].clip(lower=0))
    session_df["log_ability"] = np.log1p((session_df["ability"] - session_df["ability"].min()).clip(lower=0))
    return session_df


def load_non_llm_records() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    compress_summary = read_json(NON_LLM_PROCESSED / "compress_arc_predictions_evaluation_summary.json")
    compress_metrics = compress_summary["metrics"]
    guess_values = list((compress_summary.get("ranked_guess_numbers") or {}).values())
    guess_mean = float(np.mean(guess_values)) if guess_values else np.nan
    guess_median = float(np.median(guess_values)) if guess_values else np.nan
    rows.append(
        {
            "source_family": "non_llm",
            "subtype": "compress_arc",
            "label": "Compress ARC",
            "performance_rate": compress_metrics["final_pick_pass2"]["percentage"] / 100.0,
            "final_top1_rate": compress_metrics["final_pick_pass1"]["percentage"] / 100.0,
            "final_top2_rate": compress_metrics["final_pick_pass2"]["percentage"] / 100.0,
            "oracle_rate": compress_metrics["ranked_candidate_solved_anywhere"]["percentage"] / 100.0,
            "primary_effort": float(compress_summary["iterations"]),
            "secondary_effort": guess_mean,
            "guess_number_mean": guess_mean,
            "guess_number_median": guess_median,
            "task_count": float(compress_summary["task_count"]),
            "iterations": float(compress_summary["iterations"]),
        }
    )

    varc_summary = read_json(NON_LLM_PROCESSED / "varc_predictions_summary.json")
    for result in varc_summary["results"]:
        candidate_stats = result["candidate_count_stats"]
        attempt_dirs = result.get("attempt_dirs", [])
        metrics = result["metrics"]
        pass4 = metrics.get("pass@4", {})
        rows.append(
            {
                "source_family": "non_llm",
                "subtype": "varc",
                "label": Path(result["model_dir"]).name,
                "split": result.get("split"),
                "performance_rate": pass4.get("task_percentage", np.nan) / 100.0,
                "pass1_rate": metrics.get("pass@1", {}).get("task_percentage", np.nan) / 100.0,
                "pass2_rate": metrics.get("pass@2", {}).get("task_percentage", np.nan) / 100.0,
                "pass3_rate": metrics.get("pass@3", {}).get("task_percentage", np.nan) / 100.0,
                "pass4_rate": pass4.get("task_percentage", np.nan) / 100.0,
                "oracle_rate": result["candidate_pool_oracle"]["task_percentage"] / 100.0,
                "primary_effort": float(candidate_stats["max"] if candidate_stats["max"] is not None else np.nan),
                "secondary_effort": float(len(attempt_dirs)),
                "candidate_count_min": candidate_stats["min"],
                "candidate_count_max": candidate_stats["max"],
                "candidate_count_unique": len(candidate_stats.get("unique", [])),
                "attempt_dir_count": float(len(attempt_dirs)),
                "task_count": float(result["task_count"]),
            }
        )

    trm_summary = read_json(NON_LLM_PROCESSED / "trm_arc_agi_ii_progression.json")
    for result in trm_summary["results"]:
        pair_accuracy = result["pair_accuracy"]
        rows.append(
            {
                "source_family": "non_llm",
                "subtype": "trm",
                "label": result["folder"],
                "step": float(result["step"]),
                "performance_rate": pair_accuracy["attempt_1_or_2_percentage"] / 100.0,
                "pair1_rate": pair_accuracy["attempt_1_percentage"] / 100.0,
                "pair2_rate": pair_accuracy["attempt_1_or_2_percentage"] / 100.0,
                "oracle_rate": result["kaggle_score"] / 100.0,
                "primary_effort": float(result["step"]),
                "secondary_effort": float(result["pair_count"]),
                "pair_count": float(result["pair_count"]),
                "task_count": float(result["task_count"]),
                "kaggle_score": float(result["kaggle_score"]) / 100.0,
            }
        )

    return pd.DataFrame(rows)


def summarize_source(df: pd.DataFrame, group_col: str = "source_family") -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df.groupby(group_col)[numeric_cols].agg(["count", "mean", "median", "std"]).copy()
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def correlation_block(df: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    available = [c for c in cols if c in df.columns]
    if len(available) < 2:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for i, x in enumerate(available):
        for y in available[i + 1 :]:
            subset = df[[x, y]].dropna()
            if len(subset) < 3:
                continue
            if subset[x].nunique(dropna=True) < 2 or subset[y].nunique(dropna=True) < 2:
                rows.append(
                    {
                        "group": label,
                        "metric_x": x,
                        "metric_y": y,
                        "n": int(len(subset)),
                        "pearson_r": np.nan,
                        "pearson_p": np.nan,
                        "spearman_r": np.nan,
                        "spearman_p": np.nan,
                    }
                )
                continue
            p = pearsonr(subset[x], subset[y])
            s = spearmanr(subset[x], subset[y])
            rows.append(
                {
                    "group": label,
                    "metric_x": x,
                    "metric_y": y,
                    "n": int(len(subset)),
                    "pearson_r": float(p.statistic),
                    "pearson_p": float(p.pvalue),
                    "spearman_r": float(s.statistic),
                    "spearman_p": float(s.pvalue),
                }
            )
    return pd.DataFrame(rows)


def _clean_pair_arrays(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    subset = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if subset.empty:
        return np.array([]), np.array([])
    return subset[x_col].to_numpy(dtype=float), subset[y_col].to_numpy(dtype=float)


def _bootstrap_corr_interval(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    n_boot: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        xs = x[idx]
        ys = y[idx]
        if method == "pearson":
            value = pearsonr(xs, ys).statistic
        else:
            value = spearmanr(xs, ys).statistic
        samples[i] = value

    samples = samples[np.isfinite(samples)]
    if not len(samples):
        return np.nan, np.nan
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _bootstrap_corr_difference(
    df: pd.DataFrame,
    x_col: str,
    y_left: str,
    y_right: str,
    method: str = "spearman",
    n_boot: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float]:
    subset = df[[x_col, y_left, y_right]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 3:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    x = subset[x_col].to_numpy(dtype=float)
    y0 = subset[y_left].to_numpy(dtype=float)
    y1 = subset[y_right].to_numpy(dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(subset), size=len(subset))
        xs = x[idx]
        left = y0[idx]
        right = y1[idx]
        if method == "pearson":
            left_corr = pearsonr(xs, left).statistic
            right_corr = pearsonr(xs, right).statistic
        else:
            left_corr = spearmanr(xs, left).statistic
            right_corr = spearmanr(xs, right).statistic
        deltas[i] = left_corr - right_corr

    deltas = deltas[np.isfinite(deltas)]
    if not len(deltas):
        return np.nan, np.nan, np.nan
    center = float(np.nanmean(deltas))
    return center, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def summarize_selected_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    analysis: str,
    group: str,
    n_boot: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    subset = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(subset))
    if n < 3 or subset[x_col].nunique(dropna=True) < 2 or subset[y_col].nunique(dropna=True) < 2:
        return {
            "analysis": analysis,
            "group": group,
            "metric_x": x_col,
            "metric_y": y_col,
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "pearson_ci_low": np.nan,
            "pearson_ci_high": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "spearman_ci_low": np.nan,
            "spearman_ci_high": np.nan,
        }

    pearson = pearsonr(subset[x_col], subset[y_col])
    spearman = spearmanr(subset[x_col], subset[y_col])
    pearson_ci = _bootstrap_corr_interval(subset[x_col].to_numpy(dtype=float), subset[y_col].to_numpy(dtype=float), "pearson", n_boot=n_boot, seed=seed)
    spearman_ci = _bootstrap_corr_interval(subset[x_col].to_numpy(dtype=float), subset[y_col].to_numpy(dtype=float), "spearman", n_boot=n_boot, seed=seed + 1)
    return {
        "analysis": analysis,
        "group": group,
        "metric_x": x_col,
        "metric_y": y_col,
        "n": n,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "pearson_ci_low": pearson_ci[0],
        "pearson_ci_high": pearson_ci[1],
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "spearman_ci_low": spearman_ci[0],
        "spearman_ci_high": spearman_ci[1],
    }


def fit_nested_ols_models(task_df: pd.DataFrame, target: str, feature_sets: list[tuple[str, list[str]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    row_counts: dict[str, int] = {}

    for model_name, features in feature_sets:
        present = [col for col in features if col in task_df.columns]
        if len(present) < 2:
            continue
        subset = task_df[[target] + present].replace([np.inf, -np.inf], np.nan).dropna(subset=[target] + present)
        if len(subset) < 10:
            continue
        X = sm.add_constant(subset[present], has_constant="add")
        y = subset[target]
        fit = sm.OLS(y, X).fit()
        fitted[model_name] = fit
        row_counts[model_name] = int(len(subset))
        model_rows.append(
            {
                "target": target,
                "model": model_name,
                "n": int(len(subset)),
                "feature_count": int(len(present)),
                "r2": float(fit.rsquared),
                "adj_r2": float(fit.rsquared_adj),
                "aic": float(fit.aic),
                "bic": float(fit.bic),
                "f_statistic": float(fit.fvalue) if fit.fvalue is not None else np.nan,
                "f_pvalue": float(fit.f_pvalue) if fit.f_pvalue is not None else np.nan,
            }
        )

    for (base_name, _), (full_name, _) in zip(feature_sets[:-1], feature_sets[1:]):
        if base_name not in fitted or full_name not in fitted:
            continue
        base = fitted[base_name]
        full = fitted[full_name]
        anova = anova_lm(base, full)
        if len(anova.index) < 2:
            continue
        comparison_rows.append(
            {
                "target": target,
                "base_model": base_name,
                "full_model": full_name,
                "n": int(row_counts[full_name]),
                "delta_r2": float(full.rsquared - base.rsquared),
                "df_diff": float(anova.loc[1, "df_diff"]),
                "ss_diff": float(anova.loc[1, "ss_diff"]),
                "f_statistic": float(anova.loc[1, "F"]),
                "p_value": float(anova.loc[1, "Pr(>F)"]),
            }
        )

    return pd.DataFrame(model_rows), pd.DataFrame(comparison_rows)


def fit_shared_task_pca(task_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "human_solve_rate",
        "human_mean_duration_seconds",
        "human_mean_submissions",
        "llm_mean_score",
        "llm_mean_duration_seconds",
        "llm_mean_cost",
        "llm_mean_total_tokens",
        "trm_best_task_score",
    ]
    work = task_df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if work.shape[0] < 3 or work.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame()

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(2, work.shape[1], work.shape[0]))),
        ]
    )
    scores = pipeline.fit_transform(work)
    pca = pipeline.named_steps["pca"]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=work.columns,
        columns=[f"pc{i + 1}" for i in range(pca.n_components_)],
    ).reset_index(names="feature")
    loadings["group"] = "shared_arc2"
    loadings["explained_variance_ratio_pc1"] = float(pca.explained_variance_ratio_[0])
    loadings["explained_variance_ratio_pc2"] = float(pca.explained_variance_ratio_[1]) if pca.n_components_ > 1 else np.nan

    score_df = pd.DataFrame(scores, columns=[f"shared_arc2_pc{i + 1}" for i in range(scores.shape[1])])
    score_df["task_ID"] = task_df.loc[work.index, "task_ID"].to_numpy()
    score_df["human_solve_rate"] = task_df.loc[work.index, "human_solve_rate"].to_numpy()
    score_df["llm_mean_score"] = task_df.loc[work.index, "llm_mean_score"].to_numpy()
    return loadings, score_df


def latex_escape(text: str) -> str:
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    escaped = text
    for old, new in replacements:
        escaped = escaped.replace(old, new)
    return escaped


def latex_table(df: pd.DataFrame, caption: str, label: str, float_format: str = "%.3f") -> str:
    if df.empty:
        body = "\\multicolumn{1}{c}{No data}"
        columns = 1
    else:
        def cell_to_text(value: Any) -> str:
            if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
                return "NA"
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric = float(value)
                if np.isfinite(numeric) and abs(numeric - round(numeric)) < 1e-9:
                    return str(int(round(numeric)))
                return float_format % numeric
            return latex_escape(str(value))

        columns = len(df.columns)
        header = " & ".join(latex_escape(str(col)) for col in df.columns) + r" \\"
        rows = [" & ".join(cell_to_text(value) for value in row) + r" \\" for row in df.itertuples(index=False, name=None)]
        body = "\n".join([header] + rows)
    align = "l" * columns
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{latex_escape(caption)}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{align}}}\n"
        "\\toprule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


def collect_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).copy()
    return numeric.replace([np.inf, -np.inf], np.nan)


def fit_source_classifier(df: pd.DataFrame) -> dict[str, Any]:
    feature_df = collect_numeric_features(df)
    feature_df = feature_df.drop(columns=[c for c in ["task_ID", "task_pair_id"] if c in feature_df.columns], errors="ignore")
    feature_df = feature_df.loc[:, feature_df.nunique(dropna=True) > 1]

    target = df["source_family"].astype(str)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )

    n_splits = min(5, int(target.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    preds = cross_val_predict(model, feature_df, target, cv=cv)
    model.fit(feature_df, target)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(target, preds)),
        "macro_f1": float(f1_score(target, preds, average="macro")),
        "micro_f1": float(f1_score(target, preds, average="micro")),
        "classification_report": classification_report(target, preds, output_dict=True, zero_division=0),
        "classes": model.named_steps["clf"].classes_.tolist(),
        "feature_count": int(feature_df.shape[1]),
        "row_count": int(feature_df.shape[0]),
        "class_counts": target.value_counts().to_dict(),
    }


def fit_shared_source_classifier(df: pd.DataFrame) -> dict[str, Any]:
    feature_cols = ["performance_rate", "primary_effort", "secondary_effort", "effort_ratio"]
    feature_df = df[feature_cols].copy()
    target = df["source_family"].astype(str)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )
    n_splits = min(5, int(target.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    preds = cross_val_predict(model, feature_df, target, cv=cv)
    model.fit(feature_df, target)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(target, preds)),
        "macro_f1": float(f1_score(target, preds, average="macro")),
        "micro_f1": float(f1_score(target, preds, average="micro")),
        "classification_report": classification_report(target, preds, output_dict=True, zero_division=0),
        "classes": model.named_steps["clf"].classes_.tolist(),
        "feature_count": int(feature_df.shape[1]),
        "row_count": int(feature_df.shape[0]),
        "class_counts": target.value_counts().to_dict(),
        "features": feature_cols,
    }


def fit_pca_summary(df: pd.DataFrame, feature_cols: list[str], label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if work.shape[1] < 2 or work.shape[0] < 3:
        return pd.DataFrame(), pd.DataFrame()

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(2, work.shape[1], work.shape[0]))),
        ]
    )
    scores = pipeline.fit_transform(work)
    pca = pipeline.named_steps["pca"]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=work.columns,
        columns=[f"pc{i + 1}" for i in range(pca.n_components_)],
    ).reset_index(names="feature")
    loadings["group"] = label
    loadings["explained_variance_ratio_pc1"] = float(pca.explained_variance_ratio_[0])
    loadings["explained_variance_ratio_pc2"] = float(pca.explained_variance_ratio_[1]) if pca.n_components_ > 1 else np.nan
    score_df = pd.DataFrame(scores, columns=[f"{label}_pc{i + 1}" for i in range(scores.shape[1])])
    score_df["group"] = label
    score_df["row_index"] = work.index.to_numpy()
    return loadings, score_df


def fit_task_regression(task_df: pd.DataFrame, target: str, feature_sets: list[tuple[str, list[str]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, features in feature_sets:
        present = [col for col in features if col in task_df.columns]
        if len(present) < 2:
            continue
        subset = task_df[[target] + present].dropna(subset=[target])
        if len(subset) < 10:
            continue
        X = subset[present]
        y = subset[target]
        cv = KFold(n_splits=min(5, len(subset)), shuffle=True, random_state=0)
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        preds = cross_val_predict(pipe, X, y, cv=cv)
        pipe.fit(X, y)
        rows.append(
            {
                "target": target,
                "model": model_name,
                "n": int(len(subset)),
                "feature_count": int(len(present)),
                "r2": float(r2_score(y, preds)),
                "mae": float(mean_absolute_error(y, preds)),
                "pearson_r": float(pearsonr(y, preds).statistic),
                "spearman_r": float(spearmanr(y, preds).statistic),
                "features": present,
            }
        )
    return pd.DataFrame(rows)


def build_shared_task_table(human_public_df: pd.DataFrame, llm_task_df: pd.DataFrame, trm_summary: dict[str, Any]) -> pd.DataFrame:
    grouped = (
        human_public_df.groupby("task_ID")
        .agg(
            task_set=("task_set", "first"),
            n_pairs=("task_pair_id", "count"),
            human_solve_rate=("solve_rate", "mean"),
            human_mean_duration_seconds=("mean_duration_seconds", "mean"),
            human_mean_submissions=("mean_submissions", "mean"),
            human_mean_pred_prob=("mean_pred_prob", "mean"),
            human_outfit=("outfit", "mean"),
            human_difficulty=("difficulty", "mean"),
            human_point_biserial=("point_biserial", "mean"),
            human_ability_gap=("ability_gap", "mean"),
            n_train_pairs=("n_train_pairs", "mean"),
            n_test_pairs=("n_test_pairs", "mean"),
            input_rows=("input_rows", "mean"),
            input_cols=("input_cols", "mean"),
            input_cells=("input_cells", "mean"),
            input_colors=("input_colors", "mean"),
            output_rows=("output_rows", "mean"),
            output_cols=("output_cols", "mean"),
            output_cells=("output_cells", "mean"),
            output_colors=("output_colors", "mean"),
            size_change_ratio=("size_change_ratio", "mean"),
        )
        .reset_index()
    )

    llm_agg = (
        llm_task_df.groupby("task_ID")
        .agg(
            llm_model_count=("model_name", "nunique"),
            llm_mean_score=("task_score", "mean"),
            llm_median_score=("task_score", "median"),
            llm_score_std=("task_score", "std"),
            llm_mean_cost=("task_cost", "mean"),
            llm_mean_attempts=("task_attempts", "mean"),
            llm_mean_output_tokens=("task_output_tokens", "mean"),
            llm_mean_total_tokens=("task_total_tokens", "mean"),
            llm_mean_duration_seconds=("task_duration_seconds", "mean"),
            llm_mean_empty_attempts=("task_empty_attempts", "mean"),
        )
        .reset_index()
    )
    llm_agg["llm_effort_per_success_seconds"] = llm_agg["llm_mean_duration_seconds"] / llm_agg["llm_mean_score"].replace(0, np.nan)
    llm_agg["llm_effort_per_success_cost"] = llm_agg["llm_mean_cost"] / llm_agg["llm_mean_score"].replace(0, np.nan)
    llm_agg["llm_effort_per_success_tokens"] = llm_agg["llm_mean_total_tokens"] / llm_agg["llm_mean_score"].replace(0, np.nan)

    trm_best = max(trm_summary["results"], key=lambda item: item["kaggle_score"])
    trm_rows = pd.DataFrame(
        [{"task_ID": task_id, "trm_best_task_score": float(score)} for task_id, score in trm_best["task_fractional_scores"].items()]
    )

    merged = grouped.merge(llm_agg, on="task_ID", how="inner")
    merged = merged.merge(trm_rows, on="task_ID", how="left")
    merged["human_effort_per_success_seconds"] = merged["human_mean_duration_seconds"] / merged["human_solve_rate"].replace(0, np.nan)
    merged["human_effort_per_success_submissions"] = merged["human_mean_submissions"] / merged["human_solve_rate"].replace(0, np.nan)
    merged["llm_human_duration_ratio"] = merged["llm_mean_duration_seconds"] / merged["human_mean_duration_seconds"].replace(0, np.nan)
    return merged


def source_specific_pca(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    loadings_all: list[pd.DataFrame] = []
    scores_all: list[pd.DataFrame] = []
    groups = {
        "llm": ["performance_rate", "avg_duration_per_task", "avg_total_tokens_per_task", "avg_cost_per_task", "attempts_per_task", "empty_attempt_rate"],
        "human": ["performance_rate", "mean_duration_seconds", "mean_submissions", "ability", "outfit"],
        "non_llm": ["performance_rate", "primary_effort", "secondary_effort", "oracle_rate", "kaggle_score"],
    }
    for source, cols in groups.items():
        subset = df.loc[df["source_family"] == source, cols].copy()
        if subset.empty:
            continue
        loadings, scores = fit_pca_summary(subset, list(subset.columns), source)
        if not loadings.empty:
            loadings_all.append(loadings)
        if not scores.empty:
            scores_all.append(scores)
    loadings_df = pd.concat(loadings_all, ignore_index=True) if loadings_all else pd.DataFrame()
    scores_df = pd.concat(scores_all, ignore_index=True) if scores_all else pd.DataFrame()
    return loadings_df, scores_df


def plot_source_tradeoff(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = {"llm": "#2E86AB", "human": "#E76F51", "non_llm": "#1ABC9C"}
    for source, group in df.groupby("source_family"):
        x = np.log1p(group["primary_effort"].clip(lower=0))
        y = group["performance_rate"]
        ax.scatter(x, y, s=35, alpha=0.75, label=source, color=palette.get(source, "#555555"))
        if len(group) >= 3:
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 3:
                slope, intercept = np.polyfit(x[m], y[m], 1)
                xs = np.linspace(float(x[m].min()), float(x[m].max()), 100)
                ax.plot(xs, slope * xs + intercept, color=palette.get(source, "#555555"), linewidth=2)
    ax.set_xlabel("log1p(primary effort)")
    ax.set_ylabel("performance rate")
    ax.set_title("Efficiency tradeoff by source family")
    ax.legend(title="source")
    ax.grid(True, alpha=0.2)
    out = FIGURES_DIR / "source_tradeoff.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_task_correlations(task_df: pd.DataFrame) -> Path:
    cols = [
        "human_solve_rate",
        "human_mean_duration_seconds",
        "human_mean_submissions",
        "llm_mean_score",
        "llm_mean_duration_seconds",
        "llm_mean_cost",
        "llm_mean_total_tokens",
        "trm_best_task_score",
    ]
    corr = task_df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax, cmap="vlag", center=0, annot=False, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Shared ARC-2 task correlations")
    out = FIGURES_DIR / "task_correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_shared_task_pca(shared_task_df: pd.DataFrame, score_df: pd.DataFrame) -> Path:
    if score_df.empty:
        return FIGURES_DIR / "shared_task_latent_map.png"

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        score_df["shared_arc2_pc1"],
        score_df["shared_arc2_pc2"] if "shared_arc2_pc2" in score_df.columns else np.zeros(len(score_df)),
        c=score_df["human_solve_rate"],
        cmap="viridis",
        s=45,
        alpha=0.82,
        edgecolor="none",
    )
    ax.set_xlabel("shared ARC-2 PC1")
    ax.set_ylabel("shared ARC-2 PC2")
    ax.set_title("Shared-task latent map colored by human solve rate")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
    cbar.set_label("human solve rate")
    ax.grid(True, alpha=0.18)
    out = FIGURES_DIR / "shared_task_latent_map.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_trm_progression(trm_summary: dict[str, Any]) -> Path:
    result_df = pd.DataFrame(trm_summary["results"]).sort_values("step")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(result_df["step"], result_df["kaggle_score"], marker="o", color="#2E86AB", label="kaggle score")
    ax1.set_xlabel("step")
    ax1.set_ylabel("kaggle score (%)", color="#2E86AB")
    ax2 = ax1.twinx()
    ax2.plot(
        result_df["step"],
        result_df["pair_accuracy"].apply(lambda x: x["attempt_1_or_2_percentage"]),
        marker="s",
        color="#E76F51",
        label="pair accuracy",
    )
    ax2.set_ylabel("pair accuracy (%)", color="#E76F51")
    ax1.set_title("TRM progression over steps")
    out = FIGURES_DIR / "trm_progression.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def build_report(
    inventory: pd.DataFrame,
    source_summary: pd.DataFrame,
    source_correlations: pd.DataFrame,
    task_correlations: pd.DataFrame,
    classifier_results: dict[str, Any],
    shared_classifier_results: dict[str, Any],
    regression_results: pd.DataFrame,
    regression_ols: pd.DataFrame,
    regression_compare: pd.DataFrame,
    selected_stats: pd.DataFrame,
    pca_loadings: pd.DataFrame,
    shared_task_pca_loadings: pd.DataFrame,
    shared_task_df: pd.DataFrame,
    llm_run_df: pd.DataFrame,
    human_session_df: pd.DataFrame,
    non_llm_df: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    def fmt_float(value: Any, digits: int = 3) -> str:
        if value is None or pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"

    def fmt_pvalue(value: Any) -> str:
        if value is None or pd.isna(value):
            return "nan"
        value = float(value)
        if value < 0.001:
            return "<0.001"
        return f"{value:.3f}"

    def source_row(source: str) -> pd.Series:
        row = source_summary.loc[source_summary["source_family"] == source]
        if row.empty:
            return pd.Series(dtype=float)
        return row.iloc[0]

    def pick_stats(analysis: str, metric_x: str, metric_y: str) -> pd.Series:
        row = selected_stats.loc[
            (selected_stats["analysis"] == analysis)
            & (selected_stats["metric_x"] == metric_x)
            & (selected_stats["metric_y"] == metric_y)
        ]
        if row.empty:
            return pd.Series(dtype=float)
        return row.iloc[0]

    lines: list[str] = []
    lines.append("# Efficiency comparison report")
    lines.append("")
    lines.append("## Summary")
    shared_llm = pick_stats("Shared-task human vs LLM score", "human_solve_rate", "llm_mean_score")
    public_overlap = pick_stats("Public-eval human vs average model", "solve_rate", "lm_mean")
    llm_perf = pick_stats("LLM performance vs thinking rank", "performance_rate", "thinking_rank")
    human_perf = pick_stats("Human performance vs ability", "performance_rate", "ability")
    lines.append(
        f"- Latest shared-task human-vs-LLM alignment: Spearman={fmt_float(shared_llm.get('spearman_r'))} "
        f"(p={fmt_pvalue(shared_llm.get('spearman_p'))}, 95% bootstrap CI [{fmt_float(shared_llm.get('spearman_ci_low'))}, {fmt_float(shared_llm.get('spearman_ci_high'))}])"
    )
    lines.append(
        f"- Earlier public-eval overlap check: Pearson={fmt_float(public_overlap.get('pearson_r'))} "
        f"(p={fmt_pvalue(public_overlap.get('pearson_p'))})"
    )
    lines.append(
        f"- Within LLMs, performance tracks thinking rank strongly: Spearman={fmt_float(llm_perf.get('spearman_r'))} "
        f"(p={fmt_pvalue(llm_perf.get('spearman_p'))})"
    )
    lines.append(
        f"- Within humans, performance tracks latent ability strongly: Spearman={fmt_float(human_perf.get('spearman_r'))} "
        f"(p={fmt_pvalue(human_perf.get('spearman_p'))})"
    )
    lines.append("")
    lines.append("## Data inventory")
    for _, row in inventory.iterrows():
        lines.append(
            f"- {row['source_family']} / {row['subtype']}: {int(row['rows'])} rows, "
            f"performance=`{row['performance_metric']}`, effort=`{row['primary_effort_metric']}`"
        )
    lines.append("")
    lines.append("## Distribution snapshot")
    for source, label in [("llm", "LLM"), ("human", "Human"), ("non_llm", "Non-LLM")]:
        row = source_row(source)
        if row.empty:
            continue
        if source == "llm":
            lines.append(
                f"- {label}: performance mean={fmt_float(row.get('performance_rate_mean'))}, median={fmt_float(row.get('performance_rate_median'))}; "
                f"duration mean={fmt_float(row.get('avg_duration_per_task_mean'))}, median={fmt_float(row.get('avg_duration_per_task_median'))}; "
                f"cost mean={fmt_float(row.get('avg_cost_per_task_mean'))}, median={fmt_float(row.get('avg_cost_per_task_median'))}"
            )
        elif source == "human":
            lines.append(
                f"- {label}: performance mean={fmt_float(row.get('performance_rate_mean'))}, median={fmt_float(row.get('performance_rate_median'))}; "
                f"duration mean={fmt_float(row.get('mean_duration_seconds_mean'))}, median={fmt_float(row.get('mean_duration_seconds_median'))}; "
                f"ability mean={fmt_float(row.get('ability_mean'))}, median={fmt_float(row.get('ability_median'))}"
            )
        else:
            lines.append(
                f"- {label}: performance mean={fmt_float(row.get('performance_rate_mean'))}, median={fmt_float(row.get('performance_rate_median'))}; "
                f"primary effort mean={fmt_float(row.get('primary_effort_mean'))}, median={fmt_float(row.get('primary_effort_median'))}; "
                f"oracle mean={fmt_float(row.get('oracle_rate_mean'))}, median={fmt_float(row.get('oracle_rate_median'))}"
            )
    lines.append("")
    lines.append("## Hypotheses entertained")
    lines.append("- Shared ARC task difficulty should align across humans, LLMs, and TRM.")
    lines.append("- Better performance should usually come with more resource use, but the sign of that tradeoff may differ by source.")
    lines.append("- A small number of latent axes should summarize most of the efficiency variation.")
    lines.append("- Generic efficiency features should still identify the source family.")
    lines.append("")
    lines.append("## Cross-source alignment")
    lines.append("The current data do not support a direct person-level latent correlation between model theta and human ability, because those estimates live on different rows and scales. The cleanest comparison is task-level alignment.")
    for analysis in ["Shared-task human vs LLM score", "Shared-task human vs LLM duration", "Shared-task human vs LLM cost", "Shared-task human vs TRM score", "Public-eval human vs average model", "Public-eval human vs best single model"]:
        block = selected_stats.loc[selected_stats["analysis"] == analysis]
        if block.empty:
            continue
        lines.append(f"### {analysis}")
        for _, row in block.iterrows():
            lines.append(
                f"- {row['metric_x']} vs {row['metric_y']}: "
                f"Pearson={fmt_float(row['pearson_r'])} (p={fmt_pvalue(row['pearson_p'])}, CI [{fmt_float(row['pearson_ci_low'])}, {fmt_float(row['pearson_ci_high'])}]); "
                f"Spearman={fmt_float(row['spearman_r'])} (p={fmt_pvalue(row['spearman_p'])}, CI [{fmt_float(row['spearman_ci_low'])}, {fmt_float(row['spearman_ci_high'])}])"
            )
    diff = _bootstrap_corr_difference(shared_task_df, "human_solve_rate", "llm_mean_score", "trm_best_task_score", method="spearman", n_boot=3000, seed=1)
    lines.append(
        f"- Shared-task alignment is materially stronger for LLM score than TRM score: delta Spearman={fmt_float(diff[0])} "
        f"(95% bootstrap CI [{fmt_float(diff[1])}, {fmt_float(diff[2])}])"
    )
    lines.append("")
    lines.append("## Within-source efficiency")
    for analysis in ["LLM performance vs thinking rank", "LLM duration vs thinking rank", "LLM performance vs duration", "LLM duration vs cost", "Human performance vs ability", "Human performance vs duration", "Human performance vs outfit", "TRM performance vs step", "TRM performance vs oracle"]:
        block = selected_stats.loc[selected_stats["analysis"] == analysis]
        if block.empty:
            continue
        lines.append(f"### {analysis}")
        for _, row in block.iterrows():
            lines.append(
                f"- {row['metric_x']} vs {row['metric_y']}: "
                f"Spearman={fmt_float(row['spearman_r'])} (p={fmt_pvalue(row['spearman_p'])}, CI [{fmt_float(row['spearman_ci_low'])}, {fmt_float(row['spearman_ci_high'])}])"
            )
    lines.append("")
    lines.append("## Predictive models")
    for target in regression_results["target"].drop_duplicates():
        target_block = regression_results.loc[regression_results["target"] == target].sort_values("r2", ascending=False)
        if target_block.empty:
            continue
        best = target_block.iloc[0]
        lines.append(
            f"- Best CV model for {target}: {best['model']} with R2={fmt_float(best['r2'])}, MAE={fmt_float(best['mae'])}, Pearson={fmt_float(best['pearson_r'])}, Spearman={fmt_float(best['spearman_r'])}"
        )
    if not regression_compare.empty:
        lines.append("")
        lines.append("Nested OLS comparisons:")
        for _, row in regression_compare.iterrows():
            lines.append(
                f"- {row['target']}: {row['base_model']} -> {row['full_model']} "
                f"delta R2={fmt_float(row['delta_r2'])}, F={fmt_float(row['f_statistic'])}, p={fmt_pvalue(row['p_value'])}"
            )
    lines.append("")
    lines.append("## Latent structure")
    if not pca_loadings.empty:
        for source in pca_loadings["group"].drop_duplicates():
            block = pca_loadings.loc[pca_loadings["group"] == source].copy()
            lines.append(
                f"- {source}: PC1 variance={fmt_float(block['explained_variance_ratio_pc1'].iloc[0])}, PC2 variance={fmt_float(block['explained_variance_ratio_pc2'].iloc[0])}"
            )
            top_pc1 = block.sort_values("pc1", key=lambda s: s.abs(), ascending=False).head(4)
            for _, item in top_pc1.iterrows():
                lines.append(f"  - pc1 {item['feature']}: {fmt_float(item['pc1'])}")
    if not shared_task_pca_loadings.empty:
        lines.append("- shared ARC-2 task PCA:")
        lines.append(
            f"  - PC1 variance={fmt_float(shared_task_pca_loadings['explained_variance_ratio_pc1'].iloc[0])}, "
            f"PC2 variance={fmt_float(shared_task_pca_loadings['explained_variance_ratio_pc2'].iloc[0])}"
        )
        top_shared_pc1 = shared_task_pca_loadings.sort_values("pc1", key=lambda s: s.abs(), ascending=False).head(5)
        top_shared_pc2 = shared_task_pca_loadings.sort_values("pc2", key=lambda s: s.abs(), ascending=False).head(5)
        lines.append("  - PC1 loadings:")
        for _, item in top_shared_pc1.iterrows():
            lines.append(f"    - {item['feature']}: {fmt_float(item['pc1'])}")
        lines.append("  - PC2 loadings:")
        for _, item in top_shared_pc2.iterrows():
            lines.append(f"    - {item['feature']}: {fmt_float(item['pc2'])}")
    lines.append("")
    lines.append("## Weak or discarded analyses")
    lines.append("- The full-feature source classifier is trivial, because raw telemetry makes the source family obvious.")
    lines.append("- The shared 4-feature classifier is the informative one; it still separates sources but not perfectly.")
    lines.append("- Geometry-only prediction of human solve rate is weak.")
    lines.append("- Adding LLM effort or TRM score after LLM performance does not materially improve the shared-task prediction.")
    lines.append("- Human duration is not explained well by geometry or LLM features.")
    lines.append("- Compress ARC has only one row, so it is descriptive rather than inferential.")
    lines.append("- VARC has only four rows, so correlation claims there are fragile.")
    lines.append("")
    lines.append("## Figures")
    for path in figure_paths:
        lines.append(f"- {path.as_posix()}")
    return "\n".join(lines)


def build_latex_report(
    inventory: pd.DataFrame,
    source_summary: pd.DataFrame,
    regression_results: pd.DataFrame,
    regression_compare: pd.DataFrame,
    selected_stats: pd.DataFrame,
    pca_loadings: pd.DataFrame,
    shared_task_pca_loadings: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    def format_ci_row(row: pd.Series, prefix: str) -> str:
        return f"{float(row[f'{prefix}_r']):.3f} [{float(row[f'{prefix}_ci_low']):.3f}, {float(row[f'{prefix}_ci_high']):.3f}]"

    def format_pvalue(value: Any) -> str:
        if value is None or pd.isna(value):
            return "nan"
        value = float(value)
        if value < 0.001:
            return "<0.001"
        return f"{value:.3f}"

    cross_groups = ["Shared-task human vs LLM score", "Shared-task human vs LLM duration", "Shared-task human vs LLM cost", "Shared-task human vs TRM score", "Public-eval human vs average model", "Public-eval human vs best single model"]
    cross_table = selected_stats.loc[selected_stats["analysis"].isin(cross_groups)].copy()
    if not cross_table.empty:
        cross_table["pearson"] = cross_table.apply(lambda row: format_ci_row(row, "pearson"), axis=1)
        cross_table["spearman"] = cross_table.apply(lambda row: format_ci_row(row, "spearman"), axis=1)
        cross_table["pearson_p"] = cross_table["pearson_p"].apply(format_pvalue)
        cross_table["spearman_p"] = cross_table["spearman_p"].apply(format_pvalue)
        cross_table = cross_table[["analysis", "group", "metric_x", "metric_y", "n", "pearson_p", "pearson", "spearman_p", "spearman"]]

    source_table = selected_stats.loc[selected_stats["analysis"].isin([
        "LLM performance vs thinking rank",
        "LLM duration vs thinking rank",
        "LLM performance vs duration",
        "LLM duration vs cost",
        "Human performance vs ability",
        "Human performance vs duration",
        "Human performance vs outfit",
        "TRM performance vs step",
        "TRM performance vs oracle",
    ])].copy()
    if not source_table.empty:
        source_table["spearman"] = source_table.apply(lambda row: format_ci_row(row, "spearman"), axis=1)
        source_table["spearman_p"] = source_table["spearman_p"].apply(format_pvalue)
        source_table = source_table[["analysis", "group", "metric_x", "metric_y", "n", "spearman_p", "spearman"]]

    regression_summary = regression_results[["target", "model", "n", "r2", "mae", "pearson_r", "spearman_r"]].copy()
    regression_summary = regression_summary.sort_values(["target", "r2"], ascending=[True, False])

    compare_table = regression_compare[["target", "base_model", "full_model", "n", "delta_r2", "f_statistic", "p_value"]].copy() if not regression_compare.empty else pd.DataFrame()
    if not compare_table.empty:
        compare_table["p_value"] = compare_table["p_value"].apply(format_pvalue)

    pca_table = shared_task_pca_loadings[["feature", "pc1", "pc2", "explained_variance_ratio_pc1", "explained_variance_ratio_pc2"]].copy()
    source_pca_rows: list[dict[str, Any]] = []
    if not pca_loadings.empty:
        for source in pca_loadings["group"].drop_duplicates():
            block = pca_loadings.loc[pca_loadings["group"] == source].copy()
            top = block.sort_values("pc1", key=lambda s: s.abs(), ascending=False).head(2)
            source_pca_rows.append(
                {
                    "source": source,
                    "pc1_variance": float(block["explained_variance_ratio_pc1"].iloc[0]),
                    "pc2_variance": float(block["explained_variance_ratio_pc2"].iloc[0]),
                    "top_pc1_feature": top.iloc[0]["feature"] if not top.empty else "",
                    "top_pc1_loading": float(top.iloc[0]["pc1"]) if not top.empty else np.nan,
                    "second_pc1_feature": top.iloc[1]["feature"] if len(top) > 1 else "",
                    "second_pc1_loading": float(top.iloc[1]["pc1"]) if len(top) > 1 else np.nan,
                }
            )
    source_pca_table = pd.DataFrame(source_pca_rows)

    inventory_latex = latex_table(inventory, "Data inventory for the efficiency analysis.", "tab:inventory")
    cross_latex = latex_table(cross_table, "Cross-source alignment on shared ARC-2 and public-eval overlaps.", "tab:cross_source")
    source_latex = latex_table(source_table, "Within-source efficiency correlations.", "tab:source_corr")
    regression_latex = latex_table(regression_summary, "Cross-validated regression summaries for task-level prediction.", "tab:regression_summary")
    compare_latex = latex_table(compare_table, "Nested OLS comparisons for the key task-level models.", "tab:regression_compare") if not compare_table.empty else ""
    source_pca_latex = latex_table(source_pca_table, "Source-specific PCA summaries.", "tab:source_pca") if not source_pca_table.empty else ""
    pca_latex = latex_table(pca_table, "Shared-task PCA loadings for the efficiency latent space.", "tab:shared_pca")

    lines: list[str] = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=1in]{geometry}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{longtable}")
    lines.append(r"\usepackage{array}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\usepackage{amsmath}")
    lines.append(r"\usepackage{siunitx}")
    lines.append(r"\sisetup{detect-all}")
    lines.append(r"\title{Efficiency comparison across LLM, human, and non-LLM ARC analyses}")
    lines.append(r"\author{Codex analysis}")
    lines.append(r"\date{\today}")
    lines.append(r"\begin{document}")
    lines.append(r"\maketitle")
    lines.append(r"\begin{abstract}")
    lines.append(
        "We compare human, LLM, and non-LLM efficiency signals on ARC-derived data. The strongest cross-source signal is shared task difficulty: human solve rate and mean LLM score correlate moderately, while effort alignment is much weaker. Within LLMs, more thinking budget tracks better performance but also more cost and duration. Within humans, higher ability corresponds to better performance and shorter times. These differences suggest a shared difficulty axis plus source-specific efficiency regimes."
    )
    lines.append(r"\end{abstract}")
    lines.append(r"\section{Data}")
    lines.append("The analysis covers the LLM run logs, human session summaries, public-eval task pairs, and the non-LLM Compress/VARC/TRM runs.")
    lines.append(r"\begin{itemize}")
    for _, row in inventory.iterrows():
        lines.append(
            f"\\item {latex_escape(str(row['source_family']))} / {latex_escape(str(row['subtype']))}: {int(row['rows'])} rows; performance metric {latex_escape(str(row['performance_metric']))}; effort metric {latex_escape(str(row['primary_effort_metric']))}."
        )
    lines.append(r"\end{itemize}")
    lines.append(inventory_latex)
    lines.append(r"\section{Hypotheses}")
    lines.append(r"\begin{itemize}")
    lines.append(r"\item Shared ARC task difficulty should align across humans, LLMs, and TRM.")
    lines.append(r"\item Better performance should trade off against more resource use, but the sign of that tradeoff may differ by source.")
    lines.append(r"\item A small number of latent axes should summarize the efficiency structure.")
    lines.append(r"\item Generic efficiency statistics should still identify the source family.")
    lines.append(r"\end{itemize}")
    lines.append(r"\section{Cross-source alignment}")
    lines.append("There is no direct person-level latent correlation between model theta and human ability in the current data structure, so the shared comparison is task-level alignment.")
    lines.append(cross_latex)
    lines.append(r"\section{Within-source efficiency}")
    lines.append(source_latex)
    lines.append(r"\section{Predictive models}")
    lines.append(regression_latex)
    if compare_latex:
        lines.append(compare_latex)
    lines.append(r"\section{Latent structure}")
    lines.append("Source-specific PCA shows a common efficiency axis within each family. Humans load positively on performance and ability, negatively on time and submissions. LLMs load positively on performance, duration, and cost, which suggests more capability comes with more compute consumption. Non-LLM TRM loads on performance and oracle rate, opposite primary effort.")
    if source_pca_latex:
        lines.append(source_pca_latex)
    lines.append(pca_latex)
    lines.append("The shared-task PCA is more revealing: PC1 is mostly a compute-burden axis driven by LLM duration, cost, and tokens, while PC2 is a human-efficiency axis with human solve rate positive and human duration/submissions negative.")
    lines.append(r"\begin{itemize}")
    for _, row in shared_task_pca_loadings.iterrows():
        lines.append(
            f"\\item {latex_escape(str(row['feature']))}: PC1={float(row['pc1']):.3f}, PC2={float(row['pc2']):.3f}."
        )
    lines.append(r"\end{itemize}")
    lines.append(r"\section{Figures}")
    lines.append(r"\begin{itemize}")
    for path in figure_paths:
        lines.append(f"\\item \\texttt{{{latex_escape(path.as_posix())}}}")
    lines.append(r"\end{itemize}")
    lines.append(r"\includegraphics[width=\linewidth]{figures/source_tradeoff.png}")
    lines.append(r"\includegraphics[width=\linewidth]{figures/task_correlation_heatmap.png}")
    lines.append(r"\includegraphics[width=\linewidth]{figures/shared_task_latent_map.png}")
    lines.append(r"\includegraphics[width=\linewidth]{figures/trm_progression.png}")
    lines.append(r"\section{Negative results}")
    lines.append(r"\begin{itemize}")
    lines.append(r"\item The full-feature source classifier is trivial, because raw telemetry makes the source family obvious.")
    lines.append(r"\item Geometry-only prediction of human solve rate is weak.")
    lines.append(r"\item Adding TRM after LLM score does not materially improve the shared-task model.")
    lines.append(r"\item Human duration is not well predicted by geometry or LLM features.")
    lines.append(r"\item Compress ARC has only one row, so it is descriptive only.")
    lines.append(r"\item VARC has only four rows, so its correlation structure is fragile.")
    lines.append(r"\end{itemize}")
    lines.append(r"\end{document}")
    return "\n".join(lines)


def main() -> int:
    ensure_dirs()

    session_df, item_df, public_eval_df, model_pair_df = load_human_tables()
    llm_run_df, llm_task_df = load_llm_run_summaries()
    human_session_df = load_human_sessions()
    non_llm_df = load_non_llm_records()
    trm_summary = read_json(NON_LLM_PROCESSED / "trm_arc_agi_ii_progression.json")

    shared_task_df = build_shared_task_table(
        public_eval_df,
        llm_task_df.loc[llm_task_df["split"] == "arc_agi_v2_public_eval"].copy(),
        trm_summary,
    )
    shared_task_pca_loadings, shared_task_pca_scores = fit_shared_task_pca(shared_task_df)

    inventory = pd.DataFrame(
        [
            {
                "source_family": "llm",
                "subtype": "model_runs",
                "rows": len(llm_run_df),
                "performance_metric": "task_solve_rate / pair_accuracy",
                "primary_effort_metric": "avg_duration_per_task",
            },
            {
                "source_family": "llm",
                "subtype": "ARC-2 task rollup",
                "rows": len(llm_task_df.loc[llm_task_df["split"] == "arc_agi_v2_public_eval"]),
                "performance_metric": "task_score",
                "primary_effort_metric": "task_duration_seconds",
            },
            {
                "source_family": "human",
                "subtype": "sessions",
                "rows": len(human_session_df),
                "performance_metric": "solve_rate",
                "primary_effort_metric": "mean_duration_seconds",
            },
            {
                "source_family": "human",
                "subtype": "public_eval task pairs",
                "rows": len(public_eval_df),
                "performance_metric": "solve_rate",
                "primary_effort_metric": "mean_duration_seconds",
            },
            {
                "source_family": "non_llm",
                "subtype": "Compress ARC",
                "rows": int((non_llm_df["subtype"] == "compress_arc").sum()),
                "performance_metric": "final pick pass2",
                "primary_effort_metric": "iterations",
            },
            {
                "source_family": "non_llm",
                "subtype": "VARC",
                "rows": int((non_llm_df["subtype"] == "varc").sum()),
                "performance_metric": "pass@1..4 / oracle",
                "primary_effort_metric": "candidate count / attempt dirs",
            },
            {
                "source_family": "non_llm",
                "subtype": "TRM progression steps",
                "rows": int((non_llm_df["subtype"] == "trm").sum()),
                "performance_metric": "pair accuracy / kaggle score",
                "primary_effort_metric": "step",
            },
        ]
    )

    combined_records = pd.concat(
        [
            llm_run_df.assign(primary_effort=llm_run_df["avg_duration_per_task"], label=llm_run_df["model_name"]),
            human_session_df.assign(label=human_session_df["label"]),
            non_llm_df.assign(label=non_llm_df["label"]),
        ],
        ignore_index=True,
        sort=False,
    )
    if "primary_effort" not in combined_records.columns:
        combined_records["primary_effort"] = np.nan

    source_summary = summarize_source(combined_records)

    llm_corr = correlation_block(
        llm_run_df,
        [
            "performance_rate",
            "avg_duration_per_task",
            "avg_total_tokens_per_task",
            "avg_cost_per_task",
            "attempts_per_task",
            "empty_attempt_rate",
            "thinking_rank",
        ],
        "llm",
    )
    human_corr = correlation_block(
        human_session_df,
        [
            "performance_rate",
            "mean_duration_seconds",
            "mean_submissions",
            "tasks_per_minute",
            "ability",
            "outfit",
            "log_tasks_per_minute",
        ],
        "human",
    )
    non_llm_corr = correlation_block(
        non_llm_df,
        [
            "performance_rate",
            "primary_effort",
            "secondary_effort",
            "oracle_rate",
            "final_top1_rate",
            "final_top2_rate",
            "pass1_rate",
            "pass2_rate",
            "pass3_rate",
            "pass4_rate",
            "pair1_rate",
            "pair2_rate",
            "kaggle_score",
        ],
        "non_llm",
    )
    source_correlations = pd.concat([llm_corr, human_corr, non_llm_corr], ignore_index=True)

    pca_loadings, pca_scores = source_specific_pca(combined_records)
    classifier_results = fit_source_classifier(combined_records)
    shared_classifier_results = fit_shared_source_classifier(combined_records)

    geometry_cols = [
        "n_pairs",
        "n_train_pairs",
        "n_test_pairs",
        "input_rows",
        "input_cols",
        "input_cells",
        "input_colors",
        "output_rows",
        "output_cols",
        "output_cells",
        "output_colors",
        "size_change_ratio",
        "human_ability_gap",
        "human_point_biserial",
    ]
    model_sets = [
        ("geometry_only", geometry_cols),
        ("geometry_plus_llm_perf", geometry_cols + ["llm_mean_score", "llm_median_score"]),
        (
            "geometry_plus_llm_perf_effort",
            geometry_cols
            + [
                "llm_mean_score",
                "llm_median_score",
                "llm_mean_duration_seconds",
                "llm_mean_cost",
                "llm_mean_total_tokens",
                "llm_mean_attempts",
            ],
        ),
        (
            "geometry_plus_llm_plus_trm",
            geometry_cols
            + [
                "llm_mean_score",
                "llm_median_score",
                "llm_mean_duration_seconds",
                "llm_mean_cost",
                "llm_mean_total_tokens",
                "llm_mean_attempts",
                "trm_best_task_score",
            ],
        ),
    ]
    regression_human_solve = fit_task_regression(shared_task_df, "human_solve_rate", model_sets)
    regression_human_duration = fit_task_regression(shared_task_df, "human_mean_duration_seconds", model_sets)
    regression_results = pd.concat([regression_human_solve, regression_human_duration], ignore_index=True)

    task_corr_cols = [
        "human_solve_rate",
        "human_mean_duration_seconds",
        "human_mean_submissions",
        "llm_mean_score",
        "llm_median_score",
        "llm_mean_duration_seconds",
        "llm_mean_cost",
        "llm_mean_total_tokens",
        "llm_mean_attempts",
        "trm_best_task_score",
    ]
    task_correlations = correlation_block(shared_task_df, task_corr_cols, "shared_arc2")

    selected_stats_rows: list[dict[str, Any]] = []
    selected_specs = [
        (llm_run_df, "performance_rate", "thinking_rank", "llm", "LLM performance vs thinking rank"),
        (llm_run_df, "avg_duration_per_task", "thinking_rank", "llm", "LLM duration vs thinking rank"),
        (llm_run_df, "performance_rate", "avg_duration_per_task", "llm", "LLM performance vs duration"),
        (llm_run_df, "avg_duration_per_task", "avg_cost_per_task", "llm", "LLM duration vs cost"),
        (human_session_df, "performance_rate", "ability", "human", "Human performance vs ability"),
        (human_session_df, "performance_rate", "mean_duration_seconds", "human", "Human performance vs duration"),
        (human_session_df, "performance_rate", "outfit", "human", "Human performance vs outfit"),
        (shared_task_df, "human_solve_rate", "llm_mean_score", "shared_arc2", "Shared-task human vs LLM score"),
        (shared_task_df, "human_solve_rate", "llm_mean_duration_seconds", "shared_arc2", "Shared-task human vs LLM duration"),
        (shared_task_df, "human_solve_rate", "llm_mean_cost", "shared_arc2", "Shared-task human vs LLM cost"),
        (shared_task_df, "human_solve_rate", "trm_best_task_score", "shared_arc2", "Shared-task human vs TRM score"),
        (shared_task_df, "human_mean_duration_seconds", "llm_mean_duration_seconds", "shared_arc2", "Shared-task human vs LLM duration"),
        (shared_task_df, "human_mean_submissions", "llm_mean_attempts", "shared_arc2", "Shared-task submissions vs LLM attempts"),
        (shared_task_df, "llm_mean_score", "llm_mean_duration_seconds", "shared_arc2", "Shared-task LLM score vs duration"),
        (shared_task_df, "llm_mean_duration_seconds", "llm_mean_cost", "shared_arc2", "Shared-task LLM duration vs cost"),
        (public_eval_df.loc[public_eval_df["attempts"] >= 8].copy(), "solve_rate", "lm_mean", "human_public_eval_overlap", "Public-eval human vs average model"),
        (public_eval_df.loc[public_eval_df["attempts"] >= 8].copy(), "solve_rate", "lm_best_single_model", "human_public_eval_overlap", "Public-eval human vs best single model"),
        (non_llm_df.loc[non_llm_df["subtype"] == "trm"].copy(), "performance_rate", "primary_effort", "non_llm/trm", "TRM performance vs step"),
        (non_llm_df.loc[non_llm_df["subtype"] == "trm"].copy(), "performance_rate", "oracle_rate", "non_llm/trm", "TRM performance vs oracle"),
    ]
    for frame, x_col, y_col, group, analysis in selected_specs:
        selected_stats_rows.append(
            summarize_selected_correlation(frame, x_col, y_col, analysis=analysis, group=group, n_boot=3000, seed=0)
        )
    selected_stats = pd.DataFrame(selected_stats_rows)

    regression_ols_rows: list[pd.DataFrame] = []
    regression_compare_rows: list[pd.DataFrame] = []
    for target in ["human_solve_rate", "human_mean_duration_seconds"]:
        model_rows, compare_rows = fit_nested_ols_models(shared_task_df, target, model_sets)
        if not model_rows.empty:
            regression_ols_rows.append(model_rows)
        if not compare_rows.empty:
            regression_compare_rows.append(compare_rows)
    regression_ols = pd.concat(regression_ols_rows, ignore_index=True) if regression_ols_rows else pd.DataFrame()
    regression_compare = pd.concat(regression_compare_rows, ignore_index=True) if regression_compare_rows else pd.DataFrame()

    figure_paths = [
        plot_source_tradeoff(combined_records),
        plot_task_correlations(shared_task_df),
        plot_shared_task_pca(shared_task_df, shared_task_pca_scores),
        plot_trm_progression(trm_summary),
    ]

    tables = {
        "inventory.csv": inventory,
        "source_summary.csv": source_summary,
        "source_correlations.csv": source_correlations,
        "selected_statistics.csv": selected_stats,
        "llm_runs.csv": llm_run_df,
        "llm_arc2_task_rollup.csv": llm_task_df.loc[llm_task_df["split"] == "arc_agi_v2_public_eval"].copy(),
        "human_sessions.csv": human_session_df,
        "human_public_eval_pairs.csv": public_eval_df,
        "non_llm_runs.csv": non_llm_df,
        "shared_task_arc2.csv": shared_task_df,
        "task_correlations.csv": task_correlations,
        "task_regressions.csv": regression_results,
        "task_regression_ols.csv": regression_ols,
        "task_regression_compare.csv": regression_compare,
        "pca_loadings.csv": pca_loadings,
        "shared_task_pca_loadings.csv": shared_task_pca_loadings,
        "shared_task_pca_scores.csv": shared_task_pca_scores,
    }
    for filename, frame in tables.items():
        frame.to_csv(TABLES_DIR / filename, index=False)

    results_payload = {
        "source_classifier": classifier_results,
        "shared_source_classifier": shared_classifier_results,
        "task_regression": regression_results.to_dict(orient="records"),
        "task_regression_ols": regression_ols.to_dict(orient="records"),
        "task_regression_compare": regression_compare.to_dict(orient="records"),
        "selected_statistics": selected_stats.to_dict(orient="records"),
        "shared_task_count": int(len(shared_task_df)),
        "record_counts": {
            "llm_models": int(len(llm_run_df)),
            "human_sessions": int(len(human_session_df)),
            "non_llm_runs": int(len(non_llm_df)),
        },
        "source_summary_rows": int(len(source_summary)),
        "shared_task_pca": {
            "explained_variance_ratio_pc1": float(shared_task_pca_loadings["explained_variance_ratio_pc1"].iloc[0]) if not shared_task_pca_loadings.empty else np.nan,
            "explained_variance_ratio_pc2": float(shared_task_pca_loadings["explained_variance_ratio_pc2"].iloc[0]) if not shared_task_pca_loadings.empty else np.nan,
        },
    }
    (OUT_DIR / "analysis_results.json").write_text(json.dumps(results_payload, indent=2, default=json_default), encoding="utf-8")

    report = build_report(
        inventory=inventory,
        source_summary=source_summary,
        source_correlations=source_correlations,
        task_correlations=task_correlations,
        classifier_results=classifier_results,
        shared_classifier_results=shared_classifier_results,
        regression_results=regression_results,
        regression_ols=regression_ols,
        regression_compare=regression_compare,
        selected_stats=selected_stats,
        pca_loadings=pca_loadings,
        shared_task_pca_loadings=shared_task_pca_loadings,
        shared_task_df=shared_task_df,
        llm_run_df=llm_run_df,
        human_session_df=human_session_df,
        non_llm_df=non_llm_df,
        figure_paths=figure_paths,
    )
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    latex_report = build_latex_report(
        inventory=inventory,
        source_summary=source_summary,
        regression_results=regression_results,
        regression_compare=regression_compare,
        selected_stats=selected_stats,
        pca_loadings=pca_loadings,
        shared_task_pca_loadings=shared_task_pca_loadings,
        figure_paths=figure_paths,
    )
    (OUT_DIR / "paper.tex").write_text(latex_report, encoding="utf-8")

    print("Efficiency analysis complete")
    print(f"- record rows: {len(combined_records)}")
    print(f"- shared ARC-2 tasks: {len(shared_task_df)}")
    print(
        f"- source classifier balanced accuracy: full={classifier_results['balanced_accuracy']:.3f}, "
        f"shared={shared_classifier_results['balanced_accuracy']:.3f}"
    )
    if not selected_stats.empty:
        key = selected_stats.loc[
            (selected_stats["analysis"] == "Shared-task human vs LLM score")
            & (selected_stats["metric_x"] == "human_solve_rate")
            & (selected_stats["metric_y"] == "llm_mean_score")
        ]
        if not key.empty:
            row = key.iloc[0]
            print(
                f"- latest shared-task human-vs-LLM alignment: spearman={row['spearman_r']:.3f} "
                f"(p={row['spearman_p']:.3g})"
            )
    top_reg = regression_results.sort_values(["target", "r2"], ascending=[True, False]).groupby("target").head(1)
    for _, row in top_reg.iterrows():
        print(f"- best {row['target']} model: {row['model']} (R2={row['r2']:.3f}, MAE={row['mae']:.3f})")
    print(f"- report: {OUT_DIR / 'report.md'}")
    print(f"- latex: {OUT_DIR / 'paper.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
