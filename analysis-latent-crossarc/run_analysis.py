from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"
REPORT_PATH = BASE_DIR / "report.md"
SUMMARY_PATH = BASE_DIR / "summary.json"

HUMAN_RAW_CSV = ROOT_DIR / "data-human" / "test_pair_attempts.csv"
ARC1_TRAIN_DIR = ROOT_DIR / "data-llm" / "ARC-AGI" / "data" / "training"
ARC1_EVAL_DIR = ROOT_DIR / "data-llm" / "ARC-AGI" / "data" / "evaluation"
ARC2_TRAIN_DIR = ROOT_DIR / "data-llm" / "ARC-AGI-2" / "data" / "training"
ARC2_EVAL_DIR = ROOT_DIR / "data-llm" / "ARC-AGI-2" / "data" / "evaluation"

ARC1_LLM_MATRIX = ROOT_DIR / "analysis-python-complexity" / "llm_response_matrix_arc_agi_1_eval.csv"
ARC2_LLM_MATRIX = ROOT_DIR / "analysis-python-complexity" / "llm_response_matrix_arc_agi_2_eval.csv"
COMPLEXITY_REPORT = ROOT_DIR / "analysis-python-complexity" / "complexity_report.csv"

N_SPLIT_HALF = 120
BOOTSTRAP_REPS = 2000
RNG_SEED = 0


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def canonical_json_hash(obj: object) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(canonical).hexdigest()


def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    if pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(pair["x"], pair["y"]).statistic)
    return float(pearsonr(pair["x"], pair["y"]).statistic)


def percentile_interval(values: pd.Series | np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(arr, alpha / 2.0)), float(np.quantile(arr, 1.0 - alpha / 2.0))


def safe_logit(p: pd.Series | np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    arr = np.clip(arr, eps, 1.0 - eps)
    return np.log(arr / (1.0 - arr))


def normalize_grid_dimensions(grid: list[list[int]]) -> tuple[int, int, int]:
    if not grid:
        return 0, 0, 0
    h = len(grid)
    w = len(grid[0]) if grid[0] else 0
    return h, w, h * w


def summarize_task_json(task_obj: dict) -> dict[str, float]:
    all_pairs = list(task_obj.get("train", [])) + list(task_obj.get("test", []))
    input_cells: list[int] = []
    output_cells: list[int] = []

    for pair in all_pairs:
        _, _, input_count = normalize_grid_dimensions(pair.get("input", []))
        input_cells.append(input_count)
    for pair in task_obj.get("train", []) + [p for p in task_obj.get("test", []) if "output" in p]:
        _, _, output_count = normalize_grid_dimensions(pair.get("output", []))
        output_cells.append(output_count)

    return {
        "n_train_pairs": len(task_obj.get("train", [])),
        "n_test_pairs": len(task_obj.get("test", [])),
        "mean_input_cells": float(np.mean(input_cells)) if input_cells else float("nan"),
        "max_input_cells": float(np.max(input_cells)) if input_cells else float("nan"),
        "mean_output_cells": float(np.mean(output_cells)) if output_cells else float("nan"),
        "max_output_cells": float(np.max(output_cells)) if output_cells else float("nan"),
    }


def load_benchmark_metadata(task_dir: Path, benchmark: str, split: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task_path in sorted(task_dir.glob("*.json")):
        task_obj = json.loads(task_path.read_text(encoding="utf-8"))
        row = {
            "task_id": task_path.stem,
            "benchmark": benchmark,
            "split": split,
            "json_hash": canonical_json_hash(task_obj),
        }
        row.update(summarize_task_json(task_obj))
        rows.append(row)
    return pd.DataFrame(rows)


def build_arc_metadata_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arc_frames = [
        load_benchmark_metadata(ARC1_TRAIN_DIR, "arc1", "train"),
        load_benchmark_metadata(ARC1_EVAL_DIR, "arc1", "eval"),
        load_benchmark_metadata(ARC2_TRAIN_DIR, "arc2", "train"),
        load_benchmark_metadata(ARC2_EVAL_DIR, "arc2", "eval"),
    ]
    task_metadata = pd.concat(arc_frames, ignore_index=True)
    task_metadata.to_csv(TABLES_DIR / "arc_task_metadata.csv", index=False)

    partition_summary = (
        task_metadata.groupby(["benchmark", "split"], as_index=False)
        .agg(
            task_count=("task_id", "nunique"),
            mean_input_cells=("mean_input_cells", "mean"),
            median_input_cells=("mean_input_cells", "median"),
            mean_output_cells=("mean_output_cells", "mean"),
            mean_test_pairs=("n_test_pairs", "mean"),
        )
        .sort_values(["benchmark", "split"])
    )
    partition_summary.to_csv(TABLES_DIR / "arc_partition_summary.csv", index=False)

    split_map = {
        "arc1_train": task_metadata.query("benchmark == 'arc1' and split == 'train'"),
        "arc1_eval": task_metadata.query("benchmark == 'arc1' and split == 'eval'"),
        "arc2_train": task_metadata.query("benchmark == 'arc2' and split == 'train'"),
        "arc2_eval": task_metadata.query("benchmark == 'arc2' and split == 'eval'"),
    }

    overlap_rows: list[dict[str, object]] = []
    split_items = list(split_map.items())
    for idx, (left_name, left_df) in enumerate(split_items):
        left_ids = set(left_df["task_id"])
        for right_name, right_df in split_items[idx + 1 :]:
            right_ids = set(right_df["task_id"])
            shared_ids = sorted(left_ids & right_ids)
            overlap_rows.append(
                {
                    "left_partition": left_name,
                    "right_partition": right_name,
                    "shared_task_count": len(shared_ids),
                }
            )
    overlap_summary = pd.DataFrame(overlap_rows)
    overlap_summary.to_csv(TABLES_DIR / "arc_overlap_summary.csv", index=False)

    shared_eval = split_map["arc1_eval"].merge(
        split_map["arc2_eval"],
        on="task_id",
        suffixes=("_arc1", "_arc2"),
    )
    shared_eval_rows: list[dict[str, object]] = []
    for task_id in shared_eval["task_id"].tolist():
        arc1_obj = json.loads((ARC1_EVAL_DIR / f"{task_id}.json").read_text(encoding="utf-8"))
        arc2_obj = json.loads((ARC2_EVAL_DIR / f"{task_id}.json").read_text(encoding="utf-8"))
        shared_eval_rows.append(
            {
                "task_id": task_id,
                "same_json": arc1_obj == arc2_obj,
                "same_train_examples": arc1_obj.get("train", []) == arc2_obj.get("train", []),
                "same_test_examples": arc1_obj.get("test", []) == arc2_obj.get("test", []),
                "arc1_train_pairs": len(arc1_obj.get("train", [])),
                "arc2_train_pairs": len(arc2_obj.get("train", [])),
                "arc1_test_pairs": len(arc1_obj.get("test", [])),
                "arc2_test_pairs": len(arc2_obj.get("test", [])),
            }
        )
    shared_eval_summary = pd.DataFrame(shared_eval_rows)
    shared_eval_summary.to_csv(TABLES_DIR / "shared_eval_task_variants.csv", index=False)

    return task_metadata, partition_summary, shared_eval_summary


def load_human_attempts() -> pd.DataFrame:
    human = pd.read_csv(HUMAN_RAW_CSV)
    human["task_pair_id"] = human["task_ID"] + "__" + human["test_index"].astype(str)
    human["solved"] = (human["correct_submissions"] > 0).astype(int)
    human["is_public_eval"] = (human["task_set"] == "Public Eval").astype(int)
    human["log_duration"] = np.log1p(human["duration_seconds"].clip(lower=0))

    arc1_single_pair_ids = {
        f"{task_path.stem}__0"
        for task_path in ARC1_EVAL_DIR.glob("*.json")
        if len(json.loads(task_path.read_text(encoding="utf-8")).get("test", [])) == 1
    }
    human["benchmark_label"] = np.where(
        human["task_set"].eq("Public Eval"),
        "arc2_eval",
        np.where(human["task_pair_id"].isin(arc1_single_pair_ids), "arc1_sidecar", "arc2_train_other"),
    )
    human["dataset_key"] = human["benchmark_label"].map(
        {
            "arc1_sidecar": "arc_agi_1_eval",
            "arc2_eval": "arc_agi_2_eval",
            "arc2_train_other": "arc_agi_2_train",
        }
    )
    return human


def build_sparse_design(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    encoder: OneHotEncoder | None = None,
) -> tuple[sparse.csr_matrix, OneHotEncoder, pd.Index]:
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
        cat_matrix = encoder.fit_transform(df[categorical_cols])
    else:
        cat_matrix = encoder.transform(df[categorical_cols])
    numeric_matrix = sparse.csr_matrix(df[numeric_cols].astype(float).to_numpy())
    design = sparse.hstack([cat_matrix, numeric_matrix], format="csr")
    feature_names = pd.Index(list(encoder.get_feature_names_out(categorical_cols)) + numeric_cols)
    return design, encoder, feature_names


def extract_effect_series(coef: pd.Series, prefix: str) -> pd.Series:
    keep = coef.index.str.startswith(prefix)
    effect = coef.loc[keep].copy()
    effect.index = effect.index.str.replace(prefix, "", regex=False)
    return effect


def load_llm_eval_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    arc1 = pd.read_csv(ARC1_LLM_MATRIX, index_col=0)
    arc2 = pd.read_csv(ARC2_LLM_MATRIX, index_col=0)
    arc1 = arc1.loc[[not str(idx).startswith("_") for idx in arc1.index]].sort_index()
    arc2 = arc2.loc[[not str(idx).startswith("_") for idx in arc2.index]].sort_index()
    return arc1.astype(float), arc2.astype(float)


def fit_human_correctness_task_summary(human: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = ["session_ID", "task_ID", "task_pair_id"]
    numeric_cols = ["is_public_eval", "test_index"]

    design, _, feature_names = build_sparse_design(human, categorical_cols, numeric_cols)
    y = human["solved"].to_numpy()

    model = LogisticRegression(
        C=1.0,
        solver="saga",
        max_iter=2000,
        fit_intercept=True,
        random_state=RNG_SEED,
    )
    model.fit(design, y)

    coef = pd.Series(model.coef_[0], index=feature_names)
    task_effect = extract_effect_series(coef, "task_ID_")
    pair_effect = extract_effect_series(coef, "task_pair_id_")

    pair_meta = human[["task_ID", "task_pair_id", "test_index", "is_public_eval", "benchmark_label"]].drop_duplicates(
        "task_pair_id"
    )
    pair_meta = pair_meta.copy()
    pair_meta["latent_correct_logit"] = (
        float(model.intercept_[0])
        + pair_meta["task_ID"].map(task_effect).fillna(0.0)
        + pair_meta["task_pair_id"].map(pair_effect).fillna(0.0)
        + coef["is_public_eval"] * pair_meta["is_public_eval"]
        + coef["test_index"] * pair_meta["test_index"]
    )
    pair_meta["latent_correct_prob"] = 1.0 / (1.0 + np.exp(-pair_meta["latent_correct_logit"]))

    task_summary = (
        pair_meta.groupby("task_ID", as_index=False)
        .agg(
            benchmark_label=("benchmark_label", lambda s: s.mode().iloc[0]),
            latent_correct_logit=("latent_correct_logit", "mean"),
            latent_correct_prob=("latent_correct_prob", "mean"),
        )
        .sort_values(["benchmark_label", "task_ID"])
    )
    return task_summary


def fit_full_human_latent_tables(human: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    categorical_cols = ["session_ID", "task_ID", "task_pair_id"]
    numeric_cols = ["is_public_eval", "test_index"]
    design, _, feature_names = build_sparse_design(human, categorical_cols, numeric_cols)

    y_correct = human["solved"].to_numpy()
    correct_model = LogisticRegression(
        C=1.0,
        solver="saga",
        max_iter=2000,
        fit_intercept=True,
        random_state=RNG_SEED,
    )
    correct_model.fit(design, y_correct)
    correct_coef = pd.Series(correct_model.coef_[0], index=feature_names)

    y_duration = human["log_duration"].to_numpy()
    duration_model = Ridge(alpha=1.0)
    duration_model.fit(design, y_duration)
    duration_coef = pd.Series(duration_model.coef_, index=feature_names)

    predicted_prob = correct_model.predict_proba(design)[:, 1]
    predicted_duration = duration_model.predict(design)

    diagnostics = pd.DataFrame(
        [
            {
                "auc": float(roc_auc_score(y_correct, predicted_prob)),
                "log_loss": float(log_loss(y_correct, predicted_prob)),
                "brier": float(brier_score_loss(y_correct, predicted_prob)),
                "duration_r2": float(r2_score(y_duration, predicted_duration)),
                "duration_mae_log": float(mean_absolute_error(y_duration, predicted_duration)),
                "n_rows": int(len(human)),
                "n_sessions": int(human["session_ID"].nunique()),
                "n_tasks": int(human["task_ID"].nunique()),
                "n_task_pairs": int(human["task_pair_id"].nunique()),
            }
        ]
    )
    diagnostics.to_csv(TABLES_DIR / "human_model_diagnostics.csv", index=False)

    session_summary = (
        human.groupby("session_ID", as_index=False)
        .agg(
            attempts=("solved", "size"),
            solve_rate=("solved", "mean"),
            mean_duration_seconds=("duration_seconds", "mean"),
            task_count=("task_ID", "nunique"),
        )
        .sort_values("attempts", ascending=False)
    )
    session_summary["latent_session_ability"] = session_summary["session_ID"].map(
        extract_effect_series(correct_coef, "session_ID_")
    )
    session_summary.to_csv(TABLES_DIR / "human_session_latent_summary.csv", index=False)

    pair_summary = (
        human.groupby("task_pair_id", as_index=False)
        .agg(
            task_ID=("task_ID", "first"),
            benchmark_label=("benchmark_label", "first"),
            dataset_key=("dataset_key", "first"),
            test_index=("test_index", "first"),
            attempts=("solved", "size"),
            session_count=("session_ID", "nunique"),
            raw_solve_rate=("solved", "mean"),
            mean_duration_seconds=("duration_seconds", "mean"),
            median_duration_seconds=("duration_seconds", "median"),
            mean_submissions=("submissions", "mean"),
        )
        .sort_values(["benchmark_label", "task_ID", "test_index"])
    )
    pair_meta = pair_summary[["task_pair_id", "task_ID", "test_index"]].merge(
        human[["task_pair_id", "is_public_eval"]].drop_duplicates("task_pair_id"),
        on="task_pair_id",
        how="left",
    )
    pair_meta["latent_correct_logit"] = (
        float(correct_model.intercept_[0])
        + pair_meta["task_ID"].map(extract_effect_series(correct_coef, "task_ID_")).fillna(0.0)
        + pair_meta["task_pair_id"].map(extract_effect_series(correct_coef, "task_pair_id_")).fillna(0.0)
        + correct_coef["is_public_eval"] * pair_meta["is_public_eval"]
        + correct_coef["test_index"] * pair_meta["test_index"]
    )
    pair_meta["latent_correct_prob"] = 1.0 / (1.0 + np.exp(-pair_meta["latent_correct_logit"]))
    pair_meta["latent_duration_log"] = (
        float(duration_model.intercept_)
        + pair_meta["task_ID"].map(extract_effect_series(duration_coef, "task_ID_")).fillna(0.0)
        + pair_meta["task_pair_id"].map(extract_effect_series(duration_coef, "task_pair_id_")).fillna(0.0)
        + duration_coef["is_public_eval"] * pair_meta["is_public_eval"]
        + duration_coef["test_index"] * pair_meta["test_index"]
    )
    pair_meta["latent_duration_seconds"] = np.expm1(pair_meta["latent_duration_log"])

    pair_summary = pair_summary.merge(
        pair_meta[
            ["task_pair_id", "latent_correct_logit", "latent_correct_prob", "latent_duration_log", "latent_duration_seconds"]
        ],
        on="task_pair_id",
        how="left",
    )
    pair_summary.to_csv(TABLES_DIR / "human_pair_latent_summary.csv", index=False)

    task_summary = (
        pair_summary.groupby("task_ID", as_index=False)
        .agg(
            benchmark_label=("benchmark_label", lambda s: s.mode().iloc[0]),
            dataset_key=("dataset_key", lambda s: s.mode().iloc[0]),
            pair_count=("task_pair_id", "size"),
            attempts_total=("attempts", "sum"),
            raw_solve_rate=("raw_solve_rate", "mean"),
            mean_duration_seconds=("mean_duration_seconds", "mean"),
            mean_submissions=("mean_submissions", "mean"),
            latent_correct_logit=("latent_correct_logit", "mean"),
            latent_correct_prob=("latent_correct_prob", "mean"),
            latent_duration_log=("latent_duration_log", "mean"),
            latent_duration_seconds=("latent_duration_seconds", "mean"),
            pair_difficulty_sd=("latent_correct_logit", "std"),
            pair_prob_range=("latent_correct_prob", lambda s: float(s.max() - s.min())),
        )
        .fillna({"pair_difficulty_sd": 0.0, "pair_prob_range": 0.0})
        .sort_values(["benchmark_label", "task_ID"])
    )
    task_coverage = human.groupby("task_ID", as_index=False).agg(
        session_count=("session_ID", "nunique"),
        row_count=("solved", "size"),
    )
    task_summary = task_summary.merge(task_coverage, on="task_ID", how="left")
    task_summary["latent_human_difficulty"] = -task_summary["latent_correct_logit"]

    human_pca_input = task_summary[["latent_correct_prob", "latent_duration_log"]].copy()
    scaled = StandardScaler().fit_transform(
        np.column_stack([human_pca_input["latent_correct_prob"], -human_pca_input["latent_duration_log"]])
    )
    pca = PCA(n_components=1, random_state=RNG_SEED)
    human_ease_pc1 = pca.fit_transform(scaled)[:, 0]
    if safe_corr(pd.Series(human_ease_pc1), task_summary["latent_correct_prob"]) < 0:
        human_ease_pc1 = -human_ease_pc1
    task_summary["human_ease_pc1"] = human_ease_pc1
    task_summary["human_difficulty_pc1"] = -human_ease_pc1
    task_summary.to_csv(TABLES_DIR / "human_task_latent_summary.csv", index=False)

    benchmark_summary = (
        human.groupby("benchmark_label", as_index=False)
        .agg(
            row_count=("solved", "size"),
            task_count=("task_ID", "nunique"),
            pair_count=("task_pair_id", "nunique"),
            session_count=("session_ID", "nunique"),
            raw_solve_rate=("solved", "mean"),
            mean_duration_seconds=("duration_seconds", "mean"),
        )
        .sort_values("benchmark_label")
    )
    benchmark_summary = benchmark_summary.merge(
        task_summary.groupby("benchmark_label", as_index=False)
        .agg(
            mean_latent_human_difficulty=("latent_human_difficulty", "mean"),
            mean_human_difficulty_pc1=("human_difficulty_pc1", "mean"),
        ),
        on="benchmark_label",
        how="left",
    )
    benchmark_summary.to_csv(TABLES_DIR / "human_benchmark_summary.csv", index=False)

    return diagnostics, pair_summary, task_summary


def run_human_split_half(human: pd.DataFrame, n_sims: int = N_SPLIT_HALF) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    sessions = np.array(sorted(human["session_ID"].unique()))
    rows: list[dict[str, float | int | str]] = []

    for draw_id in range(n_sims):
        shuffled = rng.permutation(sessions)
        midpoint = len(shuffled) // 2
        session_halves = [set(shuffled[:midpoint]), set(shuffled[midpoint:])]
        half_tables: list[pd.DataFrame] = []
        for half_idx, session_ids in enumerate(session_halves):
            sub = human.loc[human["session_ID"].isin(session_ids)].copy()
            raw_task = (
                sub.groupby(["benchmark_label", "task_ID"], as_index=False)
                .agg(attempts=("solved", "size"), raw_solve_rate=("solved", "mean"))
                .sort_values(["benchmark_label", "task_ID"])
            )
            latent_task = fit_human_correctness_task_summary(sub)
            merged = raw_task.merge(
                latent_task[["task_ID", "latent_correct_logit"]],
                on="task_ID",
                how="left",
            )
            merged["half"] = half_idx
            half_tables.append(merged)

        for benchmark_label in sorted(human["benchmark_label"].unique()):
            left = half_tables[0].query("benchmark_label == @benchmark_label").set_index("task_ID")
            right = half_tables[1].query("benchmark_label == @benchmark_label").set_index("task_ID")
            joined = left.join(right, lsuffix="_a", rsuffix="_b", how="inner")
            joined = joined.loc[(joined["attempts_a"] >= 2) & (joined["attempts_b"] >= 2)]
            if len(joined) < 20:
                continue
            rows.append(
                {
                    "draw_id": draw_id,
                    "benchmark_label": benchmark_label,
                    "n_tasks": int(len(joined)),
                    "raw_task_correlation": safe_corr(joined["raw_solve_rate_a"], joined["raw_solve_rate_b"]),
                    "latent_task_correlation": safe_corr(
                        joined["latent_correct_logit_a"], joined["latent_correct_logit_b"]
                    ),
                }
            )

    draw_df = pd.DataFrame(rows)
    draw_df.to_csv(TABLES_DIR / "human_split_half_draws.csv", index=False)

    summary = (
        draw_df.groupby("benchmark_label", as_index=False)
        .agg(
            completed_draws=("draw_id", "nunique"),
            mean_task_count=("n_tasks", "mean"),
            raw_corr_mean=("raw_task_correlation", "mean"),
            raw_corr_median=("raw_task_correlation", "median"),
            latent_corr_mean=("latent_task_correlation", "mean"),
            latent_corr_median=("latent_task_correlation", "median"),
        )
        .sort_values("benchmark_label")
    )
    summary["raw_corr_ci_lo"], summary["raw_corr_ci_hi"] = zip(
        *draw_df.groupby("benchmark_label")["raw_task_correlation"].apply(percentile_interval).tolist()
    )
    summary["latent_corr_ci_lo"], summary["latent_corr_ci_hi"] = zip(
        *draw_df.groupby("benchmark_label")["latent_task_correlation"].apply(percentile_interval).tolist()
    )
    summary["latent_minus_raw"] = summary["latent_corr_mean"] - summary["raw_corr_mean"]
    summary.to_csv(TABLES_DIR / "human_split_half_summary.csv", index=False)
    return draw_df, summary


def build_llm_latent_tables(arc1: pd.DataFrame, arc2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_models = sorted(set(arc1.index) & set(arc2.index))
    common_arc1 = arc1.loc[common_models].copy()
    common_arc2 = arc2.loc[common_models].copy()

    model_summary = pd.DataFrame(
        {
            "model_name": common_models,
            "arc1_accuracy": common_arc1.mean(axis=1).values,
            "arc2_accuracy": common_arc2.mean(axis=1).values,
        }
    )
    model_summary["accuracy_diff_arc2_minus_arc1"] = model_summary["arc2_accuracy"] - model_summary["arc1_accuracy"]
    model_summary["arc1_rank"] = model_summary["arc1_accuracy"].rank(ascending=False, method="min")
    model_summary["arc2_rank"] = model_summary["arc2_accuracy"].rank(ascending=False, method="min")

    long_frames: list[pd.DataFrame] = []
    for benchmark_label, matrix in [("arc1_eval", common_arc1), ("arc2_eval", common_arc2)]:
        long = matrix.stack().rename("solved").reset_index()
        long.columns = ["model_name", "task_id", "solved"]
        long["benchmark_label"] = benchmark_label
        long_frames.append(long)
    llm_long = pd.concat(long_frames, ignore_index=True)

    design, _, feature_names = build_sparse_design(llm_long, ["model_name", "task_id"], [])
    llm_model = LogisticRegression(
        C=2.0,
        solver="saga",
        max_iter=4000,
        fit_intercept=True,
        random_state=RNG_SEED,
    )
    llm_model.fit(design, llm_long["solved"].to_numpy())

    coef = pd.Series(llm_model.coef_[0], index=feature_names)
    ability = extract_effect_series(coef, "model_name_")
    difficulty = -(extract_effect_series(coef, "task_id_") - extract_effect_series(coef, "task_id_").mean())
    model_summary["latent_model_ability"] = model_summary["model_name"].map(ability)
    model_summary.to_csv(TABLES_DIR / "llm_common_model_summary.csv", index=False)

    task_summary = pd.DataFrame({"task_id": difficulty.index, "llm_latent_difficulty": difficulty.values})
    task_summary["in_arc1_eval"] = task_summary["task_id"].isin(common_arc1.columns)
    task_summary["in_arc2_eval"] = task_summary["task_id"].isin(common_arc2.columns)
    task_summary["arc1_pass_rate"] = task_summary["task_id"].map(common_arc1.mean(axis=0))
    task_summary["arc2_pass_rate"] = task_summary["task_id"].map(common_arc2.mean(axis=0))
    task_summary["benchmark_membership"] = task_summary.apply(
        lambda row: "|".join(
            membership
            for membership, keep in [("arc1_eval", row["in_arc1_eval"]), ("arc2_eval", row["in_arc2_eval"])]
            if keep
        ),
        axis=1,
    )
    task_summary.to_csv(TABLES_DIR / "llm_common_task_summary.csv", index=False)

    benchmark_summary = pd.DataFrame(
        [
            {
                "benchmark_label": "arc1_eval",
                "task_count": int(common_arc1.shape[1]),
                "common_model_count": int(common_arc1.shape[0]),
                "mean_pass_rate": float(common_arc1.values.mean()),
                "mean_task_difficulty": float(task_summary.loc[task_summary["in_arc1_eval"], "llm_latent_difficulty"].mean()),
            },
            {
                "benchmark_label": "arc2_eval",
                "task_count": int(common_arc2.shape[1]),
                "common_model_count": int(common_arc2.shape[0]),
                "mean_pass_rate": float(common_arc2.values.mean()),
                "mean_task_difficulty": float(task_summary.loc[task_summary["in_arc2_eval"], "llm_latent_difficulty"].mean()),
            },
        ]
    )
    benchmark_summary.to_csv(TABLES_DIR / "llm_benchmark_summary.csv", index=False)
    return model_summary, task_summary, benchmark_summary


def build_human_llm_alignment(task_summary: pd.DataFrame, llm_task_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    llm_lookup = llm_task_summary.set_index("task_id")
    alignment_rows: list[dict[str, object]] = []
    matched_frames: list[pd.DataFrame] = []

    for benchmark_label, llm_rate_col in [("arc1_sidecar", "arc1_pass_rate"), ("arc2_eval", "arc2_pass_rate")]:
        human_subset = task_summary.loc[task_summary["benchmark_label"] == benchmark_label].copy()
        matched = human_subset.merge(
            llm_lookup[["llm_latent_difficulty", llm_rate_col]],
            left_on="task_ID",
            right_index=True,
            how="inner",
        )
        matched["benchmark_label"] = benchmark_label
        matched["llm_pass_rate"] = matched[llm_rate_col]
        matched_frames.append(matched)
        alignment_rows.append(
            {
                "benchmark_label": benchmark_label,
                "matched_task_count": int(len(matched)),
                "rawsolve_vs_llm_difficulty_pearson": safe_corr(-matched["raw_solve_rate"], matched["llm_latent_difficulty"]),
                "latent_vs_llm_difficulty_pearson": safe_corr(
                    matched["latent_human_difficulty"], matched["llm_latent_difficulty"]
                ),
                "pc1_vs_llm_difficulty_pearson": safe_corr(
                    matched["human_difficulty_pc1"], matched["llm_latent_difficulty"]
                ),
                "latent_vs_llm_difficulty_spearman": safe_corr(
                    matched["latent_human_difficulty"], matched["llm_latent_difficulty"], method="spearman"
                ),
                "duration_vs_llm_difficulty_pearson": safe_corr(
                    matched["latent_duration_log"], matched["llm_latent_difficulty"]
                ),
            }
        )

    matched_all = pd.concat(matched_frames, ignore_index=True)
    matched_all.to_csv(TABLES_DIR / "human_llm_alignment_tasks.csv", index=False)
    alignment_summary = pd.DataFrame(alignment_rows)
    alignment_summary.to_csv(TABLES_DIR / "human_llm_alignment_summary.csv", index=False)

    pooled_rows: list[dict[str, object]] = []
    for human_metric in ["raw_solve_rate", "latent_human_difficulty", "human_difficulty_pc1", "latent_duration_log"]:
        pooled = matched_all[[human_metric, "llm_latent_difficulty", "benchmark_label"]].dropna().copy()
        pooled["human_centered"] = pooled[human_metric] - pooled.groupby("benchmark_label")[human_metric].transform("mean")
        pooled["llm_centered"] = (
            pooled["llm_latent_difficulty"] - pooled.groupby("benchmark_label")["llm_latent_difficulty"].transform("mean")
        )
        pooled["human_z"] = pooled.groupby("benchmark_label")[human_metric].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0)
        )
        pooled["llm_z"] = pooled.groupby("benchmark_label")["llm_latent_difficulty"].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0)
        )
        pooled_rows.extend(
            [
                {
                    "human_metric": human_metric,
                    "pooling": "naive_concat",
                    "n": int(len(pooled)),
                    "pearson_r": safe_corr(pooled[human_metric], pooled["llm_latent_difficulty"]),
                    "spearman_r": safe_corr(pooled[human_metric], pooled["llm_latent_difficulty"], method="spearman"),
                },
                {
                    "human_metric": human_metric,
                    "pooling": "benchmark_centered",
                    "n": int(len(pooled)),
                    "pearson_r": safe_corr(pooled["human_centered"], pooled["llm_centered"]),
                    "spearman_r": safe_corr(pooled["human_centered"], pooled["llm_centered"], method="spearman"),
                },
                {
                    "human_metric": human_metric,
                    "pooling": "benchmark_z",
                    "n": int(len(pooled)),
                    "pearson_r": safe_corr(pooled["human_z"], pooled["llm_z"]),
                    "spearman_r": safe_corr(pooled["human_z"], pooled["llm_z"], method="spearman"),
                },
            ]
        )
    pd.DataFrame(pooled_rows).to_csv(TABLES_DIR / "human_llm_pooled_summary.csv", index=False)
    return matched_all, alignment_summary


def expand_complexity_report() -> pd.DataFrame:
    complexity = pd.read_csv(COMPLEXITY_REPORT)
    rows: list[dict[str, object]] = []
    for _, row in complexity.iterrows():
        memberships = str(row["dataset_membership"]).split("|")
        for dataset_key in memberships:
            expanded = row.to_dict()
            expanded["dataset_key"] = dataset_key
            rows.append(expanded)
    expanded_df = pd.DataFrame(rows)

    structure_cols = [
        "nonblank_lines",
        "token_count",
        "ast_node_count",
        "branch_node_count",
        "cyclomatic_complexity",
        "max_nesting_depth",
        "gzip_bytes",
        "halstead_volume",
        "halstead_effort",
    ]
    scaled = StandardScaler().fit_transform(expanded_df[structure_cols].astype(float))
    structure_pc1 = PCA(n_components=1, random_state=RNG_SEED).fit_transform(scaled)[:, 0]
    if safe_corr(pd.Series(structure_pc1), expanded_df["cyclomatic_complexity"]) < 0:
        structure_pc1 = -structure_pc1
    expanded_df["structure_pc1"] = structure_pc1
    expanded_df.to_csv(TABLES_DIR / "complexity_expanded.csv", index=False)
    return expanded_df


def bootstrap_corr_difference(
    df: pd.DataFrame,
    predictor_col: str,
    llm_col: str,
    human_col: str,
    n_boot: int = BOOTSTRAP_REPS,
) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    clean = df[[predictor_col, llm_col, human_col]].dropna().copy()
    if len(clean) < 6:
        return pd.DataFrame(
            [
                {
                    "predictor": predictor_col,
                    "n": int(len(clean)),
                    "human_corr": float("nan"),
                    "llm_corr": float("nan"),
                    "delta_llm_minus_human": float("nan"),
                    "delta_ci_lo": float("nan"),
                    "delta_ci_hi": float("nan"),
                }
            ]
        )

    human_corr = safe_corr(clean[predictor_col], clean[human_col])
    llm_corr = safe_corr(clean[predictor_col], clean[llm_col])
    deltas: list[float] = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, len(clean), size=len(clean))
        sample = clean.iloc[sample_idx]
        sample_h = safe_corr(sample[predictor_col], sample[human_col])
        sample_l = safe_corr(sample[predictor_col], sample[llm_col])
        if np.isfinite(sample_h) and np.isfinite(sample_l):
            deltas.append(sample_l - sample_h)
    ci_lo, ci_hi = percentile_interval(np.array(deltas))
    return pd.DataFrame(
        [
            {
                "predictor": predictor_col,
                "n": int(len(clean)),
                "human_corr": human_corr,
                "llm_corr": llm_corr,
                "delta_llm_minus_human": llm_corr - human_corr,
                "delta_ci_lo": ci_lo,
                "delta_ci_hi": ci_hi,
            }
        ]
    )


def build_structure_outputs(
    human_task_summary: pd.DataFrame,
    llm_task_summary: pd.DataFrame,
    complexity_expanded: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    human_complexity = human_task_summary.merge(
        complexity_expanded,
        left_on=["task_ID", "dataset_key"],
        right_on=["task_id", "dataset_key"],
        how="inner",
    )
    human_corr_rows: list[dict[str, object]] = []
    for benchmark_label, subset in human_complexity.groupby("benchmark_label"):
        for human_metric in ["latent_human_difficulty", "human_difficulty_pc1", "latent_duration_log"]:
            for predictor in ["cyclomatic_complexity", "structure_pc1"]:
                human_corr_rows.append(
                    {
                        "benchmark_label": benchmark_label,
                        "human_metric": human_metric,
                        "predictor": predictor,
                        "n": int(len(subset)),
                        "pearson_r": safe_corr(subset[predictor], subset[human_metric]),
                        "spearman_r": safe_corr(subset[predictor], subset[human_metric], method="spearman"),
                    }
                )
    human_corr = pd.DataFrame(human_corr_rows).sort_values(["benchmark_label", "human_metric", "predictor"])
    human_corr.to_csv(TABLES_DIR / "human_solver_structure_correlations.csv", index=False)

    pooled_human = human_complexity.loc[human_complexity["benchmark_label"].isin(["arc1_sidecar", "arc2_eval"])].copy()
    human_pooled_rows: list[dict[str, object]] = []
    for human_metric in ["latent_human_difficulty", "human_difficulty_pc1", "latent_duration_log"]:
        for predictor in ["cyclomatic_complexity", "structure_pc1"]:
            sub = pooled_human[[human_metric, predictor, "benchmark_label"]].dropna().copy()
            sub["human_centered"] = sub[human_metric] - sub.groupby("benchmark_label")[human_metric].transform("mean")
            sub["predictor_centered"] = sub[predictor] - sub.groupby("benchmark_label")[predictor].transform("mean")
            sub["human_z"] = sub.groupby("benchmark_label")[human_metric].transform(
                lambda s: (s - s.mean()) / s.std(ddof=0)
            )
            sub["predictor_z"] = sub.groupby("benchmark_label")[predictor].transform(
                lambda s: (s - s.mean()) / s.std(ddof=0)
            )
            human_pooled_rows.extend(
                [
                    {
                        "human_metric": human_metric,
                        "predictor": predictor,
                        "pooling": "naive_concat",
                        "n": int(len(sub)),
                        "pearson_r": safe_corr(sub[human_metric], sub[predictor]),
                        "spearman_r": safe_corr(sub[human_metric], sub[predictor], method="spearman"),
                    },
                    {
                        "human_metric": human_metric,
                        "predictor": predictor,
                        "pooling": "benchmark_centered",
                        "n": int(len(sub)),
                        "pearson_r": safe_corr(sub["human_centered"], sub["predictor_centered"]),
                        "spearman_r": safe_corr(sub["human_centered"], sub["predictor_centered"], method="spearman"),
                    },
                    {
                        "human_metric": human_metric,
                        "predictor": predictor,
                        "pooling": "benchmark_z",
                        "n": int(len(sub)),
                        "pearson_r": safe_corr(sub["human_z"], sub["predictor_z"]),
                        "spearman_r": safe_corr(sub["human_z"], sub["predictor_z"], method="spearman"),
                    },
                ]
            )
    pd.DataFrame(human_pooled_rows).to_csv(TABLES_DIR / "human_solver_structure_pooled_summary.csv", index=False)

    llm_membership = []
    for _, row in llm_task_summary.iterrows():
        memberships = str(row["benchmark_membership"]).split("|")
        for benchmark_label in memberships:
            if benchmark_label:
                llm_membership.append(
                    {
                        "task_id": row["task_id"],
                        "benchmark_label": benchmark_label,
                        "llm_latent_difficulty": row["llm_latent_difficulty"],
                        "llm_pass_rate": row["arc1_pass_rate"] if benchmark_label == "arc1_eval" else row["arc2_pass_rate"],
                        "dataset_key": "arc_agi_1_eval" if benchmark_label == "arc1_eval" else "arc_agi_2_eval",
                    }
                )
    llm_membership_df = pd.DataFrame(llm_membership)
    llm_complexity = llm_membership_df.merge(
        complexity_expanded,
        on=["task_id", "dataset_key"],
        how="inner",
    )
    llm_corr_rows: list[dict[str, object]] = []
    for benchmark_label, subset in llm_complexity.groupby("benchmark_label"):
        for predictor in ["cyclomatic_complexity", "structure_pc1"]:
            llm_corr_rows.append(
                {
                    "benchmark_label": benchmark_label,
                    "predictor": predictor,
                    "n": int(len(subset)),
                    "pearson_r": safe_corr(subset[predictor], subset["llm_latent_difficulty"]),
                    "spearman_r": safe_corr(subset[predictor], subset["llm_latent_difficulty"], method="spearman"),
                }
            )
    llm_corr = pd.DataFrame(llm_corr_rows).sort_values(["benchmark_label", "predictor"])
    llm_corr.to_csv(TABLES_DIR / "llm_solver_structure_correlations.csv", index=False)

    pooled_llm = llm_complexity.loc[llm_complexity["benchmark_label"].isin(["arc1_eval", "arc2_eval"])].copy()
    llm_pooled_rows: list[dict[str, object]] = []
    for llm_metric in ["llm_latent_difficulty", "llm_pass_rate"]:
        for predictor in ["cyclomatic_complexity", "structure_pc1"]:
            sub = pooled_llm[[llm_metric, predictor, "benchmark_label"]].dropna().copy()
            sub["llm_centered"] = sub[llm_metric] - sub.groupby("benchmark_label")[llm_metric].transform("mean")
            sub["predictor_centered"] = sub[predictor] - sub.groupby("benchmark_label")[predictor].transform("mean")
            sub["llm_z"] = sub.groupby("benchmark_label")[llm_metric].transform(
                lambda s: (s - s.mean()) / s.std(ddof=0)
            )
            sub["predictor_z"] = sub.groupby("benchmark_label")[predictor].transform(
                lambda s: (s - s.mean()) / s.std(ddof=0)
            )
            llm_pooled_rows.extend(
                [
                    {
                        "llm_metric": llm_metric,
                        "predictor": predictor,
                        "pooling": "naive_concat",
                        "n": int(len(sub)),
                        "pearson_r": safe_corr(sub[llm_metric], sub[predictor]),
                        "spearman_r": safe_corr(sub[llm_metric], sub[predictor], method="spearman"),
                    },
                    {
                        "llm_metric": llm_metric,
                        "predictor": predictor,
                        "pooling": "benchmark_centered",
                        "n": int(len(sub)),
                        "pearson_r": safe_corr(sub["llm_centered"], sub["predictor_centered"]),
                        "spearman_r": safe_corr(sub["llm_centered"], sub["predictor_centered"], method="spearman"),
                    },
                    {
                        "llm_metric": llm_metric,
                        "predictor": predictor,
                        "pooling": "benchmark_z",
                        "n": int(len(sub)),
                        "pearson_r": safe_corr(sub["llm_z"], sub["predictor_z"]),
                        "spearman_r": safe_corr(sub["llm_z"], sub["predictor_z"], method="spearman"),
                    },
                ]
            )
    pd.DataFrame(llm_pooled_rows).to_csv(TABLES_DIR / "llm_solver_structure_pooled_summary.csv", index=False)

    arc2_eval_human = human_task_summary.query("benchmark_label == 'arc2_eval'").copy()
    arc2_eval_human = arc2_eval_human.merge(
        complexity_expanded.query("dataset_key == 'arc_agi_2_eval'"),
        left_on=["task_ID", "dataset_key"],
        right_on=["task_id", "dataset_key"],
        how="inner",
    )
    arc2_eval_human_llm = arc2_eval_human.merge(
        llm_task_summary[["task_id", "llm_latent_difficulty"]],
        left_on="task_ID",
        right_on="task_id",
        how="inner",
        suffixes=("", "_llm"),
    )

    delta_frames = [
        bootstrap_corr_difference(
            arc2_eval_human_llm,
            predictor_col="cyclomatic_complexity",
            human_col="latent_human_difficulty",
            llm_col="llm_latent_difficulty",
        ),
        bootstrap_corr_difference(
            arc2_eval_human_llm,
            predictor_col="structure_pc1",
            human_col="latent_human_difficulty",
            llm_col="llm_latent_difficulty",
        ),
    ]
    delta_table = pd.concat(delta_frames, ignore_index=True)
    delta_table.to_csv(TABLES_DIR / "direct_overlap_structure_delta.csv", index=False)

    pooled_overlap = pooled_human.merge(
        llm_task_summary[["task_id", "llm_latent_difficulty"]],
        left_on="task_ID",
        right_on="task_id",
        how="inner",
    )
    pooled_overlap = pooled_overlap.loc[pooled_overlap["benchmark_label"].isin(["arc1_sidecar", "arc2_eval"])].copy()
    pooled_delta_rows: list[dict[str, object]] = []
    for predictor in ["cyclomatic_complexity", "structure_pc1"]:
        sub = pooled_overlap[[predictor, "latent_human_difficulty", "llm_latent_difficulty", "benchmark_label"]].dropna()
        sub = sub.copy()
        for col in [predictor, "latent_human_difficulty", "llm_latent_difficulty"]:
            sub[f"{col}_centered"] = sub[col] - sub.groupby("benchmark_label")[col].transform("mean")
            sub[f"{col}_z"] = sub.groupby("benchmark_label")[col].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
        pooled_delta_rows.extend(
            [
                {
                    "predictor": predictor,
                    "pooling": "naive_concat",
                    "n": int(len(sub)),
                    "human_corr": safe_corr(sub[predictor], sub["latent_human_difficulty"]),
                    "llm_corr": safe_corr(sub[predictor], sub["llm_latent_difficulty"]),
                },
                {
                    "predictor": predictor,
                    "pooling": "benchmark_centered",
                    "n": int(len(sub)),
                    "human_corr": safe_corr(sub[f"{predictor}_centered"], sub["latent_human_difficulty_centered"]),
                    "llm_corr": safe_corr(sub[f"{predictor}_centered"], sub["llm_latent_difficulty_centered"]),
                },
                {
                    "predictor": predictor,
                    "pooling": "benchmark_z",
                    "n": int(len(sub)),
                    "human_corr": safe_corr(sub[f"{predictor}_z"], sub["latent_human_difficulty_z"]),
                    "llm_corr": safe_corr(sub[f"{predictor}_z"], sub["llm_latent_difficulty_z"]),
                },
            ]
        )
    pooled_delta = pd.DataFrame(pooled_delta_rows)
    pooled_delta["delta_llm_minus_human"] = pooled_delta["llm_corr"] - pooled_delta["human_corr"]
    pooled_delta.to_csv(TABLES_DIR / "pooled_structure_delta_summary.csv", index=False)
    return human_corr, llm_corr, delta_table


def make_figures(
    partition_summary: pd.DataFrame,
    split_half_summary: pd.DataFrame,
    llm_model_summary: pd.DataFrame,
    human_llm_alignment: pd.DataFrame,
    human_structure_corr: pd.DataFrame,
    llm_structure_corr: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    eval_partitions = partition_summary.query("split == 'eval'").copy()
    eval_partitions["label"] = eval_partitions["benchmark"].str.upper() + " Eval"
    sns.barplot(data=eval_partitions, x="label", y="mean_input_cells", ax=ax, color="#4C7EA8")
    ax.set_title("Average Eval Input Cells")
    ax.set_xlabel("")
    ax.set_ylabel("Mean input cells")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "arc_eval_input_cells.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = split_half_summary.melt(
        id_vars=["benchmark_label"],
        value_vars=["raw_corr_mean", "latent_corr_mean"],
        var_name="metric",
        value_name="correlation",
    )
    plot_df["metric"] = plot_df["metric"].map(
        {"raw_corr_mean": "Raw solve-rate split-half", "latent_corr_mean": "Latent task split-half"}
    )
    sns.barplot(data=plot_df, x="benchmark_label", y="correlation", hue="metric", ax=ax)
    ax.set_title("Human Task Stability: Raw vs Latent")
    ax.set_xlabel("")
    ax.set_ylabel("Mean split-half Pearson r")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "human_split_half_stability.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    sns.scatterplot(data=llm_model_summary, x="arc1_accuracy", y="arc2_accuracy", ax=ax, s=60, color="#B25C2F")
    for _, row in llm_model_summary.nlargest(5, "arc2_accuracy").iterrows():
        ax.text(row["arc1_accuracy"], row["arc2_accuracy"], row["model_name"], fontsize=7)
    ax.set_title("Common-Model Accuracy Across ARC-1 and ARC-2")
    ax.set_xlabel("ARC-1 eval accuracy")
    ax.set_ylabel("ARC-2 eval accuracy")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "llm_common_model_accuracy.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    for ax, benchmark_label, title in zip(
        axes,
        ["arc1_sidecar", "arc2_eval"],
        ["ARC1 Sidecar Human vs ARC1 LLM", "ARC2 Eval Human vs ARC2 LLM"],
    ):
        subset = human_llm_alignment.query("benchmark_label == @benchmark_label").copy()
        sns.scatterplot(data=subset, x="human_difficulty_pc1", y="llm_latent_difficulty", ax=ax, s=35, color="#2B8A6D")
        ax.set_title(title)
        ax.set_xlabel("Human difficulty PC1")
        ax.set_ylabel("LLM latent difficulty")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "human_llm_alignment.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    corr_plot = pd.concat(
        [
            human_structure_corr.query(
                "human_metric == 'latent_human_difficulty' and predictor == 'cyclomatic_complexity'"
            )[["benchmark_label", "pearson_r"]].assign(series="Human"),
            llm_structure_corr.query("predictor == 'cyclomatic_complexity'")[["benchmark_label", "pearson_r"]].assign(
                series="LLM"
            ),
        ],
        ignore_index=True,
    )
    sns.barplot(data=corr_plot, x="benchmark_label", y="pearson_r", hue="series", ax=ax)
    ax.set_title("Cyclomatic Complexity Correlation")
    ax.set_xlabel("")
    ax.set_ylabel("Pearson r")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "solver_structure_cyclomatic.png", dpi=160)
    plt.close(fig)


def write_report(
    partition_summary: pd.DataFrame,
    shared_eval_summary: pd.DataFrame,
    human_diagnostics: pd.DataFrame,
    human_benchmark_summary: pd.DataFrame,
    split_half_summary: pd.DataFrame,
    llm_model_summary: pd.DataFrame,
    llm_benchmark_summary: pd.DataFrame,
    human_llm_alignment_summary: pd.DataFrame,
    structure_delta: pd.DataFrame,
) -> None:
    common_model_corr = {
        "pearson": safe_corr(llm_model_summary["arc1_accuracy"], llm_model_summary["arc2_accuracy"]),
        "spearman": safe_corr(llm_model_summary["arc1_accuracy"], llm_model_summary["arc2_accuracy"], method="spearman"),
    }
    shared_changed = int((~shared_eval_summary["same_train_examples"]).sum())
    report = f"""# Latent Cross-ARC Report

## Scope

This package revisits sparse human ARC data with latent estimates, links ARC-AGI-1 and ARC-AGI-2 through common respondents or common models where possible, and re-runs the solver-structure comparisons with wider coverage than the earlier direct-overlap slice.

## 1. Benchmark Inventory

- ARC-1 train tasks: {int(partition_summary.query("benchmark == 'arc1' and split == 'train'")["task_count"].iloc[0])}
- ARC-1 eval tasks: {int(partition_summary.query("benchmark == 'arc1' and split == 'eval'")["task_count"].iloc[0])}
- ARC-2 train tasks: {int(partition_summary.query("benchmark == 'arc2' and split == 'train'")["task_count"].iloc[0])}
- ARC-2 eval tasks: {int(partition_summary.query("benchmark == 'arc2' and split == 'eval'")["task_count"].iloc[0])}
- ARC-2 eval tasks are larger on average than ARC-1 eval tasks:
  - ARC-1 eval mean input cells: {partition_summary.query("benchmark == 'arc1' and split == 'eval'")["mean_input_cells"].iloc[0]:.1f}
  - ARC-2 eval mean input cells: {partition_summary.query("benchmark == 'arc2' and split == 'eval'")["mean_input_cells"].iloc[0]:.1f}
- The six shared eval task IDs are not identical across benchmarks:
  - shared eval IDs: {len(shared_eval_summary)}
  - changed training examples: {shared_changed}
  - unchanged test examples: {int(shared_eval_summary["same_test_examples"].sum())}

## 2. Human Latent Model

- Human attempt rows: {int(human_diagnostics["n_rows"].iloc[0])}
- Sessions: {int(human_diagnostics["n_sessions"].iloc[0])}
- Tasks with observed human coverage: {int(human_diagnostics["n_tasks"].iloc[0])}
- Pair-level rows with observed human coverage: {int(human_diagnostics["n_task_pairs"].iloc[0])}
- Correctness model diagnostics:
  - AUC: {human_diagnostics["auc"].iloc[0]:.3f}
  - Log loss: {human_diagnostics["log_loss"].iloc[0]:.3f}
  - Brier: {human_diagnostics["brier"].iloc[0]:.3f}
- Duration model diagnostics:
  - R^2: {human_diagnostics["duration_r2"].iloc[0]:.3f}
  - MAE on log-seconds: {human_diagnostics["duration_mae_log"].iloc[0]:.3f}

Human benchmark slices:

```text
{human_benchmark_summary.to_string(index=False)}
```

## 3. Raw vs Latent Stability

The main practical reason to use the latent estimates is stability under sparse coverage. Using all responses with partial pooling gives more reproducible task-level estimates than raw task solve rates.

```text
{split_half_summary.to_string(index=False)}
```

## 4. ARC-1 / ARC-2 LLM Linkage

- Common eval models across both matrices: {len(llm_model_summary)}
- Common-model ARC-1 vs ARC-2 accuracy correlation:
  - Pearson: {common_model_corr["pearson"]:.3f}
  - Spearman: {common_model_corr["spearman"]:.3f}
- Mean common-model pass rate:
  - ARC-1 eval: {llm_benchmark_summary.query("benchmark_label == 'arc1_eval'")["mean_pass_rate"].iloc[0]:.3f}
  - ARC-2 eval: {llm_benchmark_summary.query("benchmark_label == 'arc2_eval'")["mean_pass_rate"].iloc[0]:.3f}

This supports a linked scale through common models, but not a “same benchmark, just harder” simplification. ARC-2 eval is much harder, larger on average, more multi-test-pair heavy, and even the shared eval IDs come with revised training examples.

## 5. Human vs LLM Alignment

```text
{human_llm_alignment_summary.to_string(index=False)}
```

This widens the matched coverage substantially:

- ARC1 sidecar matched tasks: {int(human_llm_alignment_summary.query("benchmark_label == 'arc1_sidecar'")["matched_task_count"].iloc[0])}
- ARC2 eval matched tasks: {int(human_llm_alignment_summary.query("benchmark_label == 'arc2_eval'")["matched_task_count"].iloc[0])}

## 6. Solver-Structure Revisit

Direct ARC-2 eval human/LLM overlap deltas:

```text
{structure_delta.to_string(index=False)}
```

The headline pattern remains the same in this package: solver structure tracks LLM difficulty more strongly than human difficulty on the direct ARC-2 eval overlap.

## 7. Bottom Line

1. The sparse human data are usable without pretending raw item means are enough.
2. Latent task estimates are more stable than raw solve rates on session split-halves.
3. ARC-1 sidecar human coverage gives a much larger matched ARC-1 human/LLM slice than the earlier tiny direct-overlap analyses.
4. ARC-1 and ARC-2 can be linked on a common anchored scale, but they should not be treated as interchangeable without qualification.
5. The shared eval IDs between ARC-1 and ARC-2 preserve test examples but change the training examples, which is exactly the kind of benchmark drift that matters for interpretation.
6. The solver-structure result still looks like “LLM difficulty is more structure-loaded than human difficulty,” not just a fluke of the earlier writeup.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    task_metadata, partition_summary, shared_eval_summary = build_arc_metadata_outputs()
    human = load_human_attempts()
    human_diagnostics, _, human_task_summary = fit_full_human_latent_tables(human)
    _, split_half_summary = run_human_split_half(human)
    arc1_matrix, arc2_matrix = load_llm_eval_matrices()
    llm_model_summary, llm_task_summary, llm_benchmark_summary = build_llm_latent_tables(arc1_matrix, arc2_matrix)
    human_llm_alignment_tasks, human_llm_alignment_summary = build_human_llm_alignment(human_task_summary, llm_task_summary)
    complexity_expanded = expand_complexity_report()
    human_structure_corr, llm_structure_corr, structure_delta = build_structure_outputs(
        human_task_summary, llm_task_summary, complexity_expanded
    )

    make_figures(
        partition_summary,
        split_half_summary,
        llm_model_summary,
        human_llm_alignment_tasks,
        human_structure_corr,
        llm_structure_corr,
    )
    write_report(
        partition_summary,
        shared_eval_summary,
        human_diagnostics,
        pd.read_csv(TABLES_DIR / "human_benchmark_summary.csv"),
        split_half_summary,
        llm_model_summary,
        llm_benchmark_summary,
        human_llm_alignment_summary,
        structure_delta,
    )

    summary = {
        "human_task_counts": pd.read_csv(TABLES_DIR / "human_benchmark_summary.csv")
        .set_index("benchmark_label")["task_count"]
        .to_dict(),
        "human_split_half_latent_minus_raw": pd.read_csv(TABLES_DIR / "human_split_half_summary.csv")
        .set_index("benchmark_label")["latent_minus_raw"]
        .to_dict(),
        "common_model_accuracy_correlation_pearson": safe_corr(
            llm_model_summary["arc1_accuracy"], llm_model_summary["arc2_accuracy"]
        ),
        "common_model_accuracy_correlation_spearman": safe_corr(
            llm_model_summary["arc1_accuracy"], llm_model_summary["arc2_accuracy"], method="spearman"
        ),
        "shared_eval_task_count": int(len(shared_eval_summary)),
        "shared_eval_same_test_count": int(shared_eval_summary["same_test_examples"].sum()),
        "shared_eval_same_train_count": int(shared_eval_summary["same_train_examples"].sum()),
        "arc1_eval_mean_pass_rate": float(
            llm_benchmark_summary.query("benchmark_label == 'arc1_eval'")["mean_pass_rate"].iloc[0]
        ),
        "arc2_eval_mean_pass_rate": float(
            llm_benchmark_summary.query("benchmark_label == 'arc2_eval'")["mean_pass_rate"].iloc[0]
        ),
        "arc1_sidecar_human_llm_matched_tasks": int(
            human_llm_alignment_summary.query("benchmark_label == 'arc1_sidecar'")["matched_task_count"].iloc[0]
        ),
        "arc2_eval_human_llm_matched_tasks": int(
            human_llm_alignment_summary.query("benchmark_label == 'arc2_eval'")["matched_task_count"].iloc[0]
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
