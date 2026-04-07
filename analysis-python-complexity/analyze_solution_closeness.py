from __future__ import annotations

import json
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent
PSYCH_DIR = ROOT_DIR / "data-llm"

HUMAN_TABLE_PATH = ROOT_DIR / "analysis-human" / "analysis" / "tables" / "public_eval_human_vs_models.csv"
APPROVED_LLM_PATH = BASE_DIR / "approved_llm_complexity_join.csv"
HUMAN_OVERLAP_PATH = BASE_DIR / "human_llm_overlap_tasks.csv"

DATASETS = {
    "arc_agi_1_eval": {
        "preds_dir": PSYCH_DIR / "arc_agi_v1_public_eval",
        "truth_dir": PSYCH_DIR / "ARC-AGI" / "data" / "evaluation",
    },
    "arc_agi_2_eval": {
        "preds_dir": PSYCH_DIR / "arc_agi_v2_public_eval",
        "truth_dir": PSYCH_DIR / "ARC-AGI-2" / "data" / "evaluation",
    },
}

SOFT_METRICS = [
    "cell_accuracy_padded",
    "shape_iou",
    "color_iou",
    "component_size_iou",
    "adjacency_iou",
]
PAIR_METRICS = ["exact_current", "exact_any", *SOFT_METRICS, "soft_composite"]
TASK_BINARY_METRICS = ["exact_current", "exact_any"]
TASK_SOFT_METRICS = [*SOFT_METRICS, "soft_composite"]
TASK_METRICS = [*TASK_BINARY_METRICS, *TASK_SOFT_METRICS]

METRIC_LABELS = {
    "exact_current": "Exact match (current attempt-1-first scoring)",
    "exact_any": "Exact match (either stored attempt)",
    "cell_accuracy_padded": "Padded cell accuracy",
    "shape_iou": "Shape IoU",
    "color_iou": "Color multiset IoU",
    "component_size_iou": "Component-size IoU",
    "adjacency_iou": "Adjacency IoU",
    "soft_composite": "Soft composite mean",
}

HUMAN_OUTCOME_CONFIGS = [
    {"outcome": "solve_rate", "orientation": "ease", "label": "Human solve rate"},
    {"outcome": "difficulty", "orientation": "hardness", "label": "Human pair difficulty"},
    {"outcome": "mean_duration_seconds", "orientation": "hardness", "label": "Mean human duration"},
]

LLM_PREDICTOR_CONFIGS = [
    {"sample": "approved_eval_all", "predictor": "complexity_pc1_score", "orientation": "hardness", "label": "Complexity PC1"},
    {"sample": "approved_eval_all", "predictor": "cyclomatic_complexity", "orientation": "hardness", "label": "Cyclomatic complexity"},
    {"sample": "approved_eval_all", "predictor": "log1p_elapsed_ms_total", "orientation": "hardness", "label": "log1p runtime"},
    {"sample": "approved_arc2_overlap", "predictor": "difficulty_weighted", "orientation": "hardness", "label": "Human task difficulty"},
    {"sample": "approved_arc2_overlap", "predictor": "human_solve_rate_weighted", "orientation": "ease", "label": "Human solve rate"},
    {"sample": "approved_arc2_overlap", "predictor": "mean_duration_seconds_weighted", "orientation": "hardness", "label": "Mean human duration"},
]

SENTINEL = -1


def normalize_grid(grid: object) -> list[list[int]]:
    if not isinstance(grid, list) or not grid:
        return []
    out: list[list[int]] = []
    try:
        for row in grid:
            if not isinstance(row, list):
                return []
            out.append([int(cell) for cell in row])
    except Exception:
        return []
    if not out or any(len(row) != len(out[0]) for row in out):
        return []
    return out


def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]))


def flatten_grid(grid: list[list[int]]) -> list[int]:
    return [cell for row in grid for cell in row]


def pad_grid(grid: list[list[int]], height: int, width: int, fill: int = SENTINEL) -> np.ndarray:
    arr = np.full((height, width), fill, dtype=int)
    if not grid:
        return arr
    g_height, g_width = grid_shape(grid)
    arr[:g_height, :g_width] = np.asarray(grid, dtype=int)
    return arr


def safe_corr(x, y) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return np.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman_corr(x, y) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return safe_corr(x_rank, y_rank)


def bounded_logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(math.log((1.0 - p) / p))


def bootstrap_diff_corr(x, y_a, y_b, n_boot: int = 8000, seed: int = 0) -> tuple[float, float, float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_a_arr = np.asarray(y_a, dtype=float)
    y_b_arr = np.asarray(y_b, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_a_arr) & np.isfinite(y_b_arr)
    x_arr = x_arr[mask]
    y_a_arr = y_a_arr[mask]
    y_b_arr = y_b_arr[mask]
    if len(x_arr) < 6:
        return (np.nan, np.nan, np.nan, np.nan)

    rng = np.random.default_rng(seed)
    vals = []
    n = len(x_arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r_a = safe_corr(x_arr[idx], y_a_arr[idx])
        r_b = safe_corr(x_arr[idx], y_b_arr[idx])
        if np.isfinite(r_a) and np.isfinite(r_b):
            vals.append(r_a - r_b)
    if not vals:
        return (np.nan, np.nan, np.nan, np.nan)

    vals_arr = np.asarray(vals, dtype=float)
    est = safe_corr(x_arr, y_a_arr) - safe_corr(x_arr, y_b_arr)
    ci_low, ci_high = np.percentile(vals_arr, [2.5, 97.5])
    p_boot = 2.0 * min(np.mean(vals_arr >= 0.0), np.mean(vals_arr <= 0.0))
    p_boot = max(float(p_boot), 1.0 / len(vals_arr))
    return (float(est), float(ci_low), float(ci_high), float(p_boot))


def counter_iou(counter_a: Counter, counter_b: Counter) -> float:
    keys = set(counter_a) | set(counter_b)
    if not keys:
        return 1.0
    intersection = sum(min(counter_a.get(key, 0), counter_b.get(key, 0)) for key in keys)
    union = sum(max(counter_a.get(key, 0), counter_b.get(key, 0)) for key in keys)
    if union <= 0:
        return 0.0
    return float(intersection / union)


def component_signature(grid: list[list[int]]) -> Counter:
    signature: Counter = Counter()
    if not grid:
        return signature

    height, width = grid_shape(grid)
    visited = [[False] * width for _ in range(height)]
    for row in range(height):
        for col in range(width):
            if visited[row][col]:
                continue
            color = grid[row][col]
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row][col] = True
            size = 0
            while queue:
                cur_row, cur_col = queue.popleft()
                size += 1
                for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row = cur_row + d_row
                    next_col = cur_col + d_col
                    if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                        continue
                    if visited[next_row][next_col] or grid[next_row][next_col] != color:
                        continue
                    visited[next_row][next_col] = True
                    queue.append((next_row, next_col))
            signature[(color, size)] += 1
    return signature


def adjacency_signature(grid: list[list[int]]) -> Counter:
    signature: Counter = Counter()
    if not grid:
        return signature

    height, width = grid_shape(grid)
    for row in range(height):
        for col in range(width):
            color = grid[row][col]
            if col + 1 < width:
                other = grid[row][col + 1]
                signature[("h", min(color, other), max(color, other))] += 1
            if row + 1 < height:
                other = grid[row + 1][col]
                signature[("v", min(color, other), max(color, other))] += 1
    return signature


def single_attempt_scores(pred_grid: list[list[int]], true_grid: list[list[int]]) -> dict[str, float]:
    if not pred_grid:
        return {
            "exact": 0.0,
            "cell_accuracy_padded": 0.0,
            "shape_iou": 0.0,
            "color_iou": 0.0,
            "component_size_iou": 0.0,
            "adjacency_iou": 0.0,
        }

    pred_height, pred_width = grid_shape(pred_grid)
    true_height, true_width = grid_shape(true_grid)

    exact = float(pred_grid == true_grid)

    pad_height = max(pred_height, true_height)
    pad_width = max(pred_width, true_width)
    pred_pad = pad_grid(pred_grid, pad_height, pad_width)
    true_pad = pad_grid(true_grid, pad_height, pad_width)
    cell_accuracy_padded = float(np.mean(pred_pad == true_pad))

    overlap_area = min(pred_height, true_height) * min(pred_width, true_width)
    union_area = (pred_height * pred_width) + (true_height * true_width) - overlap_area
    shape_iou = float(overlap_area / union_area) if union_area > 0 else 0.0

    color_iou = counter_iou(Counter(flatten_grid(pred_grid)), Counter(flatten_grid(true_grid)))
    component_size_iou = counter_iou(component_signature(pred_grid), component_signature(true_grid))
    adjacency_iou = counter_iou(adjacency_signature(pred_grid), adjacency_signature(true_grid))

    return {
        "exact": exact,
        "cell_accuracy_padded": cell_accuracy_padded,
        "shape_iou": shape_iou,
        "color_iou": color_iou,
        "component_size_iou": component_size_iou,
        "adjacency_iou": adjacency_iou,
    }


def extract_attempt_answer(pred_entry: dict, attempt_key: str) -> list[list[int]]:
    attempt = pred_entry.get(attempt_key) or {}
    return normalize_grid(attempt.get("answer"))


def candidate_pair_index(candidate: dict) -> int | None:
    direct_meta = candidate.get("metadata")
    if isinstance(direct_meta, dict) and "pair_index" in direct_meta:
        try:
            return int(direct_meta["pair_index"])
        except Exception:
            return None
    for attempt_key in ("attempt_1", "attempt_2"):
        attempt_meta = (candidate.get(attempt_key) or {}).get("metadata")
        if isinstance(attempt_meta, dict) and "pair_index" in attempt_meta:
            try:
                return int(attempt_meta["pair_index"])
            except Exception:
                return None
    return None


def locate_prediction_entry(pred_obj: object, pair_index: int) -> dict:
    if not isinstance(pred_obj, list):
        return {}
    for candidate in pred_obj:
        if not isinstance(candidate, dict):
            continue
        if candidate_pair_index(candidate) == pair_index:
            return candidate
    if 0 <= pair_index < len(pred_obj) and isinstance(pred_obj[pair_index], dict):
        return pred_obj[pair_index]
    return {}


def compare_pair_entry(pred_entry: dict, true_grid: list[list[int]]) -> dict[str, float]:
    attempt1_answer = extract_attempt_answer(pred_entry, "attempt_1")
    attempt2_answer = extract_attempt_answer(pred_entry, "attempt_2")

    attempt1_scores = single_attempt_scores(attempt1_answer, true_grid)
    attempt2_scores = single_attempt_scores(attempt2_answer, true_grid)

    current_scores = attempt1_scores if attempt1_answer else attempt2_scores
    max_soft = {metric: max(attempt1_scores[metric], attempt2_scores[metric]) for metric in SOFT_METRICS}

    return {
        "exact_current": float(current_scores["exact"]),
        "exact_any": float(max(attempt1_scores["exact"], attempt2_scores["exact"])),
        "cell_accuracy_padded": float(max_soft["cell_accuracy_padded"]),
        "shape_iou": float(max_soft["shape_iou"]),
        "color_iou": float(max_soft["color_iou"]),
        "component_size_iou": float(max_soft["component_size_iou"]),
        "adjacency_iou": float(max_soft["adjacency_iou"]),
        "soft_composite": float(np.mean(list(max_soft.values()))),
    }


def classify_model(model_name: str) -> str:
    lower_name = model_name.lower()
    if "thinking-none" in lower_name:
        return "Standard"
    if any(token in lower_name for token in ("thinking", "deep", "reasoning")):
        return "Thinking"
    if "gemini" in lower_name or "gpt-5-pro" in lower_name:
        return "Thinking"
    return "Standard"


def load_truth_tasks(truth_dir: Path, task_ids: list[str]) -> list[dict]:
    rows: list[dict] = []
    for task_id in sorted(task_ids):
        truth_path = truth_dir / f"{task_id}.json"
        if not truth_path.exists():
            continue
        truth_obj = json.loads(truth_path.read_text(encoding="utf-8"))
        for pair_index, pair in enumerate(truth_obj.get("test", [])):
            rows.append(
                {
                    "task_id": task_id,
                    "pair_index": pair_index,
                    "task_pair_id": f"{task_id}__{pair_index}",
                    "true_output": normalize_grid(pair.get("output")),
                }
            )
    return rows


def collect_pair_rows(dataset_key: str, task_ids: list[str]) -> pd.DataFrame:
    dataset = DATASETS[dataset_key]
    preds_dir = dataset["preds_dir"]
    truth_pairs = load_truth_tasks(dataset["truth_dir"], task_ids)

    rows: list[dict] = []
    model_dirs = sorted(
        [path for path in preds_dir.iterdir() if path.is_dir() and not path.name.startswith(".")],
        key=lambda path: path.name.lower(),
    )

    for model_dir in model_dirs:
        model_name = model_dir.name
        model_type = classify_model(model_name)
        pred_cache: dict[str, object] = {}
        for pair_spec in truth_pairs:
            task_id = pair_spec["task_id"]
            if task_id not in pred_cache:
                pred_path = model_dir / f"{task_id}.json"
                if pred_path.exists():
                    try:
                        pred_cache[task_id] = json.loads(pred_path.read_text(encoding="utf-8"))
                    except Exception:
                        pred_cache[task_id] = []
                else:
                    pred_cache[task_id] = []
            pred_entry = locate_prediction_entry(pred_cache[task_id], pair_spec["pair_index"])
            scores = compare_pair_entry(pred_entry, pair_spec["true_output"])
            row = {
                "dataset_key": dataset_key,
                "task_id": task_id,
                "pair_index": pair_spec["pair_index"],
                "task_pair_id": pair_spec["task_pair_id"],
                "model_name": model_name,
                "model_type": model_type,
            }
            row.update(scores)
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_pair_metrics(pair_rows: pd.DataFrame) -> pd.DataFrame:
    agg_rows: list[dict] = []
    for (dataset_key, task_id, pair_index, task_pair_id), group in pair_rows.groupby(
        ["dataset_key", "task_id", "pair_index", "task_pair_id"], sort=True
    ):
        row = {
            "dataset_key": dataset_key,
            "task_id": task_id,
            "pair_index": pair_index,
            "task_pair_id": task_pair_id,
            "num_models_all": int(len(group)),
            "num_models_thinking": int(group["model_type"].eq("Thinking").sum()),
            "num_models_standard": int(group["model_type"].eq("Standard").sum()),
        }
        for metric in PAIR_METRICS:
            row[f"{metric}_mean_all"] = float(group[metric].mean())
            thinking = group.loc[group["model_type"] == "Thinking", metric]
            standard = group.loc[group["model_type"] == "Standard", metric]
            row[f"{metric}_mean_thinking"] = float(thinking.mean()) if not thinking.empty else np.nan
            row[f"{metric}_mean_standard"] = float(standard.mean()) if not standard.empty else np.nan
            row[f"{metric}_gap_thinking"] = row[f"{metric}_mean_thinking"] - row[f"{metric}_mean_standard"]
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


def build_task_metric_table(pair_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_model_rows: list[dict] = []
    for (dataset_key, task_id, model_name, model_type), group in pair_rows.groupby(
        ["dataset_key", "task_id", "model_name", "model_type"], sort=True
    ):
        row = {
            "dataset_key": dataset_key,
            "task_id": task_id,
            "model_name": model_name,
            "model_type": model_type,
            "num_task_pairs": int(len(group)),
            "exact_current": float(group["exact_current"].min()),
            "exact_any": float(group["exact_any"].min()),
        }
        for metric in TASK_SOFT_METRICS:
            row[metric] = float(group[metric].mean())
        task_model_rows.append(row)

    task_model_df = pd.DataFrame(task_model_rows)

    task_rows: list[dict] = []
    for (dataset_key, task_id), group in task_model_df.groupby(["dataset_key", "task_id"], sort=True):
        row = {
            "dataset_key": dataset_key,
            "task_id": task_id,
            "num_models_all": int(len(group)),
            "num_models_thinking": int(group["model_type"].eq("Thinking").sum()),
            "num_models_standard": int(group["model_type"].eq("Standard").sum()),
        }
        for metric in TASK_METRICS:
            mean_all = float(group[metric].mean())
            thinking = group.loc[group["model_type"] == "Thinking", metric]
            standard = group.loc[group["model_type"] == "Standard", metric]
            row[f"{metric}_mean_all"] = mean_all
            row[f"{metric}_mean_thinking"] = float(thinking.mean()) if not thinking.empty else np.nan
            row[f"{metric}_mean_standard"] = float(standard.mean()) if not standard.empty else np.nan
            row[f"{metric}_gap_thinking"] = row[f"{metric}_mean_thinking"] - row[f"{metric}_mean_standard"]
            row[f"{metric}_difficulty_logit"] = bounded_logit(mean_all)
        task_rows.append(row)

    return pd.DataFrame(task_rows), task_model_df


def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for col_a in columns:
        for col_b in columns:
            matrix.loc[col_a, col_b] = safe_corr(df[col_a], df[col_b])
    return matrix


def build_human_comparison_table(human_df: pd.DataFrame, pair_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    human_table = human_df.copy()
    if "task_id" not in human_table.columns and "task_ID" in human_table.columns:
        human_table["task_id"] = human_table["task_ID"]

    merged = human_table.merge(pair_metrics, on=["task_pair_id", "task_id"], how="left")

    candidate_metrics = [
        "exact_any_mean_all",
        "cell_accuracy_padded_mean_all",
        "shape_iou_mean_all",
        "color_iou_mean_all",
        "component_size_iou_mean_all",
        "adjacency_iou_mean_all",
        "soft_composite_mean_all",
    ]

    tests: list[dict] = []
    sample_defs = [
        ("all_public_eval_pairs", merged),
        ("public_eval_pairs_attempts_ge_8", merged.loc[merged["attempts"] >= 8].copy()),
    ]
    for sample_name, sample_df in sample_defs:
        for outcome_cfg in HUMAN_OUTCOME_CONFIGS:
            outcome = outcome_cfg["outcome"]
            for candidate in candidate_metrics:
                subset = sample_df[[outcome, "lm_mean", candidate]].dropna().copy()
                if outcome_cfg["orientation"] == "hardness":
                    subset["candidate_eval"] = 1.0 - subset[candidate]
                    subset["baseline_eval"] = 1.0 - subset["lm_mean"]
                else:
                    subset["candidate_eval"] = subset[candidate]
                    subset["baseline_eval"] = subset["lm_mean"]

                delta, ci_low, ci_high, p_value = bootstrap_diff_corr(
                    subset[outcome],
                    subset["candidate_eval"],
                    subset["baseline_eval"],
                    seed=13,
                )
                tests.append(
                    {
                        "analysis": "human_pair_level",
                        "sample": sample_name,
                        "outcome": outcome,
                        "outcome_label": outcome_cfg["label"],
                        "candidate_metric": candidate.replace("_mean_all", ""),
                        "candidate_label": METRIC_LABELS[candidate.replace("_mean_all", "")],
                        "baseline_metric": "lm_mean",
                        "n": int(len(subset)),
                        "baseline_corr": safe_corr(subset[outcome], subset["baseline_eval"]),
                        "candidate_corr": safe_corr(subset[outcome], subset["candidate_eval"]),
                        "candidate_spearman": spearman_corr(subset[outcome], subset["candidate_eval"]),
                        "delta_corr": delta,
                        "delta_ci_low": ci_low,
                        "delta_ci_high": ci_high,
                        "p_value": p_value,
                    }
                )
    tests_df = pd.DataFrame(tests)
    tests_df["q_value_bh"] = multipletests(tests_df["p_value"].fillna(1.0), method="fdr_bh")[1]
    tests_df["improves_raw_p_lt_0_05"] = (tests_df["delta_corr"] > 0) & (tests_df["p_value"] < 0.05)
    tests_df["improves_fdr_lt_0_05"] = (tests_df["delta_corr"] > 0) & (tests_df["q_value_bh"] < 0.05)
    tests_df["worsens_raw_p_lt_0_05"] = (tests_df["delta_corr"] < 0) & (tests_df["p_value"] < 0.05)
    tests_df["worsens_fdr_lt_0_05"] = (tests_df["delta_corr"] < 0) & (tests_df["q_value_bh"] < 0.05)
    tests_df = tests_df.sort_values(["sample", "outcome", "delta_corr"], ascending=[True, True, False]).reset_index(drop=True)
    return merged, tests_df


def build_llm_comparison_table(task_metrics: pd.DataFrame, approved_df: pd.DataFrame, overlap_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    approved_join = approved_df.merge(task_metrics, on=["dataset_key", "task_id"], how="left")
    overlap_join = overlap_df.merge(task_metrics, on=["dataset_key", "task_id"], how="left")

    candidate_metrics = [
        "exact_any",
        "cell_accuracy_padded",
        "shape_iou",
        "color_iou",
        "component_size_iou",
        "adjacency_iou",
        "soft_composite",
    ]

    tests: list[dict] = []
    for predictor_cfg in LLM_PREDICTOR_CONFIGS:
        source_df = approved_join if predictor_cfg["sample"] == "approved_eval_all" else overlap_join
        predictor = predictor_cfg["predictor"]
        for candidate_metric in candidate_metrics:
            if predictor_cfg["orientation"] == "ease":
                baseline_col = "pass_rate_all"
                candidate_col = f"{candidate_metric}_mean_all"
                transform_note = "easiness"
            else:
                baseline_col = "logit_difficulty_all"
                candidate_col = f"{candidate_metric}_difficulty_logit"
                transform_note = "difficulty"

            subset = source_df[[predictor, baseline_col, candidate_col]].dropna().copy()
            delta, ci_low, ci_high, p_value = bootstrap_diff_corr(
                subset[predictor],
                subset[candidate_col],
                subset[baseline_col],
                seed=29,
            )
            tests.append(
                {
                    "analysis": "llm_task_level",
                    "sample": predictor_cfg["sample"],
                    "predictor": predictor,
                    "predictor_label": predictor_cfg["label"],
                    "orientation": transform_note,
                    "candidate_metric": candidate_metric,
                    "candidate_label": METRIC_LABELS[candidate_metric],
                    "baseline_metric": baseline_col,
                    "n": int(len(subset)),
                    "baseline_corr": safe_corr(subset[predictor], subset[baseline_col]),
                    "candidate_corr": safe_corr(subset[predictor], subset[candidate_col]),
                    "candidate_spearman": spearman_corr(subset[predictor], subset[candidate_col]),
                    "delta_corr": delta,
                    "delta_ci_low": ci_low,
                    "delta_ci_high": ci_high,
                    "p_value": p_value,
                }
            )

    tests_df = pd.DataFrame(tests)
    tests_df["q_value_bh"] = multipletests(tests_df["p_value"].fillna(1.0), method="fdr_bh")[1]
    tests_df["improves_raw_p_lt_0_05"] = (tests_df["delta_corr"] > 0) & (tests_df["p_value"] < 0.05)
    tests_df["improves_fdr_lt_0_05"] = (tests_df["delta_corr"] > 0) & (tests_df["q_value_bh"] < 0.05)
    tests_df["worsens_raw_p_lt_0_05"] = (tests_df["delta_corr"] < 0) & (tests_df["p_value"] < 0.05)
    tests_df["worsens_fdr_lt_0_05"] = (tests_df["delta_corr"] < 0) & (tests_df["q_value_bh"] < 0.05)
    tests_df = tests_df.sort_values(["sample", "predictor", "delta_corr"], ascending=[True, True, False]).reset_index(drop=True)
    return approved_join, tests_df


def mean_offdiag(matrix: pd.DataFrame) -> float:
    values = matrix.to_numpy(dtype=float)
    mask = ~np.eye(values.shape[0], dtype=bool)
    return float(np.nanmean(values[mask]))


def write_report(summary: dict, human_tests: pd.DataFrame, llm_tests: pd.DataFrame, report_path: Path) -> None:
    human_sig = human_tests.loc[human_tests["improves_raw_p_lt_0_05"]].copy()
    llm_sig = llm_tests.loc[llm_tests["improves_raw_p_lt_0_05"]].copy()
    human_worse = human_tests.loc[human_tests["worsens_raw_p_lt_0_05"]].copy()
    llm_worse = llm_tests.loc[llm_tests["worsens_raw_p_lt_0_05"]].copy()

    lines = [
        "# Solution-Closeness Analysis",
        "",
        "## Scope",
        "",
        "- LLM side: true partial-credit scoring from stored prediction grids.",
        "- Human side: raw human wrong-answer grids are not present in the repo, so the human analysis asks whether softer LLM pair-level signals explain human pair outcomes better than exact-match rates do.",
        "",
        "## Metric Families",
        "",
        "- `exact_current`: existing attempt-1-first exact-match behavior used in the current tables.",
        "- `exact_any`: exact match if either stored attempt solves the pair or task.",
        "- `cell_accuracy_padded`: top-left-aligned cell agreement after padding to the larger canvas.",
        "- `shape_iou`: overlap-over-union of output canvas shape.",
        "- `color_iou`: multiset overlap of color counts.",
        "- `component_size_iou`: overlap of connected-component size signatures by color.",
        "- `adjacency_iou`: overlap of horizontal/vertical neighbor-pair signatures.",
        "- `soft_composite`: mean of the five soft metrics above.",
        "",
        "## High-Level Takeaways",
        "",
        f"- ARC-2 model-pair rows scored: `{summary['counts']['arc2_pair_model_rows']}`.",
        f"- Approved task-model rows scored: `{summary['counts']['approved_task_model_rows']}`.",
        f"- `attempt_2` already matters: pair-level exact improves over current scoring on `{summary['attempt2_effect']['pair_level_improved_share']:.1%}` of ARC-2 model-pair rows, and task-level exact improves on `{summary['attempt2_effect']['task_level_improved_share']:.1%}` of approved model-task rows.",
        f"- Human-pair metric mean off-diagonal correlation: `{summary['metric_correlations']['human_pair_mean_offdiag']:.3f}`.",
        f"- LLM-task metric mean off-diagonal correlation: `{summary['metric_correlations']['llm_task_mean_offdiag']:.3f}`.",
        f"- Human-side raw `p < 0.05` improvements: `{int(human_sig.shape[0])}` tests; FDR-significant: `{int(human_tests['improves_fdr_lt_0_05'].sum())}`.",
        f"- Human-side raw `p < 0.05` degradations: `{int(human_worse.shape[0])}` tests.",
        f"- LLM-side raw `p < 0.05` improvements: `{int(llm_sig.shape[0])}` tests; FDR-significant: `{int(llm_tests['improves_fdr_lt_0_05'].sum())}`.",
        f"- LLM-side raw `p < 0.05` degradations: `{int(llm_worse.shape[0])}` tests.",
        "",
        "## Strongest Human-Side Improvements",
        "",
    ]

    if human_sig.empty:
        lines.append("- No human-side comparison cleared raw `p < 0.05`.")
    else:
        for _, row in human_sig.sort_values("delta_corr", ascending=False).head(8).iterrows():
            lines.append(
                f"- `{row['sample']}` | `{row['outcome_label']}`: `{row['candidate_label']}` beats baseline by `delta r = {row['delta_corr']:.3f}` "
                f"(candidate `r = {row['candidate_corr']:.3f}`, baseline `r = {row['baseline_corr']:.3f}`, p `{row['p_value']:.4f}`, q `{row['q_value_bh']:.4f}`)."
            )

    lines.extend(["", "## Strongest LLM-Side Improvements", ""])
    if llm_sig.empty:
        lines.append("- No LLM-side comparison cleared raw `p < 0.05`.")
    else:
        for _, row in llm_sig.sort_values("delta_corr", ascending=False).head(8).iterrows():
            lines.append(
                f"- `{row['sample']}` | `{row['predictor_label']}`: `{row['candidate_label']}` beats baseline by `delta r = {row['delta_corr']:.3f}` "
                f"(candidate `r = {row['candidate_corr']:.3f}`, baseline `r = {row['baseline_corr']:.3f}`, p `{row['p_value']:.4f}`, q `{row['q_value_bh']:.4f}`)."
            )

    lines.extend(["", "## Largest Degradations", ""])
    degradation_rows = pd.concat(
        [
            human_worse.assign(domain="human"),
            llm_worse.assign(domain="llm"),
        ],
        ignore_index=True,
    ).sort_values("delta_corr")
    if degradation_rows.empty:
        lines.append("- No degradation cleared raw `p < 0.05`.")
    else:
        for _, row in degradation_rows.head(8).iterrows():
            label = row["outcome_label"] if row["domain"] == "human" else row["predictor_label"]
            lines.append(
                f"- `{row['domain']}` | `{row['candidate_label']}` underperforms on `{label}` by `delta r = {row['delta_corr']:.3f}` "
                f"(candidate `r = {row['candidate_corr']:.3f}`, baseline `r = {row['baseline_corr']:.3f}`, p `{row['p_value']:.4f}`)."
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The human-side results are about better task-pair alignment with human outcomes, not direct partial-credit scoring of human submitted grids.",
            "- `exact_any` isolates the effect of honoring the stored second attempt before any softer scoring is added.",
            "- All improvement tests use paired bootstrap differences of correlations on the same sampled rows.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    human_df = pd.read_csv(HUMAN_TABLE_PATH)
    approved_df = pd.read_csv(APPROVED_LLM_PATH)
    overlap_df = pd.read_csv(HUMAN_OVERLAP_PATH)

    human_task_ids = sorted(human_df["task_ID"].unique())
    human_arc2_pair_rows = collect_pair_rows("arc_agi_2_eval", human_task_ids)
    human_arc2_pair_metrics = aggregate_pair_metrics(human_arc2_pair_rows)

    approved_task_ids_by_dataset = {
        dataset_key: sorted(group["task_id"].unique())
        for dataset_key, group in approved_df.groupby("dataset_key")
    }
    approved_pair_rows = [collect_pair_rows(dataset_key, task_ids) for dataset_key, task_ids in approved_task_ids_by_dataset.items()]
    approved_pair_df = pd.concat(approved_pair_rows, ignore_index=True)
    approved_task_metrics, approved_task_model_df = build_task_metric_table(approved_pair_df)

    human_join, human_tests = build_human_comparison_table(human_df, human_arc2_pair_metrics)
    llm_join, llm_tests = build_llm_comparison_table(approved_task_metrics, approved_df, overlap_df)

    human_metric_columns = [
        "lm_mean",
        "exact_any_mean_all",
        "cell_accuracy_padded_mean_all",
        "shape_iou_mean_all",
        "color_iou_mean_all",
        "component_size_iou_mean_all",
        "adjacency_iou_mean_all",
        "soft_composite_mean_all",
    ]
    human_metric_matrix = correlation_matrix(
        human_join.loc[human_join["attempts"] >= 8].dropna(subset=human_metric_columns),
        human_metric_columns,
    )

    llm_metric_columns = [
        "pass_rate_all",
        "exact_any_mean_all",
        "cell_accuracy_padded_mean_all",
        "shape_iou_mean_all",
        "color_iou_mean_all",
        "component_size_iou_mean_all",
        "adjacency_iou_mean_all",
        "soft_composite_mean_all",
    ]
    llm_metric_matrix = correlation_matrix(llm_join.dropna(subset=llm_metric_columns), llm_metric_columns)

    pair_level_improved = float((human_arc2_pair_rows["exact_any"] > human_arc2_pair_rows["exact_current"]).mean())
    task_level_improved = float((approved_task_model_df["exact_any"] > approved_task_model_df["exact_current"]).mean())

    significant_combined = pd.concat([human_tests, llm_tests], ignore_index=True)
    significant_combined = significant_combined.loc[significant_combined["improves_raw_p_lt_0_05"]].copy()
    significant_combined = significant_combined.sort_values("delta_corr", ascending=False).reset_index(drop=True)

    summary = {
        "counts": {
            "arc2_pair_model_rows": int(len(human_arc2_pair_rows)),
            "arc2_pair_rows_aggregated": int(len(human_arc2_pair_metrics)),
            "approved_pair_model_rows": int(len(approved_pair_df)),
            "approved_task_rows_aggregated": int(len(approved_task_metrics)),
            "approved_task_model_rows": int(len(approved_task_model_df)),
            "human_overlap_rows_all": int(len(human_join)),
            "human_overlap_rows_attempts_ge_8": int((human_join["attempts"] >= 8).sum()),
        },
        "attempt2_effect": {
            "pair_level_improved_share": pair_level_improved,
            "task_level_improved_share": task_level_improved,
            "pair_level_mean_exact_current": float(human_arc2_pair_rows["exact_current"].mean()),
            "pair_level_mean_exact_any": float(human_arc2_pair_rows["exact_any"].mean()),
            "task_level_mean_exact_current": float(approved_task_model_df["exact_current"].mean()),
            "task_level_mean_exact_any": float(approved_task_model_df["exact_any"].mean()),
        },
        "metric_correlations": {
            "human_pair_mean_offdiag": mean_offdiag(human_metric_matrix),
            "llm_task_mean_offdiag": mean_offdiag(llm_metric_matrix),
        },
        "top_human_improvements_raw_p": human_tests.loc[human_tests["improves_raw_p_lt_0_05"]]
        .sort_values("delta_corr", ascending=False)
        .head(10)
        .to_dict(orient="records"),
        "top_llm_improvements_raw_p": llm_tests.loc[llm_tests["improves_raw_p_lt_0_05"]]
        .sort_values("delta_corr", ascending=False)
        .head(10)
        .to_dict(orient="records"),
    }

    human_arc2_pair_metrics.to_csv(BASE_DIR / "solution_closeness_pair_metrics_public_eval.csv", index=False)
    approved_task_metrics.to_csv(BASE_DIR / "solution_closeness_task_metrics_approved.csv", index=False)
    human_join.to_csv(BASE_DIR / "solution_closeness_human_pair_join.csv", index=False)
    llm_join.to_csv(BASE_DIR / "solution_closeness_llm_task_join.csv", index=False)
    human_tests.to_csv(BASE_DIR / "solution_closeness_human_comparison.csv", index=False)
    llm_tests.to_csv(BASE_DIR / "solution_closeness_llm_comparison.csv", index=False)
    human_metric_matrix.to_csv(BASE_DIR / "solution_closeness_human_metric_matrix.csv")
    llm_metric_matrix.to_csv(BASE_DIR / "solution_closeness_llm_metric_matrix.csv")
    significant_combined.to_csv(BASE_DIR / "solution_closeness_significant_improvements.csv", index=False)
    (BASE_DIR / "solution_closeness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, human_tests, llm_tests, BASE_DIR / "solution_closeness_report.md")

    print("Wrote solution-closeness analysis outputs to:")
    print(BASE_DIR)


if __name__ == "__main__":
    main()
