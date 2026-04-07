import json
import math
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent
PSYCH_DATA_DIR = ROOT_DIR / "data-llm"

DATASETS = {
    "arc_agi_1_eval": {
        "preds_dir": PSYCH_DATA_DIR / "arc_agi_v1_public_eval",
        "truth_dir": PSYCH_DATA_DIR / "ARC-AGI" / "data" / "evaluation",
    },
    "arc_agi_2_eval": {
        "preds_dir": PSYCH_DATA_DIR / "arc_agi_v2_public_eval",
        "truth_dir": PSYCH_DATA_DIR / "ARC-AGI-2" / "data" / "evaluation",
    },
}

COMPLEXITY_METRICS = [
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
    "halstead_effort",
    "input_cells_total",
    "output_cells_total",
    "elapsed_ms_total",
    "elapsed_ms_per_test",
    "opcode_count_dynamic",
    "branch_opcode_count_dynamic",
    "python_call_count_dynamic",
    "peak_memory_bytes",
    "opcode_per_input_cell",
    "elapsed_ms_per_input_cell",
    "complexity_pc1_score",
    "log1p_opcode_count_dynamic",
    "log1p_branch_opcode_count_dynamic",
    "log1p_python_call_count_dynamic",
    "log1p_elapsed_ms_total",
    "log1p_elapsed_ms_per_test",
    "log1p_peak_memory_bytes",
    "log1p_ast_node_count",
    "log1p_cyclomatic_complexity",
]

LLM_OUTCOMES = [
    "latent_difficulty_prev_intersection22",
    "rasch_difficulty_all_models_pooled",
    "two_pl_difficulty_all_models",
    "fail_rate_all",
    "logit_difficulty_all",
    "response_sd_all",
    "binary_entropy_bits",
    "pc1_difficulty_z",
    "pc1_discrimination",
    "item_total_corr",
    "two_pl_discrimination",
    "log1p_two_pl_discrimination",
    "two_pl_max_info",
    "log1p_two_pl_max_info",
    "two_pl_info_theta0",
    "log1p_two_pl_info_theta0",
    "rasch_infit",
    "rasch_outfit",
    "rasch_abs_z_infit",
    "rasch_abs_z_outfit",
    "rasch_abs_z_misfit",
    "rasch_rmsea_x2",
    "thinking_advantage",
    "abs_thinking_advantage",
    "thinking_logit_advantage",
    "abs_thinking_logit_advantage",
]


def normalize_grid(grid):
    if not isinstance(grid, list):
        return []
    try:
        return [[int(cell) for cell in row] for row in grid]
    except Exception:
        return []


def grids_equal(pred_grid, true_grid):
    pred = normalize_grid(pred_grid)
    truth = normalize_grid(true_grid)
    return bool(pred) and bool(truth) and pred == truth


def load_truth_cache(truth_dir: Path):
    truth_cache = {}
    for truth_path in sorted(truth_dir.glob("*.json")):
        try:
            data = json.loads(truth_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        outputs = []
        for pair in data.get("test", []):
            outputs.append(pair.get("output"))
        truth_cache[truth_path.name] = outputs
    return truth_cache


def score_dataset(dataset_key: str, preds_dir: Path, truth_dir: Path) -> pd.DataFrame:
    truth_cache = load_truth_cache(truth_dir)
    task_files = sorted(truth_cache.keys())
    task_ids = [task[:-5] for task in task_files]
    model_dirs = sorted(
        [path for path in preds_dir.iterdir() if path.is_dir() and path.name != ".git"],
        key=lambda path: path.name.lower(),
    )
    response = pd.DataFrame(0.0, index=[path.name for path in model_dirs], columns=task_ids)

    for model_dir in model_dirs:
        for pred_path in model_dir.glob("*.json"):
            if pred_path.name not in truth_cache:
                continue
            try:
                pred_data = json.loads(pred_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            true_outputs = truth_cache[pred_path.name]
            is_correct = True
            for i, true_output in enumerate(true_outputs):
                pred_entry = None
                for item in pred_data:
                    pair_index = item.get("metadata", {}).get("pair_index")
                    if str(pair_index) == str(i):
                        pred_entry = item
                        break
                if pred_entry is None and i < len(pred_data):
                    pred_entry = pred_data[i]
                if pred_entry is None:
                    is_correct = False
                    break
                attempt1 = pred_entry.get("attempt_1") or {}
                model_output = attempt1.get("answer")
                if not model_output:
                    attempt2 = pred_entry.get("attempt_2") or {}
                    model_output = attempt2.get("answer")
                if not grids_equal(model_output, true_output):
                    is_correct = False
                    break
            response.at[model_dir.name, pred_path.stem] = 1.0 if is_correct else 0.0
    return response


def get_model_type(model_name: str) -> str:
    if re.search(r"thinking-none", model_name, flags=re.IGNORECASE):
        return "Standard"
    if re.search(r"thinking|deep|reasoning|gemini|gpt-5-pro", model_name, flags=re.IGNORECASE):
        return "Thinking"
    return "Standard"


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rank_array(values):
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def spearman_corr(x, y):
    return safe_corr(rank_array(x), rank_array(y))


def smoothed_logit_difficulty(num_correct: float, num_models: float) -> float:
    p = (num_correct + 0.5) / (num_models + 1.0)
    return float(math.log((1.0 - p) / p))


def binary_entropy_bits(pass_rate: float) -> float:
    if not math.isfinite(pass_rate):
        return np.nan
    if pass_rate <= 0.0 or pass_rate >= 1.0:
        return 0.0
    return float(
        -(pass_rate * math.log2(pass_rate) + (1.0 - pass_rate) * math.log2(1.0 - pass_rate))
    )


def corrected_item_total_corr(response: pd.DataFrame, task: str) -> float:
    y = response[task].to_numpy(dtype=float)
    n_tasks = response.shape[1]
    if n_tasks < 2:
        return np.nan
    total_other = (response.sum(axis=1).to_numpy(dtype=float) - y) / (n_tasks - 1)
    return safe_corr(y, total_other)


def compute_dataset_task_measures(dataset_key: str, response: pd.DataFrame):
    model_types = pd.Series([get_model_type(name) for name in response.index], index=response.index)
    thinking_mask = (model_types == "Thinking").to_numpy()
    standard_mask = (model_types == "Standard").to_numpy()

    response_var = response.loc[:, (response.mean(axis=0) > 0) & (response.mean(axis=0) < 1)].copy()
    model_accuracy = response_var.mean(axis=1).to_numpy(dtype=float)

    pca = PCA(n_components=1)
    theta = pca.fit_transform(response_var.to_numpy(dtype=float)).reshape(-1)
    if safe_corr(theta, model_accuracy) < 0:
        theta *= -1

    rows = []
    for task in response.columns:
        y = response[task].to_numpy(dtype=float)
        num_models = len(y)
        num_correct = float(y.sum())
        pass_rate_all = num_correct / num_models

        pass_rate_thinking = float(np.mean(y[thinking_mask])) if thinking_mask.any() else np.nan
        pass_rate_standard = float(np.mean(y[standard_mask])) if standard_mask.any() else np.nan

        pc1_discrimination = safe_corr(y, theta)
        item_total = corrected_item_total_corr(response, task)

        pc1_difficulty = np.nan
        if 0 < pass_rate_all < 1:
            try:
                lr = LogisticRegression(
                    solver="lbfgs",
                    C=10.0,
                    max_iter=2000,
                )
                lr.fit(theta.reshape(-1, 1), y.astype(int))
                coef = float(lr.coef_[0, 0])
                intercept = float(lr.intercept_[0])
                if abs(coef) > 1e-8:
                    pc1_difficulty = -intercept / coef
            except Exception:
                pc1_difficulty = np.nan

        thinking_logit = smoothed_logit_difficulty(
            float(np.sum(y[thinking_mask])),
            float(np.sum(thinking_mask)),
        ) if thinking_mask.any() else np.nan
        standard_logit = smoothed_logit_difficulty(
            float(np.sum(y[standard_mask])),
            float(np.sum(standard_mask)),
        ) if standard_mask.any() else np.nan

        rows.append(
            {
                "dataset_key": dataset_key,
                "task_id": task,
                "num_models_all": num_models,
                "num_models_thinking": int(np.sum(thinking_mask)),
                "num_models_standard": int(np.sum(standard_mask)),
                "pass_rate_all": pass_rate_all,
                "fail_rate_all": 1.0 - pass_rate_all,
                "logit_difficulty_all": smoothed_logit_difficulty(num_correct, num_models),
                "response_sd_all": float(math.sqrt(pass_rate_all * (1.0 - pass_rate_all))),
                "binary_entropy_bits": binary_entropy_bits(pass_rate_all),
                "pass_rate_thinking": pass_rate_thinking,
                "fail_rate_thinking": 1.0 - pass_rate_thinking if math.isfinite(pass_rate_thinking) else np.nan,
                "pass_rate_standard": pass_rate_standard,
                "fail_rate_standard": 1.0 - pass_rate_standard if math.isfinite(pass_rate_standard) else np.nan,
                "thinking_advantage": pass_rate_thinking - pass_rate_standard,
                "abs_thinking_advantage": abs(pass_rate_thinking - pass_rate_standard),
                "thinking_logit_advantage": standard_logit - thinking_logit,
                "abs_thinking_logit_advantage": abs(standard_logit - thinking_logit),
                "pc1_discrimination": pc1_discrimination,
                "item_total_corr": item_total,
                "pc1_difficulty": pc1_difficulty,
            }
        )

    measures = pd.DataFrame(rows)
    valid_mask = np.isfinite(measures["pc1_difficulty"].to_numpy(dtype=float))
    measures["pc1_difficulty_z"] = np.nan
    if valid_mask.sum() >= 2:
        values = measures.loc[valid_mask, "pc1_difficulty"]
        measures.loc[valid_mask, "pc1_difficulty_z"] = (values - values.mean()) / values.std(ddof=0)

    summary = {
        "dataset_key": dataset_key,
        "num_models": int(response.shape[0]),
        "num_tasks": int(response.shape[1]),
        "num_variable_tasks": int(response_var.shape[1]),
        "thinking_models": int(np.sum(thinking_mask)),
        "standard_models": int(np.sum(standard_mask)),
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "mean_model_accuracy": float(response.mean(axis=1).mean()),
    }
    model_meta = pd.DataFrame(
        {
            "dataset_key": dataset_key,
            "model_name": response.index,
            "model_type": model_types.to_numpy(),
            "accuracy": response.mean(axis=1).to_numpy(dtype=float),
            "pc1_theta": theta,
        }
    )
    return measures, summary, model_meta


def run_mirt_item_suite(
    matrix_df: pd.DataFrame,
    matrix_path: Path,
    rasch_item_output: Path,
    rasch_model_output: Path,
    rasch_fit_output: Path,
    two_pl_item_output: Path,
):
    matrix_df.to_csv(matrix_path)
    r_code = textwrap.dedent(
        """
        args <- commandArgs(trailingOnly = TRUE)
        suppressPackageStartupMessages(library(mirt))
        df <- read.csv(args[1], row.names = 1, check.names = FALSE)
        df[] <- lapply(df, function(x) as.numeric(x))
        var_cols <- sapply(df, function(x) length(unique(na.omit(x))) > 1)
        df_var <- df[, var_cols, drop = FALSE]
        mod_rasch <- mirt(df_var, 1, itemtype = 'Rasch', verbose = FALSE)
        params_rasch <- coef(mod_rasch, IRTpars = TRUE, simplify = TRUE)
        items_rasch <- params_rasch$items
        rasch_item_df <- data.frame(
            item = rownames(items_rasch),
            rasch_difficulty = items_rasch[, 'b'],
            row.names = NULL
        )
        write.csv(rasch_item_df, args[2], row.names = FALSE)

        fit_infit <- itemfit(mod_rasch, fit_stats = 'infit')
        fit_x2 <- itemfit(mod_rasch, fit_stats = 'X2')
        fit_df <- merge(as.data.frame(fit_infit), as.data.frame(fit_x2), by = 'item', all = TRUE)
        write.csv(fit_df, args[4], row.names = FALSE)

        theta <- fscores(mod_rasch, full.scores = TRUE)
        model_df <- data.frame(model = rownames(df_var), rasch_theta = theta[, 1], row.names = NULL)
        write.csv(model_df, args[3], row.names = FALSE)

        base_item_df <- data.frame(
            item = colnames(df_var),
            two_pl_discrimination = rep(NA_real_, ncol(df_var)),
            two_pl_difficulty = rep(NA_real_, ncol(df_var)),
            row.names = NULL
        )
        tryCatch({
            mod_2pl <- mirt(
                df_var,
                1,
                itemtype = '2PL',
                verbose = FALSE,
                technical = list(NCYCLES = 1500)
            )
            params_2pl <- coef(mod_2pl, IRTpars = TRUE, simplify = TRUE)
            items_2pl <- params_2pl$items
            base_item_df <- data.frame(
                item = rownames(items_2pl),
                two_pl_discrimination = items_2pl[, 'a'],
                two_pl_difficulty = items_2pl[, 'b'],
                row.names = NULL
            )
        }, error = function(e) {})
        write.csv(base_item_df, args[5], row.names = FALSE)
        """
    ).strip()

    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as handle:
        handle.write(r_code)
        script_path = Path(handle.name)

    try:
        subprocess.run(
            [
                "Rscript",
                str(script_path),
                str(matrix_path),
                str(rasch_item_output),
                str(rasch_model_output),
                str(rasch_fit_output),
                str(two_pl_item_output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        if script_path.exists():
            script_path.unlink()


def compute_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in COMPLEXITY_METRICS:
        for outcome in LLM_OUTCOMES:
            if metric not in joined.columns or outcome not in joined.columns:
                continue
            x = pd.to_numeric(joined[metric], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(joined[outcome], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                continue
            pearson = safe_corr(x[mask], y[mask])
            spearman = spearman_corr(x[mask], y[mask])
            rows.append(
                {
                    "complexity_metric": metric,
                    "llm_outcome": outcome,
                    "n": int(mask.sum()),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "abs_pearson_r": abs(pearson) if math.isfinite(pearson) else np.nan,
                    "abs_spearman_rho": abs(spearman) if math.isfinite(spearman) else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["llm_outcome", "abs_pearson_r", "abs_spearman_rho"],
        ascending=[True, False, False],
    )


def outcome_correlation_table(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, left in enumerate(LLM_OUTCOMES):
        for right in LLM_OUTCOMES[i + 1 :]:
            if left not in joined.columns or right not in joined.columns:
                continue
            x = pd.to_numeric(joined[left], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(joined[right], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                continue
            rows.append(
                {
                    "left_outcome": left,
                    "right_outcome": right,
                    "n": int(mask.sum()),
                    "pearson_r": safe_corr(x[mask], y[mask]),
                    "spearman_rho": spearman_corr(x[mask], y[mask]),
                }
            )
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False)


def compute_headline_by_dataset(joined: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("log1p_cyclomatic_complexity", "logit_difficulty_all"),
        ("ast_node_count", "rasch_difficulty_all_models_pooled"),
        ("halstead_effort", "thinking_logit_advantage"),
        ("elapsed_ms_total", "rasch_rmsea_x2"),
    ]
    rows = []
    for dataset_key, group in joined.groupby("dataset_key"):
        for metric, outcome in checks:
            if metric not in group.columns or outcome not in group.columns:
                continue
            x = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(group[outcome], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                continue
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "complexity_metric": metric,
                    "llm_outcome": outcome,
                    "n": int(mask.sum()),
                    "pearson_r": safe_corr(x[mask], y[mask]),
                    "spearman_rho": spearman_corr(x[mask], y[mask]),
                }
            )
    return pd.DataFrame(rows)


def make_heatmap(correlations: pd.DataFrame, output_path: Path):
    focus_metrics = [
        "ast_node_count",
        "cyclomatic_complexity",
        "nonblank_lines",
        "halstead_volume",
        "complexity_pc1_score",
        "elapsed_ms_per_test",
        "opcode_count_dynamic",
        "opcode_per_input_cell",
        "peak_memory_bytes",
    ]
    focus_outcomes = [
        "rasch_difficulty_all_models_pooled",
        "logit_difficulty_all",
        "binary_entropy_bits",
        "two_pl_discrimination",
        "two_pl_max_info",
        "rasch_abs_z_misfit",
        "abs_thinking_advantage",
        "item_total_corr",
    ]
    plot_df = correlations[
        correlations["complexity_metric"].isin(focus_metrics)
        & correlations["llm_outcome"].isin(focus_outcomes)
    ].copy()
    pivot = plot_df.pivot(index="complexity_metric", columns="llm_outcome", values="pearson_r")
    pivot = pivot.reindex(index=focus_metrics, columns=focus_outcomes)

    rename_cols = {
        "rasch_difficulty_all_models_pooled": "Pooled Rasch\n(all models)",
        "logit_difficulty_all": "Logit fail\ndifficulty",
        "binary_entropy_bits": "Entropy",
        "two_pl_discrimination": "2PL\ndiscrimination",
        "two_pl_max_info": "2PL max\ninformation",
        "rasch_abs_z_misfit": "Rasch\nmisfit |z|",
        "abs_thinking_advantage": "Thinking gap\n(abs)",
        "item_total_corr": "Item-total\ncorrelation",
    }
    rename_rows = {
        "ast_node_count": "AST node count",
        "cyclomatic_complexity": "Cyclomatic complexity",
        "nonblank_lines": "Nonblank lines",
        "halstead_volume": "Halstead volume",
        "complexity_pc1_score": "Complexity PC1",
        "elapsed_ms_per_test": "Elapsed ms / test",
        "opcode_count_dynamic": "Opcode count",
        "opcode_per_input_cell": "Opcode / input cell",
        "peak_memory_bytes": "Peak memory",
    }
    pivot = pivot.rename(columns=rename_cols, index=rename_rows)

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 7), dpi=180)
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Solver Complexity vs LLM-Derived Task Signals", fontsize=18, weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    response_frames = {}
    dataset_measure_frames = []
    dataset_summaries = []
    model_meta_frames = []

    for dataset_key, paths in DATASETS.items():
        response = score_dataset(dataset_key, paths["preds_dir"], paths["truth_dir"])
        response_frames[dataset_key] = response
        response.to_csv(BASE_DIR / f"llm_response_matrix_{dataset_key}.csv")

        measures, summary, model_meta = compute_dataset_task_measures(dataset_key, response)
        dataset_measure_frames.append(measures)
        dataset_summaries.append(summary)
        model_meta_frames.append(model_meta)

    task_measures = pd.concat(dataset_measure_frames, ignore_index=True)
    model_meta = pd.concat(model_meta_frames, ignore_index=True)

    union_models = sorted(set().union(*[set(df.index) for df in response_frames.values()]))
    pooled_matrix = pd.concat(
        [
            response_frames[dataset_key]
            .reindex(union_models)
            .add_prefix(f"{dataset_key}__")
            for dataset_key in DATASETS
        ],
        axis=1,
    )

    pooled_matrix_path = BASE_DIR / "llm_pooled_input_matrix.csv"
    pooled_item_path = BASE_DIR / "llm_rasch_pooled_items.csv"
    pooled_model_path = BASE_DIR / "llm_rasch_pooled_model_thetas.csv"
    pooled_fit_path = BASE_DIR / "llm_rasch_pooled_item_fit.csv"
    pooled_2pl_path = BASE_DIR / "llm_2pl_pooled_items.csv"
    run_mirt_item_suite(
        pooled_matrix,
        pooled_matrix_path,
        pooled_item_path,
        pooled_model_path,
        pooled_fit_path,
        pooled_2pl_path,
    )

    pooled_items = pd.read_csv(pooled_item_path)
    pooled_items[["dataset_key", "task_id"]] = pooled_items["item"].str.split("__", n=1, expand=True)
    pooled_items = pooled_items.rename(
        columns={"rasch_difficulty": "rasch_difficulty_all_models_pooled"}
    )[["dataset_key", "task_id", "rasch_difficulty_all_models_pooled"]]

    pooled_fit = pd.read_csv(pooled_fit_path)
    pooled_fit[["dataset_key", "task_id"]] = pooled_fit["item"].str.split("__", n=1, expand=True)
    pooled_fit = pooled_fit.rename(
        columns={
            "infit": "rasch_infit",
            "outfit": "rasch_outfit",
            "z.infit": "rasch_z_infit",
            "z.outfit": "rasch_z_outfit",
            "RMSEA.X2": "rasch_rmsea_x2",
        }
    )
    pooled_fit["rasch_abs_z_infit"] = pooled_fit["rasch_z_infit"].abs()
    pooled_fit["rasch_abs_z_outfit"] = pooled_fit["rasch_z_outfit"].abs()
    pooled_fit["rasch_abs_z_misfit"] = pooled_fit[
        ["rasch_abs_z_infit", "rasch_abs_z_outfit"]
    ].max(axis=1)
    pooled_fit = pooled_fit[
        [
            "dataset_key",
            "task_id",
            "rasch_infit",
            "rasch_outfit",
            "rasch_abs_z_infit",
            "rasch_abs_z_outfit",
            "rasch_abs_z_misfit",
            "rasch_rmsea_x2",
        ]
    ]

    pooled_2pl = pd.read_csv(pooled_2pl_path)
    pooled_2pl[["dataset_key", "task_id"]] = pooled_2pl["item"].str.split("__", n=1, expand=True)
    pooled_2pl = pooled_2pl.rename(
        columns={
            "two_pl_difficulty": "two_pl_difficulty_all_models",
        }
    )
    a = pd.to_numeric(pooled_2pl["two_pl_discrimination"], errors="coerce")
    b = pd.to_numeric(pooled_2pl["two_pl_difficulty_all_models"], errors="coerce")
    p0 = 1.0 / (1.0 + np.exp(-(a * (0.0 - b))))
    pooled_2pl["two_pl_max_info"] = (a ** 2) / 4.0
    pooled_2pl["two_pl_info_theta0"] = (a ** 2) * p0 * (1.0 - p0)
    pooled_2pl["log1p_two_pl_discrimination"] = np.where(a >= 0, np.log1p(a), np.nan)
    pooled_2pl["log1p_two_pl_max_info"] = np.where(
        pooled_2pl["two_pl_max_info"] >= 0,
        np.log1p(pooled_2pl["two_pl_max_info"]),
        np.nan,
    )
    pooled_2pl["log1p_two_pl_info_theta0"] = np.where(
        pooled_2pl["two_pl_info_theta0"] >= 0,
        np.log1p(pooled_2pl["two_pl_info_theta0"]),
        np.nan,
    )
    pooled_2pl = pooled_2pl[
        [
            "dataset_key",
            "task_id",
            "two_pl_difficulty_all_models",
            "two_pl_discrimination",
            "log1p_two_pl_discrimination",
            "two_pl_max_info",
            "log1p_two_pl_max_info",
            "two_pl_info_theta0",
            "log1p_two_pl_info_theta0",
        ]
    ]

    task_measures = task_measures.merge(pooled_items, on=["dataset_key", "task_id"], how="left")
    task_measures = task_measures.merge(pooled_fit, on=["dataset_key", "task_id"], how="left")
    task_measures = task_measures.merge(pooled_2pl, on=["dataset_key", "task_id"], how="left")

    complexity = pd.read_csv(BASE_DIR / "complexity_with_latent_components.csv")
    complexity = complexity.rename(
        columns={"latent_difficulty": "latent_difficulty_prev_intersection22"}
    )
    joined = complexity.merge(task_measures, on=["dataset_key", "task_id"], how="left")

    for metric in [
        "opcode_count_dynamic",
        "branch_opcode_count_dynamic",
        "python_call_count_dynamic",
        "elapsed_ms_total",
        "elapsed_ms_per_test",
        "peak_memory_bytes",
        "ast_node_count",
        "cyclomatic_complexity",
    ]:
        values = pd.to_numeric(joined[metric], errors="coerce")
        joined[f"log1p_{metric}"] = np.where(values >= 0, np.log1p(values), np.nan)

    correlations = compute_correlations(joined)
    outcome_corr = outcome_correlation_table(joined)
    headline_by_dataset = compute_headline_by_dataset(joined)

    best_by_outcome = (
        correlations.sort_values(["llm_outcome", "abs_pearson_r"], ascending=[True, False])
        .groupby("llm_outcome", as_index=False)
        .head(8)
        .reset_index(drop=True)
    )

    task_measures.to_csv(BASE_DIR / "llm_task_measures.csv", index=False)
    model_meta.to_csv(BASE_DIR / "llm_model_metadata.csv", index=False)
    joined.to_csv(BASE_DIR / "approved_llm_complexity_join.csv", index=False)
    correlations.to_csv(BASE_DIR / "approved_llm_complexity_correlations.csv", index=False)
    outcome_corr.to_csv(BASE_DIR / "approved_llm_outcome_correlations.csv", index=False)
    best_by_outcome.to_csv(BASE_DIR / "approved_llm_complexity_best_by_outcome.csv", index=False)
    headline_by_dataset.to_csv(BASE_DIR / "approved_llm_headline_by_dataset.csv", index=False)

    make_heatmap(correlations, BASE_DIR / "chart_llm_complexity_heatmap.png")

    summary = {
        "dataset_summaries": dataset_summaries,
        "overlap_rows": int(len(joined)),
        "unique_overlap_tasks": int(joined[["dataset_key", "task_id"]].drop_duplicates().shape[0]),
        "outcome_families": {
            "difficulty": [
                "latent_difficulty_prev_intersection22",
                "rasch_difficulty_all_models_pooled",
                "two_pl_difficulty_all_models",
                "fail_rate_all",
                "logit_difficulty_all",
                "pc1_difficulty_z",
            ],
            "discrimination_information": [
                "pc1_discrimination",
                "item_total_corr",
                "two_pl_discrimination",
                "two_pl_max_info",
                "two_pl_info_theta0",
                "response_sd_all",
                "binary_entropy_bits",
            ],
            "fit": [
                "rasch_infit",
                "rasch_outfit",
                "rasch_abs_z_infit",
                "rasch_abs_z_outfit",
                "rasch_abs_z_misfit",
                "rasch_rmsea_x2",
            ],
            "group_gap": [
                "thinking_advantage",
                "abs_thinking_advantage",
                "thinking_logit_advantage",
                "abs_thinking_logit_advantage",
            ],
        },
        "best_by_outcome": {
            outcome: group.head(3).to_dict(orient="records")
            for outcome, group in best_by_outcome.groupby("llm_outcome")
        },
        "outcome_correlations_top": outcome_corr.head(12).to_dict(orient="records"),
        "headline_by_dataset": headline_by_dataset.to_dict(orient="records"),
    }
    with open(BASE_DIR / "approved_llm_complexity_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    report_lines = [
        "# LLM Difficulty vs Solver Complexity",
        "",
        f"- Overlap rows analyzed: `{len(joined)}`",
        f"- Unique overlap tasks: `{joined[['dataset_key', 'task_id']].drop_duplicates().shape[0]}`",
        "",
        "## Dataset Coverage",
        "",
    ]
    for summary_row in dataset_summaries:
        report_lines.append(
            f"- {summary_row['dataset_key']}: `{summary_row['num_models']}` models, "
            f"`{summary_row['num_tasks']}` tasks, `{summary_row['num_variable_tasks']}` variable tasks, "
            f"PC1 explains `{summary_row['pc1_explained_variance_ratio']:.3f}` of model-response variance"
        )

    report_lines.extend(["", "## Strongest Correlations By LLM Outcome", ""])
    for outcome, group in best_by_outcome.groupby("llm_outcome"):
        top = group.iloc[0]
        report_lines.append(
            f"- {outcome}: best metric is `{top['complexity_metric']}` with "
            f"Pearson `{top['pearson_r']:.3f}` and Spearman `{top['spearman_rho']:.3f}`"
        )

    family_map = {
        "difficulty": [
            "latent_difficulty_prev_intersection22",
            "rasch_difficulty_all_models_pooled",
            "two_pl_difficulty_all_models",
            "fail_rate_all",
            "logit_difficulty_all",
            "pc1_difficulty_z",
        ],
        "discrimination_information": [
            "pc1_discrimination",
            "item_total_corr",
            "two_pl_discrimination",
            "two_pl_max_info",
            "two_pl_info_theta0",
            "response_sd_all",
            "binary_entropy_bits",
        ],
        "fit": [
            "rasch_infit",
            "rasch_outfit",
            "rasch_abs_z_infit",
            "rasch_abs_z_outfit",
            "rasch_abs_z_misfit",
            "rasch_rmsea_x2",
        ],
        "group_gap": [
            "thinking_advantage",
            "abs_thinking_advantage",
            "thinking_logit_advantage",
            "abs_thinking_logit_advantage",
        ],
    }
    report_lines.extend(["", "## Outcome Families", ""])
    for family_name, outcomes in family_map.items():
        subset = best_by_outcome[best_by_outcome["llm_outcome"].isin(outcomes)]
        if subset.empty:
            continue
        top = subset.sort_values("abs_pearson_r", ascending=False).iloc[0]
        report_lines.append(
            f"- {family_name}: strongest result is `{top['llm_outcome']}` with "
            f"`{top['complexity_metric']}` at Pearson `{top['pearson_r']:.3f}`"
        )

    report_lines.extend(["", "## Headline Stability By Dataset", ""])
    for _, row in headline_by_dataset.iterrows():
        report_lines.append(
            f"- {row['dataset_key']}: `{row['complexity_metric']}` vs `{row['llm_outcome']}` "
            f"has Pearson `{row['pearson_r']:.3f}` on `n={int(row['n'])}`"
        )

    (BASE_DIR / "approved_llm_complexity_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
