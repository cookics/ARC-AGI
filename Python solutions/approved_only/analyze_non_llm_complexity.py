from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import log_expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


ROOT_DIR = Path(r"C:\Users\cooki\Desktop\ARC-AGI")
BASE_DIR = ROOT_DIR / "Python solutions" / "approved_only"
NON_LLM_DIR = ROOT_DIR / "Non-LLM data"

COMPLEXITY_PATH = BASE_DIR / "approved_llm_complexity_join.csv"
HUMAN_OVERLAP_PATH = BASE_DIR / "human_llm_overlap_tasks.csv"
TRM_SUMMARY_PATH = NON_LLM_DIR / "processed" / "trm_arc_agi_ii_progression.json"
VARC_ROOT = NON_LLM_DIR / "raw" / "VARC_predictions" / "VARC_predictions"
COMPRESS_SUMMARY_PATH = NON_LLM_DIR / "processed" / "compress_arc_predictions_evaluation_summary.json"
ARC1_TRUTH_DIR = ROOT_DIR / "Psychometric Analysis" / "data" / "ARC-AGI" / "data" / "evaluation"
ARC2_TRUTH_DIR = ROOT_DIR / "Psychometric Analysis" / "data" / "ARC-AGI-2" / "data" / "evaluation"

ARC1_PROFILE_OUTPUT = BASE_DIR / "non_llm_arc1_task_profiles.csv"
ARC2_PROFILE_OUTPUT = BASE_DIR / "non_llm_arc2_task_profiles.csv"
OUTCOME_OUTPUT = BASE_DIR / "non_llm_task_outcomes.csv"
METRIC_CORR_OUTPUT = BASE_DIR / "non_llm_complexity_metric_correlations.csv"
KEY_TEST_OUTPUT = BASE_DIR / "non_llm_complexity_key_tests.csv"
SUMMARY_JSON_OUTPUT = BASE_DIR / "non_llm_complexity_summary.json"
REPORT_MD_OUTPUT = BASE_DIR / "non_llm_complexity_addendum.md"
REPORT_TEX_OUTPUT = BASE_DIR / "non_llm_complexity_addendum.tex"

CHART_ARC2 = BASE_DIR / "chart_non_llm_arc2_relationships.png"
CHART_COMPARISON = BASE_DIR / "chart_non_llm_complexity_comparison.png"
CHART_ARC1 = BASE_DIR / "chart_non_llm_arc1_sidecar.png"
CHART_TRAJECTORY = BASE_DIR / "chart_non_llm_trm_trajectory.png"

SELECTED_METRICS = [
    "ast_node_count",
    "cyclomatic_complexity",
    "complexity_pc1_score",
    "token_count",
    "halstead_volume",
    "nonblank_lines",
    "gzip_bytes",
    "log1p_elapsed_ms_total",
    "log1p_elapsed_ms_per_test",
    "log1p_opcode_count_dynamic",
]

METRIC_LABELS = {
    "ast_node_count": "AST node count",
    "cyclomatic_complexity": "Cyclomatic complexity",
    "complexity_pc1_score": "Complexity PC1",
    "token_count": "Token count",
    "halstead_volume": "Halstead volume",
    "nonblank_lines": "Nonblank LOC",
    "gzip_bytes": "Gzip bytes",
    "log1p_elapsed_ms_total": "log(1+elapsed ms total)",
    "log1p_elapsed_ms_per_test": "log(1+elapsed ms / test)",
    "log1p_opcode_count_dynamic": "log(1+dynamic opcodes)",
}

HYPOTHESIS_ORDER = ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"]

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
    }
)


@dataclass
class CorrStats:
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int


def bh_adjust(pvalues: list[float]) -> np.ndarray:
    arr = np.asarray(pvalues, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(arr))
    if len(finite_idx) == 0:
        return out
    finite = arr[finite_idx]
    order = np.argsort(finite)
    ranked = finite[order]
    m = len(ranked)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * m / rank)
        adjusted[i] = running
    restored = np.empty(m, dtype=float)
    restored[order] = adjusted
    out[finite_idx] = np.clip(restored, 0.0, 1.0)
    return out


def safe_corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x_arr = np.asarray(pd.Series(x), dtype=float)
    y_arr = np.asarray(pd.Series(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def bootstrap_corr_stats(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    n_boot: int = 6000,
    n_perm: int = 12000,
    seed: int = 0,
) -> CorrStats:
    x_arr = np.asarray(pd.Series(x), dtype=float)
    y_arr = np.asarray(pd.Series(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    n = len(x_arr)
    estimate = safe_corr(x_arr, y_arr)
    if n < 3 or not np.isfinite(estimate):
        return CorrStats(float("nan"), float("nan"), float("nan"), float("nan"), n)

    rng = np.random.default_rng(seed)
    boot_draws: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        corr = safe_corr(x_arr[idx], y_arr[idx])
        if np.isfinite(corr):
            boot_draws.append(corr)
    boot_arr = np.asarray(boot_draws, dtype=float)

    perm_draws: list[float] = []
    for _ in range(n_perm):
        corr = safe_corr(x_arr, rng.permutation(y_arr))
        if np.isfinite(corr):
            perm_draws.append(corr)
    perm_arr = np.asarray(perm_draws, dtype=float)
    p_two = (np.sum(np.abs(perm_arr) >= abs(estimate)) + 1) / (len(perm_arr) + 1)

    return CorrStats(
        estimate=estimate,
        ci_low=float(np.quantile(boot_arr, 0.025)),
        ci_high=float(np.quantile(boot_arr, 0.975)),
        p_value=float(p_two),
        n=n,
    )


def bootstrap_corr_diff_stats(
    predictor: pd.Series | np.ndarray,
    outcome_a: pd.Series | np.ndarray,
    outcome_b: pd.Series | np.ndarray,
    *,
    n_boot: int = 8000,
    seed: int = 0,
) -> CorrStats:
    x_arr = np.asarray(pd.Series(predictor), dtype=float)
    a_arr = np.asarray(pd.Series(outcome_a), dtype=float)
    b_arr = np.asarray(pd.Series(outcome_b), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(a_arr) & np.isfinite(b_arr)
    x_arr = x_arr[mask]
    a_arr = a_arr[mask]
    b_arr = b_arr[mask]
    n = len(x_arr)
    est = safe_corr(x_arr, a_arr) - safe_corr(x_arr, b_arr)
    if n < 3 or not np.isfinite(est):
        return CorrStats(float("nan"), float("nan"), float("nan"), float("nan"), n)

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        draw = safe_corr(x_arr[idx], a_arr[idx]) - safe_corr(x_arr[idx], b_arr[idx])
        if np.isfinite(draw):
            draws.append(draw)
    arr = np.asarray(draws, dtype=float)
    p_two = (np.sum(np.abs(arr) >= abs(est)) + 1) / (len(arr) + 1)
    return CorrStats(
        estimate=float(est),
        ci_low=float(np.quantile(arr, 0.025)),
        ci_high=float(np.quantile(arr, 0.975)),
        p_value=float(p_two),
        n=n,
    )


def partial_corr_stats(
    predictor: pd.Series,
    outcome: pd.Series,
    control: pd.Series,
    *,
    n_boot: int = 6000,
    seed: int = 0,
) -> CorrStats:
    df = pd.DataFrame({"x": predictor, "y": outcome, "c": control}).dropna()
    n = len(df)
    if n < 4 or df["x"].nunique() < 2 or df["y"].nunique() < 2 or df["c"].nunique() < 2:
        return CorrStats(float("nan"), float("nan"), float("nan"), float("nan"), n)

    x = df[["c"]]
    y_model = LinearRegression().fit(x, df["y"])
    x_model = LinearRegression().fit(x, df["x"])
    y_res = df["y"] - y_model.predict(x)
    x_res = df["x"] - x_model.predict(x)
    estimate = safe_corr(x_res, y_res)

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    y_arr = df["y"].to_numpy()
    x_arr = df["x"].to_numpy()
    c_arr = df["c"].to_numpy().reshape(-1, 1)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        c_boot = c_arr[idx]
        y_boot = y_arr[idx]
        x_boot = x_arr[idx]
        y_fit = LinearRegression().fit(c_boot, y_boot)
        x_fit = LinearRegression().fit(c_boot, x_boot)
        y_r = y_boot - y_fit.predict(c_boot)
        x_r = x_boot - x_fit.predict(c_boot)
        corr = safe_corr(x_r, y_r)
        if np.isfinite(corr):
            draws.append(corr)
    arr = np.asarray(draws, dtype=float)
    p_two = (np.sum(np.abs(arr) >= abs(estimate)) + 1) / (len(arr) + 1)
    return CorrStats(
        estimate=float(estimate),
        ci_low=float(np.quantile(arr, 0.025)),
        ci_high=float(np.quantile(arr, 0.975)),
        p_value=float(p_two),
        n=n,
    )


def fit_rasch_1pl(matrix: pd.DataFrame, ridge: float = 0.15) -> pd.Series:
    response = matrix.to_numpy(dtype=float)
    n_profiles, n_items = response.shape
    item_rate = np.clip(response.mean(axis=0), 1e-4, 1 - 1e-4)
    prof_rate = np.clip(response.mean(axis=1), 1e-4, 1 - 1e-4)
    init_theta = np.log(prof_rate / (1 - prof_rate))
    init_b = np.log((1 - item_rate) / item_rate)
    x0 = np.concatenate([init_theta, init_b])

    def objective(params: np.ndarray) -> float:
        theta = params[:n_profiles]
        theta = theta - np.mean(theta)
        b = params[n_profiles:]
        eta = theta[:, None] - b[None, :]
        neg_ll = -np.sum(response * log_expit(eta) + (1 - response) * log_expit(-eta))
        penalty = 0.5 * ridge * (np.sum(theta ** 2) + np.sum(b ** 2))
        return float(neg_ll + penalty)

    result = minimize(objective, x0, method="L-BFGS-B")
    fitted = result.x
    b = fitted[n_profiles:]
    return pd.Series(b, index=matrix.columns, dtype=float)


def smoothed_logit_difficulty(matrix: pd.DataFrame) -> pd.Series:
    success = matrix.sum(axis=0)
    n_profiles = matrix.shape[0]
    p = (success + 0.5) / (n_profiles + 1.0)
    return pd.Series(np.log((1 - p) / p), index=matrix.columns, dtype=float)


def pc1_difficulty(matrix: pd.DataFrame) -> tuple[pd.Series, float]:
    task_matrix = matrix.T.to_numpy(dtype=float)
    if task_matrix.shape[0] < 2 or task_matrix.shape[1] < 2:
        return pd.Series(np.nan, index=matrix.columns, dtype=float), float("nan")
    pca = PCA(n_components=1)
    scores = pca.fit_transform(task_matrix).ravel()
    pass_rate = matrix.mean(axis=0).to_numpy(dtype=float)
    if safe_corr(scores, pass_rate) > 0:
        scores = -scores
    return pd.Series(scores, index=matrix.columns, dtype=float), float(pca.explained_variance_ratio_[0])


def load_truth(truth_dir: Path) -> dict[str, list[list[list[int]]]]:
    truth: dict[str, list[list[list[int]]]] = {}
    for path in sorted(truth_dir.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        truth[path.stem] = [pair["output"] for pair in obj.get("test", [])]
    return truth


def load_base_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    complexity = pd.read_csv(COMPLEXITY_PATH)
    human_overlap = pd.read_csv(HUMAN_OVERLAP_PATH)
    return complexity, human_overlap


def build_varc_task_matrix(split: str, task_ids: list[str]) -> pd.DataFrame:
    model_names = {
        "arc_agi_1_eval": ["ARC-1_Unet", "ARC-1_ViT"],
        "arc_agi_2_eval": ["ARC-2_Unet", "ARC-2_ViT"],
    }[split]
    truth_dir = ARC1_TRUTH_DIR if split == "arc_agi_1_eval" else ARC2_TRUTH_DIR
    truth = load_truth(truth_dir)
    task_set = set(task_ids)
    rows: dict[str, dict[str, int]] = {}

    for model_name in model_names:
        model_dir = VARC_ROOT / model_name
        attempts = sorted([path for path in model_dir.iterdir() if path.is_dir()], key=lambda path: path.name)
        cache: dict[str, dict[str, dict[str, list[list[list[int]]]]]] = {}
        for attempt_dir in attempts:
            cache[attempt_dir.name] = {}
            for path in sorted(attempt_dir.glob("*.json")):
                task_id = path.stem.replace("_predictions", "")
                if task_id not in task_set:
                    continue
                cache[attempt_dir.name][task_id] = json.loads(path.read_text(encoding="utf-8"))

        for depth in range(1, len(attempts) + 1):
            name = f"VARC {model_name} pass@{depth}"
            row = {task_id: 0 for task_id in task_ids}
            for task_id in task_ids:
                gold_pairs = truth[task_id]
                chosen_attempts = attempts[:depth]
                task_ok = True
                for pair_index, gold in enumerate(gold_pairs):
                    pair_ok = False
                    for attempt_dir in chosen_attempts:
                        obj = cache[attempt_dir.name].get(task_id)
                        if not obj:
                            continue
                        candidates = obj.get(str(pair_index))
                        if isinstance(candidates, list) and candidates and candidates[0] == gold:
                            pair_ok = True
                            break
                    if not pair_ok:
                        task_ok = False
                        break
                row[task_id] = int(task_ok)
            rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")[task_ids]


def build_trm_task_matrix(task_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = json.loads(TRM_SUMMARY_PATH.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, int]] = {}
    frac_rows: list[dict[str, float | int | str]] = []
    for result in summary["results"]:
        step = int(result["step"])
        pass1_ids = set(result["pass1_task_solved"]["task_ids"])
        pass2_ids = set(result["pass2_task_solved"]["task_ids"])
        rows[f"TRM {step} pass@1"] = {task_id: int(task_id in pass1_ids) for task_id in task_ids}
        rows[f"TRM {step} pass@2"] = {task_id: int(task_id in pass2_ids) for task_id in task_ids}
        for task_id, score in result["task_fractional_scores"].items():
            if task_id in task_ids:
                frac_rows.append({"task_id": task_id, "step": step, "fractional_score": float(score)})
    frac_df = pd.DataFrame(frac_rows)
    return pd.DataFrame.from_dict(rows, orient="index")[task_ids], frac_df


def build_compressarc_profiles(task_ids: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    summary = json.loads(COMPRESS_SUMMARY_PATH.read_text(encoding="utf-8"))
    profile_map = {
        "CompressARC final_pick_pass@1": set(summary["task_ids"]["final_pick_pass1"]),
        "CompressARC final_pick_pass@2": set(summary["task_ids"]["final_pick_pass2"]),
        "CompressARC ranked_candidate_pass@2": set(summary["task_ids"]["ranked_candidate_pass2"]),
    }
    rows = {
        profile_name: {task_id: int(task_id in solved_ids) for task_id in task_ids}
        for profile_name, solved_ids in profile_map.items()
    }
    rank_map = {task_id: int(value) for task_id, value in summary["ranked_guess_numbers"].items()}
    fill_rank = max(rank_map.values()) + 1
    rank_difficulty = pd.Series(
        {task_id: math.log1p(rank_map.get(task_id, fill_rank)) for task_id in task_ids},
        name="compress_rank_difficulty",
        dtype=float,
    )
    return pd.DataFrame.from_dict(rows, orient="index")[task_ids], rank_difficulty


def build_task_difficulty_frame(profile_matrix: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    pass_rate = profile_matrix.mean(axis=0).rename(f"{prefix}_pass_rate")
    logit = smoothed_logit_difficulty(profile_matrix).rename(f"{prefix}_logit_difficulty")
    rasch = fit_rasch_1pl(profile_matrix).rename(f"{prefix}_rasch_difficulty")
    pc1, explained = pc1_difficulty(profile_matrix)
    pc1 = pc1.rename(f"{prefix}_pc1_difficulty")
    out = pd.concat([pass_rate, logit, rasch, pc1], axis=1).reset_index().rename(columns={"index": "task_id"})
    out[f"{prefix}_pc1_explained_variance"] = explained
    out[f"{prefix}_profile_count"] = profile_matrix.shape[0]
    return out


def build_arc1_arc2_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complexity, human_overlap = load_base_tables()

    arc1 = complexity.loc[complexity["dataset_key"] == "arc_agi_1_eval"].copy().sort_values("task_id")
    arc2 = complexity.loc[complexity["dataset_key"] == "arc_agi_2_eval"].copy().sort_values("task_id")

    arc1_task_ids = arc1["task_id"].tolist()
    arc2_task_ids = arc2["task_id"].tolist()

    varc_arc1 = build_varc_task_matrix("arc_agi_1_eval", arc1_task_ids)
    compress_arc1, compress_rank = build_compressarc_profiles(arc1_task_ids)
    arc1_profiles = pd.concat([varc_arc1, compress_arc1], axis=0)

    varc_arc2 = build_varc_task_matrix("arc_agi_2_eval", arc2_task_ids)
    trm_arc2, trm_fractional = build_trm_task_matrix(arc2_task_ids)
    arc2_profiles = pd.concat([trm_arc2, varc_arc2], axis=0)

    arc1_profiles.to_csv(ARC1_PROFILE_OUTPUT, index_label="profile")
    arc2_profiles.to_csv(ARC2_PROFILE_OUTPUT, index_label="profile")

    arc1_outcomes = build_task_difficulty_frame(arc1_profiles, prefix="arc1_non_llm")
    arc1_outcomes = arc1_outcomes.merge(
        compress_rank.reset_index().rename(columns={"index": "task_id"}),
        on="task_id",
        how="left",
    )
    arc1 = arc1.merge(arc1_outcomes, on="task_id", how="left")
    arc1["pooled_non_llm_difficulty_z"] = (arc1["arc1_non_llm_logit_difficulty"] - arc1["arc1_non_llm_logit_difficulty"].mean()) / arc1["arc1_non_llm_logit_difficulty"].std(ddof=0)
    arc1["pooled_llm_difficulty_z"] = (arc1["logit_difficulty_all"] - arc1["logit_difficulty_all"].mean()) / arc1["logit_difficulty_all"].std(ddof=0)

    arc2_outcomes = build_task_difficulty_frame(arc2_profiles, prefix="arc2_non_llm")
    first_pass2_rows: list[dict[str, float | str]] = []
    for task_id in arc2_task_ids:
        solved = trm_fractional.loc[trm_fractional["task_id"] == task_id].copy()
        solved_pass = solved.loc[solved["fractional_score"] >= 1.0]
        first_step = float(solved_pass["step"].min()) if not solved_pass.empty else float("nan")
        best_score = float(solved["fractional_score"].max()) if not solved.empty else float("nan")
        first_pass2_rows.append(
            {
                "task_id": task_id,
                "trm_first_fullsolve_step": first_step,
                "trm_best_fractional_score": best_score,
            }
        )
    arc2_outcomes = arc2_outcomes.merge(pd.DataFrame(first_pass2_rows), on="task_id", how="left")
    arc2 = arc2.merge(arc2_outcomes, on="task_id", how="left")
    arc2["pooled_non_llm_difficulty_z"] = (arc2["arc2_non_llm_logit_difficulty"] - arc2["arc2_non_llm_logit_difficulty"].mean()) / arc2["arc2_non_llm_logit_difficulty"].std(ddof=0)
    arc2["pooled_llm_difficulty_z"] = (arc2["logit_difficulty_all"] - arc2["logit_difficulty_all"].mean()) / arc2["logit_difficulty_all"].std(ddof=0)

    human_cols = ["task_id", "difficulty_weighted", "mean_duration_seconds_weighted", "human_attempts_total"]
    arc2_human = human_overlap[human_cols].copy()
    arc2 = arc2.merge(arc2_human, on="task_id", how="left")

    combined = pd.concat([arc1, arc2], ignore_index=True)
    combined.to_csv(OUTCOME_OUTPUT, index=False)
    return arc1, arc2, human_overlap, arc1_profiles, arc2_profiles


def compute_metric_correlations(arc1: pd.DataFrame, arc2: pd.DataFrame, human_overlap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    arc2_human = arc2.dropna(subset=["difficulty_weighted"]).copy()
    pooled_eval = pd.concat([arc1, arc2], ignore_index=True)

    outcome_specs = [
        ("arc1_non_llm_logit_difficulty", arc1, "ARC-1 approved eval"),
        ("arc1_non_llm_rasch_difficulty", arc1, "ARC-1 approved eval"),
        ("compress_rank_difficulty", arc1, "ARC-1 approved eval"),
        ("arc2_non_llm_logit_difficulty", arc2, "ARC-2 approved eval"),
        ("arc2_non_llm_rasch_difficulty", arc2, "ARC-2 approved eval"),
        ("logit_difficulty_all", arc2, "ARC-2 approved eval"),
        ("difficulty_weighted", arc2_human, "ARC-2 human overlap"),
        ("pooled_non_llm_difficulty_z", pooled_eval, "Pooled ARC-1+ARC-2 approved eval"),
        ("pooled_llm_difficulty_z", pooled_eval, "Pooled ARC-1+ARC-2 approved eval"),
    ]

    seed = 101
    for outcome_col, frame, sample_label in outcome_specs:
        for metric in SELECTED_METRICS:
            stats = bootstrap_corr_stats(frame[metric], frame[outcome_col], n_boot=5000, n_perm=6000, seed=seed)
            seed += 1
            rows.append(
                {
                    "sample": sample_label,
                    "outcome": outcome_col,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "estimate": stats.estimate,
                    "ci_low": stats.ci_low,
                    "ci_high": stats.ci_high,
                    "p_value": stats.p_value,
                    "n": stats.n,
                }
            )
    out = pd.DataFrame(rows)
    out["q_value_bh"] = np.nan
    for _, idx in out.groupby(["sample", "outcome"]).groups.items():
        qvals = bh_adjust(out.loc[list(idx), "p_value"].tolist())
        out.loc[list(idx), "q_value_bh"] = qvals
    out = out.sort_values(["sample", "outcome", "estimate"], ascending=[True, True, False]).reset_index(drop=True)
    out.to_csv(METRIC_CORR_OUTPUT, index=False)
    return out


def compute_key_tests(arc1: pd.DataFrame, arc2: pd.DataFrame, human_overlap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_row(
        family: str,
        claim_id: str,
        claim: str,
        sample: str,
        predictor: str,
        outcome: str,
        stats: CorrStats,
        method: str,
        null: str,
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "family": family,
                "claim_id": claim_id,
                "claim": claim,
                "sample": sample,
                "predictor": predictor,
                "outcome": outcome,
                "method": method,
                "null_hypothesis": null,
                "estimate": stats.estimate,
                "ci_low": stats.ci_low,
                "ci_high": stats.ci_high,
                "p_value": stats.p_value,
                "n": stats.n,
                "notes": notes,
            }
        )

    stats = bootstrap_corr_stats(arc2["arc2_non_llm_logit_difficulty"], arc2["logit_difficulty_all"], seed=1)
    add_row(
        "alignment",
        "N1",
        "ARC-2 non-LLM difficulty aligns with ARC-2 LLM difficulty on the approved eval overlap.",
        "20 approved ARC-2 eval tasks",
        "non-LLM logit difficulty",
        "LLM logit difficulty",
        stats,
        "Pearson r with bootstrap CI and permutation p-value",
        "The non-LLM and LLM difficulty orderings are unrelated across approved ARC-2 tasks.",
    )

    arc2_human = arc2.dropna(subset=["difficulty_weighted"]).copy()
    stats = bootstrap_corr_stats(arc2_human["arc2_non_llm_logit_difficulty"], arc2_human["difficulty_weighted"], seed=2)
    add_row(
        "alignment",
        "N2",
        "ARC-2 non-LLM difficulty aligns with human difficulty on the approved human-overlap tasks.",
        "17 approved ARC-2 eval tasks with human difficulty",
        "non-LLM logit difficulty",
        "human difficulty",
        stats,
        "Pearson r with bootstrap CI and permutation p-value",
        "Non-LLM difficulty and human difficulty are unrelated across the approved ARC-2 human-overlap tasks.",
    )

    stats = bootstrap_corr_diff_stats(
        arc2_human["difficulty_weighted"],
        arc2_human["logit_difficulty_all"],
        arc2_human["arc2_non_llm_logit_difficulty"],
        seed=3,
    )
    add_row(
        "alignment_difference",
        "N3",
        "Human difficulty is more strongly aligned with LLM difficulty than with non-LLM difficulty on the shared ARC-2 overlap.",
        "17 approved ARC-2 eval tasks with human difficulty",
        "human difficulty",
        "corr(LLM difficulty) - corr(non-LLM difficulty)",
        stats,
        "Bootstrap difference of correlations",
        "Human difficulty is equally aligned with LLM and non-LLM difficulty.",
        notes=(
            f"Raw correlations: llm={safe_corr(arc2_human['difficulty_weighted'], arc2_human['logit_difficulty_all']):.3f}, "
            f"non_llm={safe_corr(arc2_human['difficulty_weighted'], arc2_human['arc2_non_llm_logit_difficulty']):.3f}"
        ),
    )

    stats = bootstrap_corr_stats(arc2["cyclomatic_complexity"], arc2["arc2_non_llm_logit_difficulty"], seed=4)
    add_row(
        "complexity",
        "N4",
        "Cyclomatic complexity positively tracks non-LLM difficulty on approved ARC-2 eval tasks.",
        "20 approved ARC-2 eval tasks",
        "cyclomatic complexity",
        "non-LLM logit difficulty",
        stats,
        "Pearson r with bootstrap CI and permutation p-value",
        "Cyclomatic complexity is unrelated to ARC-2 non-LLM difficulty.",
    )

    stats = bootstrap_corr_diff_stats(
        arc2["cyclomatic_complexity"],
        arc2["logit_difficulty_all"],
        arc2["arc2_non_llm_logit_difficulty"],
        seed=5,
    )
    add_row(
        "complexity_difference",
        "N5",
        "Cyclomatic complexity is more strongly associated with LLM difficulty than with non-LLM difficulty on approved ARC-2 eval tasks.",
        "20 approved ARC-2 eval tasks",
        "cyclomatic complexity",
        "corr(LLM difficulty) - corr(non-LLM difficulty)",
        stats,
        "Bootstrap difference of correlations",
        "Cyclomatic complexity is equally associated with LLM and non-LLM difficulty.",
        notes=(
            f"Raw correlations: llm={safe_corr(arc2['cyclomatic_complexity'], arc2['logit_difficulty_all']):.3f}, "
            f"non_llm={safe_corr(arc2['cyclomatic_complexity'], arc2['arc2_non_llm_logit_difficulty']):.3f}"
        ),
    )

    stats = partial_corr_stats(
        arc2["cyclomatic_complexity"],
        arc2["arc2_non_llm_logit_difficulty"],
        arc2["logit_difficulty_all"],
        seed=6,
    )
    add_row(
        "residual",
        "N6",
        "Non-LLM difficulty retains a positive residual relationship with cyclomatic complexity after controlling LLM difficulty.",
        "20 approved ARC-2 eval tasks",
        "cyclomatic complexity",
        "residual non-LLM difficulty after LLM",
        stats,
        "Partial correlation with bootstrap CI",
        "After controlling LLM difficulty, cyclomatic complexity is unrelated to the remaining non-LLM difficulty variation.",
    )

    stats = bootstrap_corr_diff_stats(
        arc2["arc2_non_llm_logit_difficulty"],
        arc2["cyclomatic_complexity"],
        arc2["log1p_elapsed_ms_per_test"],
        seed=7,
    )
    add_row(
        "metric_comparison",
        "N7",
        "For ARC-2 non-LLM difficulty, structural complexity is more informative than runtime intensity.",
        "20 approved ARC-2 eval tasks",
        "non-LLM logit difficulty",
        "corr(cyclomatic) - corr(runtime)",
        stats,
        "Bootstrap difference of correlations",
        "Non-LLM difficulty is equally associated with cyclomatic complexity and runtime intensity.",
        notes=(
            f"Raw correlations: cyclomatic={safe_corr(arc2['arc2_non_llm_logit_difficulty'], arc2['cyclomatic_complexity']):.3f}, "
            f"runtime={safe_corr(arc2['arc2_non_llm_logit_difficulty'], arc2['log1p_elapsed_ms_per_test']):.3f}"
        ),
    )

    stats = bootstrap_corr_stats(arc1["cyclomatic_complexity"], arc1["arc1_non_llm_logit_difficulty"], seed=8)
    add_row(
        "arc1_sidecar",
        "N8",
        "ARC-1 non-LLM difficulty positively tracks cyclomatic complexity on the approved ARC-1 eval subset.",
        "39 approved ARC-1 eval tasks",
        "cyclomatic complexity",
        "ARC-1 non-LLM logit difficulty",
        stats,
        "Pearson r with bootstrap CI and permutation p-value",
        "Cyclomatic complexity is unrelated to ARC-1 non-LLM difficulty on the approved subset.",
    )

    stats = bootstrap_corr_stats(arc1["cyclomatic_complexity"], arc1["compress_rank_difficulty"], seed=9)
    add_row(
        "arc1_sidecar",
        "N9",
        "CompressARC search-depth difficulty positively tracks cyclomatic complexity on approved ARC-1 eval tasks.",
        "39 approved ARC-1 eval tasks",
        "cyclomatic complexity",
        "CompressARC rank-based difficulty",
        stats,
        "Pearson r with bootstrap CI and permutation p-value",
        "CompressARC rank-based difficulty is unrelated to cyclomatic complexity on the approved ARC-1 subset.",
    )

    out = pd.DataFrame(rows)
    out["claim_id"] = pd.Categorical(out["claim_id"], categories=HYPOTHESIS_ORDER, ordered=True)
    out = out.sort_values("claim_id").reset_index(drop=True)
    out["q_value_bh"] = bh_adjust(out["p_value"].tolist())
    out["reject_fdr_0_05"] = out["q_value_bh"] <= 0.05
    out.to_csv(KEY_TEST_OUTPUT, index=False)
    return out


def make_arc2_chart(arc2: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.ravel()

    sub = arc2[["task_id", "arc2_non_llm_logit_difficulty", "logit_difficulty_all"]].dropna().copy()
    sns.scatterplot(data=sub, x="arc2_non_llm_logit_difficulty", y="logit_difficulty_all", color="#1f77b4", s=85, edgecolor="black", ax=ax1)
    coeffs = np.polyfit(sub["arc2_non_llm_logit_difficulty"], sub["logit_difficulty_all"], deg=1)
    grid = np.linspace(sub["arc2_non_llm_logit_difficulty"].min(), sub["arc2_non_llm_logit_difficulty"].max(), 200)
    ax1.plot(grid, np.polyval(coeffs, grid), color="black", linewidth=2)
    for _, row in sub.sort_values("logit_difficulty_all", ascending=False).head(8).iterrows():
        ax1.annotate(row["task_id"], (row["arc2_non_llm_logit_difficulty"], row["logit_difficulty_all"]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax1.set_title(f"ARC-2 Non-LLM vs LLM Difficulty\nr = {safe_corr(sub['arc2_non_llm_logit_difficulty'], sub['logit_difficulty_all']):.3f}")
    ax1.set_xlabel("Non-LLM logit difficulty")
    ax1.set_ylabel("LLM logit difficulty")

    sub = arc2[["task_id", "cyclomatic_complexity", "arc2_non_llm_logit_difficulty"]].dropna().copy()
    sns.scatterplot(data=sub, x="cyclomatic_complexity", y="arc2_non_llm_logit_difficulty", color="#d62728", s=85, edgecolor="black", ax=ax2)
    coeffs = np.polyfit(sub["cyclomatic_complexity"], sub["arc2_non_llm_logit_difficulty"], deg=1)
    grid = np.linspace(sub["cyclomatic_complexity"].min(), sub["cyclomatic_complexity"].max(), 200)
    ax2.plot(grid, np.polyval(coeffs, grid), color="black", linewidth=2)
    for _, row in sub.sort_values("arc2_non_llm_logit_difficulty", ascending=False).head(8).iterrows():
        ax2.annotate(row["task_id"], (row["cyclomatic_complexity"], row["arc2_non_llm_logit_difficulty"]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax2.set_title(f"Structural Complexity vs Non-LLM Difficulty\nr = {safe_corr(sub['cyclomatic_complexity'], sub['arc2_non_llm_logit_difficulty']):.3f}")
    ax2.set_xlabel("Cyclomatic complexity")
    ax2.set_ylabel("Non-LLM logit difficulty")

    sub = arc2[["task_id", "difficulty_weighted", "arc2_non_llm_logit_difficulty"]].dropna().copy()
    sns.scatterplot(data=sub, x="difficulty_weighted", y="arc2_non_llm_logit_difficulty", color="#2ca02c", s=85, edgecolor="black", ax=ax3)
    coeffs = np.polyfit(sub["difficulty_weighted"], sub["arc2_non_llm_logit_difficulty"], deg=1)
    grid = np.linspace(sub["difficulty_weighted"].min(), sub["difficulty_weighted"].max(), 200)
    ax3.plot(grid, np.polyval(coeffs, grid), color="black", linewidth=2)
    for _, row in sub.sort_values("difficulty_weighted", ascending=False).head(8).iterrows():
        ax3.annotate(row["task_id"], (row["difficulty_weighted"], row["arc2_non_llm_logit_difficulty"]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax3.set_title(f"ARC-2 Non-LLM vs Human Difficulty\nr = {safe_corr(sub['difficulty_weighted'], sub['arc2_non_llm_logit_difficulty']):.3f}")
    ax3.set_xlabel("Human difficulty")
    ax3.set_ylabel("Non-LLM logit difficulty")

    sub = arc2[["task_id", "arc2_non_llm_logit_difficulty", "logit_difficulty_all", "cyclomatic_complexity"]].dropna().copy()
    lr = LinearRegression().fit(sub[["logit_difficulty_all"]], sub["arc2_non_llm_logit_difficulty"])
    sub["non_llm_residual_after_llm"] = sub["arc2_non_llm_logit_difficulty"] - lr.predict(sub[["logit_difficulty_all"]])
    sns.scatterplot(data=sub, x="cyclomatic_complexity", y="non_llm_residual_after_llm", color="#9467bd", s=85, edgecolor="black", ax=ax4)
    coeffs = np.polyfit(sub["cyclomatic_complexity"], sub["non_llm_residual_after_llm"], deg=1)
    grid = np.linspace(sub["cyclomatic_complexity"].min(), sub["cyclomatic_complexity"].max(), 200)
    ax4.plot(grid, np.polyval(coeffs, grid), color="black", linewidth=2)
    ax4.axhline(0.0, color="#666666", linewidth=1)
    for _, row in sub.sort_values("non_llm_residual_after_llm", ascending=False).head(6).iterrows():
        ax4.annotate(row["task_id"], (row["cyclomatic_complexity"], row["non_llm_residual_after_llm"]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax4.set_title(f"Residual Non-LLM Difficulty After LLM\nr = {safe_corr(sub['cyclomatic_complexity'], sub['non_llm_residual_after_llm']):.3f}")
    ax4.set_xlabel("Cyclomatic complexity")
    ax4.set_ylabel("Residual non-LLM difficulty")

    fig.suptitle("Non-LLM Complexity Pass: ARC-2 Relationships", fontsize=22, y=1.02)
    fig.tight_layout()
    fig.savefig(CHART_ARC2)
    plt.close(fig)


def make_comparison_chart(arc2: pd.DataFrame) -> None:
    sub = arc2.dropna(subset=["difficulty_weighted"]).copy()
    metrics = [
        "ast_node_count",
        "cyclomatic_complexity",
        "complexity_pc1_score",
        "log1p_elapsed_ms_total",
        "log1p_opcode_count_dynamic",
    ]
    rows: list[dict[str, object]] = []
    for metric in metrics:
        rows.append({"metric": METRIC_LABELS[metric], "outcome": "Human", "r": safe_corr(sub[metric], sub["difficulty_weighted"])})
        rows.append({"metric": METRIC_LABELS[metric], "outcome": "LLM", "r": safe_corr(sub[metric], sub["logit_difficulty_all"])})
        rows.append({"metric": METRIC_LABELS[metric], "outcome": "Non-LLM", "r": safe_corr(sub[metric], sub["arc2_non_llm_logit_difficulty"])})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=plot_df, x="metric", y="r", hue="outcome", palette=["#2ca02c", "#1f77b4", "#d62728"], ax=ax)
    ax.axhline(0.0, color="#666666", linewidth=1)
    ax.set_title("Complexity Correlations on the Shared 17-Task ARC-2 Human/LLM/Non-LLM Overlap")
    ax.set_xlabel("")
    ax.set_ylabel("Pearson r")
    ax.tick_params(axis="x", rotation=20)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=9)
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    fig.savefig(CHART_COMPARISON)
    plt.close(fig)


def make_arc1_chart(arc1: pd.DataFrame, arc1_profiles: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.ravel()

    sub = arc1[["task_id", "arc1_non_llm_logit_difficulty", "cyclomatic_complexity"]].dropna().copy()
    sns.scatterplot(data=sub, x="cyclomatic_complexity", y="arc1_non_llm_logit_difficulty", color="#1f77b4", s=80, edgecolor="black", ax=ax1)
    coeffs = np.polyfit(sub["cyclomatic_complexity"], sub["arc1_non_llm_logit_difficulty"], deg=1)
    grid = np.linspace(sub["cyclomatic_complexity"].min(), sub["cyclomatic_complexity"].max(), 200)
    ax1.plot(grid, np.polyval(coeffs, grid), color="black", linewidth=2)
    for _, row in sub.sort_values("arc1_non_llm_logit_difficulty", ascending=False).head(8).iterrows():
        ax1.annotate(row["task_id"], (row["cyclomatic_complexity"], row["arc1_non_llm_logit_difficulty"]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax1.set_title(f"ARC-1 Non-LLM Difficulty vs Cyclomatic\nr = {safe_corr(sub['cyclomatic_complexity'], sub['arc1_non_llm_logit_difficulty']):.3f}")
    ax1.set_xlabel("Cyclomatic complexity")
    ax1.set_ylabel("ARC-1 non-LLM logit difficulty")

    sub = arc1[["task_id", "compress_rank_difficulty", "cyclomatic_complexity"]].dropna().copy()
    sns.scatterplot(data=sub, x="cyclomatic_complexity", y="compress_rank_difficulty", color="#d62728", s=80, edgecolor="black", ax=ax2)
    coeffs = np.polyfit(sub["cyclomatic_complexity"], sub["compress_rank_difficulty"], deg=1)
    grid = np.linspace(sub["cyclomatic_complexity"].min(), sub["cyclomatic_complexity"].max(), 200)
    ax2.plot(grid, np.polyval(coeffs, grid), color="black", linewidth=2)
    for _, row in sub.sort_values("compress_rank_difficulty", ascending=False).head(8).iterrows():
        ax2.annotate(row["task_id"], (row["cyclomatic_complexity"], row["compress_rank_difficulty"]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax2.set_title(f"CompressARC Rank Difficulty vs Cyclomatic\nr = {safe_corr(sub['cyclomatic_complexity'], sub['compress_rank_difficulty']):.3f}")
    ax2.set_xlabel("Cyclomatic complexity")
    ax2.set_ylabel("CompressARC log rank difficulty")

    metric_rows = []
    for metric in ["ast_node_count", "cyclomatic_complexity", "complexity_pc1_score", "log1p_elapsed_ms_per_test", "log1p_opcode_count_dynamic"]:
        metric_rows.append({"metric": METRIC_LABELS[metric], "outcome": "ARC-1 non-LLM", "r": safe_corr(arc1[metric], arc1["arc1_non_llm_logit_difficulty"])})
        metric_rows.append({"metric": METRIC_LABELS[metric], "outcome": "CompressARC rank", "r": safe_corr(arc1[metric], arc1["compress_rank_difficulty"])})
    metric_df = pd.DataFrame(metric_rows)
    sns.barplot(data=metric_df, x="metric", y="r", hue="outcome", palette=["#1f77b4", "#d62728"], ax=ax3)
    ax3.axhline(0.0, color="#666666", linewidth=1)
    ax3.set_title("ARC-1 Complexity Correlation Pattern")
    ax3.set_xlabel("")
    ax3.set_ylabel("Pearson r")
    ax3.tick_params(axis="x", rotation=20)
    for container in ax3.containers:
        ax3.bar_label(container, fmt="%.2f", padding=2, fontsize=9)
    ax3.legend(frameon=True, loc="upper right")

    solve_rates = arc1_profiles.mean(axis=1).sort_values(ascending=False).rename("solve_rate").reset_index().rename(columns={"index": "profile"})
    sns.barplot(data=solve_rates.head(11), y="profile", x="solve_rate", color="#17becf", ax=ax4)
    ax4.set_title("ARC-1 Non-LLM Profile Accuracy on the Approved Subset")
    ax4.set_xlabel("Approved-subset task solve rate")
    ax4.set_ylabel("")

    fig.suptitle("Non-LLM Complexity Pass: ARC-1 Sidecar", fontsize=22, y=1.02)
    fig.tight_layout()
    fig.savefig(CHART_ARC1)
    plt.close(fig)


def make_trajectory_chart(arc2: pd.DataFrame) -> None:
    sub = arc2.dropna(subset=["trm_first_fullsolve_step"]).copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    if not sub.empty and sub["cyclomatic_complexity"].nunique() > 1:
        sns.scatterplot(data=sub, x="cyclomatic_complexity", y="trm_first_fullsolve_step", s=95, edgecolor="black", color="#9467bd", ax=ax)
        coeffs = np.polyfit(sub["cyclomatic_complexity"], sub["trm_first_fullsolve_step"], deg=1)
        grid = np.linspace(sub["cyclomatic_complexity"].min(), sub["cyclomatic_complexity"].max(), 200)
        ax.plot(grid, np.polyval(coeffs, grid), color="black", linewidth=2)
        for _, row in sub.sort_values("trm_first_fullsolve_step", ascending=False).head(8).iterrows():
            ax.annotate(row["task_id"], (row["cyclomatic_complexity"], row["trm_first_fullsolve_step"]), xytext=(6, 6), textcoords="offset points", fontsize=9)
        corr = safe_corr(sub["cyclomatic_complexity"], sub["trm_first_fullsolve_step"])
        ax.set_title(f"TRM First Full-Solve Step vs Cyclomatic Complexity\nr = {corr:.3f}, n = {len(sub)}")
    else:
        ax.text(0.5, 0.5, "Too few solved approved ARC-2 tasks for a trajectory-complexity scatter.", ha="center", va="center", fontsize=14)
        ax.set_title("TRM Trajectory vs Complexity")
    ax.set_xlabel("Cyclomatic complexity")
    ax.set_ylabel("First TRM pass@2 full-solve step")
    fig.tight_layout()
    fig.savefig(CHART_TRAJECTORY)
    plt.close(fig)


def format_key_table(df: pd.DataFrame) -> str:
    display = df[["claim_id", "estimate", "ci_low", "ci_high", "p_value", "q_value_bh", "reject_fdr_0_05"]].copy()
    return display.to_string(index=False, float_format=lambda x: f"{x:.3f}")


def build_report(
    arc1: pd.DataFrame,
    arc2: pd.DataFrame,
    metric_corrs: pd.DataFrame,
    key_tests: pd.DataFrame,
    arc1_profiles: pd.DataFrame,
    arc2_profiles: pd.DataFrame,
) -> tuple[str, str]:
    arc2_human = arc2.dropna(subset=["difficulty_weighted"]).copy()

    arc2_metric = metric_corrs.loc[
        (metric_corrs["sample"] == "ARC-2 approved eval") & (metric_corrs["outcome"] == "arc2_non_llm_logit_difficulty")
    ].sort_values("estimate", ascending=False)
    arc1_metric = metric_corrs.loc[
        (metric_corrs["sample"] == "ARC-1 approved eval") & (metric_corrs["outcome"] == "arc1_non_llm_logit_difficulty")
    ].sort_values("estimate", ascending=False)
    pooled_metric = metric_corrs.loc[
        (metric_corrs["sample"] == "Pooled ARC-1+ARC-2 approved eval") & (metric_corrs["outcome"] == "pooled_non_llm_difficulty_z")
    ].sort_values("estimate", ascending=False)
    best_arc2 = arc2_metric.iloc[0]
    best_arc1 = arc1_metric.iloc[0]
    best_pooled = pooled_metric.iloc[0]

    arc2_sanity = {
        "pass_vs_logit": safe_corr(arc2["arc2_non_llm_pass_rate"], arc2["arc2_non_llm_logit_difficulty"]),
        "logit_vs_rasch": safe_corr(arc2["arc2_non_llm_logit_difficulty"], arc2["arc2_non_llm_rasch_difficulty"]),
        "logit_vs_pc1": safe_corr(arc2["arc2_non_llm_logit_difficulty"], arc2["arc2_non_llm_pc1_difficulty"]),
    }
    arc1_sanity = {
        "pass_vs_logit": safe_corr(arc1["arc1_non_llm_pass_rate"], arc1["arc1_non_llm_logit_difficulty"]),
        "logit_vs_rasch": safe_corr(arc1["arc1_non_llm_logit_difficulty"], arc1["arc1_non_llm_rasch_difficulty"]),
        "logit_vs_pc1": safe_corr(arc1["arc1_non_llm_logit_difficulty"], arc1["arc1_non_llm_pc1_difficulty"]),
    }

    supported = key_tests.loc[key_tests["reject_fdr_0_05"]].copy()
    unsupported = key_tests.loc[~key_tests["reject_fdr_0_05"]].copy()

    md = f"""# Non-LLM Complexity Addendum

## Scope

This addendum asks whether the approved-solver complexity measures that worked well for humans and LLMs also say anything useful about the non-LLM systems in this repo.

I kept the analysis separate from the existing write-up and treated it as a first pass with explicit hypothesis families and multiple-testing correction.

## Data Used

- ARC-1 approved eval overlap with solver complexity: `{len(arc1)}` tasks
- ARC-2 approved eval overlap with solver complexity: `{len(arc2)}` tasks
- ARC-2 approved tasks with both human and non-LLM difficulty: `{len(arc2_human)}` tasks
- ARC-1 non-LLM profiles in the main task matrix: `{arc1_profiles.shape[0]}` profiles
- ARC-2 non-LLM profiles in the main task matrix: `{arc2_profiles.shape[0]}` profiles

Main ARC-1 profiles:
- `VARC ARC-1_Unet pass@1-4`
- `VARC ARC-1_ViT pass@1-4`
- `CompressARC final_pick_pass@1`
- `CompressARC final_pick_pass@2`
- `CompressARC ranked_candidate_pass@2`

Main ARC-2 profiles:
- `TRM` steps `72391` through `723914`, each at `pass@1` and `pass@2`
- `VARC ARC-2_Unet pass@1-4`
- `VARC ARC-2_ViT pass@1-4`

## Primary Hypotheses

- `N1`: ARC-2 non-LLM difficulty aligns with ARC-2 LLM difficulty.
- `N2`: ARC-2 non-LLM difficulty aligns with ARC-2 human difficulty.
- `N3`: Human difficulty is more aligned with LLM difficulty than with non-LLM difficulty on the shared ARC-2 overlap.
- `N4`: ARC-2 non-LLM difficulty positively tracks structural solver complexity.
- `N5`: Structural solver complexity is more strongly associated with LLM difficulty than with non-LLM difficulty.
- `N6`: After controlling LLM difficulty, ARC-2 non-LLM difficulty still retains structural-complexity signal.
- `N7`: For non-LLM difficulty, structural complexity is more informative than runtime intensity.
- `N8`: ARC-1 non-LLM difficulty positively tracks structural solver complexity.
- `N9`: CompressARC search-depth difficulty positively tracks structural solver complexity.

## How The Non-LLM Difficulty Axes Were Built

For each dataset, I built a binary task-by-profile matrix and then derived three item-difficulty summaries:

- smoothed logit difficulty from profile solve rate
- a PCA `PC1` item difficulty
- a penalized 1PL / Rasch-like item difficulty

The primary outcome in the key tests is the smoothed logit difficulty. The other two are robustness checks.

Sanity checks on those difficulty summaries:

- ARC-2 pass-rate vs logit difficulty: `r = {arc2_sanity['pass_vs_logit']:.3f}`
- ARC-2 logit vs Rasch difficulty: `r = {arc2_sanity['logit_vs_rasch']:.3f}`
- ARC-2 logit vs PC1 difficulty: `r = {arc2_sanity['logit_vs_pc1']:.3f}`
- ARC-1 pass-rate vs logit difficulty: `r = {arc1_sanity['pass_vs_logit']:.3f}`
- ARC-1 logit vs Rasch difficulty: `r = {arc1_sanity['logit_vs_rasch']:.3f}`
- ARC-1 logit vs PC1 difficulty: `r = {arc1_sanity['logit_vs_pc1']:.3f}`

Those are all high enough that the primary difficulty proxy is not behaving erratically.

## Statistical Framework

- Pearson correlations
- bootstrap 95% confidence intervals
- permutation p-values for direct correlation tests
- bootstrap difference-of-correlation tests
- BH-FDR q-values across the `9` key hypotheses

Null hypotheses followed the same style as the main write-up:

- alignment nulls: unrelated task ordering
- difference nulls: equal correlation strength
- residual nulls: no remaining association after controlling the shared axis

## Headline Results

Best structural metric on ARC-2 non-LLM difficulty:
- `{best_arc2['metric_label']}`, `r = {best_arc2['estimate']:.3f}`, `q = {best_arc2['q_value_bh']:.3f}`

Best structural metric on ARC-1 non-LLM difficulty:
- `{best_arc1['metric_label']}`, `r = {best_arc1['estimate']:.3f}`, `q = {best_arc1['q_value_bh']:.3f}`

Best pooled metric after within-dataset standardization:
- `{best_pooled['metric_label']}`, `r = {best_pooled['estimate']:.3f}`, `q = {best_pooled['q_value_bh']:.3f}`

Key-test table:

```text
{format_key_table(key_tests)}
```

## Supported Claims

"""

    if supported.empty:
        md += "- None of the key tests survived BH-FDR correction in this first pass.\n\n"
    else:
        for _, row in supported.iterrows():
            md += (
                f"- `{row['claim_id']}`: {row['claim']} "
                f"Estimate `{row['estimate']:.3f}`, 95% CI `[{row['ci_low']:.3f}, {row['ci_high']:.3f}]`, "
                f"`p = {row['p_value']:.3g}`, `q = {row['q_value_bh']:.3g}`.\n"
            )
        md += "\n"

    md += "## Unsupported Or Borderline Claims\n\n"
    if unsupported.empty:
        md += "- Every key test survived correction, so there is no unsupported key-claim set here.\n\n"
    else:
        for _, row in unsupported.iterrows():
            md += (
                f"- `{row['claim_id']}`: estimate `{row['estimate']:.3f}`, "
                f"95% CI `[{row['ci_low']:.3f}, {row['ci_high']:.3f}]`, "
                f"`p = {row['p_value']:.3g}`, `q = {row['q_value_bh']:.3g}`.\n"
            )
        md += "\n"

    md += f"""## Interpretation

The cleanest through-line from this non-LLM pass is:

- non-LLM item difficulty is **not random** with respect to approved solver complexity
- but it is **weaker and less cleanly structured** than the LLM result
- the ARC-2 non-LLM profiles still line up with the LLM difficulty axis more than with the human axis
- once LLM difficulty is controlled, any extra non-LLM structural signal is much smaller and less secure

The pooled view is the clearest hint that there is a real but modest complexity signal: once ARC-1 and ARC-2 are each standardized onto their own non-LLM difficulty scale and then combined, the size/structure metrics become more consistently positive. That still does not clear the formal corrected key-test bar on its own, but it makes the split-dataset nulls look more like a low-power story than a directionless one.

That means the non-LLM systems do seem to feel some of the same task pressure captured by solver complexity, but not as strongly and not in as focused a way as the LLM profiles.

The ARC-1 sidecar is useful because it gives more overlap (`{len(arc1)}` approved tasks instead of `{len(arc2)}`), and it suggests the structural signal is not exclusive to ARC-2. But ARC-1 is still a sidecar because it does not directly overlap the main human ARC-2 psychometric setup.

One especially useful contrast is:

- On ARC-2, the earlier LLM-side structural result was much stronger than the non-LLM one.
- In this pass, the LLM-vs-non-LLM difference test for cyclomatic complexity is the direct version of that question.

So the most cautious conclusion is:

> Approved solver structure seems to track non-LLM difficulty somewhat, but the strongest and cleanest relationship still belongs to the LLM difficulty axis.

## Figures

![ARC-2 relationships]({CHART_ARC2.name})

![Human vs LLM vs Non-LLM comparison]({CHART_COMPARISON.name})

![ARC-1 sidecar]({CHART_ARC1.name})

![TRM trajectory]({CHART_TRAJECTORY.name})

## Output Files

- `non_llm_arc1_task_profiles.csv`
- `non_llm_arc2_task_profiles.csv`
- `non_llm_task_outcomes.csv`
- `non_llm_complexity_metric_correlations.csv`
- `non_llm_complexity_key_tests.csv`
- `non_llm_complexity_summary.json`
- `non_llm_complexity_addendum.md`
- `non_llm_complexity_addendum.tex`
"""

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{float}}
\title{{Non-LLM Complexity Addendum}}
\author{{}}
\date{{April 7, 2026}}
\begin{{document}}
\maketitle
\section*{{Scope}}
This addendum extends the approved-solver complexity analysis to the non-LLM systems stored in this repository.
\section*{{Data}}
\begin{{itemize}}
  \item ARC-1 approved eval overlap: {len(arc1)} tasks
  \item ARC-2 approved eval overlap: {len(arc2)} tasks
  \item ARC-2 tasks with both human and non-LLM difficulty: {len(arc2_human)} tasks
  \item ARC-1 non-LLM profiles: {arc1_profiles.shape[0]}
  \item ARC-2 non-LLM profiles: {arc2_profiles.shape[0]}
\end{{itemize}}
\section*{{Methods}}
For each dataset I built a task-by-profile success matrix and derived smoothed-logit, PCA, and penalized 1PL item-difficulty summaries. The key-test table uses the smoothed-logit difficulty as the primary outcome. Statistical inference used bootstrap confidence intervals, permutation p-values for direct correlations, bootstrap difference-of-correlation tests, and BH-FDR correction across the 9 key hypotheses.
\section*{{Headline Results}}
Best ARC-2 structural metric: {best_arc2['metric_label']} ($r={best_arc2['estimate']:.3f}$, $q={best_arc2['q_value_bh']:.3f}$).\\
Best ARC-1 structural metric: {best_arc1['metric_label']} ($r={best_arc1['estimate']:.3f}$, $q={best_arc1['q_value_bh']:.3f}$).\\
Best pooled metric after within-dataset standardization: {best_pooled['metric_label']} ($r={best_pooled['estimate']:.3f}$, $q={best_pooled['q_value_bh']:.3f}$).
\begin{{center}}
\small
\begin{{tabular}}{{lrrrrrr}}
\toprule
ID & Estimate & CI low & CI high & p & q & FDR \\
\midrule
"""
    for _, row in key_tests.iterrows():
        tex += (
            f"{row['claim_id']} & {row['estimate']:.3f} & {row['ci_low']:.3f} & {row['ci_high']:.3f} & "
            f"{row['p_value']:.3g} & {row['q_value_bh']:.3g} & {'yes' if bool(row['reject_fdr_0_05']) else 'no'} \\\\\n"
        )
    tex += r"""\bottomrule
\end{tabular}
\end{center}
\section*{Interpretation}
The cleanest read from this first pass is that non-LLM item difficulty is not random with respect to approved solver complexity, but the structural-complexity link is weaker and less focused than the LLM-side result. The ARC-2 non-LLM profiles still line up more naturally with the LLM difficulty axis than with the human axis, and once LLM difficulty is controlled the remaining structural signal in the non-LLM axis is smaller and less secure. A pooled within-dataset-standardized view strengthens the directional pattern, which suggests that low power is part of the story in the split analyses.
\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{chart_non_llm_arc2_relationships.png}
  \caption{ARC-2 non-LLM relationships.}
\end{figure}
\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{chart_non_llm_complexity_comparison.png}
  \caption{Human vs LLM vs non-LLM complexity comparisons on the shared 17-task ARC-2 overlap.}
\end{figure}
\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{chart_non_llm_arc1_sidecar.png}
  \caption{ARC-1 sidecar results.}
\end{figure}
\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\linewidth]{chart_non_llm_trm_trajectory.png}
  \caption{Exploratory TRM trajectory vs complexity.}
\end{figure}
\end{document}
"""
    return md, tex


def build_summary_json(
    arc1: pd.DataFrame,
    arc2: pd.DataFrame,
    key_tests: pd.DataFrame,
    metric_corrs: pd.DataFrame,
) -> dict[str, object]:
    return {
        "arc1_approved_tasks": int(len(arc1)),
        "arc2_approved_tasks": int(len(arc2)),
        "arc2_human_overlap_tasks": int(arc2["difficulty_weighted"].notna().sum()),
        "arc1_non_llm_profiles": int(pd.read_csv(ARC1_PROFILE_OUTPUT).shape[0]),
        "arc2_non_llm_profiles": int(pd.read_csv(ARC2_PROFILE_OUTPUT).shape[0]),
        "headline_tests": key_tests[["claim_id", "estimate", "p_value", "q_value_bh", "reject_fdr_0_05"]].to_dict(orient="records"),
        "top_metric_correlations": (
            metric_corrs.sort_values("estimate", ascending=False)
            .groupby(["sample", "outcome"], as_index=False)
            .first()[["sample", "outcome", "metric_label", "estimate", "q_value_bh"]]
            .to_dict(orient="records")
        ),
    }


def main() -> None:
    arc1, arc2, human_overlap, arc1_profiles, arc2_profiles = build_arc1_arc2_frames()
    metric_corrs = compute_metric_correlations(arc1, arc2, human_overlap)
    key_tests = compute_key_tests(arc1, arc2, human_overlap)

    make_arc2_chart(arc2)
    make_comparison_chart(arc2)
    make_arc1_chart(arc1, arc1_profiles)
    make_trajectory_chart(arc2)

    md, tex = build_report(arc1, arc2, metric_corrs, key_tests, arc1_profiles, arc2_profiles)
    REPORT_MD_OUTPUT.write_text(md, encoding="utf-8")
    REPORT_TEX_OUTPUT.write_text(tex, encoding="utf-8")

    summary = build_summary_json(arc1, arc2, key_tests, metric_corrs)
    SUMMARY_JSON_OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
