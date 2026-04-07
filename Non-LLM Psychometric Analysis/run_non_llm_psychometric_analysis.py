from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ANALYSIS_DIR / "figures"
TABLES_DIR = ANALYSIS_DIR / "tables"

ARC2_TRUTH_DIR = ROOT_DIR / "Psychometric Analysis" / "data" / "ARC-AGI-2" / "data" / "evaluation"
LLM_PREDS_DIR = ROOT_DIR / "Psychometric Analysis" / "data" / "arc_agi_v2_public_eval"
HUMAN_RAW_CSV = ROOT_DIR / "Human data" / "test_pair_attempts.csv"
HUMAN_PUBLIC_EVAL_CSV = ROOT_DIR / "Human data" / "analysis" / "tables" / "public_eval_human_vs_models.csv"

TRM_ROOT = ROOT_DIR / "Non-LLM data" / "raw" / "TRM-ARC-AGI-II"
VARC_ROOT = ROOT_DIR / "Non-LLM data" / "raw" / "VARC_predictions" / "VARC_predictions"
COMPRESS_ARC_SUMMARY = ROOT_DIR / "Non-LLM data" / "processed" / "compress_arc_predictions_evaluation_summary.json"

PRIMARY_THRESHOLD = 8
THRESHOLDS = [2, 3, 5, 8]
FEATURE_COLUMNS = [
    "input_cells",
    "input_colors",
    "output_cells",
    "output_colors",
    "n_train_pairs",
    "size_change_ratio",
]
FEATURE_LABELS = {
    "input_cells": "Input cells",
    "input_colors": "Input colors",
    "output_cells": "Output cells",
    "output_colors": "Output colors",
    "n_train_pairs": "Train pairs",
    "size_change_ratio": "Size change",
}


@dataclass
class HumanSubset:
    threshold: int
    pairs: list[str]
    human: pd.Series
    frame: pd.DataFrame
    raw_attempts: pd.DataFrame


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def frame_to_text_table(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def normalize_grid(grid: object) -> str:
    if not isinstance(grid, list) or not grid:
        return "EMPTY"
    return ",".join(str(cell) for row in grid for cell in row)


def family_for_model(model: str) -> str:
    if model.startswith(("gpt-5", "gpt-4")):
        return "GPT"
    if model.startswith("claude-opus"):
        return "Claude Opus"
    if model.startswith("claude-sonnet"):
        return "Claude Sonnet"
    if model.startswith("claude-haiku"):
        return "Claude Haiku"
    if model.startswith("gemini"):
        return "Gemini"
    if model.startswith("grok"):
        return "Grok"
    if model.startswith("qwen"):
        return "Qwen"
    return "Other"


def system_kind(name: str) -> str:
    if name.startswith("TRM"):
        return "TRM"
    if name.startswith("VARC"):
        return "VARC"
    if name.startswith("Family "):
        return "LLM family"
    if name == "LLM average":
        return "LLM aggregate"
    return "LLM"


def corr_or_nan(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, method: str = "pearson") -> float:
    s1 = pd.Series(a)
    s2 = pd.Series(b)
    if s1.nunique() < 2 or s2.nunique() < 2:
        return float("nan")
    return float(s1.corr(s2, method=method))


def bootstrap_corr(y: np.ndarray, x: np.ndarray, n_boot: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    x_boot = x[idx]
    y_boot = y[idx]
    x_centered = x_boot - x_boot.mean(axis=1, keepdims=True)
    y_centered = y_boot - y_boot.mean(axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.sum(x_centered * x_centered, axis=1) * np.sum(y_centered * y_centered, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = numerator / denominator
    return draws[np.isfinite(draws)]


def partial_corr(y: pd.Series, x: pd.Series, controls: pd.DataFrame | pd.Series) -> float:
    if np.array_equal(np.asarray(x), np.asarray(controls).ravel()):
        return float("nan")
    controls_df = pd.DataFrame(controls).copy()
    controls_df = controls_df.fillna(0.0)
    lr = LinearRegression().fit(controls_df, y)
    y_res = y - lr.predict(controls_df)
    lr = LinearRegression().fit(controls_df, x)
    x_res = x - lr.predict(controls_df)
    return corr_or_nan(y_res, x_res)


def load_arc2_truth() -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict] = []
    truth_outputs: dict[str, str] = {}
    for path in sorted(ARC2_TRUTH_DIR.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        for idx, pair in enumerate(obj.get("test", [])):
            pair_id = f"{path.stem}__{idx}"
            truth_outputs[pair_id] = normalize_grid(pair["output"])
            rows.append({"task_pair_id": pair_id, "task_id": path.stem, "test_index": idx})
    return pd.DataFrame(rows), truth_outputs


def load_human_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(HUMAN_RAW_CSV)
    raw = raw[raw["task_set"] == "Public Eval"].copy()
    raw["task_pair_id"] = raw["task_ID"] + "__" + raw["test_index"].astype(str)
    raw["solved"] = (raw["correct_submissions"] > 0).astype(int)

    item_attempts = (
        raw.groupby("task_pair_id")
        .agg(human_attempts=("solved", "size"), human_solve_rate=("solved", "mean"))
        .reset_index()
    )

    public_eval = pd.read_csv(HUMAN_PUBLIC_EVAL_CSV)
    public_eval = public_eval.rename(
        columns={
            "task_ID": "task_id",
            "attempts": "human_attempts_existing",
            "solve_rate": "human_solve_rate_existing",
        }
    )
    public_eval = public_eval.merge(item_attempts, on="task_pair_id", how="left")
    return raw, public_eval


def build_llm_matrix(pair_ids: list[str], truth_outputs: dict[str, str]) -> pd.DataFrame:
    pair_set = set(pair_ids)
    rows: dict[str, dict[str, int]] = {}
    for model_dir in sorted(LLM_PREDS_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        row = {pair_id: 0 for pair_id in pair_ids}
        for pred_path in model_dir.glob("*.json"):
            pred_obj = json.loads(pred_path.read_text(encoding="utf-8"))
            for idx in range(len(pred_obj)):
                pair_id = f"{pred_path.stem}__{idx}"
                if pair_id not in pair_set:
                    continue
                pred_entry = None
                for candidate in pred_obj:
                    if candidate.get("metadata", {}).get("pair_index") == idx:
                        pred_entry = candidate
                        break
                if pred_entry is None and idx < len(pred_obj):
                    pred_entry = pred_obj[idx]
                answer = None
                if pred_entry:
                    answer = (pred_entry.get("attempt_1") or {}).get("answer")
                    if not answer:
                        answer = (pred_entry.get("attempt_2") or {}).get("answer")
                row[pair_id] = int(normalize_grid(answer) == truth_outputs[pair_id])
        rows[model_dir.name] = row
    return pd.DataFrame.from_dict(rows, orient="index")[pair_ids]


def build_trm_matrix(pair_ids: list[str], truth_outputs: dict[str, str]) -> pd.DataFrame:
    rows: dict[str, dict[str, int]] = {}
    pair_set = set(pair_ids)
    for submission_path in sorted(TRM_ROOT.glob("evaluator_ARC_step_*/submission.json")):
        step = int(submission_path.parent.name.split("_")[-1])
        submission = json.loads(submission_path.read_text(encoding="utf-8"))
        pass1 = {pair_id: 0 for pair_id in pair_ids}
        pass2 = {pair_id: 0 for pair_id in pair_ids}
        for task_id, entries in submission.items():
            for idx, pred_entry in enumerate(entries):
                pair_id = f"{task_id}__{idx}"
                if pair_id not in pair_set or not isinstance(pred_entry, dict):
                    continue
                a1 = pred_entry.get("attempt_1")
                a2 = pred_entry.get("attempt_2")
                gold = truth_outputs[pair_id]
                a1_ok = normalize_grid(a1) == gold
                a2_ok = normalize_grid(a2) == gold
                pass1[pair_id] = int(a1_ok)
                pass2[pair_id] = int(a1_ok or a2_ok)
        rows[f"TRM {step} pass@1"] = pass1
        rows[f"TRM {step} pass@2"] = pass2
    return pd.DataFrame.from_dict(rows, orient="index")[pair_ids]


def build_varc_matrix(pair_ids: list[str], truth_outputs: dict[str, str]) -> pd.DataFrame:
    rows: dict[str, dict[str, int]] = {}
    for model_name in ["ARC-2_Unet", "ARC-2_ViT"]:
        model_dir = VARC_ROOT / model_name
        attempt_dirs = sorted([path for path in model_dir.iterdir() if path.is_dir()], key=lambda path: path.name)
        cache: dict[str, dict[str, dict]] = {}
        for attempt_dir in attempt_dirs:
            cache[attempt_dir.name] = {
                path.stem.replace("_predictions", ""): json.loads(path.read_text(encoding="utf-8"))
                for path in attempt_dir.glob("*.json")
            }
        for depth in range(1, len(attempt_dirs) + 1):
            row = {pair_id: 0 for pair_id in pair_ids}
            for pair_id in pair_ids:
                task_id, test_index = pair_id.split("__")
                gold = truth_outputs[pair_id]
                ok = False
                for attempt_dir in attempt_dirs[:depth]:
                    task_obj = cache[attempt_dir.name].get(task_id)
                    if not task_obj:
                        continue
                    candidates = task_obj.get(test_index)
                    if isinstance(candidates, list) and candidates and normalize_grid(candidates[0]) == gold:
                        ok = True
                        break
                row[pair_id] = int(ok)
            rows[f"VARC {model_name} pass@{depth}"] = row
    return pd.DataFrame.from_dict(rows, orient="index")[pair_ids]


def build_family_averages(llm_matrix: pd.DataFrame) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {"LLM average": llm_matrix.mean(axis=0)}
    families: dict[str, list[str]] = {}
    for model in llm_matrix.index:
        family = family_for_model(model)
        if family == "Other":
            continue
        families.setdefault(family, []).append(model)
    for family, members in sorted(families.items()):
        if len(members) < 2:
            continue
        rows[f"Family {family}"] = llm_matrix.loc[members].mean(axis=0)
    return pd.DataFrame(rows).T[llm_matrix.columns]


def build_human_subset(human_meta: pd.DataFrame, human_raw: pd.DataFrame, threshold: int) -> HumanSubset:
    frame = human_meta.loc[human_meta["human_attempts"] >= threshold].copy()
    frame = frame.sort_values("task_pair_id")
    pairs = frame["task_pair_id"].tolist()
    human = frame.set_index("task_pair_id")["human_solve_rate"]
    raw_attempts = human_raw.loc[human_raw["task_pair_id"].isin(pairs)].copy()
    return HumanSubset(threshold=threshold, pairs=pairs, human=human, frame=frame, raw_attempts=raw_attempts)


def split_half_correlations(human_attempts: pd.DataFrame, n_sims: int = 3000, seed: int = 0) -> pd.DataFrame:
    sessions = np.array(sorted(human_attempts["session_ID"].unique()))
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for _ in range(n_sims):
        perm = rng.permutation(sessions)
        half_a = set(perm[: len(perm) // 2])
        part_a = human_attempts.loc[human_attempts["session_ID"].isin(half_a)].groupby("task_pair_id").agg(
            rate=("solved", "mean"),
            n=("solved", "size"),
        )
        part_b = human_attempts.loc[~human_attempts["session_ID"].isin(half_a)].groupby("task_pair_id").agg(
            rate=("solved", "mean"),
            n=("solved", "size"),
        )
        merged = part_a.join(part_b, lsuffix="_a", rsuffix="_b", how="inner")
        merged = merged.loc[(merged["n_a"] >= 2) & (merged["n_b"] >= 2)]
        if len(merged) < 20:
            continue
        rows.append(
            {
                "pearson": float(merged["rate_a"].corr(merged["rate_b"])),
                "spearman": float(merged["rate_a"].corr(merged["rate_b"], method="spearman")),
                "n_items": int(len(merged)),
            }
        )
    return pd.DataFrame(rows)


def summarize_systems(
    subset: HumanSubset,
    system_matrix: pd.DataFrame,
    n_boot: int = 4000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_halves = split_half_correlations(subset.raw_attempts, n_sims=3000, seed=subset.threshold)
    human = subset.human
    llm_average = system_matrix.loc["LLM average", subset.pairs].astype(float)
    feature_controls = subset.frame.set_index("task_pair_id").loc[subset.pairs, FEATURE_COLUMNS].fillna(0.0)

    rows: list[dict] = []
    for seed, system in enumerate(system_matrix.index):
        profile = system_matrix.loc[system, subset.pairs].astype(float)
        pearson = corr_or_nan(profile, human)
        spearman = corr_or_nan(profile, human, method="spearman")
        draws = bootstrap_corr(human.to_numpy(), profile.to_numpy(), n_boot=n_boot, seed=seed)
        percentile_vs_split = float((split_halves["pearson"] <= pearson).mean()) if not split_halves.empty else float("nan")
        rows.append(
            {
                "threshold": subset.threshold,
                "n_pairs": len(subset.pairs),
                "system": system,
                "kind": system_kind(system),
                "family": family_for_model(system) if system_kind(system) == "LLM" else "",
                "pair_accuracy": float(profile.mean()),
                "human_pearson": pearson,
                "human_spearman": spearman,
                "bootstrap_ci_lo": float(np.quantile(draws, 0.025)) if len(draws) else float("nan"),
                "bootstrap_ci_hi": float(np.quantile(draws, 0.975)) if len(draws) else float("nan"),
                "percentile_vs_human_split": percentile_vs_split,
                "corr_with_llm_average": corr_or_nan(profile, llm_average),
                "corr_with_human_residual": corr_or_nan(profile, human - llm_average),
                "partial_corr_raw_features": partial_corr(human, profile, feature_controls),
                "partial_corr_given_llm_average": (
                    float("nan") if system == "LLM average" else partial_corr(human, profile, llm_average)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["human_pearson", "pair_accuracy"], ascending=[False, False]), split_halves


def build_threshold_sensitivity(
    human_meta: pd.DataFrame,
    human_raw: pd.DataFrame,
    system_matrix: pd.DataFrame,
    systems: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitivity_rows: list[dict] = []
    split_rows: list[dict] = []
    for threshold in THRESHOLDS:
        subset = build_human_subset(human_meta, human_raw, threshold)
        split_halves = split_half_correlations(subset.raw_attempts, n_sims=3000, seed=threshold)
        split_rows.append(
            {
                "threshold": threshold,
                "n_pairs": len(subset.pairs),
                "n_draws": len(split_halves),
                "human_split_median": float(split_halves["pearson"].median()),
                "human_split_ci_lo": float(split_halves["pearson"].quantile(0.025)),
                "human_split_ci_hi": float(split_halves["pearson"].quantile(0.975)),
            }
        )
        llm_average = system_matrix.loc["LLM average", subset.pairs].astype(float)
        for system in systems:
            profile = system_matrix.loc[system, subset.pairs].astype(float)
            pearson = corr_or_nan(profile, subset.human)
            sensitivity_rows.append(
                {
                    "threshold": threshold,
                    "n_pairs": len(subset.pairs),
                    "system": system,
                    "pair_accuracy": float(profile.mean()),
                    "human_pearson": pearson,
                    "human_spearman": corr_or_nan(profile, subset.human, method="spearman"),
                    "percentile_vs_human_split": float((split_halves["pearson"] <= pearson).mean()),
                    "corr_with_llm_average": corr_or_nan(profile, llm_average),
                    "corr_with_human_residual": corr_or_nan(profile, subset.human - llm_average),
                }
            )
    return pd.DataFrame(sensitivity_rows), pd.DataFrame(split_rows)


def build_feature_sensitivity(subset: HumanSubset, system_matrix: pd.DataFrame, systems: list[str]) -> pd.DataFrame:
    frame = subset.frame.set_index("task_pair_id").loc[subset.pairs]
    rows: list[dict] = []
    for system in systems:
        if system == "Human":
            profile = subset.human
        else:
            profile = system_matrix.loc[system, subset.pairs].astype(float)
        for feature in FEATURE_COLUMNS + ["mean_duration_seconds"]:
            rows.append(
                {
                    "system": system,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, "Mean duration"),
                    "pearson": corr_or_nan(profile, frame[feature]),
                    "spearman": corr_or_nan(profile, frame[feature], method="spearman"),
                }
            )
    return pd.DataFrame(rows)


def build_difficulty_bins(subset: HumanSubset, system_matrix: pd.DataFrame, systems: list[str]) -> pd.DataFrame:
    frame = subset.frame.set_index("task_pair_id").loc[subset.pairs].copy()
    frame["Human"] = subset.human
    frame["difficulty_bin"] = pd.qcut(frame["Human"], q=4, duplicates="drop")
    rows: list[dict] = []
    for system in systems:
        if system != "Human":
            frame[system] = system_matrix.loc[system, subset.pairs].astype(float)
        grouped = frame.groupby("difficulty_bin", observed=False)[system].mean().reset_index()
        for _, row in grouped.iterrows():
            rows.append({"system": system, "difficulty_bin": str(row["difficulty_bin"]), "mean_success": float(row[system])})
    return pd.DataFrame(rows)


def build_geometry(subset: HumanSubset, system_matrix: pd.DataFrame, systems: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles: dict[str, pd.Series] = {"Human": subset.human}
    for system in systems:
        profiles[system] = system_matrix.loc[system, subset.pairs].astype(float)
    profile_df = pd.DataFrame(profiles).T
    corr_df = profile_df.T.corr()
    centered = profile_df.sub(profile_df.mean(axis=1), axis=0)
    scaled = centered.div(profile_df.std(axis=1).replace(0, 1), axis=0)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled.to_numpy())
    coord_df = pd.DataFrame(coords, columns=["pc1", "pc2"], index=scaled.index).reset_index(names="system")
    coord_df["kind"] = coord_df["system"].map(lambda name: "Human" if name == "Human" else system_kind(name))
    coord_df["explained_variance_ratio"] = (
        f"{pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}"
    )
    return corr_df, coord_df


def build_trm_trajectory(primary_summary: pd.DataFrame) -> pd.DataFrame:
    trm = primary_summary.loc[primary_summary["kind"] == "TRM"].copy()
    trm["step"] = trm["system"].str.extract(r"TRM (\d+)").astype(int)
    trm["score_mode"] = trm["system"].str.extract(r"(pass@\d)")
    return trm.sort_values(["score_mode", "step"])


def build_varc_depth(primary_summary: pd.DataFrame) -> pd.DataFrame:
    varc = primary_summary.loc[primary_summary["kind"] == "VARC"].copy()
    varc["model"] = varc["system"].str.extract(r"VARC (ARC-2_[^ ]+)")
    varc["depth"] = varc["system"].str.extract(r"pass@(\d+)").astype(int)
    return varc.sort_values(["model", "depth"])


def build_complementarity_table(
    subset: HumanSubset,
    system_matrix: pd.DataFrame,
    best_score_llm: str,
    best_trm: str,
    best_varc: str,
) -> pd.DataFrame:
    frame = subset.frame.set_index("task_pair_id").loc[subset.pairs].copy()
    frame["Human"] = subset.human
    frame["LLM average"] = system_matrix.loc["LLM average", subset.pairs].astype(float)
    frame[best_score_llm] = system_matrix.loc[best_score_llm, subset.pairs].astype(float)
    frame[best_trm] = system_matrix.loc[best_trm, subset.pairs].astype(float)
    frame[best_varc] = system_matrix.loc[best_varc, subset.pairs].astype(float)
    frame["human_easy"] = frame["Human"] >= 0.7
    frame["llm_hard"] = frame["LLM average"] <= 0.1
    table = frame.loc[frame["human_easy"] & frame["llm_hard"]].copy()
    table["rescued_by_trm"] = table[best_trm] > 0
    table["rescued_by_varc"] = table[best_varc] > 0
    table["rescued_by_any_nonllm"] = table["rescued_by_trm"] | table["rescued_by_varc"]
    keep = ["Human", "LLM average", best_score_llm, best_trm, best_varc, "rescued_by_trm", "rescued_by_varc", "rescued_by_any_nonllm"]
    return table[keep].sort_values(["rescued_by_any_nonllm", "Human"], ascending=[False, False])


def plot_accuracy_alignment_scatter(summary: pd.DataFrame, selected_systems: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    llm = summary.loc[summary["kind"] == "LLM"].copy()
    ax.scatter(llm["pair_accuracy"], llm["human_pearson"], s=45, alpha=0.35, color="#8D99AE", label="Single LLMs")

    palette = {"LLM aggregate": "#F58518", "TRM": "#4C78A8", "VARC": "#54A24B", "LLM family": "#B279A2"}
    for system in selected_systems:
        row = summary.loc[summary["system"] == system].iloc[0]
        color = palette.get(row["kind"], "#333333")
        ax.scatter(row["pair_accuracy"], row["human_pearson"], s=110, color=color, edgecolor="white", linewidth=1.0)
        ax.text(row["pair_accuracy"] + 0.004, row["human_pearson"] + 0.006, system, fontsize=9)

    ax.set_title("ARC-2 Public Eval: accuracy vs human-alignment")
    ax.set_xlabel("Pair accuracy on the human-overlap subset")
    ax.set_ylabel("Pearson correlation with human solve rates")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)


def plot_threshold_sensitivity(sensitivity: pd.DataFrame, split_summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for system, group in sensitivity.groupby("system"):
        ax.plot(group["threshold"], group["human_pearson"], marker="o", linewidth=2, label=system)
    ax.plot(split_summary["threshold"], split_summary["human_split_median"], color="#1F3552", linestyle="--", linewidth=2, label="Human split-half median")
    ax.fill_between(
        split_summary["threshold"],
        split_summary["human_split_ci_lo"],
        split_summary["human_split_ci_hi"],
        color="#9FD0CB",
        alpha=0.20,
        label="Human split-half 95% interval",
    )
    ax.set_title("Human-alignment is stable across coverage thresholds")
    ax.set_xlabel("Minimum human attempts per ARC-2 test pair")
    ax.set_ylabel("Pearson correlation with human solve rate")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(out_path)
    plt.close(fig)


def plot_bootstrap_context(
    split_halves: pd.DataFrame,
    summary: pd.DataFrame,
    systems: list[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(split_halves["pearson"], bins=32, color="#4C78A8", alpha=0.70, ax=ax)
    ax.axvline(float(split_halves["pearson"].median()), color="#1F3552", linewidth=2.5, label="Human split-half median")
    ax.axvspan(
        float(split_halves["pearson"].quantile(0.025)),
        float(split_halves["pearson"].quantile(0.975)),
        color="#4C78A8",
        alpha=0.10,
        label="Human split-half 95% interval",
    )
    colors = {"LLM average": "#F58518", "TRM": "#4C78A8", "VARC": "#54A24B", "LLM": "#E45756"}
    for system in systems:
        row = summary.loc[summary["system"] == system].iloc[0]
        color = colors.get(system_kind(system), "#333333")
        ax.axvspan(row["bootstrap_ci_lo"], row["bootstrap_ci_hi"], color=color, alpha=0.12)
        ax.axvline(row["human_pearson"], color=color, linestyle="--", linewidth=2.5, label=system)
    ax.set_title("Human-vs-system correlations in split-half context")
    ax.set_xlabel("Pearson correlation across ARC-2 test pairs")
    ax.set_ylabel("Random human split-half simulations")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_heatmap(feature_df: pd.DataFrame, out_path: Path) -> None:
    pivot = feature_df.pivot(index="system", columns="feature_label", values="pearson")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0.0, vmin=-0.5, vmax=0.5, ax=ax)
    ax.set_title("Feature sensitivity differs across humans, LLMs, and non-LLMs")
    fig.savefig(out_path)
    plt.close(fig)


def plot_difficulty_curves(bin_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    order = list(dict.fromkeys(bin_df["difficulty_bin"]))
    sns.lineplot(data=bin_df, x="difficulty_bin", y="mean_success", hue="system", marker="o", linewidth=2, ax=ax, sort=False)
    ax.set_title("Non-LLM systems do not follow the human difficulty gradient cleanly")
    ax.set_xlabel("Human solve-rate quartile")
    ax.set_ylabel("Mean success")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(out_path)
    plt.close(fig)


def plot_geometry(coord_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"Human": "#1F3552", "LLM aggregate": "#F58518", "LLM family": "#B279A2", "LLM": "#E45756", "TRM": "#4C78A8", "VARC": "#54A24B"}
    for _, row in coord_df.iterrows():
        ax.scatter(row["pc1"], row["pc2"], s=110, color=colors.get(row["kind"], "#333333"))
        ax.text(row["pc1"] + 0.02, row["pc2"] + 0.02, row["system"], fontsize=9)
    ax.axhline(0, color="#DDDDDD", linewidth=1)
    ax.axvline(0, color="#DDDDDD", linewidth=1)
    ax.set_title("System geometry on ARC-2 item profiles")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.savefig(out_path)
    plt.close(fig)


def plot_trm_trajectory(trm_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    sns.lineplot(data=trm_df, x="step", y="pair_accuracy", hue="score_mode", marker="o", linewidth=2, ax=axes[0])
    axes[0].set_title("TRM ARC-2 score rises over training")
    axes[0].set_ylabel("Pair accuracy")
    axes[0].legend(frameon=False)

    sns.lineplot(data=trm_df, x="step", y="human_pearson", hue="score_mode", marker="o", linewidth=2, ax=axes[1], legend=False)
    axes[1].set_title("Human-alignment peaks earlier and then drops")
    axes[1].set_ylabel("Pearson correlation with human solve rate")
    fig.savefig(out_path)
    plt.close(fig)


def plot_varc_depth(varc_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    sns.lineplot(data=varc_df, x="depth", y="pair_accuracy", hue="model", marker="o", linewidth=2, ax=axes[0])
    axes[0].set_title("VARC accuracy improves with more guesses")
    axes[0].set_ylabel("Pair accuracy")
    axes[0].legend(frameon=False)

    sns.lineplot(data=varc_df, x="depth", y="human_pearson", hue="model", marker="o", linewidth=2, ax=axes[1], legend=False)
    axes[1].set_title("VARC human-alignment rises only modestly")
    axes[1].set_ylabel("Pearson correlation with human solve rate")
    fig.savefig(out_path)
    plt.close(fig)


def write_report(
    primary_summary: pd.DataFrame,
    split_halves: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    split_thresholds: pd.DataFrame,
    feature_sensitivity: pd.DataFrame,
    complementarity: pd.DataFrame,
    trm_trajectory: pd.DataFrame,
    compress_arc_summary: dict,
    selected_systems: list[str],
    best_score_llm: str,
    best_aligned_llm: str,
    best_trm_score: str,
    best_trm_alignment: str,
    best_varc: str,
) -> None:
    llm_average = primary_summary.loc[primary_summary["system"] == "LLM average"].iloc[0]
    best_score_llm_row = primary_summary.loc[primary_summary["system"] == best_score_llm].iloc[0]
    best_aligned_llm_row = primary_summary.loc[primary_summary["system"] == best_aligned_llm].iloc[0]
    best_trm_score_row = primary_summary.loc[primary_summary["system"] == best_trm_score].iloc[0]
    best_trm_alignment_row = primary_summary.loc[primary_summary["system"] == best_trm_alignment].iloc[0]
    best_varc_row = primary_summary.loc[primary_summary["system"] == best_varc].iloc[0]
    split_median = float(split_halves["pearson"].median())
    split_ci_lo = float(split_halves["pearson"].quantile(0.025))
    split_ci_hi = float(split_halves["pearson"].quantile(0.975))
    complementarity_count = int(complementarity["rescued_by_any_nonllm"].sum())
    complementarity_total = int(len(complementarity))
    trm_best_step = trm_trajectory.loc[trm_trajectory["human_pearson"].idxmax(), "step"]
    trm_peak_corr = trm_trajectory["human_pearson"].max()

    lines = [
        "# Non-LLM Psychometric Analysis",
        "",
        "## Setup",
        "",
        "- Main comparison space: ARC-AGI-2 Public Eval test pairs, because that is where the human attempt data and the stored LLM prediction corpus overlap.",
        f"- Primary threshold: at least {PRIMARY_THRESHOLD} human attempts per test pair (`{int(primary_summary['n_pairs'].iloc[0])}` pairs).",
        "- Main human benchmark: item-level solve rates, with human split-half correlations used as a reliability reference.",
        "- Main non-LLM sources: TRM ARC-AGI-II evaluator submissions and VARC ARC-2 prediction dumps.",
        "- CompressARC is included only as an ARC-1 sidecar, because it does not overlap the human ARC-2 set.",
        "",
        "## Main Findings",
        "",
        f"- The strongest human-like profile is still the `LLM average`, with Pearson `{llm_average['human_pearson']:.3f}` and Spearman `{llm_average['human_spearman']:.3f}` on the well-sampled ARC-2 subset.",
        f"- That lands at the `{100 * llm_average['percentile_vs_human_split']:.1f}`th percentile of the human split-half distribution; the human split-half median is `{split_median:.3f}` with a 95% interval of `[{split_ci_lo:.3f}, {split_ci_hi:.3f}]`.",
        f"- The best-score single LLM on this subset is `{best_score_llm}` at pair accuracy `{best_score_llm_row['pair_accuracy']:.3f}` and human-correlation `{best_score_llm_row['human_pearson']:.3f}`.",
        f"- The most human-aligned single LLM is `{best_aligned_llm}` at Pearson `{best_aligned_llm_row['human_pearson']:.3f}` and pair accuracy `{best_aligned_llm_row['pair_accuracy']:.3f}`.",
        f"- The best-score TRM profile is `{best_trm_score}` at pair accuracy `{best_trm_score_row['pair_accuracy']:.3f}`, but its human-correlation is only `{best_trm_score_row['human_pearson']:.3f}`.",
        f"- The most human-aligned TRM profile is `{best_trm_alignment}` at Pearson `{best_trm_alignment_row['human_pearson']:.3f}`, still far below the LLM average and below many single LLMs.",
        f"- The best VARC profile is `{best_varc}` at pair accuracy `{best_varc_row['pair_accuracy']:.3f}` and human-correlation `{best_varc_row['human_pearson']:.3f}`.",
        "",
        "## Interpretation",
        "",
        "- On ARC-2, the current non-LLM systems do not reproduce the human difficulty structure nearly as well as the LLM consensus profile does.",
        "- But they are not just trivial copies of the LLM average either: they solve a few human-easy, LLM-hard items and add some orthogonal signal.",
        f"- On the primary subset there are `{complementarity_total}` human-easy / LLM-hard pairs, and the best non-LLM systems rescue `{complementarity_count}` of them.",
        f"- TRM is especially interesting because human-alignment peaks in the middle of training (around step `{int(trm_best_step)}`; peak Pearson `{trm_peak_corr:.3f}`) and then drops as ARC score continues to rise. That suggests optimization is moving the model away from the human item profile, not toward it.",
        "- So the cleanest answer from the current data is: humans and the LLM average share a stronger common difficulty axis than humans and these non-LLM systems do.",
        "",
        "## Threshold Sensitivity",
        "",
        "- I reran the main item-correlation comparison at thresholds of 2, 3, 5, and 8 human attempts per pair.",
        "- The ranking is stable: the LLM average stays on top, while TRM and VARC remain much lower across thresholds.",
        "",
        frame_to_text_table(
            threshold_sensitivity[
                ["threshold", "n_pairs", "system", "pair_accuracy", "human_pearson", "percentile_vs_human_split"]
            ].round(3)
        ),
        "",
        "## Feature Pattern",
        "",
        "- The LLM average still tracks human-like structure after controlling for raw item features.",
        "- The non-LLM systems are much less tied to the human difficulty gradient and show weaker or stranger feature sensitivities, especially around color-count cues.",
        "",
        frame_to_text_table(
            feature_sensitivity.loc[
                feature_sensitivity["feature"].isin(["input_colors", "output_colors", "input_cells", "mean_duration_seconds"])
            ][["system", "feature_label", "pearson"]].round(3)
        ),
        "",
        "## ARC-1 Sidecar",
        "",
        f"- CompressARC is the one clearly valid non-LLM prediction artifact we have on ARC-1 rather than ARC-2.",
        f"- Stored ARC-1 scores: final top-1 `{compress_arc_summary['metrics']['final_pick_pass1']['percentage']:.2f}%`, final top-2 `{compress_arc_summary['metrics']['final_pick_pass2']['percentage']:.2f}%`, ranked-anywhere `{compress_arc_summary['metrics']['ranked_candidate_solved_anywhere']['percentage']:.2f}%`.",
        "- It is useful as a bona fide non-LLM prediction archive, but it cannot answer the ARC-2 human-vs-LLM question directly.",
        "",
        "## Bottom Line",
        "",
        "- The best evidence here favors a mixed conclusion: humans and the LLM consensus are measurably similar in item difficulty structure on ARC-2, but the non-LLM systems we have do not currently share that similarity to the same degree.",
        "- Non-LLM systems are not useless or redundant; they add some complementary successes. But in psychometric terms they look more like partial, idiosyncratic alternative solvers than human-like replicas.",
        "",
        "## Selected Systems",
        "",
        frame_to_text_table(
            primary_summary.loc[primary_summary["system"].isin(selected_systems)][
                ["system", "pair_accuracy", "human_pearson", "human_spearman", "percentile_vs_human_split", "corr_with_human_residual"]
            ].round(3)
        ),
        "",
    ]
    (ANALYSIS_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_style()
    ensure_dirs()

    _, truth_outputs = load_arc2_truth()
    human_raw, human_meta = load_human_inputs()
    all_pair_ids = sorted(human_meta["task_pair_id"].unique())

    llm_matrix = build_llm_matrix(all_pair_ids, truth_outputs)
    trm_matrix = build_trm_matrix(all_pair_ids, truth_outputs)
    varc_matrix = build_varc_matrix(all_pair_ids, truth_outputs)
    family_matrix = build_family_averages(llm_matrix)
    system_matrix = pd.concat([llm_matrix, family_matrix, trm_matrix, varc_matrix], axis=0)
    system_matrix = system_matrix.loc[~system_matrix.index.duplicated(keep="first")]

    primary_subset = build_human_subset(human_meta, human_raw, PRIMARY_THRESHOLD)
    primary_summary, split_halves = summarize_systems(primary_subset, system_matrix, n_boot=4000)

    llm_only = primary_summary.loc[primary_summary["kind"] == "LLM"].copy()
    best_score_llm = llm_only.sort_values(["pair_accuracy", "human_pearson"], ascending=[False, False]).iloc[0]["system"]
    best_aligned_llm = llm_only.sort_values(["human_pearson", "pair_accuracy"], ascending=[False, False]).iloc[0]["system"]

    trm_only = primary_summary.loc[primary_summary["kind"] == "TRM"].copy()
    best_trm_score = trm_only.sort_values(["pair_accuracy", "human_pearson"], ascending=[False, False]).iloc[0]["system"]
    trm_pass2 = trm_only.loc[trm_only["system"].str.endswith("pass@2")].copy()
    best_trm_alignment = trm_pass2.sort_values(["human_pearson", "pair_accuracy"], ascending=[False, False]).iloc[0]["system"]

    varc_only = primary_summary.loc[primary_summary["kind"] == "VARC"].copy()
    best_varc = varc_only.sort_values(["pair_accuracy", "human_pearson"], ascending=[False, False]).iloc[0]["system"]

    selected_systems = list(
        dict.fromkeys(
            [
                "LLM average",
                best_score_llm,
                best_aligned_llm,
                best_trm_score,
                best_trm_alignment,
                best_varc,
                "Family GPT",
                "Family Claude Opus",
                "Family Gemini",
            ]
        )
    )
    selected_systems = [system for system in selected_systems if system in system_matrix.index]

    threshold_sensitivity, split_thresholds = build_threshold_sensitivity(
        human_meta=human_meta,
        human_raw=human_raw,
        system_matrix=system_matrix,
        systems=[system for system in selected_systems if system in system_matrix.index and not system.startswith("Family ")],
    )
    feature_sensitivity = build_feature_sensitivity(
        subset=primary_subset,
        system_matrix=system_matrix,
        systems=["Human", "LLM average", best_score_llm, best_trm_score, best_trm_alignment, best_varc],
    )
    difficulty_bins = build_difficulty_bins(
        subset=primary_subset,
        system_matrix=system_matrix,
        systems=["Human", "LLM average", best_score_llm, best_trm_score, best_trm_alignment, best_varc],
    )
    corr_matrix, geometry_coords = build_geometry(primary_subset, system_matrix, selected_systems)
    trm_trajectory = build_trm_trajectory(primary_summary)
    varc_depth = build_varc_depth(primary_summary)
    complementarity = build_complementarity_table(
        subset=primary_subset,
        system_matrix=system_matrix,
        best_score_llm=best_score_llm,
        best_trm=best_trm_score,
        best_varc=best_varc,
    )
    compress_arc_summary = json.loads(COMPRESS_ARC_SUMMARY.read_text(encoding="utf-8"))

    primary_summary.to_csv(TABLES_DIR / "system_summary.csv", index=False)
    split_halves.to_csv(TABLES_DIR / "split_half_draws_primary.csv", index=False)
    threshold_sensitivity.to_csv(TABLES_DIR / "threshold_sensitivity.csv", index=False)
    split_thresholds.to_csv(TABLES_DIR / "split_half_thresholds.csv", index=False)
    feature_sensitivity.to_csv(TABLES_DIR / "feature_sensitivity.csv", index=False)
    difficulty_bins.to_csv(TABLES_DIR / "difficulty_bin_summary.csv", index=False)
    corr_matrix.to_csv(TABLES_DIR / "geometry_correlation_matrix.csv")
    geometry_coords.to_csv(TABLES_DIR / "geometry_coordinates.csv", index=False)
    trm_trajectory.to_csv(TABLES_DIR / "trm_trajectory.csv", index=False)
    varc_depth.to_csv(TABLES_DIR / "varc_depth.csv", index=False)
    complementarity.to_csv(TABLES_DIR / "complementarity_items.csv")

    plot_accuracy_alignment_scatter(primary_summary, selected_systems, FIGURES_DIR / "fig01_accuracy_alignment_scatter.png")
    plot_threshold_sensitivity(threshold_sensitivity, split_thresholds, FIGURES_DIR / "fig02_threshold_sensitivity.png")
    plot_bootstrap_context(
        split_halves=split_halves,
        summary=primary_summary,
        systems=["LLM average", best_score_llm, best_trm_score, best_varc],
        out_path=FIGURES_DIR / "fig03_bootstrap_context.png",
    )
    plot_feature_heatmap(feature_sensitivity, FIGURES_DIR / "fig04_feature_sensitivity_heatmap.png")
    plot_difficulty_curves(difficulty_bins, FIGURES_DIR / "fig05_difficulty_curves.png")
    plot_geometry(geometry_coords, FIGURES_DIR / "fig06_geometry_map.png")
    plot_trm_trajectory(trm_trajectory, FIGURES_DIR / "fig07_trm_trajectory.png")
    plot_varc_depth(varc_depth, FIGURES_DIR / "fig08_varc_depth.png")

    write_report(
        primary_summary=primary_summary,
        split_halves=split_halves,
        threshold_sensitivity=threshold_sensitivity,
        split_thresholds=split_thresholds,
        feature_sensitivity=feature_sensitivity,
        complementarity=complementarity,
        trm_trajectory=trm_trajectory,
        compress_arc_summary=compress_arc_summary,
        selected_systems=selected_systems,
        best_score_llm=best_score_llm,
        best_aligned_llm=best_aligned_llm,
        best_trm_score=best_trm_score,
        best_trm_alignment=best_trm_alignment,
        best_varc=best_varc,
    )

    print(f"Done. Outputs written to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
