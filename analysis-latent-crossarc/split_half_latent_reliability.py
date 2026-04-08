from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
TABLES_DIR = BASE_DIR / "tables"

HUMAN_RAW_CSV = ROOT_DIR / "data-human" / "test_pair_attempts.csv"
ARC1_TRUTH_DIR = ROOT_DIR / "data-llm" / "ARC-AGI" / "data" / "evaluation"
ARC2_TRUTH_DIR = ROOT_DIR / "data-llm" / "ARC-AGI-2" / "data" / "evaluation"
ARC1_LLM_DIR = ROOT_DIR / "data-llm" / "arc_agi_v1_public_eval"
ARC2_LLM_DIR = ROOT_DIR / "data-llm" / "arc_agi_v2_public_eval"

ARC1_LLM_CACHE = TABLES_DIR / "arc1_llm_matrix.csv"
ARC2_LLM_CACHE = TABLES_DIR / "arc2_llm_matrix.csv"
SUMMARY_CSV = TABLES_DIR / "latent_split_half_summary.csv"
DRAWS_CSV = TABLES_DIR / "latent_split_half_draws.csv"
REPORT_MD = BASE_DIR / "latent_split_half_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated split-half latent reliability analysis for humans and LLMs.")
    parser.add_argument("--n-sims", type=int, default=500, help="Number of random split-half simulations per analysis.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for split generation.")
    parser.add_argument(
        "--min-human-attempts",
        type=int,
        default=8,
        help="Minimum total human attempts required for an item to enter the analysis.",
    )
    parser.add_argument(
        "--min-item-obs-per-half",
        type=int,
        default=2,
        help="Minimum per-item human observations required inside each half-split.",
    )
    parser.add_argument(
        "--min-items",
        type=int,
        default=20,
        help="Minimum number of overlapping items required for a split to count.",
    )
    parser.add_argument(
        "--rebuild-matrices",
        action="store_true",
        help="Ignore cached ARC1/ARC2 LLM matrices and rebuild them from prediction JSONs.",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def normalize_grid(grid: object) -> str:
    if not isinstance(grid, list) or not grid:
        return "EMPTY"
    return ",".join(str(cell) for row in grid for cell in row)


def load_human_attempts() -> pd.DataFrame:
    raw = pd.read_csv(HUMAN_RAW_CSV)
    raw["task_pair_id"] = raw["task_ID"] + "__" + raw["test_index"].astype(str)
    raw["solved"] = (raw["correct_submissions"] > 0).astype(int)
    return raw


def load_truth_outputs(truth_dir: Path, single_pair_only: bool = False) -> dict[str, str]:
    truth_outputs: dict[str, str] = {}
    for path in sorted(truth_dir.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        test_pairs = obj.get("test", [])
        if single_pair_only and len(test_pairs) != 1:
            continue
        for idx, pair in enumerate(test_pairs):
            truth_outputs[f"{path.stem}__{idx}"] = normalize_grid(pair["output"])
    return truth_outputs


def build_human_arc2_subset(raw: pd.DataFrame, min_human_attempts: int) -> tuple[pd.DataFrame, list[str]]:
    arc2 = raw.loc[raw["task_set"] == "Public Eval"].copy()
    counts = arc2.groupby("task_pair_id")["solved"].size()
    pair_ids = sorted(counts.loc[counts >= min_human_attempts].index)
    return arc2.loc[arc2["task_pair_id"].isin(pair_ids)].copy(), pair_ids


def build_human_arc1_sidecar_subset(raw: pd.DataFrame, min_human_attempts: int) -> tuple[pd.DataFrame, list[str]]:
    arc1_truth = load_truth_outputs(ARC1_TRUTH_DIR, single_pair_only=True)
    arc1_pair_ids = set(arc1_truth)
    arc1 = raw.loc[(raw["task_set"] == "Public Train") & (raw["task_pair_id"].isin(arc1_pair_ids))].copy()
    counts = arc1.groupby("task_pair_id")["solved"].size()
    pair_ids = sorted(counts.loc[counts >= min_human_attempts].index)
    return arc1.loc[arc1["task_pair_id"].isin(pair_ids)].copy(), pair_ids


def extract_prediction_answer(pred_entry: object) -> object:
    if not isinstance(pred_entry, dict):
        return None
    attempt_1 = pred_entry.get("attempt_1")
    if isinstance(attempt_1, dict) and attempt_1.get("answer"):
        return attempt_1.get("answer")
    attempt_2 = pred_entry.get("attempt_2")
    if isinstance(attempt_2, dict) and attempt_2.get("answer"):
        return attempt_2.get("answer")
    return None


def build_llm_matrix(pred_dir: Path, truth_outputs: dict[str, str], pair_ids: list[str]) -> pd.DataFrame:
    pair_set = set(pair_ids)
    rows: dict[str, dict[str, int]] = {}

    for model_dir in sorted(pred_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue

        row = {pair_id: 0 for pair_id in pair_ids}
        for pred_path in model_dir.glob("*.json"):
            try:
                pred_obj = json.loads(pred_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(pred_obj, list):
                continue

            indexed_by_metadata: dict[int, dict] = {}
            for candidate in pred_obj:
                if not isinstance(candidate, dict):
                    continue
                pair_index = candidate.get("metadata", {}).get("pair_index")
                if pair_index is None:
                    continue
                try:
                    indexed_by_metadata[int(pair_index)] = candidate
                except (TypeError, ValueError):
                    continue

            for idx in range(len(pred_obj)):
                pair_id = f"{pred_path.stem}__{idx}"
                if pair_id not in pair_set:
                    continue

                pred_entry = indexed_by_metadata.get(idx)
                if pred_entry is None and idx < len(pred_obj):
                    pred_entry = pred_obj[idx]
                answer = extract_prediction_answer(pred_entry)
                row[pair_id] = int(normalize_grid(answer) == truth_outputs[pair_id])

        rows[model_dir.name] = row

    matrix = pd.DataFrame.from_dict(rows, orient="index")
    matrix = matrix.reindex(columns=pair_ids).fillna(0).astype(int)
    matrix.index.name = "model_name"
    return matrix.sort_index()


def load_or_build_llm_matrix(
    cache_path: Path,
    pred_dir: Path,
    truth_outputs: dict[str, str],
    pair_ids: list[str],
    rebuild: bool,
) -> pd.DataFrame:
    if cache_path.exists() and not rebuild:
        matrix = pd.read_csv(cache_path, index_col=0)
        matrix = matrix.reindex(columns=pair_ids).fillna(0).astype(int)
        missing_models = matrix.shape[0] == 0
        missing_columns = matrix.isnull().all(axis=0).any()
        if not missing_models and not missing_columns:
            return matrix.sort_index()

    matrix = build_llm_matrix(pred_dir, truth_outputs, pair_ids)
    matrix.to_csv(cache_path)
    return matrix


def fit_latent_item_difficulty(
    long_df: pd.DataFrame,
    respondent_col: str = "respondent_id",
    item_col: str = "task_pair_id",
    outcome_col: str = "solved",
) -> pd.Series:
    if long_df.empty:
        return pd.Series(dtype=float)
    if long_df[respondent_col].nunique() < 2 or long_df[item_col].nunique() < 2 or long_df[outcome_col].nunique() < 2:
        return pd.Series(dtype=float)

    encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    design = encoder.fit_transform(long_df[[respondent_col, item_col]])
    outcomes = long_df[outcome_col].to_numpy()

    model = LogisticRegression(
        C=2.0,
        solver="saga",
        max_iter=4000,
        fit_intercept=True,
        random_state=0,
    )
    model.fit(design, outcomes)

    feature_names = pd.Index(encoder.get_feature_names_out([respondent_col, item_col]))
    coefficients = pd.Series(model.coef_[0], index=feature_names)

    item_prefix = f"{item_col}_"
    item_ease = coefficients.loc[feature_names.str.startswith(item_prefix)]
    item_ease.index = item_ease.index.str.replace(item_prefix, "", regex=False)
    item_difficulty = -(item_ease - item_ease.mean())
    return item_difficulty.sort_index()


def long_split_latent_correlations(
    long_df: pd.DataFrame,
    respondent_col: str,
    n_sims: int,
    seed: int,
    min_item_obs_per_half: int,
    min_items: int,
) -> pd.DataFrame:
    respondents = np.array(sorted(long_df[respondent_col].unique()))
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for sim in range(n_sims):
        perm = rng.permutation(respondents)
        cut = len(perm) // 2
        half_a_ids = set(perm[:cut])

        half_a = long_df.loc[long_df[respondent_col].isin(half_a_ids)].copy()
        half_b = long_df.loc[~long_df[respondent_col].isin(half_a_ids)].copy()

        counts_a = half_a.groupby("task_pair_id")["solved"].size().rename("n_a")
        counts_b = half_b.groupby("task_pair_id")["solved"].size().rename("n_b")
        eligible = pd.concat([counts_a, counts_b], axis=1, join="inner").dropna()
        eligible = eligible.loc[(eligible["n_a"] >= min_item_obs_per_half) & (eligible["n_b"] >= min_item_obs_per_half)]

        if len(eligible) < min_items:
            continue

        half_a = half_a.loc[half_a["task_pair_id"].isin(eligible.index)]
        half_b = half_b.loc[half_b["task_pair_id"].isin(eligible.index)]

        diff_a = fit_latent_item_difficulty(half_a, respondent_col=respondent_col)
        diff_b = fit_latent_item_difficulty(half_b, respondent_col=respondent_col)

        merged = pd.concat([diff_a.rename("difficulty_a"), diff_b.rename("difficulty_b")], axis=1, join="inner").dropna()
        if len(merged) < min_items or merged["difficulty_a"].nunique() < 2 or merged["difficulty_b"].nunique() < 2:
            continue

        rows.append(
            {
                "simulation": sim,
                "pearson": float(merged["difficulty_a"].corr(merged["difficulty_b"])),
                "spearman": float(merged["difficulty_a"].corr(merged["difficulty_b"], method="spearman")),
                "n_items": int(len(merged)),
            }
        )

    return pd.DataFrame(rows)


def matrix_split_latent_correlations(matrix: pd.DataFrame, n_sims: int, seed: int, min_items: int) -> pd.DataFrame:
    respondents = np.array(sorted(matrix.index))
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for sim in range(n_sims):
        perm = rng.permutation(respondents)
        cut = len(perm) // 2
        half_a = matrix.loc[perm[:cut]]
        half_b = matrix.loc[perm[cut:]]

        long_a = half_a.stack(future_stack=True).rename("solved").reset_index()
        long_a.columns = ["respondent_id", "task_pair_id", "solved"]
        long_b = half_b.stack(future_stack=True).rename("solved").reset_index()
        long_b.columns = ["respondent_id", "task_pair_id", "solved"]

        diff_a = fit_latent_item_difficulty(long_a)
        diff_b = fit_latent_item_difficulty(long_b)
        merged = pd.concat([diff_a.rename("difficulty_a"), diff_b.rename("difficulty_b")], axis=1, join="inner").dropna()

        if len(merged) < min_items or merged["difficulty_a"].nunique() < 2 or merged["difficulty_b"].nunique() < 2:
            continue

        rows.append(
            {
                "simulation": sim,
                "pearson": float(merged["difficulty_a"].corr(merged["difficulty_b"])),
                "spearman": float(merged["difficulty_a"].corr(merged["difficulty_b"], method="spearman")),
                "n_items": int(len(merged)),
            }
        )

    return pd.DataFrame(rows)


def summarize_draws(
    draws: pd.DataFrame,
    benchmark: str,
    population: str,
    subset_note: str,
    n_respondents: int,
    n_items: int,
    requested_sims: int,
) -> dict[str, object]:
    median_pearson = float(draws["pearson"].median())
    return {
        "benchmark": benchmark,
        "population": population,
        "subset_note": subset_note,
        "n_respondents": n_respondents,
        "n_items": n_items,
        "requested_sims": requested_sims,
        "completed_sims": int(len(draws)),
        "pearson_mean": float(draws["pearson"].mean()),
        "pearson_median": median_pearson,
        "pearson_sd": float(draws["pearson"].std(ddof=1)),
        "pearson_ci_lo": float(draws["pearson"].quantile(0.025)),
        "pearson_ci_hi": float(draws["pearson"].quantile(0.975)),
        "spearman_mean": float(draws["spearman"].mean()),
        "spearman_median": float(draws["spearman"].median()),
        "spearman_ci_lo": float(draws["spearman"].quantile(0.025)),
        "spearman_ci_hi": float(draws["spearman"].quantile(0.975)),
        "median_items_per_split": float(draws["n_items"].median()),
        "spearman_brown_from_median_pearson": float((2 * median_pearson) / (1 + median_pearson)),
    }


def write_report(summary: pd.DataFrame, min_human_attempts: int, n_sims: int) -> None:
    display = summary[
        [
            "benchmark",
            "population",
            "n_respondents",
            "n_items",
            "completed_sims",
            "pearson_mean",
            "pearson_median",
            "pearson_ci_lo",
            "pearson_ci_hi",
            "spearman_mean",
            "spearman_brown_from_median_pearson",
        ]
    ].copy()

    lines = [
        "# Split-Half Latent Reliability",
        "",
        "This note estimates how reproducible the recovered latent item-difficulty axis is when we randomly split respondents/models into two halves and refit the same 1D person-item logistic model in each half.",
        "",
        "## Setup",
        "",
        f"- Each benchmark uses `{n_sims}` random split-half simulations.",
        f"- Human items must have at least `{min_human_attempts}` total human attempts to enter the analysis.",
        "- Human ARC-2 uses Public Eval task pairs from the canonical human testing file.",
        "- Human ARC-1 is a sidecar subset: ARC-1 single-pair evaluation tasks reused inside the ARC-AGI-2 Public Train human testing file.",
        "- LLM ARC-1 and ARC-2 use the same item sets as the corresponding human analyses and the local public-eval prediction folders as respondents.",
        "- The reported correlation is the item-difficulty correlation between the two independently fit halves.",
        "",
        "## Summary",
        "",
        "```text",
        display.round(3).to_string(index=False),
        "```",
        "",
        "## Readout",
        "",
        "- Higher values mean the latent item ordering is more stable across random halves of the population.",
        "- The Spearman-Brown column converts the median split-half Pearson correlation into an estimated full-length reliability for the same latent scale.",
        "- ARC-1 human results should be read as a sidecar estimate rather than a dedicated ARC-1 human benchmark.",
        "",
    ]

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    human_raw = load_human_attempts()
    human_arc1, arc1_pair_ids = build_human_arc1_sidecar_subset(human_raw, args.min_human_attempts)
    human_arc2, arc2_pair_ids = build_human_arc2_subset(human_raw, args.min_human_attempts)

    arc1_truth = load_truth_outputs(ARC1_TRUTH_DIR, single_pair_only=True)
    arc2_truth = load_truth_outputs(ARC2_TRUTH_DIR)

    llm_arc1 = load_or_build_llm_matrix(
        ARC1_LLM_CACHE,
        ARC1_LLM_DIR,
        arc1_truth,
        arc1_pair_ids,
        rebuild=args.rebuild_matrices,
    )
    llm_arc2 = load_or_build_llm_matrix(
        ARC2_LLM_CACHE,
        ARC2_LLM_DIR,
        arc2_truth,
        arc2_pair_ids,
        rebuild=args.rebuild_matrices,
    )

    analyses = [
        {
            "benchmark": "ARC1",
            "population": "human",
            "subset_note": "ARC1 sidecar from ARC-AGI-2 Public Train human attempts",
            "n_respondents": int(human_arc1["session_ID"].nunique()),
            "n_items": len(arc1_pair_ids),
            "draws": long_split_latent_correlations(
                human_arc1,
                respondent_col="session_ID",
                n_sims=args.n_sims,
                seed=args.seed + 11,
                min_item_obs_per_half=args.min_item_obs_per_half,
                min_items=args.min_items,
            ),
        },
        {
            "benchmark": "ARC2",
            "population": "human",
            "subset_note": "Public Eval task pairs with robust human exposure",
            "n_respondents": int(human_arc2["session_ID"].nunique()),
            "n_items": len(arc2_pair_ids),
            "draws": long_split_latent_correlations(
                human_arc2,
                respondent_col="session_ID",
                n_sims=args.n_sims,
                seed=args.seed + 23,
                min_item_obs_per_half=args.min_item_obs_per_half,
                min_items=args.min_items,
            ),
        },
        {
            "benchmark": "ARC1",
            "population": "llm",
            "subset_note": "Local ARC1 public-eval prediction folder on the ARC1 human-sidecar items",
            "n_respondents": int(llm_arc1.shape[0]),
            "n_items": int(llm_arc1.shape[1]),
            "draws": matrix_split_latent_correlations(
                llm_arc1,
                n_sims=args.n_sims,
                seed=args.seed + 37,
                min_items=args.min_items,
            ),
        },
        {
            "benchmark": "ARC2",
            "population": "llm",
            "subset_note": "Local ARC2 public-eval prediction folder on the robust ARC2 human items",
            "n_respondents": int(llm_arc2.shape[0]),
            "n_items": int(llm_arc2.shape[1]),
            "draws": matrix_split_latent_correlations(
                llm_arc2,
                n_sims=args.n_sims,
                seed=args.seed + 53,
                min_items=args.min_items,
            ),
        },
    ]

    draw_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for analysis in analyses:
        draws = analysis["draws"].copy()
        draws["benchmark"] = analysis["benchmark"]
        draws["population"] = analysis["population"]
        draws["subset_note"] = analysis["subset_note"]
        draw_frames.append(draws)
        summary_rows.append(
            summarize_draws(
                draws,
                benchmark=analysis["benchmark"],
                population=analysis["population"],
                subset_note=analysis["subset_note"],
                n_respondents=analysis["n_respondents"],
                n_items=analysis["n_items"],
                requested_sims=args.n_sims,
            )
        )

    all_draws = pd.concat(draw_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(["benchmark", "population"]).reset_index(drop=True)

    all_draws.to_csv(DRAWS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    write_report(summary, min_human_attempts=args.min_human_attempts, n_sims=args.n_sims)

    print(summary.round(4).to_string(index=False))
    print(f"\nSaved draws to: {DRAWS_CSV}")
    print(f"Saved summary to: {SUMMARY_CSV}")
    print(f"Saved report to: {REPORT_MD}")


if __name__ == "__main__":
    main()
