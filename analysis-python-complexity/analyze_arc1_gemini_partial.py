from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT_PATH = ANALYSIS_DIR / "analyze_arc1_dsl_human_llm.py"
EXTRA_JOIN_PATH = ANALYSIS_DIR / "arc1_dsl_extra_llms_join.csv"

GEMINI_RUN = (
    ANALYSIS_DIR.parent / "GemmaData" / "runs" / "20260409T045708Z_gemini-3-1-flash-lite-preview_thinking-high_400tasks"
)

GEMINI_PARTIAL_JOIN_PATH = ANALYSIS_DIR / "arc1_dsl_gemini_partial_join.csv"
GEMINI_PARTIAL_CORR_PATH = ANALYSIS_DIR / "arc1_dsl_gemini_partial_correlations.csv"
GEMINI_PARTIAL_COMPARE_PATH = ANALYSIS_DIR / "arc1_dsl_gemini_partial_comparison.csv"
GEMINI_PARTIAL_SUMMARY_PATH = ANALYSIS_DIR / "arc1_dsl_gemini_partial_summary.json"
GEMINI_PARTIAL_REPORT_PATH = ANALYSIS_DIR / "arc1_dsl_gemini_partial_report.md"

LOCKED_METRICS = [
    "complexity_pc1_score",
    "log1p_cyclomatic_complexity",
    "ast_node_count",
    "log1p_branch_opcode_count_dynamic",
]


def show_progress(current: int, total: int, label: str) -> None:
    width = 32
    ratio = 0.0 if total <= 0 else current / total
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\r[{bar}] {current}/{total} {label}")
    if current >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def show_stage(stage_index: int, stage_total: int, label: str) -> None:
    show_progress(stage_index, stage_total, f"Stage: {label}")


def load_base_module():
    spec = importlib.util.spec_from_file_location("arc1_dsl_base", BASE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import base analysis module from {BASE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def smoothed_difficulty_from_counts(successes: pd.Series, totals: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=successes.index, dtype=float)
    valid = totals.astype(float) > 0
    ease = (successes[valid].astype(float) + 0.5) / (totals[valid].astype(float) + 1.0)
    out.loc[valid] = -np.log(ease / (1.0 - ease))
    return out


def load_gemini_partial(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    task_dir = run_dir / "tasks"
    rows: list[dict[str, Any]] = []
    for path in sorted(task_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        pair_matches = [int(bool(flag)) for flag in (record.get("pair_matches") or [])]
        rows.append(
            {
                "task_id": record["task_id"],
                "gemini_solved": int(bool(record.get("exact_match"))),
                "gemini_solved_pair_mean": float(np.mean(pair_matches)) if pair_matches else 0.0,
                "gemini_status": record.get("status"),
                "gemini_has_artifact": 1,
                "gemini_error_flag": int(record.get("status") != "ok"),
            }
        )

    frame = pd.DataFrame(rows)
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    diagnostics = {
        "run_dir": str(run_dir),
        "task_artifact_count": int(len(frame)),
        "tasks_completed": int(status.get("tasks_completed", len(frame))),
        "tasks_solved": int(status.get("tasks_solved", frame["gemini_solved"].sum())),
        "tasks_errors": int(status.get("tasks_errors", frame["gemini_error_flag"].sum())),
        "solve_rate_on_available": float(frame["gemini_solved"].mean()) if len(frame) else float("nan"),
        "error_rate_on_available": float(frame["gemini_error_flag"].mean()) if len(frame) else float("nan"),
    }
    return frame, diagnostics


def add_gemini_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gemini_failure"] = 1.0 - out["gemini_solved"]
    out["gemini_pair_successes"] = out["gemini_solved_pair_mean"] * out["test_pairs"]
    out["gemini_pair_difficulty"] = smoothed_difficulty_from_counts(
        out["gemini_pair_successes"],
        out["test_pairs"].where(out["gemini_solved_pair_mean"].notna(), 0),
    )

    pair_mean_cols = [
        "gpt_solved_pair_mean",
        "claude_solved_pair_mean",
        "nemotron_solved_pair_mean",
        "gemma_solved_pair_mean",
        "gemini_solved_pair_mean",
    ]
    task_cols = ["gpt_solved", "claude_solved", "nemotron_solved", "gemma_solved", "gemini_solved"]

    out["llm5_available_models"] = out[task_cols].notna().sum(axis=1)
    out["llm5_success_count"] = out[task_cols].fillna(0).sum(axis=1)
    out["llm5_task_difficulty"] = smoothed_difficulty_from_counts(out["llm5_success_count"], out["llm5_available_models"])

    out["llm5_pair_available_models"] = out[pair_mean_cols].notna().sum(axis=1)
    out["llm5_pair_successes"] = out[pair_mean_cols].mul(out["test_pairs"], axis=0).fillna(0).sum(axis=1)
    out["llm5_pair_total"] = out["llm5_pair_available_models"] * out["test_pairs"]
    out["llm5_pair_difficulty"] = smoothed_difficulty_from_counts(out["llm5_pair_successes"], out["llm5_pair_total"])
    out["llm5_pair_mean_rate"] = out[pair_mean_cols].mean(axis=1, skipna=True)
    out["human_llm5_pair_gap"] = (out["human_difficulty_complete_solve_rate"] - out["llm5_pair_mean_rate"]).abs()
    return out


def effective_model_count(binary_df: pd.DataFrame) -> tuple[float, float]:
    corr = binary_df.corr()
    if len(corr.columns) < 2:
        return float("nan"), float("nan")
    mask = np.triu(np.ones(corr.shape, dtype=bool), 1)
    vals = corr.where(mask).stack()
    mean_rho = float(vals.mean()) if len(vals) else float("nan")
    k = len(binary_df.columns)
    eff_n = float(k / (1.0 + (k - 1.0) * mean_rho)) if not math.isnan(mean_rho) else float("nan")
    return mean_rho, eff_n


def correlation_table(df: pd.DataFrame, targets: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in LOCKED_METRICS:
        for target_col, target_label in targets.items():
            pair = df[[metric, target_col]].dropna()
            rows.append(
                {
                    "complexity_metric": metric,
                    "target_metric": target_col,
                    "target_label": target_label,
                    "n": int(len(pair)),
                    "pearson_r": safe_corr(df[metric], df[target_col], "pearson"),
                    "spearman_rho": safe_corr(df[metric], df[target_col], "spearman"),
                }
            )
    return pd.DataFrame(rows)


def compare_targets(
    df: pd.DataFrame,
    base_module,
    metric: str,
    left_target: str,
    right_target: str,
    subset_name: str,
) -> dict[str, Any]:
    pair = df[[metric, left_target, right_target]].dropna()
    left_r = safe_corr(pair[metric], pair[left_target], "pearson")
    right_r = safe_corr(pair[metric], pair[right_target], "pearson")
    yz_corr = safe_corr(pair[left_target], pair[right_target], "pearson")
    t_value, p_value = base_module.williams_test(left_r, right_r, yz_corr, len(pair))
    return {
        "subset_name": subset_name,
        "complexity_metric": metric,
        "left_target": left_target,
        "right_target": right_target,
        "n": int(len(pair)),
        "left_r": left_r,
        "right_r": right_r,
        "difference_right_minus_left": right_r - left_r,
        "williams_t": t_value,
        "williams_p": p_value,
    }


def main() -> None:
    stage_total = 5

    show_stage(1, stage_total, "Loading saved extra-LLM join")
    print("Loading saved extra-LLM join...", file=sys.stderr, flush=True)
    base_module = load_base_module()
    extra = pd.read_csv(EXTRA_JOIN_PATH)

    show_stage(2, stage_total, "Loading Gemini partial artifacts")
    print("Loading Gemini partial artifacts...", file=sys.stderr, flush=True)
    gemini_df, gemini_diag = load_gemini_partial(GEMINI_RUN)

    show_stage(3, stage_total, "Building completed-task subset")
    print("Building completed-task subset...", file=sys.stderr, flush=True)
    merged = extra.merge(gemini_df, on="task_id", how="inner")
    merged = add_gemini_columns(merged)
    merged.to_csv(GEMINI_PARTIAL_JOIN_PATH, index=False)

    show_stage(4, stage_total, "Computing reduced-set correlations")
    print("Computing reduced-set correlations...", file=sys.stderr, flush=True)
    targets = {
        "human_difficulty_complete": "Human latent difficulty",
        "llm4_pair_difficulty": "Pooled GPT+Claude+Nemotron+Gemma pair difficulty",
        "llm5_pair_difficulty": "Pooled GPT+Claude+Nemotron+Gemma+Gemini pair difficulty",
        "gemini_pair_difficulty": "Gemini pair difficulty",
        "complexity_pc1_score": "Complexity PC1 self-check",
    }
    corr_df = correlation_table(merged, targets)
    corr_df.to_csv(GEMINI_PARTIAL_CORR_PATH, index=False)

    compare_rows: list[dict[str, Any]] = []
    full_subset = merged.copy()
    gap30_subset = merged[merged["human_llm5_pair_gap"] <= 0.30].copy()
    for subset_name, subset in [("gemini_completed_set", full_subset), ("gemini_completed_gap_le_0.30", gap30_subset)]:
        for metric in LOCKED_METRICS:
            compare_rows.append(
                compare_targets(subset, base_module, metric, "llm4_pair_difficulty", "llm5_pair_difficulty", subset_name)
            )
            compare_rows.append(
                compare_targets(subset, base_module, metric, "human_difficulty_complete", "llm5_pair_difficulty", subset_name)
            )
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(GEMINI_PARTIAL_COMPARE_PATH, index=False)

    show_stage(5, stage_total, "Writing summary")
    print("Writing summary...", file=sys.stderr, flush=True)
    mean_rho, eff_n = effective_model_count(
        merged[["gpt_solved", "claude_solved", "nemotron_solved", "gemma_solved", "gemini_solved"]].dropna()
    )
    summary = {
        "gemini": {
            **gemini_diag,
            "validated_coverage": int(len(merged)),
        },
        "reduced_set": {
            "n": int(len(merged)),
            "gap30_n": int(len(gap30_subset)),
            "human_mean_solve_rate": float(merged["human_difficulty_complete_solve_rate"].mean()),
            "llm4_pair_mean_rate": float(merged["llm4_pair_mean_rate"].mean()),
            "llm5_pair_mean_rate": float(merged["llm5_pair_mean_rate"].mean()),
            "gemini_pair_mean_rate": float(merged["gemini_solved_pair_mean"].mean()),
        },
        "pairwise_dependence": {
            "five_model_mean_pairwise_corr": mean_rho,
            "five_model_effective_n": eff_n,
        },
        "headline": {
            "complexity_pc1_human_r": float(
                safe_corr(merged["complexity_pc1_score"], merged["human_difficulty_complete"], "pearson")
            ),
            "complexity_pc1_llm4_r": float(
                safe_corr(merged["complexity_pc1_score"], merged["llm4_pair_difficulty"], "pearson")
            ),
            "complexity_pc1_llm5_r": float(
                safe_corr(merged["complexity_pc1_score"], merged["llm5_pair_difficulty"], "pearson")
            ),
            "complexity_pc1_gemini_r": float(
                safe_corr(merged["complexity_pc1_score"], merged["gemini_pair_difficulty"], "pearson")
            ),
        },
    }
    GEMINI_PARTIAL_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# ARC-1 Gemini Partial Analysis",
        "",
        f"- Gemini completed-task subset size: {summary['reduced_set']['n']}",
        f"- Gemini solve rate on completed artifacts: {summary['gemini']['solve_rate_on_available']:.3f}",
        f"- Gemini pair mean rate on subset: {summary['reduced_set']['gemini_pair_mean_rate']:.3f}",
        f"- Human mean solve rate on subset: {summary['reduced_set']['human_mean_solve_rate']:.3f}",
        f"- 4-model pooled pair mean rate on subset: {summary['reduced_set']['llm4_pair_mean_rate']:.3f}",
        f"- 5-model pooled pair mean rate on subset: {summary['reduced_set']['llm5_pair_mean_rate']:.3f}",
        "",
        "## Complexity PC1",
        "",
        f"- Human: r = {summary['headline']['complexity_pc1_human_r']:.3f}",
        f"- 4-model pooled LLM: r = {summary['headline']['complexity_pc1_llm4_r']:.3f}",
        f"- 5-model pooled LLM: r = {summary['headline']['complexity_pc1_llm5_r']:.3f}",
        f"- Gemini alone: r = {summary['headline']['complexity_pc1_gemini_r']:.3f}",
        "",
        "## Locked Metric Comparisons",
        "",
    ]
    for _, row in compare_df.iterrows():
        lines.append(
            f"- {row['subset_name']} / `{row['complexity_metric']}`: {row['left_target']} r = {row['left_r']:.3f}, "
            f"{row['right_target']} r = {row['right_r']:.3f}, Williams p = {row['williams_p']:.4g}."
        )
    GEMINI_PARTIAL_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Saved Gemini partial outputs:", file=sys.stderr, flush=True)
    print(f"  - {GEMINI_PARTIAL_JOIN_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {GEMINI_PARTIAL_CORR_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {GEMINI_PARTIAL_COMPARE_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {GEMINI_PARTIAL_SUMMARY_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {GEMINI_PARTIAL_REPORT_PATH.name}", file=sys.stderr, flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
