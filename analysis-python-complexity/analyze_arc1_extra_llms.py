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
EXPANDED_JOIN_PATH = ANALYSIS_DIR / "arc1_dsl_expanded_oldstyle_join.csv"

EXTRA_JOIN_PATH = ANALYSIS_DIR / "arc1_dsl_extra_llms_join.csv"
EXTRA_CORR_PATH = ANALYSIS_DIR / "arc1_dsl_extra_llms_correlations.csv"
EXTRA_COMPARE_PATH = ANALYSIS_DIR / "arc1_dsl_extra_llms_comparison.csv"
EXTRA_SUMMARY_PATH = ANALYSIS_DIR / "arc1_dsl_extra_llms_summary.json"
EXTRA_REPORT_PATH = ANALYSIS_DIR / "arc1_dsl_extra_llms_report.md"

NEMOTRON_RUN = (
    ANALYSIS_DIR.parent / "nemotronData" / "runs" / "20260409T034637Z_nvidia-nemotron-3-super-120b-a12b-free_400tasks"
)
GEMMA_RUN = ANALYSIS_DIR.parent / "GemmaData" / "runs" / "20260409T042406Z_gemma-4-31b-it_thinking-high_400tasks"

LOCKED_METRICS = [
    "log1p_cyclomatic_complexity",
    "ast_node_count",
    "complexity_pc1_score",
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


def smoothed_difficulty_from_counts(successes: pd.Series, totals: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=successes.index, dtype=float)
    valid = totals.astype(float) > 0
    ease = (successes[valid].astype(float) + 0.5) / (totals[valid].astype(float) + 1.0)
    out.loc[valid] = -np.log(ease / (1.0 - ease))
    return out


def load_task_run(run_dir: Path, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    task_dir = run_dir / "tasks"
    files = sorted(task_dir.glob("*.json"))
    rows: list[dict[str, Any]] = []
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        pair_matches = record.get("pair_matches") or []
        pair_flags = [int(bool(flag)) for flag in pair_matches]
        rows.append(
            {
                "task_id": record["task_id"],
                f"{prefix}_solved": int(bool(record.get("exact_match"))),
                f"{prefix}_solved_pair_mean": float(np.mean(pair_flags)) if pair_flags else 0.0,
                f"{prefix}_status": record.get("status"),
                f"{prefix}_has_artifact": 1,
                f"{prefix}_error_flag": int(record.get("status") != "ok"),
            }
        )

    frame = pd.DataFrame(rows)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    diagnostics = {
        "run_dir": str(run_dir),
        "task_artifact_count": int(len(frame)),
        "model": config.get("model"),
        "thinking_level": config.get("thinking_level"),
        "solve_rate_on_available": float(frame[f"{prefix}_solved"].mean()) if len(frame) else float("nan"),
        "error_rate_on_available": float(frame[f"{prefix}_error_flag"].mean()) if len(frame) else float("nan"),
    }
    return frame, diagnostics


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def add_extra_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pair_mean_cols = [
        "gpt_solved_pair_mean",
        "claude_solved_pair_mean",
        "nemotron_solved_pair_mean",
        "gemma_solved_pair_mean",
    ]
    task_cols = ["gpt_solved", "claude_solved", "nemotron_solved", "gemma_solved"]

    for prefix in ("nemotron", "gemma"):
        out[f"{prefix}_failure"] = 1.0 - out[f"{prefix}_solved"]
        out[f"{prefix}_pair_successes"] = out[f"{prefix}_solved_pair_mean"] * out["test_pairs"]
        out[f"{prefix}_pair_difficulty"] = smoothed_difficulty_from_counts(
            out[f"{prefix}_pair_successes"],
            out["test_pairs"].where(out[f"{prefix}_solved_pair_mean"].notna(), 0),
        )

    out["llm4_available_models"] = out[task_cols].notna().sum(axis=1)
    out["llm4_success_count"] = out[task_cols].fillna(0).sum(axis=1)
    out["llm4_task_difficulty"] = smoothed_difficulty_from_counts(
        out["llm4_success_count"], out["llm4_available_models"]
    )

    out["llm4_pair_available_models"] = out[pair_mean_cols].notna().sum(axis=1)
    out["llm4_pair_successes"] = out[pair_mean_cols].mul(out["test_pairs"], axis=0).fillna(0).sum(axis=1)
    out["llm4_pair_total"] = out["llm4_pair_available_models"] * out["test_pairs"]
    out["llm4_pair_difficulty"] = smoothed_difficulty_from_counts(out["llm4_pair_successes"], out["llm4_pair_total"])
    out["llm4_pair_mean_rate"] = out[pair_mean_cols].mean(axis=1, skipna=True)
    out["llm4_pair_failure"] = 1.0 - out["llm4_pair_mean_rate"]
    out["human_llm4_pair_gap"] = (out["human_difficulty_complete_solve_rate"] - out["llm4_pair_mean_rate"]).abs()

    common4_mask = out[pair_mean_cols].notna().all(axis=1)
    out["llm4_common_mask"] = common4_mask.astype(int)
    return out


def effective_model_count(binary_df: pd.DataFrame) -> tuple[float, float]:
    corr = binary_df.corr()
    if len(corr.columns) < 2:
        return float("nan"), float("nan")
    mask = np.triu(np.ones(corr.shape, dtype=bool), 1)
    values = corr.where(mask).stack()
    mean_rho = float(values.mean()) if len(values) else float("nan")
    k = len(binary_df.columns)
    eff_n = float(k / (1.0 + (k - 1.0) * mean_rho)) if not math.isnan(mean_rho) else float("nan")
    return mean_rho, eff_n


def build_correlation_rows(df: pd.DataFrame, metrics: list[str], targets: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
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


def compare_metric_targets(
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


def matched_subset(df: pd.DataFrame, gap: float) -> pd.DataFrame:
    return df[df["human_llm4_pair_gap"] <= gap].copy()


def write_report(summary: dict[str, Any], compare_df: pd.DataFrame) -> None:
    lines = [
        "# ARC-1 Extra LLMs Analysis",
        "",
        "## Added Models",
        "",
        (
            f"- Nemotron coverage: {summary['nemotron']['task_artifact_count']} task artifacts, "
            f"{summary['nemotron']['validated_coverage']} validated tasks, "
            f"solve rate on validated tasks = {summary['nemotron']['solve_rate_validated']:.3f}."
        ),
        (
            f"- Gemma coverage: {summary['gemma']['task_artifact_count']} task artifacts, "
            f"{summary['gemma']['validated_coverage']} validated tasks, "
            f"solve rate on validated tasks = {summary['gemma']['solve_rate_validated']:.3f}."
        ),
        "",
        "## Pooled Difficulty",
        "",
        (
            f"- Two-model pooled pair mean rate = {summary['pooled_rates']['two_model_pair_mean_rate']:.3f}; "
            f"four-model pooled pair mean rate = {summary['pooled_rates']['four_model_pair_mean_rate']:.3f}."
        ),
        (
            f"- Two-model effective model count = {summary['pairwise_dependence']['two_model_effective_n']:.3f}; "
            f"four-model effective model count = {summary['pairwise_dependence']['four_model_effective_n']:.3f}."
        ),
        "",
        "## Locked Metric Comparison",
        "",
    ]
    for _, row in compare_df.iterrows():
        lines.append(
            f"- {row['subset_name']} / `{row['complexity_metric']}`: "
            f"{row['left_target']} r = {row['left_r']:.3f}, "
            f"{row['right_target']} r = {row['right_r']:.3f}, "
            f"Williams p = {row['williams_p']:.4g}."
        )
    lines.extend(
        [
            "",
            "## Headline",
            "",
            (
                f"- Full-set `log1p_cyclomatic_complexity`: two-model pooled r = "
                f"{summary['headline']['full_log1p_cyclomatic_two_model_r']:.3f}, four-model pooled r = "
                f"{summary['headline']['full_log1p_cyclomatic_four_model_r']:.3f}."
            ),
            (
                f"- Matched (`gap<=0.30`) `log1p_cyclomatic_complexity`: two-model pooled r = "
                f"{summary['headline']['gap30_log1p_cyclomatic_two_model_r']:.3f}, four-model pooled r = "
                f"{summary['headline']['gap30_log1p_cyclomatic_four_model_r']:.3f}."
            ),
            (
                f"- Human vs four-model pooled pair difficulty on `gap<=0.30` = "
                f"{summary['headline']['gap30_human_vs_four_model_pair_difficulty_r']:.3f}."
            ),
        ]
    )
    EXTRA_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    stage_total = 6

    show_stage(1, stage_total, "Loading saved expanded ARC-1 join")
    print("Loading saved expanded ARC-1 join...", file=sys.stderr, flush=True)
    base_module = load_base_module()
    joined = pd.read_csv(EXPANDED_JOIN_PATH)

    show_stage(2, stage_total, "Loading Nemotron task artifacts")
    print("Loading Nemotron task artifacts...", file=sys.stderr, flush=True)
    nemotron_df, nemotron_diag = load_task_run(NEMOTRON_RUN, "nemotron")

    show_stage(3, stage_total, "Loading Gemma task artifacts")
    print("Loading Gemma task artifacts...", file=sys.stderr, flush=True)
    gemma_df, gemma_diag = load_task_run(GEMMA_RUN, "gemma")

    show_stage(4, stage_total, "Merging extra LLM outputs")
    print("Merging extra LLM outputs...", file=sys.stderr, flush=True)
    extra = joined.merge(nemotron_df, on="task_id", how="left").merge(gemma_df, on="task_id", how="left")
    extra = add_extra_model_columns(extra)
    extra.to_csv(EXTRA_JOIN_PATH, index=False)

    show_stage(5, stage_total, "Computing correlations and comparisons")
    print("Computing correlations and comparisons...", file=sys.stderr, flush=True)
    targets = {
        "human_difficulty_complete": "Human latent difficulty",
        "pooled_pair_difficulty": "Pooled GPT+Claude pair difficulty",
        "llm4_pair_difficulty": "Pooled GPT+Claude+Nemotron+Gemma pair difficulty",
        "gpt_pair_difficulty": "GPT pair difficulty",
        "claude_pair_difficulty": "Claude pair difficulty",
        "nemotron_pair_difficulty": "Nemotron pair difficulty",
        "gemma_pair_difficulty": "Gemma pair difficulty",
    }
    corr_df = build_correlation_rows(extra, LOCKED_METRICS, targets)
    corr_df.to_csv(EXTRA_CORR_PATH, index=False)

    full_subset = extra.copy()
    gap30_subset = matched_subset(extra, 0.30)
    compare_rows: list[dict[str, Any]] = []
    for subset_name, subset in [("full_set", full_subset), ("gap_le_0.30", gap30_subset)]:
        for metric in LOCKED_METRICS:
            compare_rows.append(
                compare_metric_targets(
                    subset,
                    base_module,
                    metric,
                    "pooled_pair_difficulty",
                    "llm4_pair_difficulty",
                    subset_name,
                )
            )
            compare_rows.append(
                compare_metric_targets(
                    subset,
                    base_module,
                    metric,
                    "human_difficulty_complete",
                    "llm4_pair_difficulty",
                    subset_name,
                )
            )
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(EXTRA_COMPARE_PATH, index=False)

    pairwise_two = extra[["gpt_solved", "claude_solved"]].dropna()
    pairwise_four = extra[["gpt_solved", "claude_solved", "nemotron_solved", "gemma_solved"]].dropna()
    two_rho, two_eff_n = effective_model_count(pairwise_two)
    four_rho, four_eff_n = effective_model_count(pairwise_four)

    summary = {
        "nemotron": {
            **nemotron_diag,
            "validated_coverage": int(extra["nemotron_has_artifact"].fillna(0).sum()),
            "solve_rate_validated": float(extra["nemotron_solved"].mean()),
        },
        "gemma": {
            **gemma_diag,
            "validated_coverage": int(extra["gemma_has_artifact"].fillna(0).sum()),
            "solve_rate_validated": float(extra["gemma_solved"].mean()),
        },
        "pooled_rates": {
            "two_model_pair_mean_rate": float(extra["llm_pair_mean_rate"].mean()),
            "four_model_pair_mean_rate": float(extra["llm4_pair_mean_rate"].mean()),
            "two_model_task_solve_rate": float((extra["gpt_solved"] + extra["claude_solved"]).mean() / 2.0),
            "four_model_task_solve_rate": float(
                extra[["gpt_solved", "claude_solved", "nemotron_solved", "gemma_solved"]].mean(axis=1, skipna=True).mean()
            ),
        },
        "pairwise_dependence": {
            "two_model_mean_pairwise_corr": two_rho,
            "two_model_effective_n": two_eff_n,
            "four_model_mean_pairwise_corr": four_rho,
            "four_model_effective_n": four_eff_n,
        },
        "headline": {
            "full_log1p_cyclomatic_two_model_r": float(
                safe_corr(full_subset["log1p_cyclomatic_complexity"], full_subset["pooled_pair_difficulty"], "pearson")
            ),
            "full_log1p_cyclomatic_four_model_r": float(
                safe_corr(full_subset["log1p_cyclomatic_complexity"], full_subset["llm4_pair_difficulty"], "pearson")
            ),
            "gap30_log1p_cyclomatic_two_model_r": float(
                safe_corr(gap30_subset["log1p_cyclomatic_complexity"], gap30_subset["pooled_pair_difficulty"], "pearson")
            ),
            "gap30_log1p_cyclomatic_four_model_r": float(
                safe_corr(gap30_subset["log1p_cyclomatic_complexity"], gap30_subset["llm4_pair_difficulty"], "pearson")
            ),
            "gap30_human_vs_four_model_pair_difficulty_r": float(
                safe_corr(
                    gap30_subset["human_difficulty_complete"],
                    gap30_subset["llm4_pair_difficulty"],
                    "pearson",
                )
            ),
            "gap30_n": int(len(gap30_subset)),
        },
    }

    EXTRA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    show_stage(6, stage_total, "Writing report")
    print("Writing report...", file=sys.stderr, flush=True)
    write_report(summary, compare_df)

    print("Saved extra-LLM outputs:", file=sys.stderr, flush=True)
    print(f"  - {EXTRA_JOIN_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {EXTRA_CORR_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {EXTRA_COMPARE_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {EXTRA_SUMMARY_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {EXTRA_REPORT_PATH.name}", file=sys.stderr, flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
