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

GPT_OSS_RUN = ANALYSIS_DIR.parent / "nemotronData" / "runs" / "20260409T223455Z_openai-gpt-oss-120b-free_8tasks"
GEMINI_RUN = ANALYSIS_DIR.parent / "GemmaData" / "runs" / "20260409T045708Z_gemini-3-1-flash-lite-preview_thinking-high_400tasks"
GPT5_NANO_RUN = ANALYSIS_DIR.parent / "nemotronData" / "runs" / "20260410T013436Z_gpt-5-4-nano-2026-03-17_400tasks"

JOIN_PATH = ANALYSIS_DIR / "arc1_dsl_gptoss_gemini_join.csv"
CORR_PATH = ANALYSIS_DIR / "arc1_dsl_gptoss_gemini_correlations.csv"
COMPARE_PATH = ANALYSIS_DIR / "arc1_dsl_gptoss_gemini_comparison.csv"
SUMMARY_PATH = ANALYSIS_DIR / "arc1_dsl_gptoss_gemini_summary.json"
REPORT_PATH = ANALYSIS_DIR / "arc1_dsl_gptoss_gemini_report.md"

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


def load_task_run(run_dir: Path, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "tasks").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        pair_matches = [int(bool(flag)) for flag in (record.get("pair_matches") or [])]
        rows.append(
            {
                "task_id": record["task_id"],
                f"{prefix}_solved": int(bool(record.get("exact_match"))),
                f"{prefix}_solved_pair_mean": float(np.mean(pair_matches)) if pair_matches else 0.0,
                f"{prefix}_status": record.get("status"),
                f"{prefix}_has_artifact": 1,
                f"{prefix}_error_flag": int(record.get("status") != "ok"),
            }
        )
    frame = pd.DataFrame(rows)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    diagnostics = {
        "run_dir": str(run_dir),
        "task_artifact_count": int(len(frame)),
        "model": summary.get("model"),
        "thinking_level": summary.get("thinking_level"),
        "reported_accuracy": float(summary.get("accuracy")) if summary.get("accuracy") is not None else float("nan"),
        "reported_solved_tasks": int(summary.get("solved_tasks")) if summary.get("solved_tasks") is not None else int(frame[f"{prefix}_solved"].sum()),
        "reported_error_tasks": int(summary.get("error_tasks")) if summary.get("error_tasks") is not None else int(frame[f"{prefix}_error_flag"].sum()),
        "solve_rate_on_available": float(frame[f"{prefix}_solved"].mean()) if len(frame) else float("nan"),
        "error_rate_on_available": float(frame[f"{prefix}_error_flag"].mean()) if len(frame) else float("nan"),
    }
    return frame, diagnostics


def add_model_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_failure"] = 1.0 - out[f"{prefix}_solved"]
    out[f"{prefix}_pair_successes"] = out[f"{prefix}_solved_pair_mean"] * out["test_pairs"]
    out[f"{prefix}_pair_difficulty"] = smoothed_difficulty_from_counts(
        out[f"{prefix}_pair_successes"],
        out["test_pairs"].where(out[f"{prefix}_solved_pair_mean"].notna(), 0),
    )
    return out


def add_pool_columns(df: pd.DataFrame, prefixes: list[str], label: str) -> pd.DataFrame:
    out = df.copy()
    task_cols = [f"{prefix}_solved" for prefix in prefixes]
    pair_cols = [f"{prefix}_solved_pair_mean" for prefix in prefixes]

    out[f"{label}_available_models"] = out[task_cols].notna().sum(axis=1)
    out[f"{label}_success_count"] = out[task_cols].fillna(0).sum(axis=1)
    out[f"{label}_task_difficulty"] = smoothed_difficulty_from_counts(
        out[f"{label}_success_count"], out[f"{label}_available_models"]
    )

    out[f"{label}_pair_available_models"] = out[pair_cols].notna().sum(axis=1)
    out[f"{label}_pair_successes"] = out[pair_cols].mul(out["test_pairs"], axis=0).fillna(0).sum(axis=1)
    out[f"{label}_pair_total"] = out[f"{label}_pair_available_models"] * out["test_pairs"]
    out[f"{label}_pair_difficulty"] = smoothed_difficulty_from_counts(
        out[f"{label}_pair_successes"], out[f"{label}_pair_total"]
    )
    out[f"{label}_pair_mean_rate"] = out[pair_cols].mean(axis=1, skipna=True)
    out[f"human_{label}_pair_gap"] = (out["human_difficulty_complete_solve_rate"] - out[f"{label}_pair_mean_rate"]).abs()
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


def compare_targets(df: pd.DataFrame, base_module, metric: str, left_target: str, right_target: str, subset_name: str) -> dict[str, Any]:
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
    stage_total = 7

    show_stage(1, stage_total, "Loading saved extra-LLM join")
    print("Loading saved extra-LLM join...", file=sys.stderr, flush=True)
    base_module = load_base_module()
    df = pd.read_csv(EXTRA_JOIN_PATH)

    show_stage(2, stage_total, "Loading Gemini full run")
    print("Loading Gemini full run...", file=sys.stderr, flush=True)
    gemini_df, gemini_diag = load_task_run(GEMINI_RUN, "gemini")

    show_stage(3, stage_total, "Loading GPT OSS full run")
    print("Loading GPT OSS full run...", file=sys.stderr, flush=True)
    gpt_oss_df, gpt_oss_diag = load_task_run(GPT_OSS_RUN, "gpt_oss")

    show_stage(4, stage_total, "Loading GPT-5.4 Nano full run")
    print("Loading GPT-5.4 Nano full run...", file=sys.stderr, flush=True)
    gpt5_nano_df, gpt5_nano_diag = load_task_run(GPT5_NANO_RUN, "gpt5_nano")

    show_stage(5, stage_total, "Merging new LLMs")
    print("Merging new LLMs...", file=sys.stderr, flush=True)
    merged = df.drop(columns=[c for c in df.columns if c.startswith("gemini_")], errors="ignore")
    merged = merged.drop(columns=[c for c in merged.columns if c.startswith("gpt5_nano_")], errors="ignore")
    merged = (
        merged.merge(gemini_df, on="task_id", how="left")
        .merge(gpt_oss_df, on="task_id", how="left")
        .merge(gpt5_nano_df, on="task_id", how="left")
    )
    merged = add_model_columns(merged, "gemini")
    merged = add_model_columns(merged, "gpt_oss")
    merged = add_model_columns(merged, "gpt5_nano")
    llm6_prefixes = ["gpt", "claude", "nemotron", "gemma", "gemini", "gpt5_nano"]
    llm7_prefixes = llm6_prefixes + ["gpt_oss"]
    merged = add_pool_columns(merged, llm6_prefixes, "llm6")
    merged = add_pool_columns(merged, llm7_prefixes, "llm7")
    JOIN_PATH.write_text("", encoding="utf-8")
    merged.to_csv(JOIN_PATH, index=False)

    show_stage(6, stage_total, "Computing correlations")
    print("Computing correlations...", file=sys.stderr, flush=True)
    targets = {
        "human_difficulty_complete": "Human latent difficulty",
        "llm6_pair_difficulty": "Pooled LLM difficulty without GPT OSS",
        "llm7_pair_difficulty": "Pooled LLM difficulty with GPT OSS",
        "gpt_oss_pair_difficulty": "GPT OSS pair difficulty",
        "gemini_pair_difficulty": "Gemini pair difficulty",
        "gpt5_nano_pair_difficulty": "GPT-5.4 Nano pair difficulty",
    }
    corr_df = correlation_table(merged, targets)
    corr_df.to_csv(CORR_PATH, index=False)

    full_subset = merged.copy()
    gap30_subset = merged[merged["human_llm7_pair_gap"] <= 0.30].copy()
    compare_rows: list[dict[str, Any]] = []
    for subset_name, subset in [("full_set", full_subset), ("gap_le_0.30", gap30_subset)]:
        for metric in LOCKED_METRICS:
            compare_rows.append(compare_targets(subset, base_module, metric, "llm6_pair_difficulty", "llm7_pair_difficulty", subset_name))
            compare_rows.append(compare_targets(subset, base_module, metric, "human_difficulty_complete", "llm7_pair_difficulty", subset_name))
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(COMPARE_PATH, index=False)

    show_stage(7, stage_total, "Writing report")
    print("Writing report...", file=sys.stderr, flush=True)
    six_rho, six_eff = effective_model_count(merged[[f"{p}_solved" for p in llm6_prefixes]].dropna())
    seven_rho, seven_eff = effective_model_count(merged[[f"{p}_solved" for p in llm7_prefixes]].dropna())
    summary = {
        "current_llms": [
            "gpt-4o",
            "claude-3.5-sonnet",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "gemma-4-31b-it",
            "gemini-3.1-flash-lite-preview",
            "gpt-5.4-nano-2026-03-17",
            "openai/gpt-oss-120b:free",
        ],
        "note": "IceCuber remains separate and is not pooled with the LLMs.",
        "gemini": {
            **gemini_diag,
            "validated_coverage": int(merged["gemini_has_artifact"].fillna(0).sum()),
            "solve_rate_validated": float(merged["gemini_solved"].mean()),
        },
        "gpt_oss": {
            **gpt_oss_diag,
            "validated_coverage": int(merged["gpt_oss_has_artifact"].fillna(0).sum()),
            "solve_rate_validated": float(merged["gpt_oss_solved"].mean()),
        },
        "gpt5_nano": {
            **gpt5_nano_diag,
            "validated_coverage": int(merged["gpt5_nano_has_artifact"].fillna(0).sum()),
            "solve_rate_validated": float(merged["gpt5_nano_solved"].mean()),
        },
        "pooled_rates": {
            "llm6_pair_mean_rate": float(merged["llm6_pair_mean_rate"].mean()),
            "llm7_pair_mean_rate": float(merged["llm7_pair_mean_rate"].mean()),
        },
        "pairwise_dependence": {
            "llm6_mean_pairwise_corr": six_rho,
            "llm6_effective_n": six_eff,
            "llm7_mean_pairwise_corr": seven_rho,
            "llm7_effective_n": seven_eff,
        },
        "headline": {
            "complexity_pc1_human_r": float(safe_corr(merged["complexity_pc1_score"], merged["human_difficulty_complete"], "pearson")),
            "complexity_pc1_llm6_r": float(safe_corr(merged["complexity_pc1_score"], merged["llm6_pair_difficulty"], "pearson")),
            "complexity_pc1_llm7_r": float(safe_corr(merged["complexity_pc1_score"], merged["llm7_pair_difficulty"], "pearson")),
            "complexity_pc1_gpt_oss_r": float(safe_corr(merged["complexity_pc1_score"], merged["gpt_oss_pair_difficulty"], "pearson")),
            "complexity_pc1_gpt5_nano_r": float(safe_corr(merged["complexity_pc1_score"], merged["gpt5_nano_pair_difficulty"], "pearson")),
            "gap30_n": int(len(gap30_subset)),
            "gap30_complexity_pc1_human_r": float(safe_corr(gap30_subset["complexity_pc1_score"], gap30_subset["human_difficulty_complete"], "pearson")),
            "gap30_complexity_pc1_llm6_r": float(safe_corr(gap30_subset["complexity_pc1_score"], gap30_subset["llm6_pair_difficulty"], "pearson")),
            "gap30_complexity_pc1_llm7_r": float(safe_corr(gap30_subset["complexity_pc1_score"], gap30_subset["llm7_pair_difficulty"], "pearson")),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# ARC-1 GPT OSS + Gemini Analysis",
        "",
        "## Current LLM Pool",
        "",
    ]
    for name in summary["current_llms"]:
        lines.append(f"- {name}")
    lines.extend(
        [
            f"- {summary['note']}",
            "",
            "## Coverage",
            "",
            f"- Gemini 3.1 validated coverage: {summary['gemini']['validated_coverage']} tasks, solve rate = {summary['gemini']['solve_rate_validated']:.3f}.",
            f"- GPT-5.4 Nano validated coverage: {summary['gpt5_nano']['validated_coverage']} tasks, solve rate = {summary['gpt5_nano']['solve_rate_validated']:.3f}.",
            f"- GPT OSS 120B validated coverage: {summary['gpt_oss']['validated_coverage']} tasks, solve rate = {summary['gpt_oss']['solve_rate_validated']:.3f}.",
            "",
            "## Complexity PC1",
            "",
            f"- Human: r = {summary['headline']['complexity_pc1_human_r']:.3f}",
            f"- LLM pool without GPT OSS: r = {summary['headline']['complexity_pc1_llm6_r']:.3f}",
            f"- LLM pool with GPT OSS: r = {summary['headline']['complexity_pc1_llm7_r']:.3f}",
            f"- GPT-5.4 Nano alone: r = {summary['headline']['complexity_pc1_gpt5_nano_r']:.3f}",
            f"- GPT OSS alone: r = {summary['headline']['complexity_pc1_gpt_oss_r']:.3f}",
            "",
            "## GPT OSS Effect",
            "",
        ]
    )
    for _, row in compare_df[compare_df["complexity_metric"].eq("complexity_pc1_score")].iterrows():
        lines.append(
            f"- {row['subset_name']}: {row['left_target']} r = {row['left_r']:.3f}, "
            f"{row['right_target']} r = {row['right_r']:.3f}, Williams p = {row['williams_p']:.4g}."
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Saved GPT OSS + Gemini outputs:", file=sys.stderr, flush=True)
    print(f"  - {JOIN_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {CORR_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {COMPARE_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {SUMMARY_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {REPORT_PATH.name}", file=sys.stderr, flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
