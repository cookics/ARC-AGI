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
MERGED_PATH = ANALYSIS_DIR / "arc1_dsl_complexity_task_join.csv"

OVERLAP_JOIN_PATH = ANALYSIS_DIR / "arc1_dsl_overlap_task_join.csv"
METRIC_CATALOG_PATH = ANALYSIS_DIR / "arc1_dsl_metric_catalog.csv"
SUBSET_RESULTS_PATH = ANALYSIS_DIR / "arc1_dsl_overlap_subset_correlations.csv"
ALIGNMENT_PATH = ANALYSIS_DIR / "arc1_dsl_overlap_difficulty_alignment.csv"
SUMMARY_PATH = ANALYSIS_DIR / "arc1_dsl_overlap_summary.json"
REPORT_PATH = ANALYSIS_DIR / "arc1_dsl_overlap_report.md"


STATIC_PYTHON_METRICS = {
    "source_bytes",
    "total_lines",
    "nonblank_lines",
    "comment_lines",
    "token_count",
    "ast_node_count",
    "function_count",
    "call_count_static",
    "comprehension_count",
    "branch_node_count",
    "return_count",
    "name_load_count",
    "constant_count",
    "assignment_node_count",
    "cyclomatic_complexity",
    "max_nesting_depth",
    "max_line_length",
    "mean_line_length",
    "gzip_bytes",
    "source_line_count",
}

DSL_STRUCTURE_METRICS = {
    "assignment_count",
    "temp_var_count",
    "ast_call_count",
    "named_call_count",
    "closure_call_count",
    "distinct_primitive_count",
    "max_dependency_depth",
    "max_fan_in",
    "higher_order_count",
    "object_op_count",
    "selection_op_count",
    "geometry_op_count",
    "set_op_count",
    "decision_op_count",
}

TASK_SHAPE_METRICS = {
    "train_pairs",
    "test_pairs",
    "example_count",
    "input_cells_total",
    "output_cells_total",
}

EXECUTION_METRICS = {
    "elapsed_ms_total",
    "elapsed_ms_per_example",
    "peak_memory_bytes",
    "current_memory_bytes",
    "solver_opcode_count_dynamic",
    "dsl_opcode_count_dynamic",
    "primitives_opcode_count_dynamic",
    "bundle_opcode_count_dynamic",
    "solver_branch_opcode_count_dynamic",
    "dsl_branch_opcode_count_dynamic",
    "primitives_branch_opcode_count_dynamic",
    "bundle_branch_opcode_count_dynamic",
    "solver_python_call_count_dynamic",
    "dsl_python_call_count_dynamic",
    "primitives_python_call_count_dynamic",
    "bundle_python_call_count_dynamic",
    "solver_distinct_function_count_dynamic",
    "dsl_distinct_function_count_dynamic",
    "primitives_distinct_function_count_dynamic",
    "bundle_distinct_function_count_dynamic",
    "elapsed_ms_per_input_cell",
    "elapsed_ms_per_output_cell",
    "bundle_opcode_per_input_cell",
    "bundle_opcode_per_output_cell",
    "bundle_branch_opcode_per_input_cell",
    "bundle_python_calls_per_example",
    "peak_memory_per_input_cell",
}


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
    ease = (successes.astype(float) + 0.5) / (totals.astype(float) + 1.0)
    return -np.log(ease / (1.0 - ease))


def add_overlap_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gpt_pair_successes"] = out["gpt_solved_pair_mean"] * out["test_pairs"]
    out["claude_pair_successes"] = out["claude_solved_pair_mean"] * out["test_pairs"]
    out["icecuber_pair_successes"] = out["icecuber_solved_pair_mean"] * out["test_pairs"]
    out["pooled_pair_successes"] = out["gpt_pair_successes"] + out["claude_pair_successes"]

    out["gpt_pair_difficulty"] = smoothed_difficulty_from_counts(out["gpt_pair_successes"], out["test_pairs"])
    out["claude_pair_difficulty"] = smoothed_difficulty_from_counts(out["claude_pair_successes"], out["test_pairs"])
    out["icecuber_pair_difficulty"] = smoothed_difficulty_from_counts(out["icecuber_pair_successes"], out["test_pairs"])
    out["pooled_pair_difficulty"] = smoothed_difficulty_from_counts(out["pooled_pair_successes"], 2 * out["test_pairs"])

    out["llm_pair_mean_rate"] = (out["gpt_solved_pair_mean"] + out["claude_solved_pair_mean"]) / 2.0
    out["llm_pair_failure"] = 1.0 - out["llm_pair_mean_rate"]
    out["human_gpt_pair_gap"] = (out["human_difficulty_complete_solve_rate"] - out["gpt_solved_pair_mean"]).abs()
    out["human_claude_pair_gap"] = (out["human_difficulty_complete_solve_rate"] - out["claude_solved_pair_mean"]).abs()
    out["human_llm_pair_gap"] = (out["human_difficulty_complete_solve_rate"] - out["llm_pair_mean_rate"]).abs()
    out["human_ice_pair_gap"] = (out["human_difficulty_complete_solve_rate"] - out["icecuber_solved_pair_mean"]).abs()
    return out


def metric_category(metric: str) -> str:
    if metric == "dsl_complexity_pc1":
        return "composite"
    if metric.startswith("prim_"):
        return "dsl_primitive_usage"
    if metric.startswith("halstead_"):
        return "python_static"
    if metric in STATIC_PYTHON_METRICS:
        return "python_static"
    if metric in DSL_STRUCTURE_METRICS:
        return "dsl_structure"
    if metric in TASK_SHAPE_METRICS:
        return "task_shape"
    if metric in EXECUTION_METRICS:
        return "dynamic_execution"
    return "other_numeric"


def build_metric_catalog(metric_columns: list[str]) -> pd.DataFrame:
    rows = [{"metric": metric, "category": metric_category(metric)} for metric in metric_columns]
    return pd.DataFrame(rows).sort_values(["category", "metric"]).reset_index(drop=True)


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def best_single_metric(
    subset: pd.DataFrame,
    metric_columns: list[str],
    target_col: str,
    min_n: int = 25,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for metric in metric_columns:
        pair = subset[[metric, target_col]].dropna()
        if len(pair) < min_n:
            continue
        if pair[metric].nunique() < 2 or pair[target_col].nunique() < 2:
            continue
        pearson_r = safe_corr(pair[metric], pair[target_col], "pearson")
        spearman_rho = safe_corr(pair[metric], pair[target_col], "spearman")
        if math.isnan(pearson_r):
            continue
        if best is None or abs(pearson_r) > abs(best["pearson_r"]):
            best = {
                "metric": metric,
                "pearson_r": pearson_r,
                "spearman_rho": spearman_rho,
                "n": int(len(pair)),
                "category": metric_category(metric),
            }
    if best is None:
        return {
            "metric": None,
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "n": 0,
            "category": None,
        }
    return best


def subset_defs(df: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    defs: list[tuple[str, str, pd.DataFrame]] = [("all_tasks", "full_set", df.copy())]
    for gap in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        subset = df[df["human_llm_pair_gap"] <= gap].copy()
        defs.append(("matched_pooled_gap", f"gap_le_{gap:.2f}", subset))
    return defs


def build_alignment_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, subset_name, subset in subset_defs(df):
        rows.append(
            {
                "subset_family": family,
                "subset_name": subset_name,
                "n": int(len(subset)),
                "human_mean_solve_rate": float(subset["human_difficulty_complete_solve_rate"].mean()),
                "gpt_mean_pair_rate": float(subset["gpt_solved_pair_mean"].mean()),
                "claude_mean_pair_rate": float(subset["claude_solved_pair_mean"].mean()),
                "pooled_mean_pair_rate": float(subset["llm_pair_mean_rate"].mean()),
                "icecuber_mean_pair_rate": float(subset["icecuber_solved_pair_mean"].mean()),
                "mean_human_pooled_gap": float(subset["human_llm_pair_gap"].mean()),
                "human_vs_gpt_pair_difficulty_r": safe_corr(
                    subset["human_difficulty_complete"], subset["gpt_pair_difficulty"], "pearson"
                ),
                "human_vs_claude_pair_difficulty_r": safe_corr(
                    subset["human_difficulty_complete"], subset["claude_pair_difficulty"], "pearson"
                ),
                "human_vs_pooled_pair_difficulty_r": safe_corr(
                    subset["human_difficulty_complete"], subset["pooled_pair_difficulty"], "pearson"
                ),
                "human_vs_icecuber_pair_difficulty_r": safe_corr(
                    subset["human_difficulty_complete"], subset["icecuber_pair_difficulty"], "pearson"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_subset_result_table(df: pd.DataFrame, metric_columns: list[str], base_module) -> pd.DataFrame:
    target_info = {
        "human_difficulty_complete": "Human latent difficulty",
        "gpt_pair_difficulty": "GPT pair-level smoothed difficulty",
        "claude_pair_difficulty": "Claude pair-level smoothed difficulty",
        "pooled_pair_difficulty": "Pooled GPT+Claude pair-level smoothed difficulty",
    }
    subset_list = subset_defs(df)
    total_steps = len(subset_list) * len(target_info)
    step = 0
    rows: list[dict[str, Any]] = []

    for family, subset_name, subset in subset_list:
        for target_col, target_label in target_info.items():
            step += 1
            show_progress(step, total_steps, f"Evaluating subset correlations ({subset_name}, {target_col})")
            if len(subset) < 25:
                rows.append(
                    {
                        "subset_family": family,
                        "subset_name": subset_name,
                        "target_metric": target_col,
                        "target_label": target_label,
                        "n": int(len(subset)),
                        "human_mean_solve_rate": float(subset["human_difficulty_complete_solve_rate"].mean()),
                        "gpt_mean_pair_rate": float(subset["gpt_solved_pair_mean"].mean()),
                        "claude_mean_pair_rate": float(subset["claude_solved_pair_mean"].mean()),
                        "pooled_mean_pair_rate": float(subset["llm_pair_mean_rate"].mean()),
                        "mean_human_pooled_gap": float(subset["human_llm_pair_gap"].mean()),
                        "pc1_pearson_r": float("nan"),
                        "pc1_spearman_rho": float("nan"),
                        "best_metric": None,
                        "best_metric_category": None,
                        "best_metric_pearson_r": float("nan"),
                        "best_metric_spearman_rho": float("nan"),
                        "best_metric_human_pearson_r": float("nan"),
                        "best_metric_human_spearman_rho": float("nan"),
                        "best_metric_williams_p_vs_human": float("nan"),
                    }
                )
                continue

            pc1_pearson = safe_corr(subset["dsl_complexity_pc1"], subset[target_col], "pearson")
            pc1_spearman = safe_corr(subset["dsl_complexity_pc1"], subset[target_col], "spearman")
            best = best_single_metric(subset, metric_columns, target_col)

            best_metric_human_pearson = float("nan")
            best_metric_human_spearman = float("nan")
            best_metric_williams_p = float("nan")
            if best["metric"] is not None and target_col != "human_difficulty_complete":
                best_metric_human_pearson = safe_corr(
                    subset[best["metric"]], subset["human_difficulty_complete"], "pearson"
                )
                best_metric_human_spearman = safe_corr(
                    subset[best["metric"]], subset["human_difficulty_complete"], "spearman"
                )
                yz_corr = safe_corr(subset[target_col], subset["human_difficulty_complete"], "pearson")
                _, best_metric_williams_p = base_module.williams_test(
                    best["pearson_r"],
                    best_metric_human_pearson,
                    yz_corr,
                    int(len(subset[[best["metric"], target_col, "human_difficulty_complete"]].dropna())),
                )

            rows.append(
                {
                    "subset_family": family,
                    "subset_name": subset_name,
                    "target_metric": target_col,
                    "target_label": target_label,
                    "n": int(len(subset)),
                    "human_mean_solve_rate": float(subset["human_difficulty_complete_solve_rate"].mean()),
                    "gpt_mean_pair_rate": float(subset["gpt_solved_pair_mean"].mean()),
                    "claude_mean_pair_rate": float(subset["claude_solved_pair_mean"].mean()),
                    "pooled_mean_pair_rate": float(subset["llm_pair_mean_rate"].mean()),
                    "mean_human_pooled_gap": float(subset["human_llm_pair_gap"].mean()),
                    "pc1_pearson_r": pc1_pearson,
                    "pc1_spearman_rho": pc1_spearman,
                    "best_metric": best["metric"],
                    "best_metric_category": best["category"],
                    "best_metric_pearson_r": best["pearson_r"],
                    "best_metric_spearman_rho": best["spearman_rho"],
                    "best_metric_human_pearson_r": best_metric_human_pearson,
                    "best_metric_human_spearman_rho": best_metric_human_spearman,
                    "best_metric_williams_p_vs_human": best_metric_williams_p,
                }
            )
    return pd.DataFrame(rows)


def build_summary(
    metric_catalog: pd.DataFrame,
    alignment: pd.DataFrame,
    subset_results: pd.DataFrame,
) -> dict[str, Any]:
    full_pooled = subset_results[
        subset_results["subset_name"].eq("full_set")
        & subset_results["target_metric"].eq("pooled_pair_difficulty")
    ].iloc[0]
    best_matched_pooled = subset_results[
        subset_results["target_metric"].eq("pooled_pair_difficulty")
        & subset_results["subset_family"].eq("matched_pooled_gap")
        & subset_results["n"].ge(50)
    ].sort_values(["best_metric_pearson_r", "n"], ascending=[False, False]).iloc[0]

    full_alignment = alignment[alignment["subset_name"].eq("full_set")].iloc[0]
    matched_alignment = alignment[alignment["subset_name"].eq(best_matched_pooled["subset_name"])].iloc[0]

    summary = {
        "metric_category_counts": metric_catalog["category"].value_counts().sort_index().to_dict(),
        "full_set": {
            "n": int(full_pooled["n"]),
            "pooled_best_metric": full_pooled["best_metric"],
            "pooled_best_metric_r": float(full_pooled["best_metric_pearson_r"]),
            "pooled_pc1_r": float(full_pooled["pc1_pearson_r"]),
            "human_vs_pooled_pair_difficulty_r": float(full_alignment["human_vs_pooled_pair_difficulty_r"]),
        },
        "best_matched_pooled_subset": {
            "subset_name": str(best_matched_pooled["subset_name"]),
            "n": int(best_matched_pooled["n"]),
            "human_mean_solve_rate": float(best_matched_pooled["human_mean_solve_rate"]),
            "pooled_mean_pair_rate": float(best_matched_pooled["pooled_mean_pair_rate"]),
            "mean_human_pooled_gap": float(best_matched_pooled["mean_human_pooled_gap"]),
            "pooled_best_metric": str(best_matched_pooled["best_metric"]),
            "pooled_best_metric_category": str(best_matched_pooled["best_metric_category"]),
            "pooled_best_metric_r": float(best_matched_pooled["best_metric_pearson_r"]),
            "pooled_pc1_r": float(best_matched_pooled["pc1_pearson_r"]),
            "same_metric_human_r": float(best_matched_pooled["best_metric_human_pearson_r"]),
            "same_metric_williams_p_vs_human": float(best_matched_pooled["best_metric_williams_p_vs_human"]),
            "human_vs_pooled_pair_difficulty_r": float(matched_alignment["human_vs_pooled_pair_difficulty_r"]),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_report(summary: dict[str, Any], alignment: pd.DataFrame, subset_results: pd.DataFrame) -> None:
    full = summary["full_set"]
    matched = summary["best_matched_pooled_subset"]
    matched_alignment = alignment[alignment["subset_name"].eq(matched["subset_name"])].iloc[0]
    matched_rows = subset_results[
        subset_results["subset_name"].eq(matched["subset_name"])
        & subset_results["target_metric"].isin(
            [
                "human_difficulty_complete",
                "gpt_pair_difficulty",
                "claude_pair_difficulty",
                "pooled_pair_difficulty",
            ]
        )
    ].copy()

    lines = [
        "# ARC-1 Overlap-Regime Complexity Analysis",
        "",
        "## What This Follow-Up Does",
        "",
        "- Starts from the saved validated ARC-1 task join rather than rerunning solver validation.",
        "- Uses pair-level LLM solve rates and smoothed pair-level difficulties, not just binary task solved flags.",
        "- Builds shared-regime subsets where pooled GPT+Claude pair success is close to the human solve rate.",
        "",
        "## Metric Inventory",
        "",
    ]
    for category, count in sorted(summary["metric_category_counts"].items()):
        lines.append(f"- {category}: {count} metrics")
    lines.extend(
        [
            "",
            "## Full Set Vs Best Matched Subset",
            "",
            f"- Full set pooled pair-difficulty: best single metric = `{full['pooled_best_metric']}` with Pearson r = {full['pooled_best_metric_r']:.3f}; PC1 = {full['pooled_pc1_r']:.3f}; human vs pooled pair-difficulty = {full['human_vs_pooled_pair_difficulty_r']:.3f}.",
            (
                f"- Best shared-regime pooled subset = `{matched['subset_name']}` with n = {matched['n']}, "
                f"human mean solve rate = {matched['human_mean_solve_rate']:.3f}, pooled GPT+Claude mean pair rate = {matched['pooled_mean_pair_rate']:.3f}, "
                f"mean rate gap = {matched['mean_human_pooled_gap']:.3f}."
            ),
            (
                f"- In that matched subset, best single metric = `{matched['pooled_best_metric']}` "
                f"({matched['pooled_best_metric_category']}) with pooled pair-difficulty Pearson r = {matched['pooled_best_metric_r']:.3f}."
            ),
            (
                f"- Using the same metric on humans gives Pearson r = {matched['same_metric_human_r']:.3f}; "
                f"Williams p for pooled-vs-human difference on that metric = {matched['same_metric_williams_p_vs_human']:.4g}."
            ),
            (
                f"- Human latent difficulty vs pooled pair-level difficulty jumps to Pearson r = "
                f"{matched['human_vs_pooled_pair_difficulty_r']:.3f} in the matched subset."
            ),
            "",
            "## Matched-Subset Target Snapshot",
            "",
        ]
    )
    for _, row in matched_rows.sort_values("target_metric").iterrows():
        lines.append(
            f"- {row['target_label']}: PC1 = {row['pc1_pearson_r']:.3f}; best single = `{row['best_metric']}` "
            f"with Pearson r = {row['best_metric_pearson_r']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Latent-Style Alignment Across Gap Thresholds",
            "",
        ]
    )
    for _, row in alignment[alignment["subset_family"].eq("matched_pooled_gap")].iterrows():
        lines.append(
            f"- {row['subset_name']}: n = {int(row['n'])}, human mean = {row['human_mean_solve_rate']:.3f}, "
            f"pooled mean = {row['pooled_mean_pair_rate']:.3f}, human vs pooled pair-difficulty = "
            f"{row['human_vs_pooled_pair_difficulty_r']:.3f}."
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    stage_total = 5
    show_stage(1, stage_total, "Loading saved ARC-1 task join")
    print("Loading saved ARC-1 task join...", file=sys.stderr, flush=True)
    base_module = load_base_module()
    merged = pd.read_csv(MERGED_PATH)
    metric_columns = base_module.complexity_metric_columns(merged)

    show_stage(2, stage_total, "Building overlap-aware task table")
    print("Building overlap-aware task table...", file=sys.stderr, flush=True)
    enriched = add_overlap_columns(merged)
    enriched.to_csv(OVERLAP_JOIN_PATH, index=False)

    show_stage(3, stage_total, "Cataloging complexity metrics")
    print("Cataloging complexity metrics...", file=sys.stderr, flush=True)
    metric_catalog = build_metric_catalog(metric_columns)
    metric_catalog.to_csv(METRIC_CATALOG_PATH, index=False)

    show_stage(4, stage_total, "Computing overlap subsets and correlations")
    print("Computing overlap subsets and correlations...", file=sys.stderr, flush=True)
    alignment = build_alignment_table(enriched)
    subset_results = build_subset_result_table(enriched, metric_columns, base_module)
    alignment.to_csv(ALIGNMENT_PATH, index=False)
    subset_results.to_csv(SUBSET_RESULTS_PATH, index=False)

    show_stage(5, stage_total, "Writing overlap report")
    print("Writing overlap report...", file=sys.stderr, flush=True)
    summary = build_summary(metric_catalog, alignment, subset_results)
    write_report(summary, alignment, subset_results)

    print("Saved overlap-aware outputs:", file=sys.stderr, flush=True)
    print(f"  - {OVERLAP_JOIN_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {METRIC_CATALOG_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {ALIGNMENT_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {SUBSET_RESULTS_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {SUMMARY_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {REPORT_PATH.name}", file=sys.stderr, flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
