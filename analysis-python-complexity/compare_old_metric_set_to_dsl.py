from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent
OLD_CORRELATIONS_PATH = ANALYSIS_DIR / "approved_llm_complexity_correlations.csv"
CURRENT_JOIN_PATH = ANALYSIS_DIR / "arc1_dsl_overlap_task_join.csv"

COMPATIBILITY_PATH = ANALYSIS_DIR / "arc1_dsl_old_metric_compatibility.csv"
LONG_RESULTS_PATH = ANALYSIS_DIR / "arc1_dsl_old_metric_target_correlations.csv"
SUMMARY_PATH = ANALYSIS_DIR / "arc1_dsl_old_metric_summary.json"
REPORT_PATH = ANALYSIS_DIR / "arc1_dsl_old_metric_report.md"


OLD_METRICS = [
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

DIRECT_METRICS = {
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
    "peak_memory_bytes",
    "elapsed_ms_per_input_cell",
}

DERIVED_NOTES = {
    "elapsed_ms_per_test": "Derived as elapsed_ms_total / test_pairs.",
    "opcode_count_dynamic": "Mapped to bundle_opcode_count_dynamic to count solver + DSL + primitive opcodes.",
    "branch_opcode_count_dynamic": "Mapped to bundle_branch_opcode_count_dynamic.",
    "python_call_count_dynamic": "Mapped to bundle_python_call_count_dynamic.",
    "opcode_per_input_cell": "Derived from bundle opcode count per input cell.",
    "complexity_pc1_score": "Mapped to dsl_complexity_pc1.",
    "log1p_opcode_count_dynamic": "Derived from mapped opcode_count_dynamic.",
    "log1p_branch_opcode_count_dynamic": "Derived from mapped branch opcode count.",
    "log1p_python_call_count_dynamic": "Derived from mapped python call count.",
    "log1p_elapsed_ms_total": "Derived from elapsed_ms_total.",
    "log1p_elapsed_ms_per_test": "Derived from elapsed_ms_per_test.",
    "log1p_peak_memory_bytes": "Derived from peak_memory_bytes.",
    "log1p_ast_node_count": "Derived from ast_node_count.",
    "log1p_cyclomatic_complexity": "Derived from cyclomatic_complexity.",
}


def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def build_current_compatibility_frame() -> pd.DataFrame:
    df = pd.read_csv(CURRENT_JOIN_PATH).copy()
    df["elapsed_ms_per_test"] = df["elapsed_ms_total"] / df["test_pairs"].clip(lower=1)
    df["opcode_count_dynamic"] = df["bundle_opcode_count_dynamic"]
    df["branch_opcode_count_dynamic"] = df["bundle_branch_opcode_count_dynamic"]
    df["python_call_count_dynamic"] = df["bundle_python_call_count_dynamic"]
    df["opcode_per_input_cell"] = df["bundle_opcode_count_dynamic"] / df["input_cells_total"].clip(lower=1)
    df["complexity_pc1_score"] = df["dsl_complexity_pc1"]

    for col in [
        "opcode_count_dynamic",
        "branch_opcode_count_dynamic",
        "python_call_count_dynamic",
        "elapsed_ms_total",
        "elapsed_ms_per_test",
        "peak_memory_bytes",
        "ast_node_count",
        "cyclomatic_complexity",
    ]:
        df[f"log1p_{col}"] = np.log1p(pd.to_numeric(df[col], errors="coerce").clip(lower=0))
    return df


def old_metric_best_rows(old_correlations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in OLD_METRICS:
        subset = old_correlations[old_correlations["complexity_metric"].eq(metric)].copy()
        if subset.empty:
            rows.append(
                {
                    "metric": metric,
                    "old_best_outcome": None,
                    "old_best_pearson_r": float("nan"),
                    "old_best_spearman_rho": float("nan"),
                    "old_n": 0,
                }
            )
            continue
        best = subset.iloc[subset["abs_pearson_r"].argmax()]
        rows.append(
            {
                "metric": metric,
                "old_best_outcome": best["llm_outcome"],
                "old_best_pearson_r": float(best["pearson_r"]),
                "old_best_spearman_rho": float(best["spearman_rho"]),
                "old_n": int(best["n"]),
            }
        )
    return pd.DataFrame(rows)


def long_current_results(df: pd.DataFrame) -> pd.DataFrame:
    subset_defs = {
        "full_set": df.copy(),
        "gap_le_0.30": df[df["human_llm_pair_gap"] <= 0.30].copy(),
    }
    target_defs = {
        "human_difficulty_complete": "Human latent difficulty",
        "gpt_pair_difficulty": "GPT pair-level smoothed difficulty",
        "claude_pair_difficulty": "Claude pair-level smoothed difficulty",
        "pooled_pair_difficulty": "Pooled GPT+Claude pair-level smoothed difficulty",
    }

    rows: list[dict[str, Any]] = []
    for subset_name, subset in subset_defs.items():
        for metric in OLD_METRICS:
            for target_col, target_label in target_defs.items():
                rows.append(
                    {
                        "metric": metric,
                        "subset_name": subset_name,
                        "target_metric": target_col,
                        "target_label": target_label,
                        "n": int(subset[[metric, target_col]].dropna().shape[0]),
                        "pearson_r": safe_corr(subset[metric], subset[target_col], "pearson"),
                        "spearman_rho": safe_corr(subset[metric], subset[target_col], "spearman"),
                    }
                )
    return pd.DataFrame(rows)


def build_compatibility_table(old_best: pd.DataFrame, current: pd.DataFrame, long_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in OLD_METRICS:
        series = pd.to_numeric(current[metric], errors="coerce")
        full_human = long_results[
            long_results["metric"].eq(metric)
            & long_results["subset_name"].eq("full_set")
            & long_results["target_metric"].eq("human_difficulty_complete")
        ].iloc[0]
        full_pooled = long_results[
            long_results["metric"].eq(metric)
            & long_results["subset_name"].eq("full_set")
            & long_results["target_metric"].eq("pooled_pair_difficulty")
        ].iloc[0]
        gap_human = long_results[
            long_results["metric"].eq(metric)
            & long_results["subset_name"].eq("gap_le_0.30")
            & long_results["target_metric"].eq("human_difficulty_complete")
        ].iloc[0]
        gap_pooled = long_results[
            long_results["metric"].eq(metric)
            & long_results["subset_name"].eq("gap_le_0.30")
            & long_results["target_metric"].eq("pooled_pair_difficulty")
        ].iloc[0]
        old_row = old_best[old_best["metric"].eq(metric)].iloc[0]

        rows.append(
            {
                "metric": metric,
                "availability_type": "direct" if metric in DIRECT_METRICS else "derived_compat",
                "compat_notes": "" if metric in DIRECT_METRICS else DERIVED_NOTES.get(metric, ""),
                "current_n": int(series.notna().sum()),
                "current_unique_count": int(series.dropna().nunique()),
                "current_nonzero_count": int((series.fillna(0) != 0).sum()),
                "current_is_degenerate": bool(series.dropna().nunique() <= 1),
                "current_min": float(series.min()) if series.notna().any() else float("nan"),
                "current_max": float(series.max()) if series.notna().any() else float("nan"),
                "old_best_outcome": old_row["old_best_outcome"],
                "old_best_pearson_r": old_row["old_best_pearson_r"],
                "old_best_spearman_rho": old_row["old_best_spearman_rho"],
                "old_n": old_row["old_n"],
                "full_human_pearson_r": full_human["pearson_r"],
                "full_pooled_pearson_r": full_pooled["pearson_r"],
                "gap30_human_pearson_r": gap_human["pearson_r"],
                "gap30_pooled_pearson_r": gap_pooled["pearson_r"],
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: dict[str, Any], compatibility: pd.DataFrame) -> None:
    degenerate = compatibility[compatibility["current_is_degenerate"]].copy()
    top_full = compatibility.sort_values("full_pooled_pearson_r", ascending=False).head(10)
    top_gap = compatibility.sort_values("gap30_pooled_pearson_r", ascending=False).head(10)

    lines = [
        "# Old Python Metric Set vs Current ARC-1 DSL Analysis",
        "",
        "## Headline",
        "",
        (
            f"- Old approved-Python analysis strongest headline was `{summary['old_headline_metric']}` "
            f"vs `{summary['old_headline_outcome']}` at Pearson r = {summary['old_headline_r']:.3f} on n = {summary['old_headline_n']}."
        ),
        (
            f"- In the current DSL analysis, {summary['degenerate_metric_count']} of the old {summary['old_metric_count']} "
            "metrics are degenerate because the DSL representation removes or compresses their variation."
        ),
        "",
        "## Degenerate Old Metrics Under The DSL Representation",
        "",
    ]
    for _, row in degenerate.iterrows():
        lines.append(f"- `{row['metric']}`: unique values = {int(row['current_unique_count'])}, notes = {row['compat_notes'] or 'direct metric'}")

    lines.extend(
        [
            "",
            "## Best Old-Compatible Metrics On The Current Full Set",
            "",
        ]
    )
    for _, row in top_full.iterrows():
        lines.append(
            f"- `{row['metric']}`: pooled pair-difficulty r = {row['full_pooled_pearson_r']:.3f}, "
            f"human latent r = {row['full_human_pearson_r']:.3f}, old best was {row['old_best_outcome']} at {row['old_best_pearson_r']:.3f}."
        )

    lines.extend(
        [
            "",
            "## Best Old-Compatible Metrics On The Gap<=0.30 Shared-Regime Subset",
            "",
        ]
    )
    for _, row in top_gap.iterrows():
        lines.append(
            f"- `{row['metric']}`: pooled pair-difficulty r = {row['gap30_pooled_pearson_r']:.3f}, "
            f"human latent r = {row['gap30_human_pearson_r']:.3f}."
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    old_correlations = pd.read_csv(OLD_CORRELATIONS_PATH)
    current = build_current_compatibility_frame()
    old_best = old_metric_best_rows(old_correlations)
    long_results = long_current_results(current)
    compatibility = build_compatibility_table(old_best, current, long_results)

    compatibility.to_csv(COMPATIBILITY_PATH, index=False)
    long_results.to_csv(LONG_RESULTS_PATH, index=False)

    old_headline = old_correlations.sort_values("abs_pearson_r", ascending=False).iloc[0]
    summary = {
        "old_metric_count": len(OLD_METRICS),
        "degenerate_metric_count": int(compatibility["current_is_degenerate"].sum()),
        "old_headline_metric": str(old_headline["complexity_metric"]),
        "old_headline_outcome": str(old_headline["llm_outcome"]),
        "old_headline_r": float(old_headline["pearson_r"]),
        "old_headline_n": int(old_headline["n"]),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, compatibility)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
