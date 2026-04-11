from __future__ import annotations

import argparse
import ast
import dis
import importlib.util
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ANALYSIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT_PATH = ANALYSIS_DIR / "analyze_arc1_dsl_human_llm.py"
OVERLAP_JOIN_PATH = ANALYSIS_DIR / "arc1_dsl_overlap_task_join.csv"

EXPANDED_METRICS_PATH = ANALYSIS_DIR / "arc1_dsl_expanded_oldstyle_metrics.csv"
EXPANDED_JOIN_PATH = ANALYSIS_DIR / "arc1_dsl_expanded_oldstyle_join.csv"
EXPANDED_CORRELATIONS_PATH = ANALYSIS_DIR / "arc1_dsl_expanded_oldstyle_correlations.csv"
EXPANDED_BEST_PATH = ANALYSIS_DIR / "arc1_dsl_expanded_oldstyle_best_metrics.csv"
EXPANDED_COMPARISON_PATH = ANALYSIS_DIR / "arc1_dsl_expanded_vs_surface_old_metrics.csv"
EXPANDED_SUMMARY_PATH = ANALYSIS_DIR / "arc1_dsl_expanded_oldstyle_summary.json"
EXPANDED_REPORT_PATH = ANALYSIS_DIR / "arc1_dsl_expanded_oldstyle_report.md"


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

PCA_METRICS = [
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
    "input_cells_total",
    "output_cells_total",
    "elapsed_ms_per_test",
    "opcode_count_dynamic",
    "branch_opcode_count_dynamic",
    "python_call_count_dynamic",
    "peak_memory_bytes",
    "opcode_per_input_cell",
    "elapsed_ms_per_input_cell",
]

TARGETS = {
    "human_difficulty_complete": "Human latent difficulty",
    "gpt_pair_difficulty": "GPT pair-level smoothed difficulty",
    "claude_pair_difficulty": "Claude pair-level smoothed difficulty",
    "pooled_pair_difficulty": "Pooled GPT+Claude pair-level smoothed difficulty",
}

BRANCH_PREFIXES = ("JUMP", "POP_JUMP", "FOR_ITER", "END_FOR", "END_SEND")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expanded old-style Python metrics for ARC-1 DSL solvers.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore any saved metrics checkpoint and recompute every task.",
    )
    return parser.parse_args()


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


def function_name_from_call(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def uppercase_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and child.id.isupper()}


def call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = function_name_from_call(child)
            if name is not None:
                names.add(name)
    return names


def function_inventory(base_module) -> dict[str, Any]:
    sections = base_module.split_bundle_sections(base_module.BUNDLE_PATH.read_text(encoding="utf-8"))
    dsl_source = sections["dsl.py"]
    solver_source = sections["solvers.py"]
    primitives_source = sections["primitives.py"]

    dsl_tree = ast.parse(dsl_source)
    solver_tree = ast.parse(solver_source)
    primitives_tree = ast.parse(primitives_source)

    dsl_funcs: dict[str, ast.FunctionDef] = {}
    dsl_func_sources: dict[str, str] = {}
    dsl_func_order: dict[str, int] = {}
    for node in dsl_tree.body:
        if isinstance(node, ast.FunctionDef):
            dsl_funcs[node.name] = node
            dsl_func_sources[node.name] = base_module.solver_source_segment(dsl_source, node)
            dsl_func_order[node.name] = node.lineno

    solver_funcs: dict[str, ast.FunctionDef] = {}
    solver_func_sources: dict[str, str] = {}
    for node in solver_tree.body:
        if isinstance(node, ast.FunctionDef) and base_module.SOLVE_NAME_RE.match(node.name):
            solver_funcs[node.name] = node
            solver_func_sources[node.name] = base_module.solver_source_segment(solver_source, node)

    constant_sources: dict[str, str] = {}
    constant_order: dict[str, int] = {}
    for node in primitives_tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if not name.isupper():
            continue
        constant_sources[name] = base_module.solver_source_segment(primitives_source, node)  # type: ignore[arg-type]
        constant_order[name] = node.lineno

    return {
        "sections": sections,
        "dsl_funcs": dsl_funcs,
        "dsl_func_sources": dsl_func_sources,
        "dsl_func_order": dsl_func_order,
        "solver_funcs": solver_funcs,
        "solver_func_sources": solver_func_sources,
        "constant_sources": constant_sources,
        "constant_order": constant_order,
    }


def expanded_source_for_task(task_id: str, inventory: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    solve_name = f"solve_{task_id}"
    solver_node = inventory["solver_funcs"][solve_name]
    needed_constants = uppercase_names(solver_node) & set(inventory["constant_sources"])
    pending = sorted(call_names(solver_node))
    seen_funcs: set[str] = set()

    while pending:
        name = pending.pop()
        if name in seen_funcs or name not in inventory["dsl_funcs"]:
            continue
        seen_funcs.add(name)
        fn_node = inventory["dsl_funcs"][name]
        needed_constants |= uppercase_names(fn_node) & set(inventory["constant_sources"])
        for child_name in call_names(fn_node):
            if child_name in inventory["dsl_funcs"] and child_name not in seen_funcs:
                pending.append(child_name)

    constant_names = sorted(needed_constants, key=lambda name: inventory["constant_order"][name])
    function_names = sorted(seen_funcs, key=lambda name: inventory["dsl_func_order"][name])

    parts = [
        "# Synthetic expanded ARC-1 solver source",
        "from codeit.dsl.arc_types import *",
        "",
    ]
    for name in constant_names:
        parts.append(inventory["constant_sources"][name])
        parts.append("")
    for name in function_names:
        parts.append(inventory["dsl_func_sources"][name])
        parts.append("")
    parts.append(inventory["solver_func_sources"][solve_name])
    parts.append("")
    return "\n".join(parts), function_names, constant_names


def surface_old_metric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["elapsed_ms_per_test"] = out["elapsed_ms_total"] / out["test_pairs"].clip(lower=1)
    out["opcode_count_dynamic"] = out["bundle_opcode_count_dynamic"]
    out["branch_opcode_count_dynamic"] = out["bundle_branch_opcode_count_dynamic"]
    out["python_call_count_dynamic"] = out["bundle_python_call_count_dynamic"]
    out["opcode_per_input_cell"] = out["bundle_opcode_count_dynamic"] / out["input_cells_total"].clip(lower=1)
    out["complexity_pc1_score"] = out["dsl_complexity_pc1"]
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
        out[f"log1p_{metric}"] = np.log1p(pd.to_numeric(out[metric], errors="coerce").clip(lower=0))
    return out


def profile_task_dynamic(base_module, bundle, task: dict[str, Any], task_id: str) -> dict[str, Any]:
    solve = getattr(bundle.solvers, f"solve_{task_id}")
    solver_filename = str(bundle.solvers.__file__)
    dsl_filename = str(bundle.dsl.__file__)
    primitives_filename = str(bundle.primitives.__file__)

    opcode_count = 0
    branch_opcode_count = 0
    python_call_count = 0
    function_names: set[str] = set()
    elapsed_ns_total = 0
    peak_memory_bytes = 0
    current_memory_bytes = 0
    input_cells_total = 0
    output_cells_total = 0

    def classify_frame(frame) -> bool:
        filename = str(frame.f_code.co_filename)
        return filename in {solver_filename, dsl_filename, primitives_filename}

    def tracer(frame, event, arg):
        nonlocal opcode_count, branch_opcode_count, python_call_count
        if not classify_frame(frame):
            return None
        if event == "call":
            python_call_count += 1
            function_names.add(frame.f_code.co_name)
            frame.f_trace_lines = False
            frame.f_trace_opcodes = True
            return tracer
        if event == "opcode":
            opcode = dis.opname[frame.f_code.co_code[frame.f_lasti]]
            opcode_count += 1
            if opcode.startswith(BRANCH_PREFIXES):
                branch_opcode_count += 1
            return tracer
        return tracer

    tracemalloc.start()
    try:
        for pair in list(task.get("train", [])) + list(task.get("test", [])):
            input_grid = base_module.canonical_grid(pair["input"])
            expected_grid = base_module.canonical_grid(pair["output"])
            input_cells_total += sum(len(row) for row in input_grid)
            output_cells_total += sum(len(row) for row in expected_grid)

            start_ns = time.perf_counter_ns()
            sys.settrace(tracer)
            try:
                _ = solve(input_grid)
            finally:
                sys.settrace(None)
            elapsed_ns_total += time.perf_counter_ns() - start_ns
            current_memory_bytes, peak = tracemalloc.get_traced_memory()
            peak_memory_bytes = max(peak_memory_bytes, peak)
            tracemalloc.reset_peak()
    finally:
        tracemalloc.stop()

    elapsed_ms_total = elapsed_ns_total / 1_000_000.0
    test_pairs = max(len(task.get("test", [])), 1)
    return {
        "dynamic_status": "ok",
        "input_cells_total": input_cells_total,
        "output_cells_total": output_cells_total,
        "elapsed_ms_total": elapsed_ms_total,
        "elapsed_ms_per_test": elapsed_ms_total / test_pairs,
        "opcode_count_dynamic": opcode_count,
        "branch_opcode_count_dynamic": branch_opcode_count,
        "python_call_count_dynamic": python_call_count,
        "peak_memory_bytes": peak_memory_bytes,
        "current_memory_bytes": current_memory_bytes,
        "opcode_per_input_cell": opcode_count / max(input_cells_total, 1),
        "elapsed_ms_per_input_cell": elapsed_ms_total / max(input_cells_total, 1),
        "branch_opcode_per_input_cell": branch_opcode_count / max(input_cells_total, 1),
        "python_calls_per_input_cell": python_call_count / max(input_cells_total, 1),
        "distinct_function_count_dynamic": len(function_names),
    }


def best_rows_by_subset_target(correlations: pd.DataFrame) -> pd.DataFrame:
    work = correlations.copy()
    work["abs_pearson_r"] = work["pearson_r"].abs()
    return (
        work.sort_values(["subset_name", "target_metric", "abs_pearson_r"], ascending=[True, True, False])
        .groupby(["subset_name", "target_metric"], as_index=False)
        .head(10)
        .reset_index(drop=True)
    )


def compute_complexity_pc1(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df[PCA_METRICS].copy()
    for metric in PCA_METRICS:
        transformed[metric] = np.log1p(pd.to_numeric(transformed[metric], errors="coerce").clip(lower=0))
    x_scaled = StandardScaler().fit_transform(transformed.to_numpy(dtype=float))
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(x_scaled).reshape(-1)
    if np.corrcoef(pc1, pd.to_numeric(df["ast_node_count"], errors="coerce").to_numpy(dtype=float))[0, 1] < 0:
        pc1 *= -1
    df = df.copy()
    df["complexity_pc1_score"] = pc1
    return df


def correlation_rows(df: pd.DataFrame, subset_name: str, target_cols: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in OLD_METRICS:
        for target_col, target_label in target_cols.items():
            pair = df[[metric, target_col]].dropna()
            rows.append(
                {
                    "subset_name": subset_name,
                    "complexity_metric": metric,
                    "target_metric": target_col,
                    "target_label": target_label,
                    "n": int(len(pair)),
                    "pearson_r": safe_corr(pair[metric], pair[target_col]),
                    "spearman_rho": safe_corr(pair[metric], pair[target_col], method="spearman"),
                }
            )
    return rows


def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def comparison_rows(surface: pd.DataFrame, expanded: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subset_name, subset_mask in {
        "full_set": surface["task_id"].notna(),
        "gap_le_0.30": surface["human_llm_pair_gap"] <= 0.30,
    }.items():
        surface_subset = surface.loc[subset_mask].copy()
        expanded_subset = expanded.loc[subset_mask].copy()
        for metric in OLD_METRICS:
            for target_col, target_label in TARGETS.items():
                rows.append(
                    {
                        "subset_name": subset_name,
                        "complexity_metric": metric,
                        "target_metric": target_col,
                        "target_label": target_label,
                        "surface_pearson_r": safe_corr(surface_subset[metric], surface_subset[target_col]),
                        "expanded_pearson_r": safe_corr(expanded_subset[metric], expanded_subset[target_col]),
                        "pearson_improvement": safe_corr(expanded_subset[metric], expanded_subset[target_col])
                        - safe_corr(surface_subset[metric], surface_subset[target_col]),
                    }
                )
    return pd.DataFrame(rows)


def write_report(summary: dict[str, Any], best: pd.DataFrame, comparison: pd.DataFrame) -> None:
    gap_best = best[
        best["subset_name"].eq("gap_le_0.30") & best["target_metric"].eq("pooled_pair_difficulty")
    ].iloc[0]
    full_best = best[
        best["subset_name"].eq("full_set") & best["target_metric"].eq("pooled_pair_difficulty")
    ].iloc[0]
    biggest_improvements = (
        comparison[comparison["target_metric"].eq("pooled_pair_difficulty")]
        .sort_values("pearson_improvement", ascending=False)
        .head(10)
    )

    lines = [
        "# Expanded Old-Style Metrics For ARC-1 DSL Solvers",
        "",
        "## Headline",
        "",
        (
            f"- Full set best old-style expanded metric for pooled GPT+Claude pair difficulty is "
            f"`{full_best['complexity_metric']}` with Pearson r = {full_best['pearson_r']:.3f}."
        ),
        (
            f"- Shared-regime gap<=0.30 best old-style expanded metric for pooled GPT+Claude pair difficulty is "
            f"`{gap_best['complexity_metric']}` with Pearson r = {gap_best['pearson_r']:.3f}."
        ),
        (
            f"- Expanded cyclomatic complexity is now non-degenerate across {summary['cyclomatic_unique_count']} "
            "distinct values."
        ),
        (
            f"- Expanded branch-opcode tracing is now non-degenerate across {summary['branch_opcode_unique_count']} "
            "distinct values."
        ),
        "",
        "## Biggest Surface -> Expanded Improvements For Pooled Pair Difficulty",
        "",
    ]
    for _, row in biggest_improvements.iterrows():
        lines.append(
            f"- {row['subset_name']} / `{row['complexity_metric']}`: "
            f"surface r = {row['surface_pearson_r']:.3f}, expanded r = {row['expanded_pearson_r']:.3f}, "
            f"gain = {row['pearson_improvement']:.3f}."
        )
    EXPANDED_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base = load_base_module()
    stage_total = 5

    show_stage(1, stage_total, "Loading saved outcomes and DSL bundle")
    print("Loading saved outcomes and DSL bundle...", file=sys.stderr, flush=True)
    overlap = pd.read_csv(OVERLAP_JOIN_PATH).sort_values("task_id").reset_index(drop=True)
    task_ids = overlap["task_id"].astype(str).tolist()
    tasks = base.load_arc1_training_tasks()
    bundle = base.load_bundle_modules()
    inventory = function_inventory(base)

    existing: pd.DataFrame | None = None
    existing_ids: set[str] = set()
    if EXPANDED_METRICS_PATH.exists() and not args.force:
        existing = pd.read_csv(EXPANDED_METRICS_PATH)
        existing_ids = set(existing["task_id"].astype(str))

    rows: list[dict[str, Any]] = [] if existing is None else existing.to_dict(orient="records")

    show_stage(2, stage_total, "Expanding primitive source and profiling tasks")
    print("Expanding primitive source and profiling tasks...", file=sys.stderr, flush=True)
    total_tasks = len(task_ids)
    for position, task_id in enumerate(task_ids, start=1):
        if task_id in existing_ids:
            show_progress(position, total_tasks, f"Expanded old-style metrics ({task_id}, cached)")
            continue
        expanded_source, function_names, constant_names = expanded_source_for_task(task_id, inventory)
        static_metrics = base.static_python_metrics(expanded_source)
        dynamic_metrics = profile_task_dynamic(base, bundle, tasks[task_id], task_id)
        rows.append(
            {
                "task_id": task_id,
                **static_metrics,
                **dynamic_metrics,
                "expanded_source_bytes": len(expanded_source.encode("utf-8")),
                "expanded_function_inventory_count": len(function_names),
                "expanded_constant_count": len(constant_names),
                "expanded_function_inventory": "|".join(function_names),
                "expanded_constant_inventory": "|".join(constant_names),
            }
        )
        pd.DataFrame(rows).sort_values("task_id").to_csv(EXPANDED_METRICS_PATH, index=False)
        show_progress(position, total_tasks, f"Expanded old-style metrics ({task_id})")

    metrics_df = pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)

    show_stage(3, stage_total, "Building old-style transformed metrics and join")
    print("Building old-style transformed metrics and join...", file=sys.stderr, flush=True)
    metrics_df = compute_complexity_pc1(metrics_df)
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
        metrics_df[f"log1p_{metric}"] = np.log1p(pd.to_numeric(metrics_df[metric], errors="coerce").clip(lower=0))

    expanded_join = overlap.drop(columns=[column for column in OLD_METRICS if column in overlap.columns], errors="ignore")
    expanded_join = expanded_join.merge(metrics_df, on="task_id", how="inner").sort_values("task_id").reset_index(drop=True)
    metrics_df.to_csv(EXPANDED_METRICS_PATH, index=False)
    expanded_join.to_csv(EXPANDED_JOIN_PATH, index=False)

    show_stage(4, stage_total, "Computing correlations and surface comparisons")
    print("Computing correlations and surface comparisons...", file=sys.stderr, flush=True)
    correlation_rows_all: list[dict[str, Any]] = []
    correlation_rows_all.extend(correlation_rows(expanded_join, "full_set", TARGETS))
    correlation_rows_all.extend(
        correlation_rows(expanded_join[expanded_join["human_llm_pair_gap"] <= 0.30].copy(), "gap_le_0.30", TARGETS)
    )
    correlations = pd.DataFrame(correlation_rows_all)
    best = best_rows_by_subset_target(correlations)
    surface = surface_old_metric_frame(overlap)
    comparison = comparison_rows(surface, expanded_join)
    correlations.to_csv(EXPANDED_CORRELATIONS_PATH, index=False)
    best.to_csv(EXPANDED_BEST_PATH, index=False)
    comparison.to_csv(EXPANDED_COMPARISON_PATH, index=False)

    show_stage(5, stage_total, "Writing summary report")
    print("Writing summary report...", file=sys.stderr, flush=True)
    summary = {
        "task_count": int(len(expanded_join)),
        "cyclomatic_unique_count": int(metrics_df["cyclomatic_complexity"].nunique()),
        "branch_opcode_unique_count": int(metrics_df["branch_opcode_count_dynamic"].nunique()),
        "full_best_pooled_metric": best[
            best["subset_name"].eq("full_set") & best["target_metric"].eq("pooled_pair_difficulty")
        ]
        .head(1)
        .to_dict(orient="records"),
        "gap30_best_pooled_metric": best[
            best["subset_name"].eq("gap_le_0.30") & best["target_metric"].eq("pooled_pair_difficulty")
        ]
        .head(1)
        .to_dict(orient="records"),
        "top_surface_to_expanded_improvements": comparison[
            comparison["target_metric"].eq("pooled_pair_difficulty")
        ]
        .sort_values("pearson_improvement", ascending=False)
        .head(10)
        .to_dict(orient="records"),
    }
    EXPANDED_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, best, comparison)

    print("Saved expanded old-style outputs:", file=sys.stderr, flush=True)
    print(f"  - {EXPANDED_METRICS_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {EXPANDED_JOIN_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {EXPANDED_CORRELATIONS_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {EXPANDED_BEST_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {EXPANDED_COMPARISON_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {EXPANDED_SUMMARY_PATH.name}", file=sys.stderr, flush=True)
    print(f"  - {EXPANDED_REPORT_PATH.name}", file=sys.stderr, flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
