from __future__ import annotations

import argparse
import ast
import copy
import csv
import dis
import gzip
import hashlib
import importlib.util
import io
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
import tokenize
import tracemalloc
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OFFICIAL_SOURCES = {
    "arc_agi_1_train": "buildtmp/ARC-AGI-1-official/data/training",
    "arc_agi_1_eval": "buildtmp/ARC-AGI-1-official/data/evaluation",
    "arc_agi_2_train": "buildtmp/ARC-AGI-2-official/data/training",
    "arc_agi_2_eval": "buildtmp/ARC-AGI-2-official/data/evaluation",
}


@dataclass(frozen=True)
class StudyCase:
    task_id: str
    solution_path: Path
    variants: list[dict[str, Any]]
    dataset_membership: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a static + dynamic complexity study for approved ARC Python solutions."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional subset of task IDs to study.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of worker subprocesses for dynamic metrics.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Per-solution timeout in seconds for dynamic metrics.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def package_root() -> Path:
    return Path(__file__).resolve().parent


def solutions_root() -> Path:
    return package_root() / "solutions"


def report_json_path() -> Path:
    return package_root() / "complexity_report.json"


def report_csv_path() -> Path:
    return package_root() / "complexity_report.csv"


def summary_json_path() -> Path:
    return package_root() / "complexity_summary.json"


def stable_task_hash(task: dict[str, Any]) -> str:
    encoded = json.dumps(task, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collect_task_variants(task_id: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for dataset_name, relative_dir in OFFICIAL_SOURCES.items():
        path = repo_root() / relative_dir / f"{task_id}.json"
        if not path.exists():
            continue
        task = json.loads(path.read_text(encoding="utf-8"))
        task.pop("name", None)
        task_hash = stable_task_hash(task)
        entry = grouped.setdefault(
            task_hash,
            {"datasets": [], "task": task},
        )
        entry["datasets"].append(dataset_name)
    return sorted(grouped.values(), key=lambda item: tuple(item["datasets"]))


def build_cases(selected_tasks: set[str] | None = None) -> list[StudyCase]:
    cases: list[StudyCase] = []
    for solution_path in sorted(solutions_root().glob("*.py")):
        task_id = solution_path.stem
        if selected_tasks is not None and task_id not in selected_tasks:
            continue
        variants = collect_task_variants(task_id)
        membership = sorted({dataset for variant in variants for dataset in variant["datasets"]})
        cases.append(
            StudyCase(
                task_id=task_id,
                solution_path=solution_path,
                variants=variants,
                dataset_membership=membership,
            )
        )
    return cases


class NestingVisitor(ast.NodeVisitor):
    FLOW_NODES = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.Match,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )

    def __init__(self) -> None:
        self.depth = 0
        self.max_depth = 0

    def generic_visit(self, node: ast.AST) -> None:
        is_flow = isinstance(node, self.FLOW_NODES)
        if is_flow:
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
        super().generic_visit(node)
        if is_flow:
            self.depth -= 1


def count_comment_lines(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            count += 1
    return count


def line_length_stats(text: str) -> tuple[int, float]:
    lengths = [len(line) for line in text.splitlines()]
    if not lengths:
        return 0, 0.0
    return max(lengths), round(statistics.mean(lengths), 3)


def compute_halstead(text: str) -> dict[str, float]:
    operator_tokens: list[str] = []
    operand_tokens: list[str] = []
    keyword_like = {
        "if",
        "elif",
        "else",
        "for",
        "while",
        "try",
        "except",
        "finally",
        "with",
        "return",
        "yield",
        "and",
        "or",
        "not",
        "in",
        "is",
        "lambda",
        "match",
        "case",
    }

    reader = io.StringIO(text).readline
    for token in tokenize.generate_tokens(reader):
        token_type = token.type
        token_string = token.string
        if token_type in {tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}:
            continue
        if token_type == tokenize.OP or token_string in keyword_like:
            operator_tokens.append(token_string)
        elif token_type in {tokenize.NAME, tokenize.NUMBER, tokenize.STRING}:
            operand_tokens.append(token_string)

    n1 = len(set(operator_tokens))
    n2 = len(set(operand_tokens))
    N1 = len(operator_tokens)
    N2 = len(operand_tokens)
    vocabulary = n1 + n2
    length = N1 + N2
    volume = length * math.log2(vocabulary) if vocabulary > 0 else 0.0
    difficulty = (n1 / 2.0) * (N2 / n2) if n2 > 0 else 0.0
    effort = volume * difficulty
    return {
        "halstead_distinct_operators": n1,
        "halstead_distinct_operands": n2,
        "halstead_total_operators": N1,
        "halstead_total_operands": N2,
        "halstead_vocabulary": vocabulary,
        "halstead_length": length,
        "halstead_volume": round(volume, 6),
        "halstead_difficulty": round(difficulty, 6),
        "halstead_effort": round(effort, 6),
    }


def compute_cyclomatic(tree: ast.AST) -> int:
    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.Match)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += max(0, len(node.values) - 1)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for generator in node.generators:
                complexity += 1
                complexity += len(generator.ifs)
    return complexity


def static_metrics(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text)

    total_lines = len(text.splitlines())
    nonblank_lines = sum(1 for line in text.splitlines() if line.strip())
    comment_lines = count_comment_lines(text)
    max_line_length, mean_line_length = line_length_stats(text)

    visitor = NestingVisitor()
    visitor.visit(tree)

    metrics = {
        "file_bytes": path.stat().st_size,
        "total_lines": total_lines,
        "nonblank_lines": nonblank_lines,
        "comment_lines": comment_lines,
        "token_count": sum(1 for _ in tokenize.generate_tokens(io.StringIO(text).readline)),
        "ast_node_count": sum(1 for _ in ast.walk(tree)),
        "function_count": sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)),
        "call_count_static": sum(isinstance(node, ast.Call) for node in ast.walk(tree)),
        "comprehension_count": sum(
            isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
            for node in ast.walk(tree)
        ),
        "branch_node_count": sum(
            isinstance(
                node,
                (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.IfExp, ast.Match, ast.ExceptHandler),
            )
            for node in ast.walk(tree)
        ),
        "cyclomatic_complexity": compute_cyclomatic(tree),
        "max_nesting_depth": visitor.max_depth,
        "max_line_length": max_line_length,
        "mean_line_length": mean_line_length,
        "gzip_bytes": len(gzip.compress(text.encode("utf-8"))),
    }
    metrics.update(compute_halstead(text))
    return metrics


def load_solution_module(solution_path: Path):
    module_name = f"complexity_solution_{solution_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, solution_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dynamic_metrics_worker() -> int:
    args = parse_args()
    payload = json.load(sys.stdin)
    output_file = args.output_file
    if output_file is None:
        raise SystemExit("--output-file is required in worker mode.")

    result: dict[str, Any]
    try:
        solution_path = Path(payload["solution_path"])
        target_filename = str(solution_path.resolve())
        module = load_solution_module(solution_path)
        solve = getattr(module, "solve", None)
        if not callable(solve):
            raise AttributeError("No callable solve(grid) function found.")

        opcode_counter = 0
        branch_opcode_counter = 0
        python_call_counter = 0
        branch_prefixes = ("JUMP", "POP_JUMP", "FOR_ITER", "END_FOR", "END_SEND")

        def tracer(frame, event, arg):
            nonlocal opcode_counter, branch_opcode_counter, python_call_counter
            if str(Path(frame.f_code.co_filename).resolve()) != target_filename:
                return None
            if event == "call":
                python_call_counter += 1
                frame.f_trace_lines = False
                frame.f_trace_opcodes = True
                return tracer
            if event == "opcode":
                opcode_counter += 1
                instruction = frame.f_code.co_code[frame.f_lasti]
                opname = dis.opname[instruction]
                if opname.startswith(branch_prefixes):
                    branch_opcode_counter += 1
                return tracer
            return tracer

        total_input_cells = 0
        total_output_cells = 0
        test_case_count = 0
        variant_count = len(payload["variants"])
        elapsed_ns_total = 0

        tracemalloc.start()
        old_tracer = sys.gettrace()
        sys.settrace(tracer)
        try:
            for variant in payload["variants"]:
                task = variant["task"]
                for pair in task["test"]:
                    test_case_count += 1
                    total_input_cells += sum(len(row) for row in pair["input"])
                    total_output_cells += sum(len(row) for row in pair["output"])
                    started = time.perf_counter_ns()
                    actual = solve(copy.deepcopy(pair["input"]))
                    elapsed_ns_total += time.perf_counter_ns() - started
                    if actual != pair["output"]:
                        raise AssertionError("Dynamic study expected approved solution to match official output.")
        finally:
            sys.settrace(old_tracer)
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        result = {
            "task_id": payload["task_id"],
            "status": "ok",
            "variant_count": variant_count,
            "test_case_count": test_case_count,
            "input_cells_total": total_input_cells,
            "output_cells_total": total_output_cells,
            "elapsed_ns_total": elapsed_ns_total,
            "elapsed_ms_total": round(elapsed_ns_total / 1_000_000, 6),
            "elapsed_ms_per_test": round((elapsed_ns_total / max(test_case_count, 1)) / 1_000_000, 6),
            "opcode_count_dynamic": opcode_counter,
            "branch_opcode_count_dynamic": branch_opcode_counter,
            "python_call_count_dynamic": python_call_counter,
            "peak_memory_bytes": peak_mem,
            "current_memory_bytes": current_mem,
        }
    except Exception as error:  # pragma: no cover - defensive runtime handling
        result = {
            "task_id": payload["task_id"],
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }

    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


def run_dynamic_case(case: StudyCase, timeout: float) -> dict[str, Any]:
    payload = {
        "task_id": case.task_id,
        "solution_path": str(case.solution_path),
        "variants": case.variants,
    }

    with tempfile.NamedTemporaryFile(
        prefix=f"complexity_{case.task_id}_",
        suffix=".json",
        delete=False,
    ) as handle:
        output_path = Path(handle.name)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--output-file",
        str(output_path),
    ]

    try:
        completed = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=repo_root(),
        )
        if output_path.exists() and output_path.stat().st_size > 0:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            result = {
                "task_id": case.task_id,
                "status": "error",
                "error_type": "WorkerOutputError",
                "error": "Worker did not produce a result file.",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        if completed.returncode != 0 and result.get("status") == "ok":
            result = {
                "task_id": case.task_id,
                "status": "error",
                "error_type": "WorkerExitError",
                "error": f"Worker exited with code {completed.returncode}",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        return result
    except subprocess.TimeoutExpired as error:
        return {
            "task_id": case.task_id,
            "status": "timeout",
            "error_type": "TimeoutExpired",
            "error": f"Dynamic metrics exceeded {timeout:.1f}s",
            "stdout": error.stdout,
            "stderr": error.stderr,
        }
    finally:
        output_path.unlink(missing_ok=True)


def write_csv(rows: list[dict[str, Any]]) -> None:
    if not rows:
        report_csv_path().write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with report_csv_path().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_n(rows: list[dict[str, Any]], key: str, n: int = 10) -> list[dict[str, Any]]:
    sortable = [row for row in rows if isinstance(row.get(key), (int, float))]
    return [
        {
            "task_id": row["task_id"],
            key: row[key],
        }
        for row in sorted(sortable, key=lambda row: row[key], reverse=True)[:n]
    ]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "nonblank_lines",
        "cyclomatic_complexity",
        "halstead_volume",
        "max_nesting_depth",
        "gzip_bytes",
        "elapsed_ms_total",
        "elapsed_ms_per_test",
        "opcode_count_dynamic",
        "branch_opcode_count_dynamic",
        "python_call_count_dynamic",
        "peak_memory_bytes",
    ]

    metrics_summary: dict[str, dict[str, float]] = {}
    for key in numeric_keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        metrics_summary[key] = {
            "min": round(min(values), 6),
            "p25": round(percentile(values, 0.25), 6),
            "median": round(percentile(values, 0.5), 6),
            "mean": round(statistics.mean(values), 6),
            "p75": round(percentile(values, 0.75), 6),
            "max": round(max(values), 6),
        }

    dataset_counter = Counter()
    for row in rows:
        membership = row["dataset_membership"]
        if isinstance(membership, str):
            datasets = [item for item in membership.split("|") if item]
        else:
            datasets = list(membership)
        for dataset in datasets:
            dataset_counter[dataset] += 1

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "file_count": len(rows),
        "dataset_membership_counts": dict(sorted(dataset_counter.items())),
        "metrics_summary": metrics_summary,
        "top_nonblank_lines": top_n(rows, "nonblank_lines"),
        "top_cyclomatic_complexity": top_n(rows, "cyclomatic_complexity"),
        "top_halstead_volume": top_n(rows, "halstead_volume"),
        "top_elapsed_ms_total": top_n(rows, "elapsed_ms_total"),
        "top_opcode_count_dynamic": top_n(rows, "opcode_count_dynamic"),
        "top_peak_memory_bytes": top_n(rows, "peak_memory_bytes"),
    }


def main() -> int:
    args = parse_args()
    if args.worker:
        return dynamic_metrics_worker()

    selected_tasks = set(args.tasks) if args.tasks else None
    cases = build_cases(selected_tasks)
    print(f"Studying {len(cases)} approved solution files.")

    static_rows: dict[str, dict[str, Any]] = {}
    for case in cases:
        row = {
            "task_id": case.task_id,
            "solution_file": str(case.solution_path),
            "dataset_membership": case.dataset_membership,
            "variant_count": len(case.variants),
            "test_case_count_total": sum(len(variant["task"]["test"]) for variant in case.variants),
        }
        row.update(static_metrics(case.solution_path))
        static_rows[case.task_id] = row

    results: dict[str, dict[str, Any]] = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_to_case = {
            executor.submit(run_dynamic_case, case, args.timeout): case
            for case in cases
        }
        for future in as_completed(future_to_case):
            case = future_to_case[future]
            dynamic = future.result()
            results[case.task_id] = dynamic
            print(f"{case.task_id}: {dynamic['status']}")

    merged_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        row = dict(static_rows[case.task_id])
        dynamic = results[case.task_id]
        if dynamic["status"] != "ok":
            failures.append(dynamic)
            row["dynamic_status"] = dynamic["status"]
        else:
            row["dynamic_status"] = "ok"
            row.update({key: value for key, value in dynamic.items() if key not in {"task_id", "status"}})
        row["dataset_membership"] = "|".join(row["dataset_membership"])
        merged_rows.append(row)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": merged_rows,
        "failures": failures,
    }
    report_json_path().write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(merged_rows)
    summary = summarize([row for row in merged_rows if row["dynamic_status"] == "ok"])
    summary["failure_count"] = len(failures)
    summary_json_path().write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print(f"CSV report: {report_csv_path()}")
    print(f"JSON report: {report_json_path()}")
    print(f"Summary: {summary_json_path()}")
    print(f"Failures: {len(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
