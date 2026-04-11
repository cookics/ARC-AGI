from __future__ import annotations

import ast
import argparse
import copy
import dis
import gzip
import io
import json
import math
import re
import statistics
import sys
import tempfile
import time
import tokenize
import tracemalloc
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent

BUNDLE_PATH = ROOT_DIR / "dataARC1TrainingPython.py"
ARC1_TRAIN_DIR = ROOT_DIR / "data-llm" / "ARC-AGI" / "data" / "training"
HRC_SUMMARY_PATH = ROOT_DIR / "HR data" / "data" / "summary_data.csv"
LEWIS_DIR = ROOT_DIR / "arc1lewish submissions"

VALIDATION_PATH = ANALYSIS_DIR / "arc1_dsl_validation_report.json"
COMPLEXITY_PATH = ANALYSIS_DIR / "arc1_dsl_complexity_metrics.csv"
CORRELATION_PATH = ANALYSIS_DIR / "arc1_dsl_complexity_correlations.csv"
DIFFERENCE_PATH = ANALYSIS_DIR / "arc1_dsl_correlation_differences.csv"
BEST_METRICS_PATH = ANALYSIS_DIR / "arc1_dsl_best_metrics_by_target.csv"
MODEL_RESULTS_PATH = ANALYSIS_DIR / "arc1_dsl_complexity_model_results.csv"
MERGED_PATH = ANALYSIS_DIR / "arc1_dsl_complexity_task_join.csv"
SUMMARY_PATH = ANALYSIS_DIR / "arc1_dsl_human_llm_summary.json"
REPORT_PATH = ANALYSIS_DIR / "arc1_dsl_human_llm_report.md"

BEGIN_RE = re.compile(r"^# --- BEGIN (?P<name>.+) ---$")
END_RE = re.compile(r"^# --- END (?P<name>.+) ---$")
SOLVE_NAME_RE = re.compile(r"^solve_([0-9a-f]{8})$")
TEMP_NAME_RE = re.compile(r"^x\d+$")

HIGHER_ORDER_NAMES = {
    "compose",
    "chain",
    "fork",
    "lbind",
    "rbind",
    "power",
    "matcher",
    "branch",
}
OBJECT_NAMES = {
    "objects",
    "partition",
    "fgpartition",
    "colorfilter",
    "sizefilter",
    "merge",
    "normalize",
    "subgrid",
    "occurrences",
    "palette",
    "asobject",
    "toindices",
    "delta",
    "backdrop",
    "inbox",
    "outbox",
}
SELECTION_NAMES = {
    "argmin",
    "argmax",
    "extract",
    "order",
    "first",
    "last",
    "mostcolor",
    "leastcolor",
    "center",
    "ulcorner",
    "urcorner",
    "llcorner",
    "lrcorner",
    "position",
}
GEOMETRY_NAMES = {
    "vmirror",
    "hmirror",
    "cmirror",
    "dmirror",
    "rot90",
    "rot180",
    "rot270",
    "crop",
    "upscale",
    "downscale",
    "shift",
    "canvas",
    "hconcat",
    "vconcat",
    "connect",
    "gravitate",
}
SET_NAMES = {
    "combine",
    "intersection",
    "difference",
    "remove",
    "insert",
    "product",
    "apply",
    "mapply",
    "sfilter",
    "mfilter",
    "rapply",
    "repeat",
}
DECISION_NAMES = {
    "branch",
    "both",
    "either",
    "equality",
    "greater",
    "less",
    "contained",
    "positive",
}
PRIMARY_COMPLEXITY_COLUMNS = [
    "named_call_count",
    "closure_call_count",
    "distinct_primitive_count",
    "assignment_count",
    "temp_var_count",
    "max_dependency_depth",
    "higher_order_count",
    "object_op_count",
    "selection_op_count",
    "geometry_op_count",
    "set_op_count",
    "decision_op_count",
    "runtime_log_ms",
    "memory_log_bytes",
]
HEADLINE_TARGETS = {
    "human_difficulty_complete": "Human latent difficulty (complete participants)",
    "gpt_failure": "GPT-4o failure rate",
    "claude_failure": "Claude 3.5 Sonnet failure rate",
    "llm_pooled_difficulty": "Pooled GPT+Claude smoothed difficulty",
    "icecuber_failure": "IceCuber failure rate",
}
NON_COMPLEXITY_COLUMNS = {
    "task_id",
    "status",
    "mismatch_split",
    "mismatch_index",
    "error_type",
    "error_message",
    "direct_primitive_names",
    "human_difficulty_complete",
    "human_difficulty_complete_solve_rate",
    "human_difficulty_complete_n_people",
    "human_difficulty_all",
    "gpt_solved",
    "gpt_solved_pair_mean",
    "claude_solved",
    "claude_solved_pair_mean",
    "icecuber_solved",
    "icecuber_solved_pair_mean",
    "gpt_failure",
    "claude_failure",
    "icecuber_failure",
    "llm_success_count",
    "llm_pooled_difficulty",
}


@dataclass(frozen=True)
class BundleModules:
    arc_types: types.ModuleType
    dsl: types.ModuleType
    primitives: types.ModuleType
    solvers: types.ModuleType
    sections: dict[str, str]


def canonical_grid(grid: Any) -> tuple[tuple[int, ...], ...]:
    if isinstance(grid, tuple):
        return tuple(tuple(int(cell) for cell in row) for row in grid)
    if isinstance(grid, list):
        return tuple(tuple(int(cell) for cell in row) for row in grid)
    raise TypeError(f"Expected list/tuple grid, got {type(grid).__name__}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARC-1 DSL complexity vs human/LLM analysis.")
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Skip validation and data assembly, and resume only the cross-validated model fitting from the saved task-join CSV.",
    )
    return parser.parse_args()


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
        if line.strip().startswith("#"):
            count += 1
    return count


def line_length_stats(text: str) -> tuple[int, float]:
    lengths = [len(line) for line in text.splitlines()]
    if not lengths:
        return 0, 0.0
    return max(lengths), round(statistics.mean(lengths), 6)


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
        if token_type in {
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
        }:
            continue
        if token_type == tokenize.OP or token_string in keyword_like:
            operator_tokens.append(token_string)
        elif token_type in {tokenize.NAME, tokenize.NUMBER, tokenize.STRING}:
            operand_tokens.append(token_string)

    n1 = len(set(operator_tokens))
    n2 = len(set(operand_tokens))
    n_1 = len(operator_tokens)
    n_2 = len(operand_tokens)
    vocabulary = n1 + n2
    length = n_1 + n_2
    volume = length * math.log2(vocabulary) if vocabulary > 0 else 0.0
    difficulty = (n1 / 2.0) * (n_2 / n2) if n2 > 0 else 0.0
    effort = volume * difficulty
    return {
        "halstead_distinct_operators": float(n1),
        "halstead_distinct_operands": float(n2),
        "halstead_total_operators": float(n_1),
        "halstead_total_operands": float(n_2),
        "halstead_vocabulary": float(vocabulary),
        "halstead_length": float(length),
        "halstead_volume": float(volume),
        "halstead_difficulty": float(difficulty),
        "halstead_effort": float(effort),
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


def static_python_metrics(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    total_lines = len(source.splitlines())
    nonblank_lines = sum(1 for line in source.splitlines() if line.strip())
    comment_lines = count_comment_lines(source)
    max_line_length, mean_line_length = line_length_stats(source)

    nesting = NestingVisitor()
    nesting.visit(tree)

    metrics = {
        "source_bytes": len(source.encode("utf-8")),
        "total_lines": total_lines,
        "nonblank_lines": nonblank_lines,
        "comment_lines": comment_lines,
        "token_count": sum(1 for _ in tokenize.generate_tokens(io.StringIO(source).readline)),
        "ast_node_count": sum(1 for _ in ast.walk(tree)),
        "function_count": sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)
        ),
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
        "return_count": sum(isinstance(node, ast.Return) for node in ast.walk(tree)),
        "name_load_count": sum(isinstance(node, ast.Name) for node in ast.walk(tree)),
        "constant_count": sum(isinstance(node, ast.Constant) for node in ast.walk(tree)),
        "assignment_node_count": sum(isinstance(node, ast.Assign) for node in ast.walk(tree)),
        "cyclomatic_complexity": compute_cyclomatic(tree),
        "max_nesting_depth": nesting.max_depth,
        "max_line_length": max_line_length,
        "mean_line_length": mean_line_length,
        "gzip_bytes": len(gzip.compress(source.encode("utf-8"))),
    }
    metrics.update(compute_halstead(source))
    return metrics


def split_bundle_sections(bundle_text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_name: str | None = None
    buffer: list[str] = []

    for line in bundle_text.splitlines():
        begin_match = BEGIN_RE.match(line)
        end_match = END_RE.match(line)
        if begin_match:
            current_name = begin_match.group("name")
            buffer = []
            continue
        if end_match:
            if current_name is None:
                raise ValueError("Encountered END marker without active section.")
            sections[current_name] = list(buffer)
            current_name = None
            buffer = []
            continue
        if current_name is not None:
            buffer.append(line)

    return {name: "\n".join(lines).strip() + "\n" for name, lines in sections.items()}


def exec_module(name: str, code: str, package: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__package__ = package
    module.__file__ = f"{BUNDLE_PATH.name}:{name}"
    sys.modules[name] = module
    exec(compile(code, module.__file__, "exec"), module.__dict__)
    return module


def load_bundle_modules() -> BundleModules:
    sections = split_bundle_sections(BUNDLE_PATH.read_text(encoding="utf-8"))
    runtime_dir = Path(tempfile.mkdtemp(prefix="arc_dsl_runtime_"))
    runtime_dsl_dir = runtime_dir / "codeit" / "dsl"
    runtime_dsl_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("arc_types.py", "dsl.py", "primitives.py", "solvers.py"):
        (runtime_dsl_dir / filename).write_text(sections[filename], encoding="utf-8")

    codeit_pkg = types.ModuleType("codeit")
    codeit_pkg.__path__ = []  # type: ignore[attr-defined]
    codeit_pkg.PROJECT_FOLDER_PATH = str(runtime_dir)  # type: ignore[attr-defined]
    sys.modules["codeit"] = codeit_pkg

    dsl_pkg = types.ModuleType("codeit.dsl")
    dsl_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["codeit.dsl"] = dsl_pkg
    codeit_pkg.dsl = dsl_pkg  # type: ignore[attr-defined]

    arc_types = exec_module("codeit.dsl.arc_types", sections["arc_types.py"], "codeit.dsl")
    dsl_pkg.arc_types = arc_types  # type: ignore[attr-defined]

    dsl = exec_module("codeit.dsl.dsl", sections["dsl.py"], "codeit.dsl")
    dsl_pkg.dsl = dsl  # type: ignore[attr-defined]

    primitives = exec_module("codeit.dsl.primitives", sections["primitives.py"], "codeit.dsl")
    dsl_pkg.primitives = primitives  # type: ignore[attr-defined]

    solvers = exec_module("codeit.dsl.solvers", sections["solvers.py"], "codeit.dsl")
    dsl_pkg.solvers = solvers  # type: ignore[attr-defined]

    return BundleModules(
        arc_types=arc_types,
        dsl=dsl,
        primitives=primitives,
        solvers=solvers,
        sections=sections,
    )


def load_arc1_training_tasks() -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for path in sorted(ARC1_TRAIN_DIR.glob("*.json")):
        tasks[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return tasks


def validate_dsl_solvers(bundle: BundleModules, tasks: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    solver_filename = str(bundle.solvers.__file__)
    dsl_filename = str(bundle.dsl.__file__)
    primitives_filename = str(bundle.primitives.__file__)
    branch_prefixes = ("JUMP", "POP_JUMP", "FOR_ITER", "END_FOR", "END_SEND")
    task_items = list(tasks.items())
    total_tasks = len(task_items)

    for task_position, (task_id, task) in enumerate(task_items, start=1):
        solve = getattr(bundle.solvers, f"solve_{task_id}", None)
        if not callable(solve):
            rows.append(
                {
                    "task_id": task_id,
                    "status": "missing_solver",
                    "train_pairs": len(task.get("train", [])),
                    "test_pairs": len(task.get("test", [])),
                }
            )
            show_progress(task_position, total_tasks, f"Validating ARC-1 DSL solvers ({task_id})")
            continue

        elapsed_ns_total = 0
        peak_memory_bytes = 0
        current_memory_bytes = 0
        status = "passed"
        mismatch_split = None
        mismatch_index = None
        error_type = None
        error_message = None
        input_cells_total = 0
        output_cells_total = 0

        module_opcode_counts = {"solver": 0, "dsl": 0, "primitives": 0}
        module_branch_opcode_counts = {"solver": 0, "dsl": 0, "primitives": 0}
        module_call_counts = {"solver": 0, "dsl": 0, "primitives": 0}
        module_function_names = {
            "solver": set(),
            "dsl": set(),
            "primitives": set(),
        }

        def classify_frame(frame) -> str | None:
            filename = str(frame.f_code.co_filename)
            if filename == solver_filename:
                return "solver"
            if filename == dsl_filename:
                return "dsl"
            if filename == primitives_filename:
                return "primitives"
            return None

        def tracer(frame, event, arg):
            module_name = classify_frame(frame)
            if module_name is None:
                return None
            if event == "call":
                module_call_counts[module_name] += 1
                module_function_names[module_name].add(frame.f_code.co_name)
                frame.f_trace_lines = False
                frame.f_trace_opcodes = module_name == "solver"
                return tracer
            if event == "opcode":
                if module_name != "solver":
                    return tracer
                module_opcode_counts[module_name] += 1
                instruction = frame.f_code.co_code[frame.f_lasti]
                opname = dis.opname[instruction]
                if opname.startswith(branch_prefixes):
                    module_branch_opcode_counts[module_name] += 1
                return tracer
            return tracer

        examples = [("train", idx, pair) for idx, pair in enumerate(task.get("train", []))]
        examples += [("test", idx, pair) for idx, pair in enumerate(task.get("test", []))]

        tracemalloc.start()
        old_tracer = sys.gettrace()
        sys.settrace(tracer)
        for split_name, pair_index, pair in examples:
            try:
                input_grid = canonical_grid(pair["input"])
                expected = canonical_grid(pair["output"])
                input_cells_total += sum(len(row) for row in input_grid)
                output_cells_total += sum(len(row) for row in expected)
                start_ns = time.perf_counter_ns()
                actual = solve(copy.deepcopy(input_grid))
                elapsed_ns = time.perf_counter_ns() - start_ns
                elapsed_ns_total += elapsed_ns
                if canonical_grid(actual) != expected:
                    status = "wrong_answer"
                    mismatch_split = split_name
                    mismatch_index = pair_index
                    break
            except Exception as exc:  # pragma: no cover - defensive runtime capture
                status = "error"
                mismatch_split = split_name
                mismatch_index = pair_index
                error_type = type(exc).__name__
                error_message = str(exc)
                break
        sys.settrace(old_tracer)
        current_memory_bytes, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rows.append(
            {
                "task_id": task_id,
                "status": status,
                "train_pairs": len(task.get("train", [])),
                "test_pairs": len(task.get("test", [])),
                "example_count": len(examples),
                "elapsed_ms_total": round(elapsed_ns_total / 1_000_000.0, 6),
                "elapsed_ms_per_example": round(
                    elapsed_ns_total / max(len(examples), 1) / 1_000_000.0,
                    6,
                ),
                "peak_memory_bytes": int(peak_memory_bytes),
                "current_memory_bytes": int(current_memory_bytes),
                "input_cells_total": int(input_cells_total),
                "output_cells_total": int(output_cells_total),
                "solver_opcode_count_dynamic": int(module_opcode_counts["solver"]),
                "dsl_opcode_count_dynamic": int(module_opcode_counts["dsl"]),
                "primitives_opcode_count_dynamic": int(module_opcode_counts["primitives"]),
                "bundle_opcode_count_dynamic": int(sum(module_opcode_counts.values())),
                "solver_branch_opcode_count_dynamic": int(module_branch_opcode_counts["solver"]),
                "dsl_branch_opcode_count_dynamic": int(module_branch_opcode_counts["dsl"]),
                "primitives_branch_opcode_count_dynamic": int(module_branch_opcode_counts["primitives"]),
                "bundle_branch_opcode_count_dynamic": int(sum(module_branch_opcode_counts.values())),
                "solver_python_call_count_dynamic": int(module_call_counts["solver"]),
                "dsl_python_call_count_dynamic": int(module_call_counts["dsl"]),
                "primitives_python_call_count_dynamic": int(module_call_counts["primitives"]),
                "bundle_python_call_count_dynamic": int(sum(module_call_counts.values())),
                "solver_distinct_function_count_dynamic": int(len(module_function_names["solver"])),
                "dsl_distinct_function_count_dynamic": int(len(module_function_names["dsl"])),
                "primitives_distinct_function_count_dynamic": int(len(module_function_names["primitives"])),
                "bundle_distinct_function_count_dynamic": int(
                    len(module_function_names["solver"] | module_function_names["dsl"] | module_function_names["primitives"])
                ),
                "mismatch_split": mismatch_split,
                "mismatch_index": mismatch_index,
                "error_type": error_type,
                "error_message": error_message,
            }
        )
        show_progress(task_position, total_tasks, f"Validating ARC-1 DSL solvers ({task_id})")

    df = pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)
    primitive_columns = [column for column in df.columns if column.startswith("prim_")]
    if primitive_columns:
        df[primitive_columns] = df[primitive_columns].fillna(0).astype(int)
    return df


def call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def assignment_target_name(node: ast.Assign) -> str | None:
    if len(node.targets) != 1:
        return None
    target = node.targets[0]
    if isinstance(target, ast.Name):
        return target.id
    return None


def referenced_temp_names(node: ast.AST) -> set[str]:
    refs: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and TEMP_NAME_RE.match(child.id):
            refs.add(child.id)
    return refs


def solver_source_segment(source: str, node: ast.FunctionDef) -> str:
    lines = source.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def build_dsl_complexity_table(source: str) -> pd.DataFrame:
    module_tree = ast.parse(source)
    rows: list[dict[str, Any]] = []

    for node in module_tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        solve_match = SOLVE_NAME_RE.match(node.name)
        if not solve_match:
            continue
        task_id = solve_match.group(1)
        solver_source = solver_source_segment(source, node)
        call_nodes = [child for child in ast.walk(node) if isinstance(child, ast.Call)]
        call_names = [name for name in (call_name(call) for call in call_nodes) if name is not None]
        named_calls = [name for name in call_names if not TEMP_NAME_RE.match(name)]
        closure_calls = [name for name in call_names if TEMP_NAME_RE.match(name)]
        distinct_named = sorted(set(named_calls))
        primitive_counts = {f"prim_{name}_count": named_calls.count(name) for name in distinct_named}

        assignment_nodes = [stmt for stmt in node.body if isinstance(stmt, ast.Assign)]
        temp_targets = [
            target_name
            for assign in assignment_nodes
            if (target_name := assignment_target_name(assign)) is not None and TEMP_NAME_RE.match(target_name)
        ]
        depth: dict[str, int] = {"I": 0}
        max_fan_in = 0
        for assign in assignment_nodes:
            target_name = assignment_target_name(assign)
            if target_name is None:
                continue
            refs = referenced_temp_names(assign.value)
            max_fan_in = max(max_fan_in, len(refs))
            depth[target_name] = 1 + max((depth.get(ref, 0) for ref in refs), default=0)

        o_depth = depth.get("O", max(depth.values(), default=0))
        line_count = solver_source.count("\n") + 1

        rows.append(
            {
                "task_id": task_id,
                **static_python_metrics(solver_source),
                "source_line_count": line_count,
                "assignment_count": len(assignment_nodes),
                "temp_var_count": len(temp_targets),
                "ast_call_count": len(call_nodes),
                "named_call_count": len(named_calls),
                "closure_call_count": len(closure_calls),
                "distinct_primitive_count": len(distinct_named),
                "max_dependency_depth": int(o_depth),
                "max_fan_in": int(max_fan_in),
                "higher_order_count": sum(name in HIGHER_ORDER_NAMES for name in named_calls),
                "object_op_count": sum(name in OBJECT_NAMES for name in named_calls),
                "selection_op_count": sum(name in SELECTION_NAMES for name in named_calls),
                "geometry_op_count": sum(name in GEOMETRY_NAMES for name in named_calls),
                "set_op_count": sum(name in SET_NAMES for name in named_calls),
                "decision_op_count": sum(name in DECISION_NAMES for name in named_calls),
                "direct_primitive_names": "|".join(distinct_named),
                **primitive_counts,
            }
        )

    return pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)


def complexity_metric_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        if column in NON_COMPLEXITY_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            columns.append(column)
    return columns


def transformed_complexity_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work = df[columns].copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for column in work.columns:
        series = work[column]
        if (series >= 0).all() and (series.max() > 20 or "bytes" in column or "count" in column or "ms" in column):
            work[column] = np.log1p(series)
    return work


def add_complexity_component(df: pd.DataFrame) -> pd.Series:
    metric_columns = complexity_metric_columns(df)
    metric_frame = transformed_complexity_frame(df, metric_columns)
    metric_frame = metric_frame.loc[:, metric_frame.nunique(dropna=False) > 1]
    z = (metric_frame - metric_frame.mean()) / metric_frame.std(ddof=0).replace(0, 1)
    _, _, vh = np.linalg.svd(z.to_numpy(), full_matrices=False)
    pc1 = pd.Series(z.to_numpy() @ vh[0], index=df.index, dtype=float)
    orient_series = pd.to_numeric(df.get("ast_node_count", df.get("named_call_count")), errors="coerce")
    if pc1.corr(orient_series) < 0:
        pc1 *= -1.0
        vh[0] *= -1.0
    df["dsl_complexity_pc1"] = pc1
    return pd.Series(vh[0], index=metric_frame.columns, name="pc1_loading").sort_values(ascending=False)


def load_hrc_responses(task_ids: set[str], complete_only: bool) -> pd.DataFrame:
    usecols = ["task_type", "task_name", "hashed_id", "attempt_number", "solved", "complete"]
    df = pd.read_csv(HRC_SUMMARY_PATH, usecols=usecols)
    df = df[df["task_type"].eq("training")].copy()
    df["task_id"] = df["task_name"].astype(str).str.replace(".json", "", regex=False)
    df = df[df["task_id"].isin(task_ids)].copy()
    if complete_only:
        df = df[df["complete"].fillna(False)].copy()
    collapsed = (
        df.groupby(["hashed_id", "task_id"], as_index=False)
        .agg(
            solved=("solved", "max"),
            max_attempt_number=("attempt_number", "max"),
            complete=("complete", "max"),
        )
        .sort_values(["hashed_id", "task_id"])
        .reset_index(drop=True)
    )
    collapsed["solved"] = collapsed["solved"].astype(int)
    return collapsed


def fit_item_difficulty(responses: pd.DataFrame, item_prefix: str) -> pd.DataFrame:
    encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    design = encoder.fit_transform(responses[["hashed_id", "task_id"]])
    y = responses["solved"].to_numpy(dtype=int)
    model = LogisticRegression(
        C=2.0,
        solver="saga",
        max_iter=10000,
        fit_intercept=True,
        random_state=0,
    )
    model.fit(design, y)

    feature_names = pd.Index(encoder.get_feature_names_out(["hashed_id", "task_id"]))
    coefficients = pd.Series(model.coef_[0], index=feature_names)
    item_ease = coefficients[feature_names.str.startswith("task_id_")]
    item_ease.index = item_ease.index.str.replace("task_id_", "", regex=False)
    item_difficulty = -(item_ease - item_ease.mean())

    solve_rate = responses.groupby("task_id")["solved"].mean()
    n_people = responses.groupby("task_id")["solved"].size()

    return pd.DataFrame(
        {
            "task_id": item_difficulty.index,
            item_prefix: item_difficulty.values,
            f"{item_prefix}_solve_rate": solve_rate.reindex(item_difficulty.index).values,
            f"{item_prefix}_n_people": n_people.reindex(item_difficulty.index).values,
        }
    )


def build_human_difficulty_table(task_ids: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    complete_responses = load_hrc_responses(task_ids, complete_only=True)
    all_responses = load_hrc_responses(task_ids, complete_only=False)

    complete_df = fit_item_difficulty(complete_responses, "human_difficulty_complete")
    all_df = fit_item_difficulty(all_responses, "human_difficulty_all")
    out = complete_df.merge(all_df[["task_id", "human_difficulty_all"]], on="task_id", how="left")

    diagnostics = {
        "complete_rows": int(len(complete_responses)),
        "all_rows": int(len(all_responses)),
        "complete_participants": int(complete_responses["hashed_id"].nunique()),
        "all_participants": int(all_responses["hashed_id"].nunique()),
        "human_complete_vs_all_corr": float(
            out[["human_difficulty_complete", "human_difficulty_all"]].corr().iloc[0, 1]
        ),
    }
    return out, diagnostics


def load_submission(name: str) -> dict[str, Any]:
    return json.loads((LEWIS_DIR / name).read_text(encoding="utf-8"))


def pair_best_exact(prediction: dict[str, Any], expected: tuple[tuple[int, ...], ...]) -> bool:
    for key in ("attempt_1", "attempt_2"):
        candidate = prediction.get(key)
        if candidate is None:
            continue
        try:
            if canonical_grid(candidate) == expected:
                return True
        except TypeError:
            continue
    return False


def score_submission(tasks: dict[str, dict[str, Any]], predictions: dict[str, Any], column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_id, task in tasks.items():
        expected_pairs = task.get("test", [])
        predicted_pairs = predictions.get(task_id, [])
        exact_flags: list[int] = []
        if isinstance(predicted_pairs, list) and len(predicted_pairs) == len(expected_pairs):
            for pair_prediction, pair in zip(predicted_pairs, expected_pairs):
                expected = canonical_grid(pair["output"])
                exact_flags.append(int(pair_best_exact(pair_prediction, expected)))
        else:
            exact_flags = [0 for _ in expected_pairs]

        solved = int(all(flag == 1 for flag in exact_flags)) if exact_flags else 0
        rows.append(
            {
                "task_id": task_id,
                column: solved,
                f"{column}_pair_mean": float(np.mean(exact_flags)) if exact_flags else 0.0,
            }
        )
    return pd.DataFrame(rows)


def smoothed_difficulty(successes: pd.Series, total_models: int) -> pd.Series:
    ease = (successes.astype(float) + 0.5) / (float(total_models) + 1.0)
    return -np.log(ease / (1.0 - ease))


def build_llm_task_table(tasks: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    gpt = score_submission(tasks, load_submission("gpt-4o_training.json"), "gpt_solved")
    claude = score_submission(tasks, load_submission("claude-35-sonnet_training.json"), "claude_solved")
    icecuber = score_submission(tasks, load_submission("icecuber-2020_training.json"), "icecuber_solved")

    llm = gpt.merge(claude, on="task_id", how="inner").merge(icecuber, on="task_id", how="inner")
    llm["gpt_failure"] = 1 - llm["gpt_solved"]
    llm["claude_failure"] = 1 - llm["claude_solved"]
    llm["icecuber_failure"] = 1 - llm["icecuber_solved"]
    llm["llm_success_count"] = llm["gpt_solved"] + llm["claude_solved"]
    llm["llm_pooled_difficulty"] = smoothed_difficulty(llm["llm_success_count"], total_models=2)

    diagnostics = {
        "gpt_solve_rate": float(llm["gpt_solved"].mean()),
        "claude_solve_rate": float(llm["claude_solved"].mean()),
        "icecuber_solve_rate": float(llm["icecuber_solved"].mean()),
        "gpt_claude_binary_corr": float(llm[["gpt_solved", "claude_solved"]].corr().iloc[0, 1]),
        "gpt_claude_same_outcome_count": int((llm["gpt_solved"] == llm["claude_solved"]).sum()),
        "gpt_claude_task_count": int(len(llm)),
        "llm_success_count_distribution": llm["llm_success_count"].value_counts().sort_index().to_dict(),
        "llm_pooled_difficulty_unique_levels": int(llm["llm_pooled_difficulty"].nunique()),
    }
    return llm, diagnostics


def safe_corr(series_a: pd.Series, series_b: pd.Series, method: str) -> float:
    pair = pd.concat([series_a, series_b], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def build_correlation_table(df: pd.DataFrame, targets: dict[str, str]) -> pd.DataFrame:
    complexity_columns = ["dsl_complexity_pc1"] + [
        column for column in complexity_metric_columns(df) if column != "dsl_complexity_pc1"
    ]
    rows: list[dict[str, Any]] = []
    for complexity_col in complexity_columns:
        for target_col, target_label in targets.items():
            subset = df[[complexity_col, target_col]].dropna()
            nonzero_count = None
            if len(subset):
                values = pd.to_numeric(subset[complexity_col], errors="coerce").fillna(0)
                nonzero_count = int((values != 0).sum())
            rows.append(
                {
                    "complexity_metric": complexity_col,
                    "target_metric": target_col,
                    "target_label": target_label,
                    "n": int(len(subset)),
                    "nonzero_count": nonzero_count,
                    "pearson_r": safe_corr(df[complexity_col], df[target_col], "pearson"),
                    "spearman_rho": safe_corr(df[complexity_col], df[target_col], "spearman"),
                }
            )
    return pd.DataFrame(rows)


def williams_test(r_xy: float, r_xz: float, r_yz: float, n: int) -> tuple[float, float]:
    if n <= 3:
        return float("nan"), float("nan")
    k = 1 - r_xy**2 - r_xz**2 - r_yz**2 + 2 * r_xy * r_xz * r_yz
    if k <= 0:
        return float("nan"), float("nan")
    numerator = (r_xy - r_xz) * math.sqrt((n - 1) * (1 + r_yz))
    denominator = math.sqrt(
        (2 * (n - 1) / (n - 3)) * k + (((r_xy + r_xz) ** 2) / 4.0) * ((1 - r_yz) ** 3)
    )
    if denominator == 0:
        return float("nan"), float("nan")
    t_value = numerator / denominator
    p_value = 2 * stats.t.sf(abs(t_value), df=n - 3)
    return float(t_value), float(p_value)


def bootstrap_corr_difference(
    x: pd.Series,
    y_a: pd.Series,
    y_b: pd.Series,
    method: str = "spearman",
    iterations: int = 4000,
    seed: int = 0,
) -> tuple[float, float, float]:
    pair = pd.concat([x, y_a, y_b], axis=1).dropna()
    values = pair.to_numpy(dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    diffs = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.integers(0, n, n)
        sample = values[idx]
        sample_df = pd.DataFrame(sample, columns=["x", "a", "b"])
        r_a = sample_df["x"].corr(sample_df["a"], method=method)
        r_b = sample_df["x"].corr(sample_df["b"], method=method)
        diffs[i] = r_a - r_b
    lower, upper = np.quantile(diffs, [0.025, 0.975])
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(lower), float(upper), float(p_value)


def build_difference_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_col, target_label in HEADLINE_TARGETS.items():
        if target_col == "human_difficulty_complete":
            continue
        subset = df[["dsl_complexity_pc1", "human_difficulty_complete", target_col]].dropna()
        pearson_human = safe_corr(subset["dsl_complexity_pc1"], subset["human_difficulty_complete"], "pearson")
        pearson_other = safe_corr(subset["dsl_complexity_pc1"], subset[target_col], "pearson")
        spearman_human = safe_corr(subset["dsl_complexity_pc1"], subset["human_difficulty_complete"], "spearman")
        spearman_other = safe_corr(subset["dsl_complexity_pc1"], subset[target_col], "spearman")
        yz_corr = safe_corr(subset["human_difficulty_complete"], subset[target_col], "pearson")
        t_value, p_value = williams_test(pearson_human, pearson_other, yz_corr, len(subset))
        boot_low, boot_high, boot_p = bootstrap_corr_difference(
            subset["dsl_complexity_pc1"],
            subset["human_difficulty_complete"],
            subset[target_col],
            method="spearman",
        )
        rows.append(
            {
                "comparison": f"human_vs_{target_col}",
                "target_label": target_label,
                "n": int(len(subset)),
                "pearson_human": pearson_human,
                "pearson_other": pearson_other,
                "pearson_difference": pearson_human - pearson_other,
                "spearman_human": spearman_human,
                "spearman_other": spearman_other,
                "spearman_difference": spearman_human - spearman_other,
                "williams_t": t_value,
                "williams_p": p_value,
                "bootstrap_diff_ci_low": boot_low,
                "bootstrap_diff_ci_high": boot_high,
                "bootstrap_p": boot_p,
            }
        )
    return pd.DataFrame(rows)


def best_metrics_by_target(correlations: pd.DataFrame) -> pd.DataFrame:
    work = correlations.copy()
    work["abs_pearson_r"] = work["pearson_r"].abs()
    work = work[(work["n"] >= 50) & ((work["nonzero_count"].fillna(work["n"])) >= 20)].copy()
    return (
        work.sort_values(["target_metric", "abs_pearson_r"], ascending=[True, False])
        .groupby("target_metric", as_index=False)
        .head(10)
        .reset_index(drop=True)
    )


def rank_array(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def spearman_array_corr(x: np.ndarray, y: np.ndarray) -> float:
    return safe_corr(pd.Series(rank_array(x)), pd.Series(rank_array(y)), "pearson")


def loo_predictions(estimator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in loo.split(x):
        model = clone(estimator)
        model.fit(x[train_idx], y[train_idx])
        preds[test_idx[0]] = float(model.predict(x[test_idx])[0])
    return preds


def model_summary(
    name: str,
    estimator,
    x: np.ndarray,
    y: np.ndarray,
    target_metric: str,
    target_label: str,
) -> dict[str, Any]:
    estimator.fit(x, y)
    train_pred = np.asarray(estimator.predict(x), dtype=float).reshape(-1)
    loo_pred = loo_predictions(estimator, x, y)
    return {
        "target_metric": target_metric,
        "target_label": target_label,
        "model": name,
        "train_pearson_r": safe_corr(pd.Series(train_pred), pd.Series(y), "pearson"),
        "train_spearman_rho": spearman_array_corr(train_pred, y),
        "loo_pearson_r": safe_corr(pd.Series(loo_pred), pd.Series(y), "pearson"),
        "loo_spearman_rho": spearman_array_corr(loo_pred, y),
        "n": int(len(y)),
    }


def build_complexity_model_results(
    df: pd.DataFrame,
    targets: dict[str, str],
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    metric_columns = complexity_metric_columns(df)
    x_frame = transformed_complexity_frame(df, metric_columns)
    x = x_frame.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    max_components = min(8, x.shape[0] - 2, x.shape[1])
    steps_per_target = 1 + (2 * max_components)
    total_steps = len(targets) * steps_per_target
    current_step = 0

    for target_col, target_label in targets.items():
        y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
        if len(y) < 10 or np.nanstd(y) == 0:
            current_step += steps_per_target
            show_progress(current_step, total_steps, f"Fitting complexity models ({target_col})")
            continue
        rows.append(
            model_summary(
                "ridge",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
                    ]
                ),
                x,
                y,
                target_col,
                target_label,
            )
        )
        current_step += 1
        show_progress(current_step, total_steps, f"Fitting complexity models ({target_col}: ridge)")
        if checkpoint_path is not None:
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        for n_components in range(1, max_components + 1):
            rows.append(
                model_summary(
                    f"pcr_{n_components}",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("pca", PCA(n_components=n_components)),
                            ("reg", LinearRegression()),
                        ]
                    ),
                    x,
                    y,
                    target_col,
                    target_label,
                )
            )
            current_step += 1
            show_progress(current_step, total_steps, f"Fitting complexity models ({target_col}: pcr_{n_components})")
            if checkpoint_path is not None:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
            rows.append(
                model_summary(
                    f"pls_{n_components}",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("pls", PLSRegression(n_components=n_components)),
                        ]
                    ),
                    x,
                    y,
                    target_col,
                    target_label,
                )
            )
            current_step += 1
            show_progress(current_step, total_steps, f"Fitting complexity models ({target_col}: pls_{n_components})")
            if checkpoint_path is not None:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
    return pd.DataFrame(rows)


def write_summary(
    validation_df: pd.DataFrame,
    component_loadings: pd.Series,
    human_diagnostics: dict[str, Any],
    llm_diagnostics: dict[str, Any],
    correlations: pd.DataFrame,
    differences: pd.DataFrame,
    model_results: pd.DataFrame,
) -> dict[str, Any]:
    headline_corr = correlations[correlations["complexity_metric"].eq("dsl_complexity_pc1")].copy()
    headline_corr = headline_corr.set_index("target_metric")
    best_by_target = best_metrics_by_target(correlations)
    best_models = (
        model_results.sort_values(["target_metric", "loo_pearson_r"], ascending=[True, False])
        .groupby("target_metric", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    summary = {
        "validated_task_count": int(len(validation_df)),
        "validation_status_counts": validation_df["status"].value_counts().to_dict(),
        "complexity_pc1_loadings": component_loadings.round(6).to_dict(),
        "human_diagnostics": human_diagnostics,
        "llm_diagnostics": llm_diagnostics,
        "headline_correlations": headline_corr[
            ["target_label", "n", "pearson_r", "spearman_rho"]
        ].round(6).to_dict(orient="index"),
        "best_single_metrics": best_by_target[
            ["target_metric", "target_label", "complexity_metric", "pearson_r", "spearman_rho", "n"]
        ].round(6).to_dict(orient="records"),
        "best_model_results": best_models.round(6).to_dict(orient="records"),
        "headline_differences": differences.round(6).to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_report(summary: dict[str, Any]) -> None:
    headline = summary["headline_correlations"]
    best_metrics = summary["best_single_metrics"]
    best_models = summary["best_model_results"]
    diffs = summary["headline_differences"]
    lines = [
        "# ARC-1 DSL Complexity vs Human and LLM Difficulty",
        "",
        f"- Validated DSL solvers: {summary['validated_task_count']} tasks.",
        f"- Validation status counts: {summary['validation_status_counts']}",
        f"- Human latent difficulty used complete-participant HRC responses; complete vs all-task difficulty correlation = {summary['human_diagnostics']['human_complete_vs_all_corr']:.3f}.",
        "",
        "## Headline Correlations (DSL complexity PC1)",
        "",
    ]
    for metric, row in headline.items():
        lines.append(
            f"- {row['target_label']}: Pearson r = {row['pearson_r']:.3f}, Spearman rho = {row['spearman_rho']:.3f}, n = {row['n']}."
        )
    lines.append("")
    lines.append("## Best Single Complexity Metrics By Target")
    lines.append("")
    seen_targets: set[str] = set()
    for row in best_metrics:
        if row["target_metric"] in seen_targets:
            continue
        seen_targets.add(row["target_metric"])
        lines.append(
            f"- {row['target_label']}: best single metric = `{row['complexity_metric']}` "
            f"(Pearson {row['pearson_r']:.3f}, Spearman {row['spearman_rho']:.3f}, n = {row['n']})."
        )
    lines.append("")
    lines.append("## Best Cross-Validated Complexity Models")
    lines.append("")
    for row in best_models:
        lines.append(
            f"- {row['target_label']}: best model = `{row['model']}` "
            f"(train Pearson {row['train_pearson_r']:.3f}, LOO Pearson {row['loo_pearson_r']:.3f}, "
            f"LOO Spearman {row['loo_spearman_rho']:.3f}, n = {row['n']})."
        )
    lines.append("")
    lines.append("## Human vs Other Correlation Differences")
    lines.append("")
    for row in diffs:
        lines.append(
            "- "
            f"{row['target_label']}: "
            f"Pearson diff = {row['pearson_difference']:.3f}, "
            f"Williams p = {row['williams_p']:.4g}, "
            f"Spearman diff = {row['spearman_difference']:.3f}, "
            f"bootstrap 95% CI = [{row['bootstrap_diff_ci_low']:.3f}, {row['bootstrap_diff_ci_high']:.3f}], "
            f"bootstrap p = {row['bootstrap_p']:.4g}."
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.models_only:
        print("Resuming from saved task join...", file=sys.stderr, flush=True)
        if not MERGED_PATH.exists():
            raise FileNotFoundError(f"Missing saved task join: {MERGED_PATH}")
        merged = pd.read_csv(MERGED_PATH)
        print("Fitting cross-validated complexity models...", file=sys.stderr, flush=True)
        model_results = build_complexity_model_results(merged, HEADLINE_TARGETS, checkpoint_path=MODEL_RESULTS_PATH)
        model_results.to_csv(MODEL_RESULTS_PATH, index=False)
        print(f"Saved model results to {MODEL_RESULTS_PATH.name}.", file=sys.stderr, flush=True)
        return

    stage_total = 6
    show_stage(1, stage_total, "Loading ARC-1 DSL bundle")
    print("Loading ARC-1 DSL bundle...", file=sys.stderr, flush=True)
    bundle = load_bundle_modules()
    show_stage(2, stage_total, "Loading ARC-1 training tasks")
    print("Loading ARC-1 training tasks...", file=sys.stderr, flush=True)
    tasks = load_arc1_training_tasks()

    show_stage(3, stage_total, "Solver validation and dynamic profiling")
    print("Running solver validation and dynamic profiling...", file=sys.stderr, flush=True)
    validation_df = validate_dsl_solvers(bundle, tasks)
    passed_ids = set(validation_df.loc[validation_df["status"].eq("passed"), "task_id"])

    show_stage(4, stage_total, "Extracting static complexity metrics")
    print("Extracting static DSL/Python complexity metrics...", file=sys.stderr, flush=True)
    complexity_df = build_dsl_complexity_table(bundle.sections["solvers.py"])
    complexity_df = complexity_df.merge(
        validation_df.drop(columns=["mismatch_split", "mismatch_index", "error_type", "error_message"]),
        on="task_id",
        how="left",
    )
    complexity_df = complexity_df[complexity_df["task_id"].isin(passed_ids)].copy()
    complexity_df["elapsed_ms_per_input_cell"] = complexity_df["elapsed_ms_total"] / complexity_df["input_cells_total"].clip(lower=1)
    complexity_df["elapsed_ms_per_output_cell"] = complexity_df["elapsed_ms_total"] / complexity_df["output_cells_total"].clip(lower=1)
    complexity_df["bundle_opcode_per_input_cell"] = complexity_df["bundle_opcode_count_dynamic"] / complexity_df["input_cells_total"].clip(lower=1)
    complexity_df["bundle_opcode_per_output_cell"] = complexity_df["bundle_opcode_count_dynamic"] / complexity_df["output_cells_total"].clip(lower=1)
    complexity_df["bundle_branch_opcode_per_input_cell"] = complexity_df["bundle_branch_opcode_count_dynamic"] / complexity_df["input_cells_total"].clip(lower=1)
    complexity_df["bundle_python_calls_per_example"] = complexity_df["bundle_python_call_count_dynamic"] / complexity_df["example_count"].clip(lower=1)
    complexity_df["peak_memory_per_input_cell"] = complexity_df["peak_memory_bytes"] / complexity_df["input_cells_total"].clip(lower=1)
    component_loadings = add_complexity_component(complexity_df)

    show_stage(5, stage_total, "Fitting human difficulty and scoring LLMs")
    print("Fitting human difficulty and scoring GPT/Claude/IceCuber...", file=sys.stderr, flush=True)
    human_df, human_diagnostics = build_human_difficulty_table(passed_ids)
    llm_df, llm_diagnostics = build_llm_task_table({task_id: tasks[task_id] for task_id in passed_ids})

    merged = complexity_df.merge(human_df, on="task_id", how="inner").merge(llm_df, on="task_id", how="inner")
    merged = merged.sort_values("task_id").reset_index(drop=True)

    show_stage(6, stage_total, "Correlations, checkpoints, and model fitting")
    print("Computing correlations and difference tests...", file=sys.stderr, flush=True)
    correlations = build_correlation_table(merged, HEADLINE_TARGETS)
    differences = build_difference_table(merged)
    best_metrics = best_metrics_by_target(correlations)

    validation_payload = {
        "task_count": int(len(validation_df)),
        "status_counts": validation_df["status"].value_counts().to_dict(),
        "failures": validation_df.loc[validation_df["status"].ne("passed")].to_dict(orient="records"),
    }
    VALIDATION_PATH.write_text(json.dumps(validation_payload, indent=2), encoding="utf-8")
    complexity_df.to_csv(COMPLEXITY_PATH, index=False)
    correlations.to_csv(CORRELATION_PATH, index=False)
    differences.to_csv(DIFFERENCE_PATH, index=False)
    best_metrics.to_csv(BEST_METRICS_PATH, index=False)
    merged.to_csv(MERGED_PATH, index=False)
    print(
        f"Saved validation to {VALIDATION_PATH.name}, metrics to {COMPLEXITY_PATH.name}, "
        f"correlations to {CORRELATION_PATH.name}, best-metric table to {BEST_METRICS_PATH.name}, "
        f"and task join to {MERGED_PATH.name}.",
        file=sys.stderr,
        flush=True,
    )

    print("Fitting cross-validated complexity models...", file=sys.stderr, flush=True)
    model_results = build_complexity_model_results(merged, HEADLINE_TARGETS, checkpoint_path=MODEL_RESULTS_PATH)
    model_results.to_csv(MODEL_RESULTS_PATH, index=False)
    print(f"Saved model results to {MODEL_RESULTS_PATH.name}.", file=sys.stderr, flush=True)

    summary = write_summary(
        validation_df=validation_df,
        component_loadings=component_loadings,
        human_diagnostics=human_diagnostics,
        llm_diagnostics=llm_diagnostics,
        correlations=correlations,
        differences=differences,
        model_results=model_results,
    )
    write_report(summary)

    print("Analysis complete.", file=sys.stderr, flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
