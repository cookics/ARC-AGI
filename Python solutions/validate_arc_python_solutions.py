from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OFFICIAL_SOURCES = {
    "arc_agi_1_train": ("buildtmp/ARC-AGI-1-official/data/training", "official"),
    "arc_agi_1_eval": ("buildtmp/ARC-AGI-1-official/data/evaluation", "official"),
    "arc_agi_2_train": ("buildtmp/ARC-AGI-2-official/data/training", "official"),
    "arc_agi_2_eval": ("buildtmp/ARC-AGI-2-official/data/evaluation", "official"),
}

LOCAL_CHALLENGE_SOURCES = {
    "arc_agi_1_train": (
        "Non-LLM data/raw/Minds AI/arc-agi-1_training_challenges.json",
        "Non-LLM data/raw/Minds AI/arc-agi-1_training_solutions.json",
    ),
    "arc_agi_1_eval": (
        "Non-LLM data/raw/Minds AI/arc-agi-1_evaluation_challenges.json",
        "Non-LLM data/raw/Minds AI/arc-agi-1_evaluation_solutions.json",
    ),
    "arc_agi_2_train": (
        "Non-LLM data/raw/Minds AI/arc-agi-2_training_challenges.json",
        "Non-LLM data/raw/Minds AI/arc-agi-2_training_solutions.json",
    ),
    "arc_agi_2_eval": (
        "Non-LLM data/raw/Minds AI/arc-agi-2_evaluation_challenges.json",
        "Non-LLM data/raw/Minds AI/arc-agi-2_evaluation_solutions.json",
    ),
}


@dataclass(frozen=True)
class ValidationCase:
    task_id: str
    solution_path: Path
    variants: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate downloaded ARC solution.py files against ARC tasks."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional subset of task IDs to validate.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of validation subprocesses to run at once.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Per-solution timeout in seconds.",
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
    return Path(__file__).resolve().parents[1]


def solutions_root() -> Path:
    return Path(__file__).resolve().parent / "solutions"


def report_path() -> Path:
    return Path(__file__).resolve().parent / "validation_report.json"


def failure_path() -> Path:
    return Path(__file__).resolve().parent / "validation_failures.json"


def shape_of(grid: Any) -> list[int] | None:
    if not isinstance(grid, list):
        return None
    if not grid:
        return [0, 0]
    if not all(isinstance(row, list) for row in grid):
        return None
    return [len(grid), len(grid[0])]


def stable_task_hash(task: dict[str, Any]) -> str:
    encoded = json.dumps(task, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rebuild_local_task(dataset_name: str, task_id: str) -> dict[str, Any]:
    challenge_rel, solution_rel = LOCAL_CHALLENGE_SOURCES[dataset_name]
    challenge_path = repo_root() / challenge_rel
    solution_path = repo_root() / solution_rel

    challenges = json.loads(challenge_path.read_text(encoding="utf-8"))
    solutions = json.loads(solution_path.read_text(encoding="utf-8"))

    challenge_task = challenges[task_id]
    solution_outputs = solutions[task_id]
    rebuilt = {"train": challenge_task["train"], "test": []}
    for index, pair in enumerate(challenge_task["test"]):
        rebuilt["test"].append(
            {
                "input": pair["input"],
                "output": solution_outputs[index],
            }
        )
    return rebuilt


def collect_task_variants(task_id: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for dataset_name, (relative_dir, source_kind) in OFFICIAL_SOURCES.items():
        task_path = repo_root() / relative_dir / f"{task_id}.json"
        if task_path.exists():
            task = json.loads(task_path.read_text(encoding="utf-8"))
            task.pop("name", None)
            task_hash = stable_task_hash(task)
            entry = grouped.setdefault(
                task_hash,
                {"datasets": [], "source": source_kind, "task": task},
            )
            entry["datasets"].append(dataset_name)

    if grouped:
        return sorted(grouped.values(), key=lambda item: tuple(item["datasets"]))

    for dataset_name in LOCAL_CHALLENGE_SOURCES:
        try:
            task = rebuild_local_task(dataset_name, task_id)
        except KeyError:
            continue
        task_hash = stable_task_hash(task)
        entry = grouped.setdefault(
            task_hash,
            {"datasets": [], "source": "local_rebuilt", "task": task},
        )
        entry["datasets"].append(dataset_name)

    return sorted(grouped.values(), key=lambda item: tuple(item["datasets"]))


def build_validation_cases(selected_tasks: set[str] | None = None) -> list[ValidationCase]:
    cases: list[ValidationCase] = []
    for solution_path in sorted(solutions_root().glob("*.py")):
        task_id = solution_path.stem
        if selected_tasks is not None and task_id not in selected_tasks:
            continue

        variants = collect_task_variants(task_id)
        if not variants:
            variants = [{"datasets": [], "source": "missing_task_data", "task": None}]

        cases.append(
            ValidationCase(
                task_id=task_id,
                solution_path=solution_path,
                variants=variants,
            )
        )

    return cases


def load_solution_module(solution_path: Path):
    module_name = f"arc_solution_{solution_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, solution_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_worker() -> int:
    args = parse_args()
    payload = json.load(sys.stdin)
    output_file = args.output_file
    if output_file is None:
        raise SystemExit("--output-file is required in worker mode.")

    result: dict[str, Any]
    try:
        solution_path = Path(payload["solution_path"])
        module = load_solution_module(solution_path)
        solve = getattr(module, "solve", None)
        if not callable(solve):
            raise AttributeError("No callable solve(grid) function found.")

        total_tests = 0
        for variant in payload["variants"]:
            task = variant["task"]
            if task is None:
                result = {
                    "task_id": payload["task_id"],
                    "status": "missing_task_data",
                    "datasets": variant["datasets"],
                    "message": "No task JSON found for this solution file.",
                }
                output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
                return 0

            for test_index, pair in enumerate(task["test"]):
                total_tests += 1
                actual = solve(copy.deepcopy(pair["input"]))
                expected = pair["output"]
                if actual != expected:
                    result = {
                        "task_id": payload["task_id"],
                        "status": "wrong_answer",
                        "datasets": variant["datasets"],
                        "source": variant["source"],
                        "test_index": test_index,
                        "expected_shape": shape_of(expected),
                        "actual_shape": shape_of(actual),
                        "expected": expected,
                        "actual": actual,
                    }
                    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    return 0

        result = {
            "task_id": payload["task_id"],
            "status": "passed",
            "variant_count": len(payload["variants"]),
            "test_count": total_tests,
            "datasets": [variant["datasets"] for variant in payload["variants"]],
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


def run_case(case: ValidationCase, timeout: float) -> dict[str, Any]:
    payload = {
        "task_id": case.task_id,
        "solution_path": str(case.solution_path),
        "variants": case.variants,
    }

    with tempfile.NamedTemporaryFile(
        prefix=f"arc_validate_{case.task_id}_",
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
                "returncode": completed.returncode,
            }

        if completed.returncode != 0 and result.get("status") == "passed":
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
            "error": f"Validation exceeded {timeout:.1f}s",
            "stdout": error.stdout,
            "stderr": error.stderr,
        }
    finally:
        output_path.unlink(missing_ok=True)


def summarize(results: list[dict[str, Any]], cases: list[ValidationCase]) -> dict[str, Any]:
    status_counts = Counter(result["status"] for result in results)
    dataset_counter = Counter()

    for case, result in zip(cases, results, strict=True):
        if result["status"] == "passed":
            for variant in case.variants:
                for dataset_name in variant["datasets"]:
                    dataset_counter[dataset_name] += 1

    failures = [result for result in results if result["status"] != "passed"]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validated_solution_count": len(cases),
        "status_counts": dict(status_counts),
        "passed_dataset_coverage": dict(sorted(dataset_counter.items())),
        "results": results,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    if args.worker:
        return validate_worker()

    selected_tasks = set(args.tasks) if args.tasks else None
    cases = build_validation_cases(selected_tasks)
    print(f"Found {len(cases)} solution files to validate.")

    results_by_task: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_to_case = {
            executor.submit(run_case, case, args.timeout): case
            for case in cases
        }
        for future in as_completed(future_to_case):
            case = future_to_case[future]
            result = future.result()
            results_by_task[case.task_id] = result
            print(f"{case.task_id}: {result['status']}")

    ordered_results = [results_by_task[case.task_id] for case in cases]
    summary = summarize(ordered_results, cases)
    report_path().write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    failure_path().write_text(
        json.dumps(summary["failures"], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print("Summary:")
    for status_name, count in sorted(summary["status_counts"].items()):
        print(f"  {status_name}: {count}")
    print(f"  report: {report_path()}")
    print(f"  failures: {failure_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
