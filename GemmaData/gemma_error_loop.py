from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = SCRIPT_DIR / "gemma_arc1_runner.py"
DEFAULT_SOURCE_TASK_DIR = SCRIPT_DIR / "invalid_rerun_159_20260409T215820Z"
DEFAULT_START_TASK_DIR = SCRIPT_DIR / "invalid_rerun_41_20260410T010534Z"
DEFAULT_PYTHON = SCRIPT_DIR.parent / ".venv" / "Scripts" / "python.exe"
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_THINKING_LEVEL = "high"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def task_ids_from_dir(dataset_dir: Path) -> list[str]:
    task_ids: list[str] = []
    for path in sorted(dataset_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        task_ids.append(path.stem)
    return task_ids


def read_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def extract_next_task_ids(summary: dict, selected_ids: Iterable[str]) -> list[str]:
    records = summary.get("records") or []
    completed_ids = {
        record.get("task_id")
        for record in records
        if record.get("task_id")
    }
    error_ids = [
        record.get("task_id")
        for record in records
        if record.get("status") == "error" and record.get("task_id")
    ]
    missing_ids = [task_id for task_id in selected_ids if task_id not in completed_ids]

    seen: set[str] = set()
    next_ids: list[str] = []
    for task_id in [*error_ids, *missing_ids]:
        if task_id not in seen:
            seen.add(task_id)
            next_ids.append(task_id)
    return next_ids


def build_subset(source_task_dir: Path, task_ids: list[str], label: str) -> Path:
    subset_dir = SCRIPT_DIR / f"{label}_{utc_stamp()}"
    subset_dir.mkdir(parents=True, exist_ok=False)

    for task_id in task_ids:
        src = source_task_dir / f"{task_id}.json"
        if not src.exists():
            raise FileNotFoundError(f"Missing source task file: {src}")
        shutil.copy2(src, subset_dir / src.name)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_task_dir": str(source_task_dir),
        "selected_task_count": len(task_ids),
        "selected_task_ids": task_ids,
        "selection_reason": "Task retry loop subset",
    }
    (subset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return subset_dir


def build_runner_args(args: argparse.Namespace, dataset_dir: Path) -> list[str]:
    cmd = [
        str(args.python_executable),
        str(RUNNER_PATH),
        "--dataset-dir",
        str(dataset_dir),
        "--model",
        args.model,
        "--thinking-level",
        args.thinking_level,
        "--workers",
        str(args.workers),
        "--max-in-flight",
        str(args.max_in_flight),
        "--rate-limit-per-minute",
        str(args.rate_limit_per_minute),
        "--transient-throttle-per-minute",
        str(args.transient_throttle_per_minute),
        "--retries",
        str(args.retries),
        "--temperature",
        str(args.temperature),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--timeout-ms",
        str(args.timeout_ms),
    ]
    if args.backoff is not None:
        cmd.extend(["--backoff", str(args.backoff)])
    if args.delay is not None:
        cmd.extend(["--delay", str(args.delay)])
    if args.transient_cooldown_seconds is not None:
        cmd.extend(["--transient-cooldown-seconds", str(args.transient_cooldown_seconds)])
    return cmd


def run_one_batch(args: argparse.Namespace, dataset_dir: Path, round_index: int) -> Path:
    cmd = build_runner_args(args, dataset_dir)
    print(f"Round {round_index}: launching {dataset_dir}", flush=True)
    print("Command: " + " ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(SCRIPT_DIR.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    run_dir: Path | None = None
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("Run directory:"):
            run_dir = Path(line.split(":", 1)[1].strip())

    return_code = proc.wait()
    if return_code != 0:
        raise SystemExit(f"Runner exited with code {return_code}")
    if run_dir is None:
        raise SystemExit("Runner did not print a run directory")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loop ARC reruns until no error tasks remain.")
    parser.add_argument(
        "--source-task-dir",
        type=Path,
        default=DEFAULT_SOURCE_TASK_DIR,
        help="Folder containing the original ARC task JSON files to copy from.",
    )
    parser.add_argument(
        "--start-dataset-dir",
        type=Path,
        default=DEFAULT_START_TASK_DIR,
        help="Folder containing the initial subset to run first.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking-level", default=DEFAULT_THINKING_LEVEL)
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable),
        help="Python interpreter to use for the runner subprocess.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-in-flight", type=int, default=4)
    parser.add_argument("--rate-limit-per-minute", type=int, default=14)
    parser.add_argument("--transient-throttle-per-minute", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=65536)
    parser.add_argument("--timeout-ms", type=int, default=600000)
    parser.add_argument("--backoff", type=float, default=2.0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--transient-cooldown-seconds", type=float, default=60.0)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Optional safety cap. 0 means keep looping until no error tasks remain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_task_dir = args.source_task_dir.resolve()
    current_dataset_dir = args.start_dataset_dir.resolve()

    if not current_dataset_dir.exists():
        raise SystemExit(f"Start dataset directory does not exist: {current_dataset_dir}")
    if not source_task_dir.exists():
        raise SystemExit(f"Source task directory does not exist: {source_task_dir}")

    round_index = 1
    while True:
        current_task_ids = task_ids_from_dir(current_dataset_dir)
        if not current_task_ids:
            raise SystemExit(f"No task files found in {current_dataset_dir}")

        run_dir = run_one_batch(args, current_dataset_dir, round_index)
        summary = read_summary(run_dir)
        next_task_ids = extract_next_task_ids(summary, current_task_ids)

        solved = summary.get("solved_tasks")
        errors = summary.get("error_tasks")
        accuracy = summary.get("accuracy")
        print(
            f"Round {round_index} finished: solved={solved} errors={errors} accuracy={accuracy:.3%}",
            flush=True,
        )

        if not next_task_ids:
            print("No remaining error or missing tasks. Loop complete.", flush=True)
            break

        if args.max_rounds and round_index >= args.max_rounds:
            print(
                f"Reached max_rounds={args.max_rounds}. Next retry set would contain {len(next_task_ids)} tasks.",
                flush=True,
            )
            break

        print(
            f"Preparing next retry set with {len(next_task_ids)} task(s): "
            + ", ".join(next_task_ids[:12])
            + (" ..." if len(next_task_ids) > 12 else ""),
            flush=True,
        )
        current_dataset_dir = build_subset(source_task_dir, next_task_ids, f"retry_subset_round_{round_index + 1:02d}")
        round_index += 1


if __name__ == "__main__":
    main()
