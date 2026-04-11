from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RUNNER_PATH = SCRIPT_DIR / "gemma_arc1_runner.py"
DEFAULT_SOURCE_TASK_DIR = REPO_ROOT / "data-llm" / "ARC-AGI" / "data" / "training"
DEFAULT_FALLBACK_SOURCE_TASK_DIR = SCRIPT_DIR / "data" / "training"
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_THINKING_LEVEL = "high"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "value"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_source_task_dir(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_SOURCE_TASK_DIR and DEFAULT_FALLBACK_SOURCE_TASK_DIR.exists():
        return DEFAULT_FALLBACK_SOURCE_TASK_DIR
    raise SystemExit(f"Source task directory does not exist: {path}")


def list_task_ids(dataset_dir: Path) -> list[str]:
    task_ids: list[str] = []
    for path in sorted(dataset_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        task_ids.append(path.stem)
    return task_ids


def scan_valid_gemini_ids(runs_dir: Path, model_slug: str) -> set[str]:
    valid_ids: set[str] = set()
    for run_dir in runs_dir.glob(f"*{model_slug}*"):
        if not run_dir.is_dir():
            continue
        tasks_dir = run_dir / "tasks"
        if not tasks_dir.is_dir():
            continue
        for task_path in tasks_dir.glob("*.json"):
            if task_path.name == "manifest.json":
                continue
            try:
                record = json.loads(task_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if record.get("status") == "ok":
                valid_ids.add(record.get("task_id") or task_path.stem)
    return valid_ids


def compute_unresolved_ids(source_task_dir: Path, runs_dir: Path, model_slug: str) -> list[str]:
    all_task_ids = list_task_ids(source_task_dir)
    valid_ids = scan_valid_gemini_ids(runs_dir, model_slug)
    return [task_id for task_id in all_task_ids if task_id not in valid_ids]


def build_subset(source_task_dir: Path, task_ids: list[str], round_index: int) -> Path:
    subset_dir = SCRIPT_DIR / f"gemini_unresolved_round_{round_index:02d}_{utc_stamp()}"
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
        "selection_reason": "Tasks that still lack a valid Gemini output",
    }
    (subset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return subset_dir


def build_runner_command(args: argparse.Namespace, dataset_dir: Path) -> list[str]:
    return [
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
        "--backoff",
        str(args.backoff),
        "--delay",
        str(args.delay),
        "--transient-cooldown-seconds",
        str(args.transient_cooldown_seconds),
    ]


def read_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_one_batch(args: argparse.Namespace, dataset_dir: Path, round_index: int) -> Path:
    cmd = build_runner_command(args, dataset_dir)
    print(f"Round {round_index}: launching {len(list_task_ids(dataset_dir))} task(s) from {dataset_dir}", flush=True)
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
    parser = argparse.ArgumentParser(
        description="Loop Gemini retries until the full ARC set has valid Gemini outputs."
    )
    parser.add_argument(
        "--source-task-dir",
        type=Path,
        default=DEFAULT_SOURCE_TASK_DIR,
        help="Directory containing the full ARC training JSON files.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=SCRIPT_DIR / "runs",
        help="Directory containing saved run outputs to scan for valid Gemini results.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable),
        help="Python interpreter used to launch the runner subprocess.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking-level", default=DEFAULT_THINKING_LEVEL)
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
        help="Optional safety cap. 0 means keep looping until no unresolved tasks remain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_task_dir = resolve_source_task_dir(args.source_task_dir.resolve())
    runs_dir = args.runs_dir.resolve()
    model_slug = slugify(args.model)

    if not runs_dir.exists():
        raise SystemExit(f"Runs directory does not exist: {runs_dir}")

    round_index = 1
    while True:
        unresolved_ids = compute_unresolved_ids(source_task_dir, runs_dir, model_slug)
        if not unresolved_ids:
            print("No unresolved Gemini outputs remain. Loop complete.", flush=True)
            break

        if args.max_rounds and round_index > args.max_rounds:
            print(
                f"Reached max_rounds={args.max_rounds}. "
                f"Next unresolved set would contain {len(unresolved_ids)} task(s).",
                flush=True,
            )
            break

        print(
            f"Round {round_index}: {len(unresolved_ids)} task(s) still lack a valid Gemini output.",
            flush=True,
        )
        subset_dir = build_subset(source_task_dir, unresolved_ids, round_index)
        run_dir = run_one_batch(args, subset_dir, round_index)
        summary = read_summary(run_dir)
        print(
            f"Round {round_index} finished: solved={summary.get('solved_tasks')} "
            f"errors={summary.get('error_tasks')} accuracy={summary.get('accuracy'):.3%}",
            flush=True,
        )
        round_index += 1


if __name__ == "__main__":
    main()
