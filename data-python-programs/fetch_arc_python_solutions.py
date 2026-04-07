from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://arc.huikang.dev/solutions/{task_id}/solution.py"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ARC-AGI solution fetcher"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
SOURCE_DIRS = {
    "arc_agi_1_train": "data-llm/ARC-AGI/data/training",
    "arc_agi_1_eval": "data-llm/ARC-AGI/data/evaluation",
    "arc_agi_2_train": "data-llm/ARC-AGI-2/data/training",
    "arc_agi_2_eval": "data-llm/ARC-AGI-2/data/evaluation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ARC Python solutions from arc.huikang.dev for all locally known task IDs."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional task IDs to fetch instead of all known tasks.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=24,
        help="Maximum parallel downloads to run at once.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even when they already exist locally.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts for transient failures.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_root() -> Path:
    return Path(__file__).resolve().parent


def load_dataset_tasks() -> dict[str, list[str]]:
    dataset_tasks: dict[str, list[str]] = {}
    for dataset_name, relative_dir in SOURCE_DIRS.items():
        path = repo_root() / relative_dir
        if not path.exists():
            raise FileNotFoundError(f"Source directory not found: {path}")

        dataset_tasks[dataset_name] = sorted(task_path.stem for task_path in path.glob("*.json"))

    return dataset_tasks


def build_task_index(dataset_tasks: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for dataset_name, task_ids in dataset_tasks.items():
        for position, task_id in enumerate(task_ids, start=1):
            entry = index.setdefault(
                task_id,
                {
                    "datasets": [],
                    "positions": {},
                    "url": BASE_URL.format(task_id=task_id),
                },
            )
            entry["datasets"].append(dataset_name)
            entry["positions"][dataset_name] = position
    return dict(sorted(index.items()))


def write_task_lists(dataset_tasks: dict[str, list[str]], task_index: dict[str, dict[str, Any]]) -> None:
    root = output_root()
    task_lists_dir = root / "task_lists"
    task_lists_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, task_ids in dataset_tasks.items():
        (task_lists_dir / f"{dataset_name}.txt").write_text(
            "\n".join(task_ids) + "\n",
            encoding="utf-8",
        )

    (root / "task_index.json").write_text(
        json.dumps(task_index, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def solution_ids_on_disk() -> set[str]:
    solutions_dir = output_root() / "solutions"
    if not solutions_dir.exists():
        solutions_dir = output_root() / "all_solutions"
    if not solutions_dir.exists():
        return set()
    return {path.stem for path in solutions_dir.glob("*.py")}


def write_availability_lists(dataset_tasks: dict[str, list[str]], available_ids: set[str]) -> None:
    availability_dir = output_root() / "availability"
    availability_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = {}
    for dataset_name, task_ids in dataset_tasks.items():
        available = [task_id for task_id in task_ids if task_id in available_ids]
        missing = [task_id for task_id in task_ids if task_id not in available_ids]

        (availability_dir / f"{dataset_name}_available.txt").write_text(
            "\n".join(available) + ("\n" if available else ""),
            encoding="utf-8",
        )
        (availability_dir / f"{dataset_name}_missing.txt").write_text(
            "\n".join(missing) + ("\n" if missing else ""),
            encoding="utf-8",
        )

        summary[dataset_name] = {
            "task_count": len(task_ids),
            "available_count": len(available),
            "missing_count": len(missing),
        }

    (availability_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def select_tasks(dataset_tasks: dict[str, list[str]], requested_tasks: list[str] | None) -> list[str]:
    all_tasks = set()
    for task_ids in dataset_tasks.values():
        all_tasks.update(task_ids)

    if not requested_tasks:
        return sorted(all_tasks)

    requested = []
    unknown = []
    for task_id in requested_tasks:
        if task_id in all_tasks:
            requested.append(task_id)
        else:
            unknown.append(task_id)

    if unknown:
        print("Unknown task IDs:", ", ".join(sorted(unknown)), file=sys.stderr)

    if not requested:
        raise SystemExit("No requested task IDs were found in the local ARC datasets.")

    return sorted(set(requested))


def should_retry(status_code: int | None, error: Exception | None) -> bool:
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    if isinstance(error, URLError):
        return True
    return False


def fetch_solution(task_id: str, *, timeout: float, retries: int, overwrite: bool) -> dict[str, Any]:
    solutions_dir = output_root() / "all_solutions"
    solutions_dir.mkdir(parents=True, exist_ok=True)
    destination = solutions_dir / f"{task_id}.py"

    if destination.exists() and not overwrite:
        return {
            "task_id": task_id,
            "status": "skipped_existing",
            "http_status": None,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "attempts": 0,
        }

    last_error: Exception | None = None
    last_status: int | None = None

    for attempt in range(1, retries + 1):
        request = Request(
            BASE_URL.format(task_id=task_id),
            headers={"User-Agent": USER_AGENT},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read()
                status_code = getattr(response, "status", response.getcode())
                if status_code != 200:
                    last_status = status_code
                    raise HTTPError(
                        response.geturl(),
                        status_code,
                        f"Unexpected HTTP status {status_code}",
                        response.headers,
                        None,
                    )

            if not body.strip():
                return {
                    "task_id": task_id,
                    "status": "empty",
                    "http_status": status_code,
                    "path": None,
                    "bytes": 0,
                    "attempts": attempt,
                }

            destination.write_bytes(body)
            return {
                "task_id": task_id,
                "status": "downloaded",
                "http_status": status_code,
                "path": str(destination),
                "bytes": len(body),
                "attempts": attempt,
            }
        except HTTPError as error:
            last_error = error
            last_status = error.code
            if not should_retry(error.code, error) or attempt == retries:
                break
        except URLError as error:
            last_error = error
            if attempt == retries:
                break
        except Exception as error:  # pragma: no cover - defensive fallback
            last_error = error
            if attempt == retries:
                break

        time.sleep(min(2 ** (attempt - 1), 5))

    return {
        "task_id": task_id,
        "status": "missing" if last_status == 404 else "error",
        "http_status": last_status,
        "path": None,
        "bytes": 0,
        "attempts": retries,
        "error": str(last_error) if last_error else "Unknown error",
    }


def summarize_results(results: list[dict[str, Any]], dataset_tasks: dict[str, list[str]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1

    missing_tasks = sorted(result["task_id"] for result in results if result["status"] == "missing")
    error_tasks = sorted(result["task_id"] for result in results if result["status"] == "error")
    empty_tasks = sorted(result["task_id"] for result in results if result["status"] == "empty")
    available_ids = solution_ids_on_disk()

    dataset_summary = {
        dataset_name: {
            "task_count": len(task_ids),
            "matching_solutions_available": sum(1 for task_id in task_ids if task_id in available_ids),
            "missing_solution_count": sum(1 for task_id in task_ids if task_id not in available_ids),
        }
        for dataset_name, task_ids in dataset_tasks.items()
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dirs": SOURCE_DIRS,
        "selected_unique_task_count": len(results),
        "solutions_on_disk_count": len(available_ids),
        "dataset_counts": {name: len(task_ids) for name, task_ids in dataset_tasks.items()},
        "result_counts": counts,
        "missing_tasks": missing_tasks,
        "error_tasks": error_tasks,
        "empty_tasks": empty_tasks,
        "datasets": dataset_summary,
        "results": results,
    }


def main() -> int:
    args = parse_args()
    dataset_tasks = load_dataset_tasks()
    task_index = build_task_index(dataset_tasks)
    write_task_lists(dataset_tasks, task_index)

    selected_tasks = select_tasks(dataset_tasks, args.tasks)

    print(f"Loaded {len(task_index)} unique task IDs from local ARC datasets.")
    print(f"Fetching {len(selected_tasks)} unique task IDs from arc.huikang.dev...")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_to_task = {
            executor.submit(
                fetch_solution,
                task_id,
                timeout=args.timeout,
                retries=max(1, args.retries),
                overwrite=args.overwrite,
            ): task_id
            for task_id in selected_tasks
        }
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)
            print(f"{result['task_id']}: {result['status']}")

    results.sort(key=lambda item: item["task_id"])
    write_availability_lists(dataset_tasks, solution_ids_on_disk())
    report = summarize_results(results, dataset_tasks)
    (output_root() / "fetch_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print("Summary:")
    for status_name, count in sorted(report["result_counts"].items()):
        print(f"  {status_name}: {count}")
    print(f"  report: {output_root() / 'fetch_report.json'}")
    print(f"  task index: {output_root() / 'task_index.json'}")
    print(f"  solutions dir: {output_root() / 'all_solutions'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
