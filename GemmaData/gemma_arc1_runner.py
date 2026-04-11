from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "data" / "training"
DEFAULT_FALLBACK_DATASET_DIR = REPO_ROOT / "data-llm" / "ARC-AGI" / "data" / "training"
DEFAULT_RUNS_DIR = SCRIPT_DIR / "runs"
DEFAULT_MODEL = "gemma-4-31b-it"
DEFAULT_THINKING_LEVEL = "high"
PRINT_LOCK = threading.Lock()
THREAD_LOCAL = threading.local()


class RateLimiter:
    def __init__(
        self,
        max_per_minute: int,
        max_in_flight: int,
        transient_cooldown_seconds: float,
        transient_throttle_per_minute: float,
    ) -> None:
        self.interval = 60.0 / max_per_minute if max_per_minute > 0 else 0.0
        self.transient_throttle_per_minute = max(transient_throttle_per_minute, 0.000001)
        self.transient_throttle_interval = 60.0 / self.transient_throttle_per_minute
        self.current_interval = self.interval
        self.max_interval = max(60.0, self.interval * 8 if self.interval > 0 else 60.0, self.transient_throttle_interval)
        self.transient_cooldown_seconds = transient_cooldown_seconds
        self._lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(max(1, max_in_flight))
        self._next_allowed = 0.0
        self._cooldown_until = 0.0
        self._active_requests = 0
        self._requests_started = 0
        self._requests_succeeded = 0
        self._requests_failed = 0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            target = max(self._next_allowed, self._cooldown_until)
            sleep_for = max(0.0, target - now)
            interval = self.current_interval if self.current_interval > 0 else self.interval
            self._next_allowed = max(now, target) + interval
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._semaphore.acquire()
        with self._lock:
            self._active_requests += 1
            self._requests_started += 1

    def release(self) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
        self._semaphore.release()

    def record_success(self) -> None:
        with self._lock:
            self._requests_succeeded += 1
            if self.current_interval > self.interval and self.interval > 0:
                self.current_interval = max(self.interval, self.current_interval * 0.9)

    def record_failure(self, error_text: str) -> None:
        transient_delay = extract_retry_delay_seconds(error_text)
        transient = is_transient_service_error(error_text)
        invalid = is_invalid_request_error(error_text)
        with self._lock:
            self._requests_failed += 1
            if transient or invalid:
                if self.current_interval <= 0:
                    self.current_interval = self.transient_throttle_interval if transient else 15.0
                else:
                    multiplier = 2.0 if transient else 1.5
                    floor = 5.0 if transient else 15.0
                    throttle_floor = self.transient_throttle_interval if transient else 15.0
                    self.current_interval = min(
                        self.max_interval,
                        max(self.current_interval * multiplier, self.current_interval + floor, throttle_floor),
                    )
                cooldown = self.transient_cooldown_seconds if transient else max(self.transient_cooldown_seconds, 300.0)
                if transient_delay is not None:
                    cooldown = max(cooldown, transient_delay)
                cooldown = max(cooldown, self.current_interval * (2.0 if transient else 4.0))
                self._cooldown_until = max(self._cooldown_until, time.monotonic() + cooldown)
                self._next_allowed = max(self._next_allowed, self._cooldown_until)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            cooldown_remaining = max(0.0, self._cooldown_until - now)
            return {
                "requests_per_minute_base": (60.0 / self.interval) if self.interval > 0 else 0.0,
                "transient_throttle_per_minute": self.transient_throttle_per_minute,
                "transient_throttle_interval_seconds": self.transient_throttle_interval,
                "current_interval_seconds": self.current_interval,
                "max_interval_seconds": self.max_interval,
                "cooldown_remaining_seconds": cooldown_remaining,
                "active_requests": self._active_requests,
                "requests_started": self._requests_started,
                "requests_succeeded": self._requests_succeeded,
                "requests_failed": self._requests_failed,
            }


def is_transient_service_error(error_text: str) -> bool:
    text = error_text.lower()
    return any(
        token in text
        for token in (
            "503",
            "unavailable",
            "high demand",
            "resource_exhausted",
            "429",
            "rate limit",
            "temporarily",
        )
    )


def is_invalid_request_error(error_text: str) -> bool:
    text = error_text.lower()
    return any(
        token in text
        for token in (
            "invalid_argument",
            "invalid request",
            "bad request",
            "malformed",
            "unsupported value",
        )
    ) or "code': 400" in text or '"code": 400' in text


def classify_error_text(error_text: str) -> str:
    text = error_text.lower()
    if is_invalid_request_error(error_text):
        return "invalid_request"
    if is_transient_service_error(error_text):
        return "transient_service"
    if "access is denied" in text or "permission denied" in text or "winerror 5" in text:
        return "telemetry_write"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "json" in text or "parse" in text or "could not locate" in text:
        return "parse_or_validation"
    return "unknown"


def summarize_exception(exc: Exception, *, stage: str) -> dict[str, Any]:
    error_text = str(exc)
    return {
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": error_text,
        "category": classify_error_text(error_text),
        "transient": is_transient_service_error(error_text),
        "invalid_request": is_invalid_request_error(error_text),
        "retry_delay_seconds": extract_retry_delay_seconds(error_text),
        "traceback": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
    }


class BatchProgress:
    def __init__(self, *, completed: int = 0, solved: int = 0, errors: int = 0) -> None:
        self._lock = threading.Lock()
        self.completed = completed
        self.solved = solved
        self.errors = errors

    def record(self, *, status: str, exact_match: bool) -> tuple[int, int, int]:
        with self._lock:
            self.completed += 1
            if exact_match:
                self.solved += 1
            if status == "error":
                self.errors += 1
            return self.completed, self.solved, self.errors


class LiveMetrics:
    def __init__(
        self,
        run_dir: Path,
        total_tasks: int,
        *,
        initial_tasks_completed: int = 0,
        initial_tasks_solved: int = 0,
        initial_tasks_errors: int = 0,
        initial_requests_started: int = 0,
        initial_requests_succeeded: int = 0,
        initial_requests_failed: int = 0,
        initial_requests_in_flight: int = 0,
        initial_transient_errors: list[dict[str, Any]] | None = None,
        initial_stop_requested: bool = False,
        initial_stop_reason: dict[str, Any] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.run_dir = run_dir
        self.status_path = run_dir / "status.json"
        self.total_tasks = total_tasks
        self.tasks_completed = initial_tasks_completed
        self.tasks_solved = initial_tasks_solved
        self.tasks_errors = initial_tasks_errors
        self.requests_started = initial_requests_started
        self.requests_succeeded = initial_requests_succeeded
        self.requests_failed = initial_requests_failed
        self.requests_in_flight = initial_requests_in_flight
        self.request_control: dict[str, Any] | None = None
        self.last_event: dict[str, Any] | None = None
        self.status_write_errors = 0
        self.last_status_write_error: str | None = None
        self.transient_errors: list[dict[str, Any]] = list(initial_transient_errors or [])
        self.stop_requested = initial_stop_requested
        self.stop_reason = initial_stop_reason
        self._persist()

    def _snapshot_unlocked(self) -> dict[str, Any]:
        progress = self.tasks_completed / self.total_tasks if self.total_tasks else 0.0
        return {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_tasks": self.total_tasks,
            "tasks_completed": self.tasks_completed,
            "tasks_solved": self.tasks_solved,
            "tasks_errors": self.tasks_errors,
            "requests_started": self.requests_started,
            "requests_succeeded": self.requests_succeeded,
            "requests_failed": self.requests_failed,
            "requests_in_flight": self.requests_in_flight,
            "request_control": self.request_control,
            "progress": progress,
            "last_event": self.last_event,
            "status_write_errors": self.status_write_errors,
            "last_status_write_error": self.last_status_write_error,
            "transient_errors": self.transient_errors,
            "stop_requested": self.stop_requested,
            "stop_reason": self.stop_reason,
        }

    def _persist(self) -> None:
        payload = self.snapshot()
        serialized = json.dumps(payload, indent=2)
        tmp_path = self.status_path.with_name(
            f"{self.status_path.stem}.tmp.{os.getpid()}.{threading.get_ident()}{self.status_path.suffix}"
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                tmp_path.write_text(serialized, encoding="utf-8")
                tmp_path.replace(self.status_path)
                return
            except OSError as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
        self.status_write_errors += 1
        if last_error is not None:
            self.last_status_write_error = f"{type(last_error).__name__}: {last_error}"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def request_started(self, *, task_id: str, attempt: int) -> None:
        with self._lock:
            self.requests_started += 1
            self.requests_in_flight += 1
            self.last_event = {
                "kind": "request_started",
                "task_id": task_id,
                "attempt": attempt,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._persist()

    def update_request_control(self, control: dict[str, Any]) -> None:
        with self._lock:
            self.request_control = control
            self._persist()

    def request_finished(self, *, task_id: str, attempt: int, status: str, error: str | None = None) -> None:
        with self._lock:
            self.requests_in_flight = max(0, self.requests_in_flight - 1)
            if status == "ok":
                self.requests_succeeded += 1
            else:
                self.requests_failed += 1
                if error and is_transient_service_error(error):
                    self.transient_errors.append(
                        {
                            "task_id": task_id,
                            "attempt": attempt,
                            "error": error,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    if len(self.transient_errors) > 50:
                        self.transient_errors = self.transient_errors[-50:]
            self.last_event = {
                "kind": "request_finished",
                "task_id": task_id,
                "attempt": attempt,
                "status": status,
                "error": error,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._persist()

    def task_completed(self, *, status: str, exact_match: bool) -> tuple[int, int, int]:
        with self._lock:
            self.tasks_completed += 1
            if exact_match:
                self.tasks_solved += 1
            if status == "error":
                self.tasks_errors += 1
            self._persist()
            return self.tasks_completed, self.tasks_solved, self.tasks_errors

    def request_stop(self, *, reason: str, task_id: str | None = None, attempt: int | None = None, error: str | None = None) -> None:
        with self._lock:
            if not self.stop_requested:
                self.stop_requested = True
                self.stop_reason = {
                    "reason": reason,
                    "task_id": task_id,
                    "attempt": attempt,
                    "error": error,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                self.last_event = {
                    "kind": "stop_requested",
                    "reason": reason,
                    "task_id": task_id,
                    "attempt": attempt,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                self._persist()


def render_progress_bar(completed: int, total: int, width: int = 24) -> str:
    if total <= 0:
        total = 1
    filled = min(width, max(0, round(width * completed / total)))
    return f"[{'#' * filled}{'.' * (width - filled)}]"

ARC_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "test": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "output": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    }
                },
                "required": ["output"],
            },
        }
    },
    "required": ["test"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ARC-AGI v1 training tasks through the Gemini API with Gemma."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing ARC-AGI v1 training JSON files. Defaults to GemmaData/data/training.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory for saving run artifacts.",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        help="Resume an existing run directory in place instead of creating a new one.",
    )
    parser.add_argument(
        "--task-id",
        help="Single ARC task id to run, without .json suffix (example: 007bbfb7).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=400,
        help="Maximum number of tasks to run when --task-id is not set.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel API calls.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMMA_MODEL", DEFAULT_MODEL),
        help="Gemma model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="Maximum output tokens.",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["minimal", "low", "medium", "high"],
        default=DEFAULT_THINKING_LEVEL,
        help="Gemini thinking level.",
    )
    parser.add_argument(
        "--timeout-ms",
        "--timeout",
        dest="timeout_ms",
        type=int,
        default=600000,
        help="Per-request timeout in milliseconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Maximum attempts per task. Use 1 to avoid retrying and wasting request budget.",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=2.0,
        help="Base seconds for retry backoff.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional sleep after each completed task.",
    )
    parser.add_argument(
        "--rate-limit-per-minute",
        type=int,
        default=12,
        help="Maximum request starts per minute across all workers.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=2,
        help="Maximum concurrent requests in flight at once.",
    )
    parser.add_argument(
        "--transient-cooldown-seconds",
        type=float,
        default=60.0,
        help="Minimum cooldown after transient provider failures like 503/429.",
    )
    parser.add_argument(
        "--transient-throttle-per-minute",
        type=float,
        default=1.0,
        help="Fallback request rate after transient provider failures, in requests per minute.",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="Optional file path whose presence stops submitting new tasks and lets in-flight requests drain.",
    )
    parser.add_argument(
        "--stop-on-transient-error",
        action="store_true",
        help="Stop submitting new tasks after the first transient provider failure like 503/429 and let in-flight requests drain.",
    )
    return parser.parse_args()


def load_api_key() -> tuple[str, str]:
    env_sources = [SCRIPT_DIR / ".env", REPO_ROOT / ".env"]
    for key_name in ("GEMINI_API_KEY2", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(key_name)
        if value:
            return value, f"environment:{key_name}"

    for env_path in env_sources:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() in {"GEMINI_API_KEY2", "GEMINI_API_KEY", "GOOGLE_API_KEY"}:
                cleaned = value.strip().strip("'\"")
                if cleaned:
                    return cleaned, str(env_path)

    raise SystemExit(
        "Set GEMINI_API_KEY2, GEMINI_API_KEY, or GOOGLE_API_KEY in the environment or in GemmaData/.env or the repo .env."
    )


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "value"


def resolve_dataset_dir(dataset_dir: Path) -> Path:
    if dataset_dir.exists():
        return dataset_dir
    if dataset_dir == DEFAULT_DATASET_DIR and DEFAULT_FALLBACK_DATASET_DIR.exists():
        return DEFAULT_FALLBACK_DATASET_DIR
    raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")


def list_task_files(dataset_dir: Path, task_id: str | None, limit: int) -> list[Path]:
    if task_id:
        task_path = dataset_dir / f"{task_id}.json"
        if not task_path.exists():
            raise SystemExit(f"Task file not found: {task_path}")
        return [task_path]

    if limit <= 0:
        raise SystemExit("--limit must be a positive integer.")

    task_files = sorted(
        task_path for task_path in dataset_dir.glob("*.json") if task_path.name != "manifest.json"
    )
    if not task_files:
        raise SystemExit(f"No ARC task files found in {dataset_dir}")
    return task_files[:limit]


def load_task_records(tasks_dir: Path) -> list[dict[str, Any]]:
    if not tasks_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for task_path in sorted(tasks_dir.glob("*.json")):
        try:
            record = json.loads(task_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(record, dict) and record.get("task_id"):
            records.append(record)
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, int]:
    completed = len(records)
    solved = sum(1 for item in records if item.get("exact_match"))
    errors = sum(1 for item in records if item.get("status") == "error")
    return {"completed": completed, "solved": solved, "errors": errors}


def build_run_dir(runs_dir: Path, model: str, thinking_level: str, task_count: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{timestamp}_{slugify(model)}_{slugify('thinking-' + thinking_level)}_{task_count}tasks"
    run_dir = runs_dir / run_name
    (run_dir / "tasks").mkdir(parents=True, exist_ok=False)
    return run_dir


def load_task(task_path: Path) -> dict[str, Any]:
    return json.loads(task_path.read_text(encoding="utf-8"))


def build_prompt(task_id: str, task_data: dict[str, Any]) -> tuple[str, str]:
    train_pairs = task_data.get("train", [])
    test_pairs = task_data.get("test", [])
    system_prompt = (
        "You solve ARC grid transformation tasks. Infer the rule from the training examples, "
        "think as thoroughly as needed, and return only valid JSON."
    )
    payload = {
        "task_id": task_id,
        "instructions": [
            "Infer the transformation rule from the training examples.",
            "Think thoroughly before answering.",
            'Return JSON only with this exact shape: {"test":[{"output":[[...]]}]}',
            "Do not include markdown fences, explanations, or extra keys.",
        ],
        "train": train_pairs,
        "test": [{"input": pair["input"]} for pair in test_pairs],
    }
    user_prompt = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return system_prompt, user_prompt


def thinking_level_from_name(name: str) -> types.ThinkingLevel:
    return getattr(types.ThinkingLevel, name.upper())


def build_config(args: argparse.Namespace, system_prompt: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        response_mime_type="application/json",
        response_schema=ARC_RESPONSE_SCHEMA,
        thinking_config=types.ThinkingConfig(
            thinking_level=thinking_level_from_name(args.thinking_level)
        ),
    )


def get_client(api_key: str, timeout: int) -> genai.Client:
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None or getattr(THREAD_LOCAL, "timeout", None) != timeout:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout),
        )
        THREAD_LOCAL.client = client
        THREAD_LOCAL.timeout = timeout
    return client


def extract_retry_delay_seconds(error_text: str) -> float | None:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"retryDelay': '(\d+(?:\.\d+)?)s'", error_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    if is_invalid_request_error(error_text):
        return 300.0
    if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
        return 60.0
    return None


def generate_with_retries(
    *,
    client: genai.Client,
    model: str,
    user_prompt: str,
    config: types.GenerateContentConfig,
    limiter: RateLimiter,
    retries: int,
    backoff: float,
    task_id: str,
    metrics: LiveMetrics,
    attempt_state: dict[str, int],
    stop_event: threading.Event | None = None,
    stop_on_transient_error: bool = False,
) -> Any:
    last_error: Exception | None = None
    attempt_log: list[dict[str, Any]] = attempt_state.setdefault("attempt_log", [])
    for attempt in range(1, retries + 1):
        try:
            attempt_state["attempts_used"] = attempt
            limiter.acquire()
            try:
                metrics.update_request_control(limiter.snapshot())
                metrics.request_started(task_id=task_id, attempt=attempt)
                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001
                error_text = str(exc)
                attempt_log.append(
                    {
                        "attempt": attempt,
                        "stage": "request",
                        "error": error_text,
                        "error_type": type(exc).__name__,
                        "error_category": classify_error_text(error_text),
                        "transient": is_transient_service_error(error_text),
                        "invalid_request": is_invalid_request_error(error_text),
                        "retry_delay_seconds": extract_retry_delay_seconds(error_text),
                    }
                )
                limiter.record_failure(error_text)
                metrics.update_request_control(limiter.snapshot())
                metrics.request_finished(task_id=task_id, attempt=attempt, status="error", error=error_text)
                if stop_on_transient_error and is_transient_service_error(error_text):
                    if stop_event is not None:
                        stop_event.set()
                    metrics.request_stop(
                        reason="transient_service_error",
                        task_id=task_id,
                        attempt=attempt,
                        error=error_text,
                    )
                raise
            else:
                limiter.record_success()
                metrics.update_request_control(limiter.snapshot())
                metrics.request_finished(task_id=task_id, attempt=attempt, status="ok")
                return response
            finally:
                limiter.release()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if stop_event is not None and stop_event.is_set():
                break
            if is_invalid_request_error(str(exc)):
                break
            if attempt >= retries:
                break
            sleep_for = backoff * (2 ** (attempt - 1))
            retry_delay = extract_retry_delay_seconds(str(exc))
            if retry_delay is not None:
                sleep_for = max(sleep_for, retry_delay)
            time.sleep(sleep_for)
    raise RuntimeError(f"Gemini request failed after {retries} attempts: {last_error}")


def serialize_response(response: Any) -> Any:
    if hasattr(response, "to_json_dict"):
        return response.to_json_dict()
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    return {"repr": repr(response)}


def extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        chunks: list[str] = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                chunks.append(part_text)
        joined = "".join(chunks).strip()
        if joined:
            return joined

    raise ValueError("Response did not contain text output.")


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json_text(text: str) -> str:
    stripped = strip_code_fences(text)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate a JSON object in the model response.")
    return stripped[start : end + 1]


def normalize_grid(grid: Any) -> list[list[int]]:
    if not isinstance(grid, list):
        raise ValueError("Grid must be a list of rows.")
    normalized: list[list[int]] = []
    for row in grid:
        if not isinstance(row, list):
            raise ValueError("Grid row must be a list.")
        normalized.append([int(cell) for cell in row])
    row_length = max((len(row) for row in normalized), default=0)
    if row_length == 0:
        return normalized
    return [row + [0] * (row_length - len(row)) for row in normalized]
    return normalized


def parse_predictions(text: str, expected_test_count: int) -> list[list[list[int]]]:
    payload = json.loads(extract_json_text(text))
    tests = payload.get("test")
    if not isinstance(tests, list):
        raise ValueError('Response JSON must contain a "test" list.')
    if len(tests) != expected_test_count:
        raise ValueError(
            f'Response "test" list length {len(tests)} does not match expected {expected_test_count}.'
        )
    outputs: list[list[list[int]]] = []
    for item in tests:
        if not isinstance(item, dict) or "output" not in item:
            raise ValueError('Each "test" entry must be an object containing "output".')
        outputs.append(normalize_grid(item["output"]))
    return outputs


def compare_outputs(
    task_data: dict[str, Any], predicted_outputs: list[list[list[int]]]
) -> tuple[bool, list[bool]]:
    expected_outputs = [normalize_grid(pair["output"]) for pair in task_data.get("test", [])]
    pair_matches = [pred == expected for pred, expected in zip(predicted_outputs, expected_outputs)]
    return all(pair_matches), pair_matches


def save_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def process_task(
    task_path: Path,
    index: int,
    total: int,
    run_dir: Path,
    args: argparse.Namespace,
    api_key: str,
    limiter: RateLimiter,
    progress: BatchProgress,
    metrics: LiveMetrics,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    started = time.time()
    task_id = task_path.stem
    task_data = load_task(task_path)
    system_prompt, user_prompt = build_prompt(task_id, task_data)
    config = build_config(args, system_prompt)
    client = get_client(api_key, args.timeout_ms)

    record: dict[str, Any] = {
        "task_id": task_id,
        "task_index": index,
        "task_path": str(task_path),
        "model": args.model,
        "thinking_level": args.thinking_level,
        "request": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "config": {
                "temperature": args.temperature,
                "max_output_tokens": args.max_output_tokens,
                "response_mime_type": "application/json",
                "thinking_level": args.thinking_level,
                "timeout_ms": args.timeout_ms,
                "rate_limit_per_minute": args.rate_limit_per_minute,
                "max_in_flight": args.max_in_flight,
                "transient_cooldown_seconds": args.transient_cooldown_seconds,
                "transient_throttle_per_minute": args.transient_throttle_per_minute,
            },
        },
    }
    attempt_state: dict[str, int] = {"attempts_used": 0}

    try:
        response = generate_with_retries(
            client=client,
            model=args.model,
            user_prompt=user_prompt,
            config=config,
            limiter=limiter,
            retries=args.retries,
            backoff=args.backoff,
            task_id=task_id,
            metrics=metrics,
            attempt_state=attempt_state,
            stop_event=stop_event,
            stop_on_transient_error=args.stop_on_transient_error,
        )
        attempts_used = attempt_state["attempts_used"]
        response_text = extract_response_text(response)
        record["response_text"] = response_text
        record["response"] = serialize_response(response)
        predicted_outputs = parse_predictions(response_text, expected_test_count=len(task_data.get("test", [])))
        exact_match, pair_matches = compare_outputs(task_data, predicted_outputs)
        record.update(
            {
                "status": "ok",
                "predicted_outputs": predicted_outputs,
                "expected_outputs": [pair["output"] for pair in task_data.get("test", [])],
                "pair_matches": pair_matches,
                "exact_match": exact_match,
                "request_attempts": attempts_used,
                "request_attempt_log": attempt_state.get("attempt_log", []),
            }
        )
    except Exception as exc:  # noqa: BLE001
        record.update(
            {
                "status": "error",
                "error": str(exc),
                "error_details": summarize_exception(exc, stage="task"),
                "exact_match": False,
                "request_attempts": attempt_state.get("attempts_used", args.retries),
                "request_attempt_log": attempt_state.get("attempt_log", []),
            }
        )

    record["duration_seconds"] = round(time.time() - started, 3)
    save_json(run_dir / "tasks" / f"{task_id}.json", record)
    metrics.task_completed(status=record["status"], exact_match=record["exact_match"])
    completed, solved, errors = progress.record(status=record["status"], exact_match=record["exact_match"])
    with PRINT_LOCK:
        bar = render_progress_bar(completed, total)
        print(
            f"{bar} {completed}/{total} solved={solved} errors={errors} "
            f"task={task_id} status={record['status']} exact_match={record['exact_match']} "
            f"attempts={record.get('request_attempts', 0)} duration={record['duration_seconds']:.3f}s",
            flush=True,
        )
    if args.delay:
        time.sleep(args.delay)
    return record


def write_summary(
    run_dir: Path,
    args: argparse.Namespace,
    dataset_dir: Path,
    metrics: LiveMetrics,
) -> dict[str, Any]:
    records = load_task_records(run_dir / "tasks")
    records_sorted = sorted(records, key=lambda item: item.get("task_index", 0))
    solved = sum(1 for item in records_sorted if item.get("exact_match"))
    errors = sum(1 for item in records_sorted if item.get("status") == "error")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "thinking_level": args.thinking_level,
        "dataset_dir": str(dataset_dir),
        "requested_tasks": len(records_sorted),
        "solved_tasks": solved,
        "error_tasks": errors,
        "accuracy": (solved / len(records_sorted)) if records_sorted else 0.0,
        "workers": args.workers,
        "rate_limit_per_minute": args.rate_limit_per_minute,
        "transient_throttle_per_minute": args.transient_throttle_per_minute,
        "request_metrics": metrics.snapshot(),
        "records": [
            {
                "task_id": item.get("task_id"),
                "task_index": item.get("task_index"),
                "status": item.get("status"),
                "exact_match": item.get("exact_match"),
                "duration_seconds": item.get("duration_seconds"),
                "error": item.get("error"),
            }
            for item in records_sorted
        ],
    }
    save_json(run_dir / "summary.json", summary)

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task_index", "task_id", "status", "exact_match", "duration_seconds", "error"],
        )
        writer.writeheader()
        for item in records_sorted:
            writer.writerow(
                {
                    "task_index": item.get("task_index"),
                    "task_id": item.get("task_id"),
                    "status": item.get("status"),
                    "exact_match": item.get("exact_match"),
                    "duration_seconds": item.get("duration_seconds"),
                    "error": item.get("error", ""),
                }
            )
    return summary


def should_stop_submitting(stop_file: Path | None, stop_event: threading.Event | None = None) -> bool:
    if stop_event is not None and stop_event.is_set():
        return True
    return stop_file is not None and stop_file.exists()


def main() -> None:
    args = parse_args()
    api_key, api_source = load_api_key()
    stop_file = args.stop_file.resolve() if args.stop_file else None
    dataset_dir = resolve_dataset_dir(args.dataset_dir.resolve())
    all_task_files = list_task_files(dataset_dir=dataset_dir, task_id=args.task_id, limit=args.limit)
    task_index_map = {task_path.stem: index for index, task_path in enumerate(all_task_files, start=1)}

    if args.resume_run_dir:
        run_dir = args.resume_run_dir.resolve()
        if not run_dir.exists():
            raise SystemExit(f"Resume run directory does not exist: {run_dir}")
        (run_dir / "tasks").mkdir(parents=True, exist_ok=True)
        print(f"Resuming run directory: {run_dir}", flush=True)
    else:
        run_dir = build_run_dir(args.runs_dir.resolve(), args.model, args.thinking_level, len(all_task_files))
        config_snapshot = {
            "model": args.model,
            "dataset_dir": str(dataset_dir),
            "runs_dir": str(args.runs_dir.resolve()),
            "task_id": args.task_id,
            "limit": args.limit,
            "workers": args.workers,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "thinking_level": args.thinking_level,
            "timeout_ms": args.timeout_ms,
            "rate_limit_per_minute": args.rate_limit_per_minute,
            "retries": args.retries,
            "backoff": args.backoff,
            "delay": args.delay,
            "max_in_flight": args.max_in_flight,
            "transient_cooldown_seconds": args.transient_cooldown_seconds,
            "transient_throttle_per_minute": args.transient_throttle_per_minute,
            "stop_file": str(stop_file) if stop_file else None,
            "stop_on_transient_error": args.stop_on_transient_error,
            "api_source": api_source,
        }
        save_json(run_dir / "config.json", config_snapshot)
        print(f"Run directory: {run_dir}", flush=True)

    existing_records = load_task_records(run_dir / "tasks")
    completed_ids = {item.get("task_id") for item in existing_records if item.get("task_id")}
    baseline = summarize_records(existing_records)
    task_files = [task_path for task_path in all_task_files if task_path.stem not in completed_ids]
    existing_transient_errors = [
        {
            "task_id": item.get("task_id"),
            "attempt": item.get("request_attempts"),
            "error": item.get("error"),
            "timestamp_utc": item.get("finished_at_utc") or item.get("updated_at_utc") or datetime.now(timezone.utc).isoformat(),
        }
        for item in existing_records
        if item.get("status") == "error" and isinstance(item.get("error"), str) and is_transient_service_error(item["error"])
    ]

    if args.resume_run_dir:
        resume_snapshot = {
            "model": args.model,
            "dataset_dir": str(dataset_dir),
            "runs_dir": str(args.runs_dir.resolve()),
            "resume_run_dir": str(run_dir),
            "task_id": args.task_id,
            "limit": args.limit,
            "workers": args.workers,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "thinking_level": args.thinking_level,
            "timeout_ms": args.timeout_ms,
            "rate_limit_per_minute": args.rate_limit_per_minute,
            "retries": args.retries,
            "backoff": args.backoff,
            "delay": args.delay,
            "max_in_flight": args.max_in_flight,
            "transient_cooldown_seconds": args.transient_cooldown_seconds,
            "transient_throttle_per_minute": args.transient_throttle_per_minute,
            "stop_file": str(stop_file) if stop_file else None,
            "stop_on_transient_error": args.stop_on_transient_error,
            "api_source": api_source,
            "baseline_completed_tasks": baseline["completed"],
            "baseline_solved_tasks": baseline["solved"],
            "baseline_error_tasks": baseline["errors"],
            "remaining_tasks": len(task_files),
            "total_tasks": len(all_task_files),
        }
        save_json(run_dir / "resume.json", resume_snapshot)

    print(f"Model: {args.model}", flush=True)
    print(f"Thinking level: {args.thinking_level}", flush=True)
    print(f"Dataset: {dataset_dir}", flush=True)
    print(f"Tasks complete already: {baseline['completed']} / {len(all_task_files)}", flush=True)
    print(f"Tasks remaining: {len(task_files)}", flush=True)

    if args.transient_throttle_per_minute <= 0:
        raise SystemExit("--transient-throttle-per-minute must be a positive number.")

    total = len(all_task_files)
    limiter = RateLimiter(
        args.rate_limit_per_minute,
        args.max_in_flight,
        args.transient_cooldown_seconds,
        args.transient_throttle_per_minute,
    )
    stop_event = threading.Event()
    progress = BatchProgress(
        completed=baseline["completed"],
        solved=baseline["solved"],
        errors=baseline["errors"],
    )
    metrics = LiveMetrics(
        run_dir=run_dir,
        total_tasks=total,
        initial_tasks_completed=baseline["completed"],
        initial_tasks_solved=baseline["solved"],
        initial_tasks_errors=baseline["errors"],
        initial_transient_errors=existing_transient_errors,
    )

    if not task_files:
        print("No remaining tasks to process.", flush=True)
    else:
        pending_tasks = iter(task_files)
        in_flight: set[Any] = set()
        max_workers = max(1, args.workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                while len(in_flight) < max_workers and not should_stop_submitting(stop_file, stop_event):
                    try:
                        task_path = next(pending_tasks)
                    except StopIteration:
                        break
                    index = task_index_map[task_path.stem]
                    future = executor.submit(
                        process_task, task_path, index, total, run_dir, args, api_key, limiter, progress, metrics, stop_event
                    )
                    in_flight.add(future)

                if not in_flight:
                    break

                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()

    summary = write_summary(run_dir=run_dir, args=args, dataset_dir=dataset_dir, metrics=metrics)
    print(
        f"Completed {summary['requested_tasks']} task(s). Solved {summary['solved_tasks']}. "
        f"Accuracy {summary['accuracy']:.3%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
