from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_OPENAI_MODEL = "gpt-5.4-nano-2026-03-17"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "data-llm" / "ARC-AGI" / "data" / "training"
DEFAULT_RUNS_DIR = SCRIPT_DIR / "runs"
PRINT_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ARC-AGI v1 training tasks through OpenRouter or OpenAI Responses."
    )
    parser.add_argument(
        "--provider",
        choices=("openrouter", "openai"),
        default=os.getenv("ARC_PROVIDER", "openrouter"),
        help="API provider to use for task runs.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing ARC-AGI v1 training JSON files.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory for saving run artifacts.",
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
        default=1,
        help="Number of parallel API calls.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Defaults depend on the provider.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Omit to use the provider default.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Maximum completion tokens.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Maximum attempts per task.",
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
        "--reasoning-effort",
        default=None,
        help='Reasoning effort to request, for example "none", "low", or "medium".',
    )
    parser.add_argument(
        "--service-tier",
        default=None,
        help='OpenAI service tier, for example "flex". Ignored for OpenRouter.',
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Request structured JSON output mode from OpenRouter.",
    )
    parser.add_argument(
        "--semantic-retries",
        type=int,
        default=0,
        help="Extra full task reruns after model-output failures such as length truncation or parse errors.",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        help="Resume an existing run directory and skip already-saved task files.",
    )
    return parser.parse_args()


def load_environment(provider: str) -> str:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(SCRIPT_DIR / ".env")
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY in the environment or .env before running.")
        return api_key

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in the environment or .env before running.")
    return api_key


def slugify_model(model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-").lower()
    return slug or "model"


def list_task_files(dataset_dir: Path, task_id: str | None, limit: int) -> list[Path]:
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
    if task_id:
        task_path = dataset_dir / f"{task_id}.json"
        if not task_path.exists():
            raise SystemExit(f"Task file not found: {task_path}")
        return [task_path]
    task_files = sorted(dataset_dir.glob("*.json"))
    if limit <= 0:
        raise SystemExit("--limit must be a positive integer.")
    return task_files[:limit]


def build_run_dir(runs_dir: Path, model: str, task_count: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{timestamp}_{slugify_model(model)}_{task_count}tasks"
    run_dir = runs_dir / run_name
    (run_dir / "tasks").mkdir(parents=True, exist_ok=False)
    return run_dir


def load_task(task_path: Path) -> dict[str, Any]:
    return json.loads(task_path.read_text(encoding="utf-8"))


def build_messages(task_id: str, task_data: dict[str, Any]) -> list[dict[str, str]]:
    train_pairs = task_data.get("train", [])
    test_pairs = task_data.get("test", [])
    prompt_payload = {
        "task_id": task_id,
        "instructions": [
            "Infer the transformation rule from the training examples.",
            "Predict the output grid for each test input.",
            'Preferred format: {"test": [{"output": [[...]]}]}.',
            "If there is only one test input, a bare grid array is also acceptable.",
            "Do not include markdown fences or explanations.",
        ],
        "train": train_pairs,
        "test": [{"input": pair["input"]} for pair in test_pairs],
    }
    system_prompt = (
        "You are solving an ARC grid transformation task. "
        "Think carefully through the training examples before answering. "
        "Infer the rule silently and return only valid JSON with integer grids."
    )
    user_prompt = json.dumps(prompt_payload, separators=(",", ":"))
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_openai_input(task_id: str, task_data: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    train_pairs = task_data.get("train", [])
    test_pairs = task_data.get("test", [])
    prompt_payload = {
        "task_id": task_id,
        "instructions": [
            "Infer the transformation rule from the training examples.",
            "Predict the output grid for each test input.",
            'Preferred format: {"test": [{"output": [[...]]}]}.',
            "If there is only one test input, a bare grid array is also acceptable.",
            "Do not include markdown fences or explanations.",
        ],
        "train": train_pairs,
        "test": [{"input": pair["input"]} for pair in test_pairs],
    }
    instructions = (
        "You are solving an ARC grid transformation task. "
        "Infer the rule silently and return only valid JSON with integer grids."
    )
    user_input = json.dumps(prompt_payload, separators=(",", ":"))
    request_payload = {
        "instructions": instructions,
        "input": user_input,
    }
    return instructions, user_input, request_payload


def post_with_retries(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    backoff: float,
    reasoning_effort: str,
    json_mode: bool,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "ARC-AGI Nemotron Runner",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "verbosity": "low",
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort, "exclude": True}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
        payload["plugins"] = [{"id": "response-healing"}]

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                response.raise_for_status()
            response.raise_for_status()
            data = response.json()
            data["_request_payload"] = payload
            data["_attempt"] = attempt
            return data
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(backoff * (2 ** (attempt - 1)))
    raise RuntimeError(f"OpenRouter request failed after {retries} attempts: {last_error}")


def post_openai_with_retries(
    api_key: str,
    model: str,
    instructions: str,
    user_input: str,
    max_tokens: int,
    timeout: int,
    retries: int,
    backoff: float,
    reasoning_effort: str,
    service_tier: str,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "ARC-AGI Nano Flex Runner",
    }
    payload = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "service_tier": service_tier,
        "reasoning": {"effort": reasoning_effort},
        "text": {"verbosity": "low"},
        "max_output_tokens": max_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                OPENAI_RESPONSES_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                response.raise_for_status()
            response.raise_for_status()
            data = response.json()
            data["_request_payload"] = payload
            data["_attempt"] = attempt
            return data
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(backoff * (2 ** (attempt - 1)))
    raise RuntimeError(f"OpenAI request failed after {retries} attempts: {last_error}")


def extract_openrouter_content(response_data: dict[str, Any]) -> str:
    choices = response_data.get("choices") or []
    if not choices:
        raise ValueError("Response contained no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        joined = "".join(text_parts).strip()
        if joined:
            return joined
    raise ValueError("Response choice did not contain text content.")


def extract_openai_content(response_data: dict[str, Any]) -> str:
    output_text = response_data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    text_parts: list[str] = []
    for item in response_data.get("output") or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"output_text", "text"}:
                    text = part.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
    joined = "".join(text_parts).strip()
    if joined:
        return joined
    raise ValueError("Response did not contain text content.")


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
    row_length: int | None = None
    for row in grid:
        if not isinstance(row, list):
            raise ValueError("Grid row must be a list.")
        normalized_row = [int(cell) for cell in row]
        if row_length is None:
            row_length = len(normalized_row)
        elif len(normalized_row) != row_length:
            raise ValueError("Grid rows must all have the same length.")
        normalized.append(normalized_row)
    return normalized


def is_grid_like(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(row, list) for row in value)


def coerce_predictions_from_payload(payload: Any, expected_test_count: int) -> list[list[list[int]]]:
    if isinstance(payload, dict) and isinstance(payload.get("test"), list):
        tests = payload["test"]
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

    if isinstance(payload, dict) and "output" in payload and expected_test_count == 1:
        return [normalize_grid(payload["output"])]

    if isinstance(payload, list):
        if expected_test_count == 1 and is_grid_like(payload):
            return [normalize_grid(payload)]
        if len(payload) == expected_test_count and all(isinstance(item, dict) and "output" in item for item in payload):
            return [normalize_grid(item["output"]) for item in payload]
        if len(payload) == expected_test_count and all(is_grid_like(item) for item in payload):
            return [normalize_grid(item) for item in payload]

    raise ValueError("Response payload could not be coerced into ARC test outputs.")


def iter_payload_candidates(text: str) -> list[Any]:
    stripped = strip_code_fences(text)
    candidates: list[Any] = []
    seen_signatures: set[str] = set()

    def add_candidate(value: Any) -> None:
        try:
            signature = json.dumps(value, sort_keys=True)
        except TypeError:
            signature = repr(value)
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            candidates.append(value)

    for parser in (json.loads, ast.literal_eval):
        try:
            add_candidate(parser(stripped))
        except Exception:  # noqa: BLE001
            pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char not in "{[":
            continue
        snippet = stripped[idx:]
        try:
            value, _ = decoder.raw_decode(snippet)
            add_candidate(value)
        except json.JSONDecodeError:
            pass

    bracket_matches = re.finditer(r"\[\s*\[.*", stripped, flags=re.DOTALL)
    for match in bracket_matches:
        snippet = match.group(0)
        for parser in (json.loads, ast.literal_eval):
            try:
                add_candidate(parser(snippet))
                break
            except Exception:  # noqa: BLE001
                continue

    return candidates


def parse_predictions(content: str, expected_test_count: int) -> list[list[list[int]]]:
    stripped = strip_code_fences(content)
    errors: list[str] = []

    try:
        json_text = extract_json_text(stripped)
        payload = json.loads(json_text)
        return coerce_predictions_from_payload(payload, expected_test_count)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))

    for candidate in iter_payload_candidates(stripped):
        try:
            return coerce_predictions_from_payload(candidate, expected_test_count)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    raise ValueError("Unable to parse model output into ARC predictions. Last errors: " + " | ".join(errors[-5:]))


def compare_outputs(task_data: dict[str, Any], predicted_outputs: list[list[list[int]]]) -> tuple[bool, list[bool]]:
    expected_outputs = [normalize_grid(pair["output"]) for pair in task_data.get("test", [])]
    pair_matches = [pred == expected for pred, expected in zip(predicted_outputs, expected_outputs)]
    return all(pair_matches), pair_matches


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_error(error_text: str) -> str:
    lowered = error_text.lower()
    if "429" in lowered or "too many requests" in lowered:
        return "rate_limit"
    if "timeout" in lowered:
        return "timeout"
    if "response choice did not contain text content" in lowered:
        return "empty_content"
    if "unable to parse model output" in lowered or "could not locate a json object" in lowered:
        return "parse"
    if "response payload could not be coerced" in lowered:
        return "parse"
    return "other"


class AdaptiveThrottle:
    def __init__(
        self,
        initial_permits: int,
        min_permits: int = 1,
        cooldown_seconds: float = 30.0,
        recover_after_successes: int = 6,
    ) -> None:
        self.max_permits = max(1, initial_permits)
        self.current_permits = self.max_permits
        self.min_permits = max(1, min_permits)
        self.cooldown_seconds = cooldown_seconds
        self.recover_after_successes = max(1, recover_after_successes)
        self.cooldown_until = 0.0
        self.success_streak = 0
        self.rate_limit_streak = 0

    def can_submit(self, active_count: int) -> bool:
        return active_count < self.current_permits and time.time() >= self.cooldown_until

    def wait_time(self, active_count: int) -> float:
        now = time.time()
        cooldown = max(0.0, self.cooldown_until - now)
        if active_count >= self.current_permits:
            return max(cooldown, 0.5)
        return cooldown

    def record_result(self, record: dict[str, Any]) -> None:
        now = time.time()
        if record.get("error_kind") == "rate_limit":
            self.rate_limit_streak += 1
            self.success_streak = 0
            reduced = max(self.min_permits, max(1, int(self.current_permits * 0.5)))
            self.current_permits = reduced
            cooldown = min(300.0, self.cooldown_seconds * (2 ** min(self.rate_limit_streak - 1, 4)))
            self.cooldown_until = max(self.cooldown_until, now + cooldown)
            return

        if record.get("status") == "ok":
            self.success_streak += 1
            if self.success_streak >= self.recover_after_successes and self.current_permits < self.max_permits:
                self.current_permits += 1
                self.success_streak = 0
            if self.rate_limit_streak > 0 and now >= self.cooldown_until:
                self.rate_limit_streak = max(0, self.rate_limit_streak - 1)


def count_rate_limit_errors(records: list[dict[str, Any]]) -> int:
    return sum(1 for record in records if record.get("error_kind") == "rate_limit")


def get_choice(response_data: dict[str, Any]) -> dict[str, Any]:
    choices = response_data.get("choices") or []
    if not choices:
        return {}
    choice = choices[0]
    return choice if isinstance(choice, dict) else {}


def get_finish_reason(provider: str, response_data: dict[str, Any] | None) -> str | None:
    if not response_data:
        return None
    if provider == "openai":
        status = response_data.get("status")
        if status is not None:
            return str(status)
        incomplete_details = response_data.get("incomplete_details")
        if isinstance(incomplete_details, dict):
            reason = incomplete_details.get("reason")
            if reason is not None:
                return str(reason)
        return None
    choice = get_choice(response_data)
    finish_reason = choice.get("finish_reason")
    return str(finish_reason) if finish_reason is not None else None


def should_retry_semantic(
    provider: str,
    response_data: dict[str, Any] | None,
    error: Exception,
    semantic_attempt: int,
    semantic_retries: int,
) -> bool:
    if semantic_attempt > semantic_retries:
        return False
    finish_reason = get_finish_reason(provider, response_data)
    if provider == "openai":
        if finish_reason == "incomplete":
            return True
        incomplete_details = (response_data or {}).get("incomplete_details")
        if isinstance(incomplete_details, dict) and incomplete_details.get("reason") == "max_output_tokens":
            return True
    else:
        if finish_reason == "length":
            return True
    error_text = str(error).lower()
    retry_markers = [
        "could not locate a json object",
        "unable to parse model output",
        "response choice did not contain text content",
        "response payload could not be coerced",
    ]
    return any(marker in error_text for marker in retry_markers)


def load_existing_records(run_dir: Path) -> list[dict[str, Any]]:
    task_dir = run_dir / "tasks"
    if not task_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(task_dir.glob("*.json")):
        try:
            records.append(load_json(path))
        except json.JSONDecodeError:
            continue
    return records


def process_task(task_path: Path, index: int, total: int, run_dir: Path, args: argparse.Namespace, api_key: str) -> dict[str, Any]:
    started = time.time()
    task_id = task_path.stem
    task_data = load_task(task_path)
    if args.provider == "openai":
        instructions, user_input, request_body = build_openai_input(task_id, task_data)
    else:
        messages = build_messages(task_id, task_data)
        request_body = messages
    record: dict[str, Any] = {
        "task_id": task_id,
        "task_index": index,
        "task_path": str(task_path),
        "provider": args.provider,
        "model": args.model,
        "request_payload": request_body,
        "max_tokens": args.max_tokens,
        "attempt_history": [],
    }
    if args.provider == "openrouter":
        record["temperature"] = args.temperature
    else:
        record["service_tier"] = args.service_tier
        record["reasoning_effort"] = args.reasoning_effort

    last_error: Exception | None = None
    for semantic_attempt in range(1, args.semantic_retries + 2):
        response_data: dict[str, Any] | None = None
        content: str | None = None
        attempt_record: dict[str, Any] = {
            "semantic_attempt": semantic_attempt,
        }
        try:
            if args.provider == "openai":
                response_data = post_openai_with_retries(
                    api_key=api_key,
                    model=args.model,
                    instructions=instructions,
                    user_input=user_input,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    retries=args.retries,
                    backoff=args.backoff,
                    reasoning_effort=args.reasoning_effort,
                    service_tier=args.service_tier,
                )
            else:
                response_data = post_with_retries(
                    api_key=api_key,
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    retries=args.retries,
                    backoff=args.backoff,
                    reasoning_effort=args.reasoning_effort,
                    json_mode=args.json_mode,
                )
            attempt_record["transport_attempt"] = response_data.get("_attempt")
            attempt_record["response"] = response_data
            attempt_record["finish_reason"] = get_finish_reason(args.provider, response_data)
            content = (
                extract_openai_content(response_data)
                if args.provider == "openai"
                else extract_openrouter_content(response_data)
            )
            attempt_record["response_text"] = content
            predicted_outputs = parse_predictions(content, expected_test_count=len(task_data.get("test", [])))
            exact_match, pair_matches = compare_outputs(task_data, predicted_outputs)
            record["attempt_history"].append(attempt_record)
            record.update(
                {
                    "status": "ok",
                    "attempt": response_data.get("_attempt"),
                    "semantic_attempt": semantic_attempt,
                    "response": response_data,
                    "response_text": content,
                    "predicted_outputs": predicted_outputs,
                    "expected_outputs": [pair["output"] for pair in task_data.get("test", [])],
                    "pair_matches": pair_matches,
                    "exact_match": exact_match,
                }
            )
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if response_data is not None:
                attempt_record["response"] = response_data
                attempt_record["transport_attempt"] = response_data.get("_attempt")
                attempt_record["finish_reason"] = get_finish_reason(args.provider, response_data)
            if content:
                attempt_record["response_text"] = content
            attempt_record["error"] = str(exc)
            record["attempt_history"].append(attempt_record)
            if not should_retry_semantic(
                provider=args.provider,
                response_data=response_data,
                error=exc,
                semantic_attempt=semantic_attempt,
                semantic_retries=args.semantic_retries,
            ):
                break

    if last_error is not None:
        error_text = str(last_error)
        record.update(
            {
                "status": "error",
                "error": error_text,
                "error_kind": classify_error(error_text),
                "exact_match": False,
            }
        )
    else:
        record["error_kind"] = None

    record["duration_seconds"] = round(time.time() - started, 3)
    save_json(run_dir / "tasks" / f"{task_id}.json", record)
    with PRINT_LOCK:
        print(
            f"[{index}/{total}] {task_id} status={record['status']} exact_match={record['exact_match']} "
            f"duration={record['duration_seconds']:.3f}s",
            flush=True,
        )
    if args.delay:
        time.sleep(args.delay)
    return record


def write_summary(
    run_dir: Path,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    dataset_dir: Path,
    requested_tasks: int,
) -> None:
    records_sorted = sorted(records, key=lambda item: item["task_index"])
    solved = sum(1 for item in records_sorted if item.get("exact_match"))
    errors = sum(1 for item in records_sorted if item.get("status") == "error")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "dataset_dir": str(dataset_dir),
        "requested_tasks": requested_tasks,
        "completed_tasks": len(records_sorted),
        "solved_tasks": solved,
        "error_tasks": errors,
        "rate_limit_errors": count_rate_limit_errors(records_sorted),
        "accuracy": (solved / len(records_sorted)) if records_sorted else 0.0,
        "workers": args.workers,
        "records": [
            {
                "task_id": item["task_id"],
                "task_index": item["task_index"],
                "status": item["status"],
                "exact_match": item["exact_match"],
                "duration_seconds": item["duration_seconds"],
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
                    "task_index": item["task_index"],
                    "task_id": item["task_id"],
                    "status": item["status"],
                    "exact_match": item["exact_match"],
                    "duration_seconds": item["duration_seconds"],
                    "error": item.get("error", ""),
                }
            )


def main() -> None:
    args = parse_args()
    if args.provider == "openai" and args.model is None:
        args.model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    elif args.provider == "openrouter" and args.model is None:
        args.model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    if args.reasoning_effort is None:
        args.reasoning_effort = "low" if args.provider == "openai" else None
    if args.provider == "openai" and args.service_tier is None:
        args.service_tier = "flex"
    api_key = load_environment(args.provider)
    dataset_dir = args.dataset_dir.resolve()
    task_files = list_task_files(dataset_dir=dataset_dir, task_id=args.task_id, limit=args.limit)
    if args.resume_run_dir:
        run_dir = args.resume_run_dir.resolve()
        (run_dir / "tasks").mkdir(parents=True, exist_ok=True)
    else:
        run_dir = build_run_dir(args.runs_dir.resolve(), args.model, len(task_files))

    existing_records = load_existing_records(run_dir)
    existing_task_ids = {record.get("task_id") for record in existing_records}
    queued_tasks = [
        (index, task_path)
        for index, task_path in enumerate(task_files, start=1)
        if task_path.stem not in existing_task_ids
    ]

    config_snapshot = {
        "provider": args.provider,
        "model": args.model,
        "dataset_dir": str(dataset_dir),
        "runs_dir": str(args.runs_dir.resolve()),
        "task_id": args.task_id,
        "limit": args.limit,
        "workers": args.workers,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "retries": args.retries,
        "semantic_retries": args.semantic_retries,
        "backoff": args.backoff,
        "delay": args.delay,
        "reasoning_effort": args.reasoning_effort,
        "service_tier": args.service_tier,
        "json_mode": args.json_mode,
        "resume_run_dir": str(run_dir) if args.resume_run_dir else None,
        "existing_completed": len(existing_records),
        "remaining_queued": len(queued_tasks),
    }
    if args.provider == "openrouter":
        config_snapshot["temperature"] = args.temperature
    save_json(run_dir / "config.json", config_snapshot)

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Provider: {args.provider}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Dataset: {dataset_dir}", flush=True)
    print(f"Tasks total: {len(task_files)}", flush=True)
    print(f"Tasks already saved: {len(existing_records)}", flush=True)
    print(f"Tasks queued now: {len(queued_tasks)}", flush=True)

    records: list[dict[str, Any]] = list(existing_records)
    total = len(task_files)
    throttle = AdaptiveThrottle(
        initial_permits=max(1, args.workers),
        min_permits=1,
        cooldown_seconds=30.0,
        recover_after_successes=6,
    )
    pending = deque(queued_tasks)
    active: dict[Any, tuple[int, Path]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        while pending or active:
            now = time.time()
            while pending and throttle.can_submit(len(active)):
                index, task_path = pending.popleft()
                future = executor.submit(process_task, task_path, index, total, run_dir, args, api_key)
                active[future] = (index, task_path)
                now = time.time()
                if len(active) >= throttle.current_permits:
                    break

            if not active:
                time.sleep(min(1.0, max(0.1, throttle.wait_time(len(active)))))
                continue

            done, _ = wait(active.keys(), timeout=1.0, return_when=FIRST_COMPLETED)
            if not done:
                continue

            for future in done:
                index, task_path = active.pop(future)
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001
                    record = {
                        "task_id": task_path.stem,
                        "task_index": index,
                        "task_path": str(task_path),
                        "model": args.model,
                        "status": "error",
                        "error": str(exc),
                        "error_kind": classify_error(str(exc)),
                        "exact_match": False,
                        "duration_seconds": 0.0,
                    }
                    save_json(run_dir / "tasks" / f"{task_path.stem}.json", record)
                records.append(record)
                throttle.record_result(record)

    write_summary(
        run_dir=run_dir,
        records=records,
        args=args,
        dataset_dir=dataset_dir,
        requested_tasks=total,
    )
    solved = sum(1 for item in records if item.get("exact_match"))
    print(
        f"Completed {len(records)} task(s). Solved {solved}. Accuracy {solved / len(records):.3%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
