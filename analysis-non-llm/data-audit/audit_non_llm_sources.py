from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data-non-llm" / "raw"
DEFAULT_OUTPUT = REPO_ROOT / "data-non-llm" / "processed" / "non_llm_source_audit.json"

CANONICAL_DIRS = OrderedDict(
    [
        ("arc_agi_1_training", REPO_ROOT / "data-llm" / "ARC-AGI" / "data" / "training"),
        ("arc_agi_1_evaluation", REPO_ROOT / "data-llm" / "ARC-AGI" / "data" / "evaluation"),
        ("arc_agi_2_training", REPO_ROOT / "data-llm" / "ARC-AGI-2" / "data" / "training"),
        ("arc_agi_2_evaluation", REPO_ROOT / "data-llm" / "ARC-AGI-2" / "data" / "evaluation"),
    ]
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(row, list) for row in value)
        and all(all(isinstance(cell, int) for cell in row) for row in value)
    )


def normalize_solution_value(value: Any) -> list[list[list[int]]] | None:
    if is_grid(value):
        return [value]
    if isinstance(value, list) and all(is_grid(item) for item in value):
        return value
    return None


def load_canonical() -> tuple[dict[str, dict[str, list[Any]]], dict[str, set[str]]]:
    canonical: dict[str, dict[str, list[Any]]] = {}
    canonical_ids: dict[str, set[str]] = {}
    for label, folder in CANONICAL_DIRS.items():
        tasks: dict[str, list[Any]] = {}
        for path in sorted(folder.glob("*.json")):
            obj = load_json(path)
            tasks[path.stem] = [pair["output"] for pair in obj.get("test", []) if "output" in pair]
        canonical[label] = tasks
        canonical_ids[label] = set(tasks)
    return canonical, canonical_ids


def infer_label(task_ids: set[str], canonical_ids: dict[str, set[str]]) -> tuple[str | None, str]:
    if not task_ids:
        return None, "empty"

    exact = [label for label, ids in canonical_ids.items() if task_ids == ids]
    if exact:
        return exact[0], "exact"

    subsets: list[tuple[int, str]] = []
    for label, ids in canonical_ids.items():
        overlap = len(task_ids & ids)
        if overlap == len(task_ids):
            subsets.append((len(ids), label))
    if subsets:
        subsets.sort()
        return subsets[0][1], "subset"

    ranked = sorted(((len(task_ids & ids), label) for label, ids in canonical_ids.items()), reverse=True)
    best_overlap, best_label = ranked[0]
    if best_overlap:
        return best_label, "partial"
    return None, "none"


def score_solution_mapping(mapping: dict[str, Any], label: str, canonical: dict[str, dict[str, list[Any]]]) -> dict[str, Any]:
    expected = canonical[label]
    task_total = 0
    task_exact = 0
    output_total = 0
    output_exact = 0
    unparsable = 0
    extra = 0
    mismatched_tasks: list[str] = []

    for task_id, value in mapping.items():
        if task_id not in expected:
            extra += 1
            continue

        pred = normalize_solution_value(value)
        if pred is None:
            unparsable += 1
            mismatched_tasks.append(task_id)
            continue

        gold = expected[task_id]
        task_total += 1
        output_total += len(gold)

        if len(pred) != len(gold):
            output_exact += sum(1 for p, g in zip(pred, gold) if p == g)
            mismatched_tasks.append(task_id)
            continue

        matches = [p == g for p, g in zip(pred, gold)]
        output_exact += sum(matches)
        if all(matches):
            task_exact += 1
        else:
            mismatched_tasks.append(task_id)

    missing_tasks = sorted(set(expected) - set(mapping))
    return {
        "task_exact": task_exact,
        "task_total": task_total,
        "task_accuracy": (task_exact / task_total) if task_total else None,
        "output_exact": output_exact,
        "output_total": output_total,
        "output_accuracy": (output_exact / output_total) if output_total else None,
        "missing_tasks": missing_tasks,
        "extra_tasks": extra,
        "unparsable_tasks": unparsable,
        "mismatched_tasks": sorted(mismatched_tasks),
    }


def make_status(row: dict[str, Any]) -> str:
    kind = row["kind"]
    if kind == "challenge_bundle":
        return "answer_bearing" if row["test_items_with_output"] else "challenge_only"
    if kind == "solution_bundle":
        if row.get("canonical_label") is None:
            return "unscored_no_local_answer_key"
        if row.get("task_exact") == row.get("task_total") and row.get("output_exact") == row.get("output_total"):
            return "matches_canonical_answer_key"
        return "partial_match_to_canonical"
    if kind == "task_directory":
        return "answer_bearing_full_task_files"
    if kind == "sample_submission":
        return "template_submission"
    if kind == "synthetic_task_lists":
        return "answer_bearing_synthetic_pairs"
    return "other"


def audit_json_files(canonical: dict[str, dict[str, list[Any]]], canonical_ids: dict[str, set[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for path in sorted(RAW_ROOT.rglob("*.json")):
        parts = path.relative_to(RAW_ROOT).parts
        rel = path.relative_to(REPO_ROOT).as_posix()

        if parts[:2] == ("VARC", "ARC-AGI") or parts[:2] == ("VARC", "ARC-AGI-2"):
            continue

        obj = load_json(path)
        row: dict[str, Any] = {"path": rel, "kind": None}

        if path.name.endswith(("_challenges.json", "-challenges.json")) and isinstance(obj, dict):
            label, match_type = infer_label(set(obj), canonical_ids)
            test_items = 0
            test_items_with_output = 0
            tasks_with_test_output = 0
            for task in obj.values():
                test_list = task.get("test", []) if isinstance(task, dict) else []
                has_output = any(isinstance(case, dict) and "output" in case for case in test_list)
                if has_output:
                    tasks_with_test_output += 1
                for case in test_list:
                    test_items += 1
                    if isinstance(case, dict) and "output" in case:
                        test_items_with_output += 1
            row.update(
                {
                    "kind": "challenge_bundle",
                    "task_count": len(obj),
                    "canonical_label": label,
                    "match_type": match_type,
                    "test_items": test_items,
                    "tasks_with_test_output": tasks_with_test_output,
                    "test_items_with_output": test_items_with_output,
                }
            )
        elif path.name.endswith(("_solutions.json", "-solutions.json")) and isinstance(obj, dict):
            label, match_type = infer_label(set(obj), canonical_ids)
            row.update(
                {
                    "kind": "solution_bundle",
                    "task_count": len(obj),
                    "canonical_label": label,
                    "match_type": match_type,
                }
            )
            if label:
                row.update(score_solution_mapping(obj, label, canonical))
        elif path.name == "sample_submission.json" and isinstance(obj, dict):
            label, match_type = infer_label(set(obj), canonical_ids)
            attempt_slots = 0
            non_zero_attempts = 0
            for attempts in obj.values():
                if not isinstance(attempts, list):
                    continue
                for attempt in attempts:
                    if not isinstance(attempt, dict):
                        continue
                    for key in ("attempt_1", "attempt_2"):
                        if key in attempt:
                            attempt_slots += 1
                            grid = attempt[key]
                            if is_grid(grid) and any(any(cell != 0 for cell in row_vals) for row_vals in grid):
                                non_zero_attempts += 1
            row.update(
                {
                    "kind": "sample_submission",
                    "task_count": len(obj),
                    "canonical_label": label,
                    "match_type": match_type,
                    "attempt_slots": attempt_slots,
                    "non_zero_attempts": non_zero_attempts,
                }
            )
        elif parts[:2] == ("VARC", "re_arc") and isinstance(obj, list):
            continue
        else:
            row.update(
                {
                    "kind": "other_json",
                    "task_count": len(obj) if hasattr(obj, "__len__") else None,
                }
            )

        row["status"] = make_status(row)
        rows.append(row)

    return rows


def audit_varc_groups(canonical: dict[str, dict[str, list[Any]]], canonical_ids: dict[str, set[str]]) -> list[dict[str, Any]]:
    groups = OrderedDict(
        [
            ("Non-LLM data/raw/VARC/ARC-AGI/data/training", RAW_ROOT / "VARC" / "ARC-AGI" / "data" / "training"),
            ("Non-LLM data/raw/VARC/ARC-AGI/data/evaluation", RAW_ROOT / "VARC" / "ARC-AGI" / "data" / "evaluation"),
            ("Non-LLM data/raw/VARC/ARC-AGI-2/data/training", RAW_ROOT / "VARC" / "ARC-AGI-2" / "data" / "training"),
            ("Non-LLM data/raw/VARC/ARC-AGI-2/data/evaluation", RAW_ROOT / "VARC" / "ARC-AGI-2" / "data" / "evaluation"),
        ]
    )

    rows: list[dict[str, Any]] = []
    for rel, folder in groups.items():
        if not folder.exists():
            continue

        json_files = sorted(folder.glob("*.json"))
        if not json_files:
            continue

        task_outputs: dict[str, list[Any]] = {}
        test_items = 0
        test_items_with_output = 0
        tasks_with_test_output = 0

        for path in json_files:
            obj = load_json(path)
            tests = obj.get("test", [])
            if any("output" in case for case in tests):
                tasks_with_test_output += 1
            outputs = []
            for case in tests:
                test_items += 1
                if "output" in case:
                    test_items_with_output += 1
                    outputs.append(case["output"])
            task_outputs[path.stem] = outputs

        label, match_type = infer_label(set(task_outputs), canonical_ids)
        row: dict[str, Any] = {
            "path": rel.replace("\\", "/"),
            "kind": "task_directory",
            "task_count": len(task_outputs),
            "canonical_label": label,
            "match_type": match_type,
            "test_items": test_items,
            "tasks_with_test_output": tasks_with_test_output,
            "test_items_with_output": test_items_with_output,
        }
        if label:
            row.update(score_solution_mapping(task_outputs, label, canonical))
        row["status"] = make_status(row)
        rows.append(row)

    re_arc_dir = RAW_ROOT / "VARC" / "re_arc" / "tasks"
    if re_arc_dir.exists():
        re_arc_files = sorted(re_arc_dir.glob("*.json"))
        if re_arc_files:
            pair_count = 0
            pairs_with_output = 0
            for path in re_arc_files:
                obj = load_json(path)
                if isinstance(obj, list):
                    pair_count += len(obj)
                    pairs_with_output += sum(1 for item in obj if isinstance(item, dict) and "output" in item)

            rows.append(
                {
                    "path": "Non-LLM data/raw/VARC/re_arc/tasks",
                    "kind": "synthetic_task_lists",
                    "task_count": len(re_arc_files),
                    "pair_count": pair_count,
                    "pairs_with_output": pairs_with_output,
                    "status": "answer_bearing_synthetic_pairs",
                }
            )
    return rows


def build_report() -> dict[str, Any]:
    canonical, canonical_ids = load_canonical()
    entries = audit_json_files(canonical, canonical_ids) + audit_varc_groups(canonical, canonical_ids)
    entries.sort(key=lambda item: item["path"])

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_sets": {
            label: {
                "task_count": len(tasks),
                "test_output_count": sum(len(outputs) for outputs in tasks.values()),
            }
            for label, tasks in canonical.items()
        },
        "entries": entries,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Non-LLM source audit")
    print()
    for entry in report["entries"]:
        line = f"- {entry['path']}: {entry['status']}"
        if entry["kind"] in {"solution_bundle", "task_directory"} and entry.get("output_total"):
            line += (
                f" | task_accuracy={entry['task_exact']}/{entry['task_total']}"
                f" | output_accuracy={entry['output_exact']}/{entry['output_total']}"
            )
        elif entry["kind"] == "challenge_bundle":
            line += (
                f" | tasks={entry['task_count']}"
                f" | test_items_with_output={entry['test_items_with_output']}/{entry['test_items']}"
            )
        elif entry["kind"] == "sample_submission":
            line += (
                f" | tasks={entry['task_count']}"
                f" | non_zero_attempts={entry['non_zero_attempts']}/{entry['attempt_slots']}"
            )
        elif entry["kind"] == "synthetic_task_lists":
            line += f" | pairs_with_output={entry['pairs_with_output']}/{entry['pair_count']}"
        print(line)


def main() -> None:
    report = build_report()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report)
    print()
    print(f"Wrote {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
