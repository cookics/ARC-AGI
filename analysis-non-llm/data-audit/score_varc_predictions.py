from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_ROOT = REPO_ROOT / "data-non-llm" / "raw" / "VARC_predictions" / "VARC_predictions"
TRUTH_DIRS = {
    "ARC-1": REPO_ROOT / "data-llm" / "ARC-AGI" / "data" / "evaluation",
    "ARC-2": REPO_ROOT / "data-llm" / "ARC-AGI-2" / "data" / "evaluation",
}
DEFAULT_OUTPUT = REPO_ROOT / "data-non-llm" / "processed" / "varc_predictions_summary.json"


def load_truth(truth_dir: Path) -> dict[str, list[list[list[int]]]]:
    truth: dict[str, list[list[list[int]]]] = {}
    for path in sorted(truth_dir.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        truth[path.stem] = [pair["output"] for pair in obj.get("test", [])]
    return truth


def score_model_dir(model_dir: Path, gold: dict[str, list[list[list[int]]]]) -> dict:
    attempts = sorted([path for path in model_dir.iterdir() if path.is_dir()], key=lambda path: path.name)
    predictions: dict[str, dict[str, dict[str, list[list[list[int]]]]]] = {}
    task_ids: set[str] = set()
    candidate_lengths: list[int] = []

    for attempt_dir in attempts:
        attempt_predictions: dict[str, dict[str, list[list[list[int]]]]] = {}
        for path in sorted(attempt_dir.glob("*.json")):
            task_id = path.stem.replace("_predictions", "")
            obj = json.loads(path.read_text(encoding="utf-8"))
            attempt_predictions[task_id] = obj
            task_ids.add(task_id)
            for value in obj.values():
                if isinstance(value, list):
                    candidate_lengths.append(len(value))
        predictions[attempt_dir.name] = attempt_predictions

    pass_metrics: dict[str, dict[str, float | int]] = {}
    for guess_count in range(1, len(attempts) + 1):
        chosen_attempts = attempts[:guess_count]
        solved_tasks = 0
        solved_pairs = 0
        total_pairs = 0

        for task_id, gold_pairs in gold.items():
            task_ok = True
            for pair_index, gold_grid in enumerate(gold_pairs):
                total_pairs += 1
                pair_key = str(pair_index)
                pair_ok = False
                for attempt_dir in chosen_attempts:
                    obj = predictions[attempt_dir.name].get(task_id)
                    if not obj:
                        continue
                    candidates = obj.get(pair_key)
                    if isinstance(candidates, list) and candidates and candidates[0] == gold_grid:
                        pair_ok = True
                        break
                if pair_ok:
                    solved_pairs += 1
                else:
                    task_ok = False
            if task_ok:
                solved_tasks += 1

        pass_metrics[f"pass@{guess_count}"] = {
            "solved_tasks": solved_tasks,
            "task_percentage": solved_tasks / len(gold) * 100,
            "solved_pairs": solved_pairs,
            "pair_percentage": solved_pairs / total_pairs * 100,
        }

    pool_solved_tasks = 0
    pool_solved_pairs = 0
    pool_total_pairs = 0
    for task_id, gold_pairs in gold.items():
        task_ok = True
        for pair_index, gold_grid in enumerate(gold_pairs):
            pool_total_pairs += 1
            pair_key = str(pair_index)
            pair_ok = False
            for attempt_dir in attempts:
                obj = predictions[attempt_dir.name].get(task_id)
                if not obj:
                    continue
                candidates = obj.get(pair_key)
                if isinstance(candidates, list) and any(candidate == gold_grid for candidate in candidates):
                    pair_ok = True
                    break
            if pair_ok:
                pool_solved_pairs += 1
            else:
                task_ok = False
        if task_ok:
            pool_solved_tasks += 1

    split = "ARC-1" if model_dir.name.startswith("ARC-1") else "ARC-2" if model_dir.name.startswith("ARC-2") else "unknown"
    return {
        "model_dir": str(model_dir),
        "split": split,
        "attempt_dirs": [attempt_dir.name for attempt_dir in attempts],
        "task_count": len(task_ids),
        "truth_task_count": len(gold),
        "exact_task_match": task_ids == set(gold),
        "candidate_count_stats": {
            "min": min(candidate_lengths) if candidate_lengths else None,
            "max": max(candidate_lengths) if candidate_lengths else None,
            "unique": sorted(set(candidate_lengths)),
        },
        "metrics": pass_metrics,
        "candidate_pool_oracle": {
            "solved_tasks": pool_solved_tasks,
            "task_percentage": pool_solved_tasks / len(gold) * 100,
            "solved_pairs": pool_solved_pairs,
            "pair_percentage": pool_solved_pairs / pool_total_pairs * 100,
        },
    }


def build_summary() -> dict:
    truth = {label: load_truth(path) for label, path in TRUTH_DIRS.items()}
    results = []
    for model_dir in sorted([path for path in PREDICTION_ROOT.iterdir() if path.is_dir()], key=lambda path: path.name):
        split = "ARC-1" if model_dir.name.startswith("ARC-1") else "ARC-2" if model_dir.name.startswith("ARC-2") else None
        if split is None:
            continue
        results.append(score_model_dir(model_dir, truth[split]))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_root": str(PREDICTION_ROOT),
        "truth_dirs": {key: str(value) for key, value in TRUTH_DIRS.items()},
        "notes": {
            "pass_metric_definition": "pass@N uses the first candidate grid from attempt_0 through attempt_(N-1) as sequential guesses for each test pair.",
            "candidate_pool_oracle_definition": "candidate_pool_oracle checks whether the true grid appears anywhere in any stored candidate list across all attempt folders.",
        },
        "results": results,
    }


def main() -> None:
    summary = build_summary()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("VARC prediction summary")
    print()
    for result in summary["results"]:
        print(result["model_dir"])
        print(f"- split: {result['split']}")
        print(f"- coverage: {result['task_count']}/{result['truth_task_count']} exact_match={result['exact_task_match']}")
        print(f"- candidates per pair: {result['candidate_count_stats']['unique']}")
        for metric_name, metric in result["metrics"].items():
            print(
                f"- {metric_name}: {metric['solved_tasks']}/{result['truth_task_count']} "
                f"({metric['task_percentage']:.2f}%)"
            )
        oracle = result["candidate_pool_oracle"]
        print(f"- candidate_pool_oracle: {oracle['solved_tasks']}/{result['truth_task_count']} ({oracle['task_percentage']:.2f}%)")
        print()

    print(f"Wrote {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
