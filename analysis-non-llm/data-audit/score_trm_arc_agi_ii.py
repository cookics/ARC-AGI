from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_ROOT = REPO_ROOT / "data-non-llm" / "raw" / "TRM-ARC-AGI-II"
TRUTH_DIR = REPO_ROOT / "data-llm" / "ARC-AGI-2" / "data" / "evaluation"
DEFAULT_OUTPUT = REPO_ROOT / "data-non-llm" / "processed" / "trm_arc_agi_ii_progression.json"


def load_truth() -> dict[str, list[list[list[int]]]]:
    truth: dict[str, list[list[list[int]]]] = {}
    for path in sorted(TRUTH_DIR.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        truth[path.stem] = [pair["output"] for pair in obj.get("test", [])]
    return truth


def score_submission(submission_path: Path, truth: dict[str, list[list[list[int]]]]) -> dict:
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    total_task_score = 0.0
    total_pairs = 0
    pair1_correct = 0
    pair2_correct = 0
    pass1_tasks: list[str] = []
    pass2_tasks: list[str] = []
    task_fractional_scores: dict[str, float] = {}

    for task_id, gold_pairs in truth.items():
        pred_entries = submission.get(task_id, [])
        task_pair_correct = 0
        pass1_ok = True
        pass2_ok = True

        for pair_index, gold_grid in enumerate(gold_pairs):
            total_pairs += 1
            pred_entry = pred_entries[pair_index] if pair_index < len(pred_entries) and isinstance(pred_entries[pair_index], dict) else {}
            attempt_1 = pred_entry.get("attempt_1")
            attempt_2 = pred_entry.get("attempt_2")
            a1_ok = attempt_1 == gold_grid
            a2_ok = attempt_2 == gold_grid

            if a1_ok:
                pair1_correct += 1
            else:
                pass1_ok = False

            if a1_ok or a2_ok:
                pair2_correct += 1
                task_pair_correct += 1
            else:
                pass2_ok = False

        task_score = task_pair_correct / len(gold_pairs)
        total_task_score += task_score
        task_fractional_scores[task_id] = task_score

        if pass1_ok:
            pass1_tasks.append(task_id)
        if pass2_ok:
            pass2_tasks.append(task_id)

    task_count = len(truth)
    return {
        "step": int(submission_path.parent.name.split("_")[-1]),
        "folder": submission_path.parent.name,
        "submission_path": str(submission_path),
        "task_count": task_count,
        "pair_count": total_pairs,
        "kaggle_score": total_task_score / task_count * 100,
        "pass1_task_solved": {
            "count": len(pass1_tasks),
            "percentage": len(pass1_tasks) / task_count * 100,
            "task_ids": pass1_tasks,
        },
        "pass2_task_solved": {
            "count": len(pass2_tasks),
            "percentage": len(pass2_tasks) / task_count * 100,
            "task_ids": pass2_tasks,
        },
        "pair_accuracy": {
            "attempt_1_correct": pair1_correct,
            "attempt_1_percentage": pair1_correct / total_pairs * 100,
            "attempt_1_or_2_correct": pair2_correct,
            "attempt_1_or_2_percentage": pair2_correct / total_pairs * 100,
        },
        "task_fractional_scores": task_fractional_scores,
    }


def build_summary() -> dict:
    truth = load_truth()
    submissions = sorted(
        SUBMISSION_ROOT.glob("evaluator_ARC_step_*/submission.json"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    results = [score_submission(path, truth) for path in submissions]

    best_kaggle = max(results, key=lambda item: item["kaggle_score"])
    best_pass2 = max(results, key=lambda item: item["pass2_task_solved"]["count"])

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "submission_root": str(SUBMISSION_ROOT),
        "truth_dir": str(TRUTH_DIR),
        "results": results,
        "best": {
            "kaggle_score": {
                "step": best_kaggle["step"],
                "folder": best_kaggle["folder"],
                "score": best_kaggle["kaggle_score"],
            },
            "pass2_task_solved": {
                "step": best_pass2["step"],
                "folder": best_pass2["folder"],
                "count": best_pass2["pass2_task_solved"]["count"],
                "percentage": best_pass2["pass2_task_solved"]["percentage"],
            },
        },
    }


def main() -> None:
    summary = build_summary()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("TRM-ARC-AGI-II progression")
    print()
    for result in summary["results"]:
        print(
            f"- {result['folder']}: kaggle={result['kaggle_score']:.4f}% | "
            f"pass1_tasks={result['pass1_task_solved']['count']}/{result['task_count']} "
            f"({result['pass1_task_solved']['percentage']:.2f}%) | "
            f"pass2_tasks={result['pass2_task_solved']['count']}/{result['task_count']} "
            f"({result['pass2_task_solved']['percentage']:.2f}%) | "
            f"pair2={result['pair_accuracy']['attempt_1_or_2_correct']}/{result['pair_count']} "
            f"({result['pair_accuracy']['attempt_1_or_2_percentage']:.2f}%)"
        )

    best = summary["best"]
    print()
    print(
        f"Best kaggle-style score: {best['kaggle_score']['folder']} "
        f"({best['kaggle_score']['score']:.4f}%)"
    )
    print(
        f"Best strict pass@2 task solve count: {best['pass2_task_solved']['folder']} "
        f"({best['pass2_task_solved']['count']}/{summary['results'][0]['task_count']}, "
        f"{best['pass2_task_solved']['percentage']:.2f}%)"
    )
    print()
    print(f"Wrote {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
