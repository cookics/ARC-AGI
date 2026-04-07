from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = REPO_ROOT / "data-non-llm" / "raw" / "Compress ARC" / "results_for_the_blog_post" / "predictions_evaluation.npz"
DEFAULT_ORDER_FILE = REPO_ROOT / "data-non-llm" / "raw" / "Compress ARC" / "dataset" / "arc-agi_evaluation_challenges.json"
DEFAULT_TRUTH_DIR = REPO_ROOT / "data-llm" / "ARC-AGI" / "data" / "evaluation"
DEFAULT_OUTPUT = REPO_ROOT / "data-non-llm" / "processed" / "compress_arc_predictions_evaluation_summary.json"

SOURCE_URLS = {
    "npz": "https://github.com/iliao2345/CompressARC/blob/master/results_for_the_blog_post/predictions_evaluation.npz",
    "task_order": "https://github.com/iliao2345/CompressARC/blob/master/dataset/arc-agi_evaluation_challenges.json",
}


def load_task_order(order_file: Path) -> list[str]:
    challenges = json.loads(order_file.read_text(encoding="utf-8"))
    return list(challenges.keys())


def solution_hash_from_local_truth(task_id: str, truth_dir: Path) -> int:
    truth = json.loads((truth_dir / f"{task_id}.json").read_text(encoding="utf-8"))
    outputs = truth.get("test", [])
    solution_tuple = tuple(tuple(tuple(row) for row in pair["output"]) for pair in outputs)
    return hash(solution_tuple)


def score_npz(npz_path: Path, order_file: Path, truth_dir: Path) -> dict:
    task_order = load_task_order(order_file)
    stored = np.load(npz_path, allow_pickle=True)
    contribution_logs = stored["solution_contribution_logs"]
    pick_histories = stored["solution_picks_histories"]

    if len(contribution_logs) != len(task_order):
        raise ValueError(
            f"Task count mismatch: NPZ has {len(contribution_logs)} tasks, task order file has {len(task_order)} tasks."
        )

    true_hashes = [solution_hash_from_local_truth(task_id, truth_dir) for task_id in task_order]

    final_pass1_ids: list[str] = []
    final_pass2_ids: list[str] = []
    ranked_any_ids: list[str] = []
    ranked_pass2_ids: list[str] = []
    ranked_guess_numbers: dict[str, int] = {}

    iterations = int(len(contribution_logs[0])) if len(contribution_logs) else 0

    for task_num, task_id in enumerate(task_order):
        true_hash_full = true_hashes[task_num]
        true_hash_shifted = true_hash_full >> 16

        final_pair = [int(x) for x in pick_histories[task_num][iterations - 1]]
        if final_pair[:1] and final_pair[0] == true_hash_full:
            final_pass1_ids.append(task_id)
        if any(x == true_hash_full for x in final_pair):
            final_pass2_ids.append(task_id)

        scores: dict[int, float] = {}
        for iteration_num in range(iterations):
            for pair_idx in range(2):
                hashed, score = contribution_logs[task_num][iteration_num][pair_idx]
                hashed = int(hashed) >> 16
                original = scores.get(hashed, -10000.0)
                scores[hashed] = float(np.logaddexp(score, original))

        if true_hash_shifted not in scores:
            continue

        ranked_any_ids.append(task_id)
        ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]))
        ordered_keys = [key for key, _ in ordered]
        solution_index = ordered_keys.index(true_hash_shifted)
        guess_number = len(ordered_keys) - solution_index
        ranked_guess_numbers[task_id] = guess_number
        if guess_number <= 2:
            ranked_pass2_ids.append(task_id)

    task_count = len(task_order)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "npz_path": str(npz_path),
            "task_order_path": str(order_file),
            "truth_dir": str(truth_dir),
            "source_urls": SOURCE_URLS,
        },
        "task_count": task_count,
        "iterations": iterations,
        "metrics": {
            "final_pick_pass1": {
                "solved_tasks": len(final_pass1_ids),
                "percentage": len(final_pass1_ids) / task_count * 100,
            },
            "final_pick_pass2": {
                "solved_tasks": len(final_pass2_ids),
                "percentage": len(final_pass2_ids) / task_count * 100,
            },
            "ranked_candidate_solved_anywhere": {
                "solved_tasks": len(ranked_any_ids),
                "percentage": len(ranked_any_ids) / task_count * 100,
            },
            "ranked_candidate_pass2": {
                "solved_tasks": len(ranked_pass2_ids),
                "percentage": len(ranked_pass2_ids) / task_count * 100,
            },
        },
        "task_ids": {
            "final_pick_pass1": final_pass1_ids,
            "final_pick_pass2": final_pass2_ids,
            "ranked_candidate_solved_anywhere": ranked_any_ids,
            "ranked_candidate_pass2": ranked_pass2_ids,
        },
        "ranked_guess_numbers": ranked_guess_numbers,
    }
    return result


def main() -> None:
    result = score_npz(DEFAULT_NPZ, DEFAULT_ORDER_FILE, DEFAULT_TRUTH_DIR)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    metrics = result["metrics"]
    print("CompressARC evaluation summary")
    print()
    print(
        f"- Final top-1: {metrics['final_pick_pass1']['solved_tasks']}/{result['task_count']} "
        f"({metrics['final_pick_pass1']['percentage']:.2f}%)"
    )
    print(
        f"- Final top-2: {metrics['final_pick_pass2']['solved_tasks']}/{result['task_count']} "
        f"({metrics['final_pick_pass2']['percentage']:.2f}%)"
    )
    print(
        f"- Ranked candidate anywhere: {metrics['ranked_candidate_solved_anywhere']['solved_tasks']}/{result['task_count']} "
        f"({metrics['ranked_candidate_solved_anywhere']['percentage']:.2f}%)"
    )
    print(
        f"- Ranked candidate pass@2: {metrics['ranked_candidate_pass2']['solved_tasks']}/{result['task_count']} "
        f"({metrics['ranked_candidate_pass2']['percentage']:.2f}%)"
    )
    print()
    print(f"Wrote {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
