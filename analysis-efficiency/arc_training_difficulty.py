from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_NPZ = REPO_ROOT / "predictions_training.npz"
DEFAULT_DATASET_ROOT = REPO_ROOT / "archive-buildtmp" / "buildtmp" / "CompressARC_tmp" / "dataset"
DEFAULT_OUT_DIR = SCRIPT_DIR / "arc_training_difficulty"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_solution_hashes(solution_map: dict[str, Any]) -> dict[str, int]:
    """Match the repo's hashing convention for solution tuples."""
    hashes: dict[str, int] = {}
    for task_id, solution in solution_map.items():
        solution_tuple = tuple(tuple(tuple(row) for row in grid) for grid in solution)
        hashes[task_id] = hash(solution_tuple)
    return hashes


def first_hit_steps(picks: np.ndarray, true_hashes: list[int], oracle_k: int) -> np.ndarray:
    """
    Return the first logged step where the true solution appears in the top-k picks.

    A value equal to n_steps means the task never hit the oracle within the archive.
    """
    if oracle_k < 1 or oracle_k > picks.shape[-1]:
        raise ValueError(f"oracle_k must be in [1, {picks.shape[-1]}]")

    n_tasks, n_steps, _ = picks.shape
    true_hashes_arr = np.asarray(true_hashes, dtype=np.int64)
    correct = np.any(picks[:, :, :oracle_k] == true_hashes_arr[:, None, None], axis=2)

    first_hit = np.full(n_tasks, n_steps, dtype=np.int32)
    has_hit = correct.any(axis=1)
    first_hit[has_hit] = correct[has_hit].argmax(axis=1)
    return first_hit


def summarize_curve(first_hit: np.ndarray, n_steps: int) -> dict[str, np.ndarray]:
    """Build cumulative curves for all tasks and for the eventually-solved subset."""
    steps = np.arange(n_steps, dtype=np.int32)
    eventual_solved = first_hit < n_steps
    solved_count_all = (first_hit[:, None] <= steps[None, :]).sum(axis=0)
    solved_fraction_all = solved_count_all / float(len(first_hit))

    if eventual_solved.any():
        solved_count_eventual = (first_hit[eventual_solved][:, None] <= steps[None, :]).sum(axis=0)
        solved_fraction_eventual = solved_count_eventual / float(eventual_solved.sum())
    else:
        solved_count_eventual = np.zeros_like(steps, dtype=np.int32)
        solved_fraction_eventual = np.zeros_like(steps, dtype=np.float64)

    return {
        "steps": steps,
        "solved_count_all": solved_count_all,
        "solved_fraction_all": solved_fraction_all,
        "solved_count_eventual": solved_count_eventual,
        "solved_fraction_eventual": solved_fraction_eventual,
        "eventual_solved_mask": eventual_solved,
    }


def normalized_difficulty(first_hit: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Map the first hit step to a [0, 1] difficulty score.

    0.0 means solved immediately at the first logged step.
    1.0 means solved only at the final logged step, or never solved.
    """
    denom = max(n_steps - 1, 1)
    difficulty = first_hit.astype(np.float64) / float(denom)
    difficulty = np.clip(difficulty, 0.0, 1.0)
    return difficulty


def write_difficulty_csv(
    out_path: Path,
    task_ids: list[str],
    first_hit: np.ndarray,
    difficulty: np.ndarray,
    n_steps: int,
    oracle_k: int,
) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_index",
                "task_id",
                "first_hit_step",
                "solved_by_end",
                "difficulty_score",
                "oracle_k",
            ],
        )
        writer.writeheader()
        for idx, task_id in enumerate(task_ids):
            writer.writerow(
                {
                    "task_index": idx,
                    "task_id": task_id,
                    "first_hit_step": int(first_hit[idx]),
                    "solved_by_end": bool(first_hit[idx] < n_steps),
                    "difficulty_score": float(difficulty[idx]),
                    "oracle_k": oracle_k,
                }
            )


def write_curve_csv(out_path: Path, curve: dict[str, np.ndarray]) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "solved_count_all",
                "solved_fraction_all",
                "solved_count_eventual",
                "solved_fraction_eventual",
            ],
        )
        writer.writeheader()
        for i, step in enumerate(curve["steps"]):
            writer.writerow(
                {
                    "step": int(step),
                    "solved_count_all": int(curve["solved_count_all"][i]),
                    "solved_fraction_all": float(curve["solved_fraction_all"][i]),
                    "solved_count_eventual": int(curve["solved_count_eventual"][i]),
                    "solved_fraction_eventual": float(curve["solved_fraction_eventual"][i]),
                }
            )


def write_summary_json(
    out_path: Path,
    task_ids: list[str],
    first_hit: np.ndarray,
    curve: dict[str, np.ndarray],
    oracle_k: int,
    npz_path: Path,
    dataset_root: Path,
) -> None:
    eventual_solved = int((first_hit < len(curve["steps"])).sum())
    summary = {
        "npz_path": str(npz_path),
        "dataset_root": str(dataset_root),
        "oracle_k": oracle_k,
        "n_tasks": len(task_ids),
        "n_steps": int(len(curve["steps"])),
        "eventual_solved": eventual_solved,
        "eventual_unsolved": int(len(task_ids) - eventual_solved),
        "final_solved_all": int(curve["solved_count_all"][-1]),
        "final_solved_fraction_all": float(curve["solved_fraction_all"][-1]),
        "final_solved_fraction_eventual": float(curve["solved_fraction_eventual"][-1]),
        "hash_seed_env": os.environ.get("PYTHONHASHSEED"),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_curve(out_path: Path, curve: dict[str, np.ndarray], oracle_k: int) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(curve["steps"], curve["solved_fraction_eventual"], label="Solved / eventual solved", color="#1f77b4", linewidth=2.0)
    plt.plot(curve["steps"], curve["solved_fraction_all"], label="Solved / all 400 tasks", color="#ff7f0e", linewidth=2.0, linestyle="--")
    plt.xlabel("Logged training step")
    plt.ylabel("Cumulative solved fraction")
    plt.title(f"ARC training cumulative solve curve (oracle top-{oracle_k})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ARC-AGI training solve curves and per-task difficulty scores.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ, help="Path to predictions_training.npz")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the ARC dataset root that contains arc-agi_training_challenges.json and arc-agi_training_solutions.json",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for outputs")
    parser.add_argument("--oracle-k", type=int, default=2, help="Count a step as solved when the true solution is in the top-k picks")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("PYTHONHASHSEED") not in {"0", "1"}:
        print(
            "Warning: PYTHONHASHSEED is not fixed. "
            "This repository uses Python's built-in hash for solution matching, so scores can change across processes."
        )

    challenges_path = args.dataset_root / "arc-agi_training_challenges.json"
    solutions_path = args.dataset_root / "arc-agi_training_solutions.json"

    problems = read_json(challenges_path)
    solutions = read_json(solutions_path)

    task_ids = list(problems.keys())
    true_hash_map = build_solution_hashes(solutions)
    true_hashes = [true_hash_map[task_id] for task_id in task_ids]

    stored = np.load(args.npz, allow_pickle=True)
    picks = stored["solution_picks_histories"]
    if len(picks) != len(task_ids):
        raise ValueError(f"Task count mismatch: NPZ has {len(picks)} tasks, dataset has {len(task_ids)} tasks")

    first_hit = first_hit_steps(picks, true_hashes, args.oracle_k)
    curve = summarize_curve(first_hit, picks.shape[1])
    difficulty = normalized_difficulty(first_hit, picks.shape[1])

    difficulty_csv = args.out_dir / "arc_training_difficulty.csv"
    curve_csv = args.out_dir / "arc_training_curve.csv"
    summary_json = args.out_dir / "arc_training_summary.json"
    curve_png = args.out_dir / "arc_training_curve.png"

    write_difficulty_csv(difficulty_csv, task_ids, first_hit, difficulty, picks.shape[1], args.oracle_k)
    write_curve_csv(curve_csv, curve)
    write_summary_json(summary_json, task_ids, first_hit, curve, args.oracle_k, args.npz, args.dataset_root)
    plot_curve(curve_png, curve, args.oracle_k)

    print(f"Wrote {difficulty_csv}")
    print(f"Wrote {curve_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {curve_png}")


if __name__ == "__main__":
    main()
