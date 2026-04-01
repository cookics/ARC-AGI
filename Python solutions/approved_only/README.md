# Approved ARC Python Solutions

This folder contains only the `approved` `solution.py` files downloaded from `https://arc.huikang.dev`.

## What Was Verified

- Local ARC task IDs in `Non-LLM data/raw/Minds AI` were checked against the official public repos:
  - ARC-AGI-1 (`fchollet/ARC-AGI`): exact ID match for `400` train and `400` eval tasks.
  - ARC-AGI-2 (`arcprize/ARC-AGI-2`): exact ID match for `1000` train and `120` eval tasks.
- The local `*_challenges.json` files use competition-style formatting and omit `test.output`.
- The paired local `*_solutions.json` files restore those missing test outputs.
- Reconstructed ARC-AGI-1 train/eval and ARC-AGI-2 train task content matched the official public task files.
- ARC-AGI-2 eval had `6` task-content differences versus the current official repo:
  - `4a21e3da`
  - `abc82100`
  - `b6f77b65`
  - `d8e07eb2`
  - `f560132c`
  - `faa9f03d`

## Full Validation Results

All downloaded `solution.py` files were validated by running each `solve(grid)` function against the corresponding official ARC task data.

- Total fetched `solution.py` files: `511`
- Passed: `127`
- Wrong answer: `380`
- Error/crash: `4`
- Timeout: `0`

The site's own status labels explained the gap almost perfectly:

- `approved`: `120 / 120` passed
- `submitted`: `1 / 236` passed
- `attempted`: `5 / 138` passed, `4` crashed
- `skipped`: `1 / 17` passed
- `correct`: `629` tasks had no stored `solution.py`

Conclusion: the site stores many draft or unverified programs. The `approved` subset is the reliable one.

## This Package

This package was built from the site's `approved` status list and contains:

- `solutions/`: the `120` approved Python files
- `approved_task_ids.txt`: plain list of approved task IDs
- `approved_task_files.csv`: task ID to filename mapping
- `approved_task_index.json`: JSON mapping of task ID to filename/status
- `summary.json`: small package summary

## Approved Breakdown

There are `120` unique approved task IDs.

By dataset membership:

- ARC-AGI 1 train: `57 / 400` (`14.25%`)
- ARC-AGI 1 eval: `39 / 400` (`9.75%`)
- ARC-AGI 2 train: `100 / 1000` (`10.0%`)
- ARC-AGI 2 eval: `20 / 120` (`16.67%`)

These counts overlap because ARC-AGI-2 train includes many ARC-AGI-1 tasks.

Overlap breakdown of the `120` approved IDs:

- `57` are in both ARC-AGI 1 train and ARC-AGI 2 train
- `37` are in both ARC-AGI 1 eval and ARC-AGI 2 train
- `18` are only in ARC-AGI 2 eval
- `6` are only in ARC-AGI 2 train
- `2` are in both ARC-AGI 1 eval and ARC-AGI 2 eval

## Supporting Files

The scripts and reports used to build and validate this package live one level up in `Python solutions/`:

- `fetch_arc_python_solutions.py`
- `validate_arc_python_solutions.py`
- `fetch_report.json`
- `validation_report.json`
- `validation_failures.json`
