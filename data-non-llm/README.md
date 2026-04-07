# Non-LLM Data

This folder is reserved for non-LLM source data that we want to keep separate from the existing ARC-AGI analysis areas.

## Layout

- `raw/`: Original input files exactly as they are received.
- `processed/`: Cleaned or transformed outputs that are ready for analysis.
- Repo-level analysis scripts for this data now live in `analysis-non-llm/`.

## Conventions

- Keep incoming files in `raw/` so the original source stays untouched.
- Put derived artifacts in `processed/` so downstream work can depend on a stable copy.
- If you need temporary scratch space, add it under a folder that is ignored locally before generating large outputs.

## Shared Loader

Use `analysis-non-llm/data-audit/compare_non_llm_datasets.py` to scan every imported dataset under `raw/` and build a normalized inventory for comparison.

## Stored Evaluation Artifact

The current trusted candidate artifact is CompressARC's ARC-AGI-1 evaluation run:

- Raw NPZ: `raw/Compress ARC/results_for_the_blog_post/predictions_evaluation.npz`
- Task order file: `raw/Compress ARC/dataset/arc-agi_evaluation_challenges.json`
- Local scorer: `analysis-non-llm/data-audit/score_compressarc_predictions.py`
- Saved summary: `processed/compress_arc_predictions_evaluation_summary.json`
