# Non-LLM Data

This folder is reserved for non-LLM source data that we want to keep separate from the existing ARC-AGI analysis areas.

## Layout

- `raw/`: Original input files exactly as they are received.
- `processed/`: Cleaned or transformed outputs that are ready for analysis.
- `analysis/`: Scripts, notebooks, or notes used to work with this data.

## Conventions

- Keep incoming files in `raw/` so the original source stays untouched.
- Put derived artifacts in `processed/` so downstream work can depend on a stable copy.
- If you need temporary scratch space, add it under a folder that is ignored locally before generating large outputs.

