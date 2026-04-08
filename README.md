# ARC-AGI Project

Project for analyzing and synthesizing ARC-AGI tasks and psychometric properties of LLMs.

## Project Structure

- `data-human/`: Canonical human-testing data and supporting source files.
- `data-llm/`: Canonical ARC task JSONs plus public-eval prediction corpora for LLM analyses.
- `data-non-llm/`: Non-LLM source data, processed summaries, and spreadsheet sidecars.
- `data-python-programs/`: Downloaded ARC Python solution corpora, validation outputs, and approved-only data bundles.
- `analysis-human/`: Human-focused analyses and local paper/report assets.
- `analysis-llm-psychometrics/`: LLM psychometric scripts, figures, tables, export package assets, and local manuscripts.
- `analysis-non-llm/`: Non-LLM audits, psychometric comparisons, and local manuscripts.
- `analysis-python-complexity/`: Complexity, closeness, comparative analyses, and manuscript writeups built on approved Python solutions.
- `analysis-efficiency/`: Cross-source efficiency analysis outputs and local paper assets.
- `papers-literature/`: Literature PDFs and reference papers.
- `notes-walkthroughs/`: Workflow notes and walkthroughs moved out of active analysis folders.
- `.venv/`: Python virtual environment (ignored).

## Recent Activities

- Split canonical data from active analyses so the repo can support verification and paper writing more cleanly.
- Consolidated human, LLM, non-LLM, and Python-program corpora into dedicated root-level data folders.
- Moved active manuscript assets into `papers/` folders inside their corresponding analysis directories.
- Preserved historical walkthroughs and archive material without deleting them.

By: @notcomplex_
