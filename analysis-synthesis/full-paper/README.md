# Full ARC Master Paper Workspace

This workspace is for the long-form master paper that is meant to absorb the full ARC analysis line in this repo, not just the short synthesis note.

## Purpose

The goal here is to support a genuinely large manuscript that can eventually reach the 50-60 page range by:

- inventorying all relevant analyses across the repo,
- recording specific notes about datasets, methods, outputs, and claims,
- designing a clean section scaffold,
- drafting each major section as its own `.tex` fragment,
- and merging those fragments into one master paper with a small Python helper.

## Layout

- `notes/`: concrete repo inventory and workstream notes
- `scaffold/`: master outline, page-budget plan, and section manifest
- `sections/`: section-level LaTeX drafts
- `scripts/`: helper scripts, including the merge script
- `output/`: generated master paper artifacts
- `references.bib`: shared bibliography for the merged paper

## Current Workflow

1. Read `notes/00_repo_inventory.md`.
2. Read `notes/01_workstream_notes.md`.
3. Read `scaffold/00_master_outline.md`.
4. Expand or revise the section fragments in `sections/`.
5. Run `python scripts/merge_master_paper.py` to generate `output/master_paper.tex`.
6. Optionally run `python scripts/merge_master_paper.py --compile` to build the PDF via `latexmk`.

## Important Framing

This workspace is intentionally broader than the earlier short paper in `analysis-synthesis/synthesis-note/`.
That earlier paper is now best treated as a compact synthesis note.
This `full-paper/` workspace is the real master-manuscript pipeline.
