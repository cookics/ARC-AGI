# Export Package

This folder is the minimal handoff package for the benchmark-generality note.

It is designed to be copied into a larger project and still make sense on its own.

## What is included

- `paper/`
  - final LaTeX note
  - compiled PDF
- `figures/`
  - the main benchmark-level figures used by the note
- `tables/`
  - the main exported tables behind the note
- `scripts/`
  - a lightweight reproduction script for the core benchmark results
- `data/`
  - the minimal source files and saved result files needed to support the note
- `notes/`
  - a findings manifest and a final completeness review

## What this package is for

This package supports the question:

`Do language models act uniformly across benchmarks, or do they show benchmark-specific abilities?`

The note's answer is:

- mostly uniform across standard benchmarks
- not purely uniform
- factor models are useful descriptively, but not clean enough to be the whole argument

## Fastest entry points

1. `paper/report.pdf`
2. `notes/findings_manifest.md`
3. `notes/completeness_review.md`
4. `scripts/reproduce_core_results.R`
