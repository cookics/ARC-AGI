# Latent Cross-ARC Analysis

This folder collects the cross-benchmark latent analyses for sparse human ARC data, ARC-AGI-1 / ARC-AGI-2 linkage, and solver-structure comparisons.

## Scope

The package is built around three goals:

1. Use all available human responses with partial pooling instead of leaning on raw sparse item means.
2. Put ARC-AGI-1 and ARC-AGI-2 on a common anchored scale where possible.
3. Re-run the human / LLM / solver-structure comparisons with wider task coverage and clearer uncertainty summaries.

## Main Inputs

- Human ARC-AGI-2 attempts: [data-human/test_pair_attempts.csv](/C:/Users/cooki/Desktop/ARC-AGI/data-human/test_pair_attempts.csv)
- ARC-AGI-1 task JSONs: [data-llm/ARC-AGI/data](/C:/Users/cooki/Desktop/ARC-AGI/data-llm/ARC-AGI/data)
- ARC-AGI-2 task JSONs: [data-llm/ARC-AGI-2/data](/C:/Users/cooki/Desktop/ARC-AGI/data-llm/ARC-AGI-2/data)
- Exact LLM task matrices reused from the complexity package:
  - [analysis-python-complexity/llm_response_matrix_arc_agi_1_eval.csv](/C:/Users/cooki/Desktop/ARC-AGI/analysis-python-complexity/llm_response_matrix_arc_agi_1_eval.csv)
  - [analysis-python-complexity/llm_response_matrix_arc_agi_2_eval.csv](/C:/Users/cooki/Desktop/ARC-AGI/analysis-python-complexity/llm_response_matrix_arc_agi_2_eval.csv)
- Approved solver complexity table: [analysis-python-complexity/complexity_report.csv](/C:/Users/cooki/Desktop/ARC-AGI/analysis-python-complexity/complexity_report.csv)

## Important Local Limitation

The local workspace contains ARC-AGI-2 human response logs, not a separate standalone ARC-AGI-1 human response table.

To still use ARC-AGI-1 human-linked data well, this package builds an `ARC1 sidecar` subset from ARC-AGI-2 Public Train attempts on tasks whose IDs and single test-pair structure match ARC-AGI-1 evaluation tasks. That gives a linked human ARC-1-style slice without pretending it is a separate dedicated ARC-1 human study.

## Outputs

Running the analysis writes:

- Tables to [analysis-latent-crossarc/tables](/C:/Users/cooki/Desktop/ARC-AGI/analysis-latent-crossarc/tables)
- Figures to [analysis-latent-crossarc/figures](/C:/Users/cooki/Desktop/ARC-AGI/analysis-latent-crossarc/figures)
- Narrative report to [analysis-latent-crossarc/report.md](/C:/Users/cooki/Desktop/ARC-AGI/analysis-latent-crossarc/report.md)
- Headline metrics to [analysis-latent-crossarc/summary.json](/C:/Users/cooki/Desktop/ARC-AGI/analysis-latent-crossarc/summary.json)

## Run

```powershell
python analysis-latent-crossarc/run_analysis.py
```

## Workstreams Covered

1. Inventory ARC-1 / ARC-2 task metadata.
2. Build coverage-aware human latent estimates from all responses.
3. Produce pair-level and task-level human summaries.
4. Compare raw vs latent task stability with split-half checks.
5. Build anchored cross-benchmark LLM summaries on common models.
6. Check whether ARC-1 and ARC-2 can be treated as one scale without qualification.
7. Revisit human-vs-LLM alignment with wider matched task coverage.
8. Expand approved solver-structure rows across benchmark membership.
9. Revisit solver-structure correlations for humans and LLMs.
10. Summarize benchmark differences, overlap quirks, and remaining limitations.
11. Write a compact report inside this folder.
