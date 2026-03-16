# Findings Manifest

This file lists the internally supported findings carried into the report.

## Standard benchmark findings

1. The latest 203-model, 13-benchmark matrix in `AI Bench more` is strongly dominated by a first general component.
Evidence:
- `tables/cross_dataset_pc1.csv`
- `tables/latest_llm_g_loadings.csv`

2. The one-factor signal is broad rather than being driven by a single benchmark.
Evidence:
- `tables/web_jackknife_generality.csv`

3. Most standard benchmarks load strongly on the general factor; the clear weak exception is hallucination resistance.
Evidence:
- `tables/latest_llm_g_loadings.csv`

4. A multi-factor structure fits better than a pure single-g model, but the better fit is layered on top of a strong shared component rather than replacing it.
Evidence:
- `tables/latest_llm_efa_sweep.csv`
- `tables/web_fit_summary.csv`

5. Rank-based and min-max transformations preserve the benchmark loading structure almost perfectly.
Evidence:
- `tables/web_transform_generality.csv`

6. The general signal collapses only when models are row-standardized, which intentionally removes each model's overall level.
Evidence:
- `tables/web_transform_generality.csv`
- `tables/profile_paradox.csv`

## Expert arena findings

7. Generality is much weaker in the expert occupational arena than in standard benchmarks.
Evidence:
- `tables/cross_dataset_pc1.csv`
- `tables/arena_summary.csv`
- `AI Bench more/LLM_arena/factor_results.txt`

8. In expert tasks, multi-factor structures outperform single-g structures.
Evidence:
- `tables/arena_best_models.csv`

9. Domain-specific variance dominates several expert categories, especially industrial/physical work.
Evidence:
- `AI Bench more/LLM_arena/arena_omega_summary.csv`

## Human comparison findings

10. Classic human intelligence batteries reject a pure unidimensional model but retain a highly stable g structure across samples.
Evidence:
- `tables/classic_fit_summary.csv`
- `tables/classic_g_congruence.csv`

11. The same broad pattern appears in the stored ICAR results: richer structures fit better than a pure single factor.
Evidence:
- `tables/icar_exhaustive_fit_referenced.csv`

## Project organization

12. The historical folders are now indexed chronologically without moving the original files.
Evidence:
- `../PROJECT_INDEX.md`
- `../Report/README.md`
