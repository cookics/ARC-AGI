# Latent Difficulty Alignment

This note compares the human item-difficulty scale to a matched latent item-difficulty scale fit on the local model-response matrix for Public Eval.

## Method

- Human difficulty is the `difficulty` estimate from `analysis-human/papers/human-testing/tables/item_summary.csv`.
- LLM difficulty is fit with the same regularized person-plus-item logistic setup, using the `data-llm/arc_agi_v2_public_eval` response matrix.
- Higher values mean harder items on both scales.
- The comparison uses the 161 Public Eval pairs that appear in both the human and model analyses.
- The `>=8` subset below is the same robustness filter used elsewhere in the human report.

## Numbers

| Metric | Value |
| --- | ---: |
| Overlap items | 161 |
| Overlap items with `>=8` human attempts | 110 |
| Pearson correlation, all overlap | 0.317 |
| Spearman correlation, all overlap | 0.329 |
| Kendall tau, all overlap | 0.234 |
| Pearson correlation, `>=8` subset | 0.412 |
| Spearman correlation, `>=8` subset | 0.450 |
| Kendall tau, `>=8` subset | 0.328 |
| Bootstrap 95% CI for Pearson, `>=8` subset | [0.221, 0.577] |
| Top-10 hardest overlap | 5 / 10 |
| Median absolute rank delta, `>=8` subset | 18.5 items |

## Readout

The answer is not "they are the same scale," but they are clearly related. The shared latent structure is moderate, and the item ordering is only partly preserved:

- The latent-difficulty correlation is around 0.41 on the robust subset.
- Rank agreement is moderate rather than tight.
- Human and model progression share a common axis, but there is still a lot of item-specific divergence.
