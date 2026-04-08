# Split-Half Latent Reliability

This note estimates how reproducible the recovered latent item-difficulty axis is when we randomly split respondents/models into two halves and refit the same 1D person-item logistic model in each half.

## Setup

- Each benchmark uses `1000` random split-half simulations.
- Human items must have at least `8` total human attempts to enter the analysis.
- Human ARC-2 uses Public Eval task pairs from the canonical human testing file.
- Human ARC-1 is a sidecar subset: ARC-1 single-pair evaluation tasks reused inside the ARC-AGI-2 Public Train human testing file.
- LLM ARC-1 and ARC-2 use the same item sets as the corresponding human analyses and the local public-eval prediction folders as respondents.
- The reported correlation is the item-difficulty correlation between the two independently fit halves.

## Summary

```text
benchmark population  n_respondents  n_items  completed_sims  pearson_mean  pearson_median  pearson_ci_lo  pearson_ci_hi  spearman_mean  spearman_brown_from_median_pearson
     ARC1      human            463      230            1000         0.431           0.431          0.345          0.516          0.347                               0.602
     ARC1        llm             25      230            1000         0.867           0.870          0.810          0.901          0.843                               0.931
     ARC2      human            385      110            1000         0.416           0.416          0.295          0.530          0.429                               0.588
     ARC2        llm             40      110            1000         0.853           0.856          0.795          0.895          0.717                               0.922
```

## Readout

- Higher values mean the latent item ordering is more stable across random halves of the population.
- The Spearman-Brown column converts the median split-half Pearson correlation into an estimated full-length reliability for the same latent scale.
- ARC-1 human results should be read as a sidecar estimate rather than a dedicated ARC-1 human benchmark.
