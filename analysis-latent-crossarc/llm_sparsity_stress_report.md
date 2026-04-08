# LLM Sparsity Stress Test

This note asks whether the lower human latent split-half reliability could mostly be a sparse-observation artifact.

## Masking Designs

- `uniform_budget`: randomly keep LLM model-item cells with probability chosen to match the human total observation budget.
- `item_count_matched`: for each task pair, randomly keep exactly as many LLM observations as humans had on that item, capped at the number of available LLM models.
- `session_pattern_matched`: assign LLM models to human sessions at random and only reveal the model responses on the exact items humans attempted, preserving human session lengths and item exposure counts.
- In each case, the same latent split-half recovery pipeline is rerun after masking.

## Dense Baselines

```text
benchmark population  pearson_median  pearson_ci_lo  pearson_ci_hi
     ARC1      human           0.431          0.345          0.516
     ARC1        llm           0.870          0.810          0.901
     ARC2      human           0.416          0.295          0.530
     ARC2        llm           0.856          0.795          0.895
```

## Masked LLM Results

```text
benchmark               mask_kind  completed_sims  human_total_observations  mean_observed_cells  pearson_median  pearson_ci_lo  pearson_ci_hi  spearman_brown_from_median_pearson
     ARC1      item_count_matched            1000                      2505             2040.818           0.662          0.585          0.721                               0.797
     ARC1 session_pattern_matched            1000                      2505             2423.813           0.652          0.578          0.719                               0.790
     ARC1          uniform_budget            1000                      2505             2480.422           0.714          0.639          0.767                               0.833
     ARC2      item_count_matched            1000                      1044             1027.099           0.546          0.384          0.677                               0.706
     ARC2 session_pattern_matched            1000                      1044             1010.828           0.505          0.311          0.655                               0.671
     ARC2          uniform_budget            1000                      1044             1005.459           0.573          0.412          0.701                               0.729
```

## Readout

- If masked LLM reliability stays far above the human baseline, sparse observation counts alone are not enough to explain the human-vs-LLM gap.
- The cleanest comparison is ARC-2, where the human item counts are all below the number of available LLM models, so the `item_count_matched` mask is exact.
