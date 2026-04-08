# Latent Cross-ARC Report

## Scope

This package revisits sparse human ARC data with latent estimates, links ARC-AGI-1 and ARC-AGI-2 through common respondents or common models where possible, and re-runs the solver-structure comparisons with wider coverage than the earlier direct-overlap slice.

## 1. Benchmark Inventory

- ARC-1 train tasks: 400
- ARC-1 eval tasks: 400
- ARC-2 train tasks: 1000
- ARC-2 eval tasks: 120
- ARC-2 eval tasks are larger on average than ARC-1 eval tasks:
  - ARC-1 eval mean input cells: 242.0
  - ARC-2 eval mean input cells: 418.3
- The six shared eval task IDs are not identical across benchmarks:
  - shared eval IDs: 6
  - changed training examples: 6
  - unchanged test examples: 6

## 2. Human Latent Model

- Human attempt rows: 4681
- Sessions: 509
- Tasks with observed human coverage: 442
- Pair-level rows with observed human coverage: 502
- Correctness model diagnostics:
  - AUC: 0.945
  - Log loss: 0.298
  - Brier: 0.088
- Duration model diagnostics:
  - R^2: 0.431
  - MAE on log-seconds: 0.520

Human benchmark slices:

```text
 benchmark_label  row_count  task_count  pair_count  session_count  raw_solve_rate  mean_duration_seconds  mean_latent_human_difficulty  mean_human_difficulty_pc1
    arc1_sidecar       2505         230         230            463        0.828343             175.865792                     -1.622423                  -0.327050
       arc2_eval       1260         115         161            392        0.613492             251.198728                     -0.733850                   0.745402
arc2_train_other        916          97         111            343        0.737991             184.513444                     -1.356415                  -0.108245
```

## 3. Raw vs Latent Stability

The main practical reason to use the latent estimates is stability under sparse coverage. Using all responses with partial pooling gives more reproducible task-level estimates than raw task solve rates.

```text
 benchmark_label  completed_draws  mean_task_count  raw_corr_mean  raw_corr_median  latent_corr_mean  latent_corr_median  raw_corr_ci_lo  raw_corr_ci_hi  latent_corr_ci_lo  latent_corr_ci_hi  latent_minus_raw
    arc1_sidecar              120       220.808333       0.372354         0.372206          0.435945            0.438778        0.283553        0.463972           0.332693           0.513510          0.063591
       arc2_eval              120       103.700000       0.372123         0.370768          0.440587            0.445815        0.263984        0.502504           0.313571           0.548462          0.068464
arc2_train_other              120        92.625000       0.351909         0.360031          0.438733            0.438964        0.235785        0.471408           0.307664           0.549117          0.086825
```

## 4. ARC-1 / ARC-2 LLM Linkage

- Common eval models across both matrices: 22
- Common-model ARC-1 vs ARC-2 accuracy correlation:
  - Pearson: 0.785
  - Spearman: 0.961
- Mean common-model pass rate:
  - ARC-1 eval: 0.407
  - ARC-2 eval: 0.047

This supports a linked scale through common models, but not a “same benchmark, just harder” simplification. ARC-2 eval is much harder, larger on average, more multi-test-pair heavy, and even the shared eval IDs come with revised training examples.

## 5. Human vs LLM Alignment

```text
benchmark_label  matched_task_count  rawsolve_vs_llm_difficulty_pearson  latent_vs_llm_difficulty_pearson  pc1_vs_llm_difficulty_pearson  latent_vs_llm_difficulty_spearman  duration_vs_llm_difficulty_pearson
   arc1_sidecar                 230                            0.296807                          0.309170                       0.362427                           0.331737                            0.314053
      arc2_eval                 115                            0.367866                          0.354763                       0.341393                           0.259781                            0.156729
```

This widens the matched coverage substantially:

- ARC1 sidecar matched tasks: 230
- ARC2 eval matched tasks: 115

## 6. Solver-Structure Revisit

Direct ARC-2 eval human/LLM overlap deltas:

```text
            predictor  n  human_corr  llm_corr  delta_llm_minus_human  delta_ci_lo  delta_ci_hi
cyclomatic_complexity 19    0.188108  0.599640               0.411532     0.026189     0.867258
        structure_pc1 19    0.173397  0.551669               0.378272    -0.055686     0.844309
```

The headline pattern remains the same in this package: solver structure tracks LLM difficulty more strongly than human difficulty on the direct ARC-2 eval overlap.

## 7. Bottom Line

1. The sparse human data are usable without pretending raw item means are enough.
2. Latent task estimates are more stable than raw solve rates on session split-halves.
3. ARC-1 sidecar human coverage gives a much larger matched ARC-1 human/LLM slice than the earlier tiny direct-overlap analyses.
4. ARC-1 and ARC-2 can be linked on a common anchored scale, but they should not be treated as interchangeable without qualification.
5. The shared eval IDs between ARC-1 and ARC-2 preserve test examples but change the training examples, which is exactly the kind of benchmark drift that matters for interpretation.
6. The solver-structure result still looks like “LLM difficulty is more structure-loaded than human difficulty,” not just a fluke of the earlier writeup.
