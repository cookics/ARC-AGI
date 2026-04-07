# Efficiency comparison report

## Summary
- Latest shared-task human-vs-LLM alignment: Spearman=0.428 (p=<0.001, 95% bootstrap CI [0.253, 0.584])
- Earlier public-eval overlap check: Pearson=0.402 (p=<0.001)
- Within LLMs, performance tracks thinking rank strongly: Spearman=0.805 (p=<0.001)
- Within humans, performance tracks latent ability strongly: Spearman=0.873 (p=<0.001)

## Data inventory
- llm / model_runs: 29 rows, performance=`task_solve_rate / pair_accuracy`, effort=`avg_duration_per_task`
- llm / ARC-2 task rollup: 2736 rows, performance=`task_score`, effort=`task_duration_seconds`
- human / sessions: 509 rows, performance=`solve_rate`, effort=`mean_duration_seconds`
- human / public_eval task pairs: 161 rows, performance=`solve_rate`, effort=`mean_duration_seconds`
- non_llm / Compress ARC: 1 rows, performance=`final pick pass2`, effort=`iterations`
- non_llm / VARC: 4 rows, performance=`pass@1..4 / oracle`, effort=`candidate count / attempt dirs`
- non_llm / TRM progression steps: 10 rows, performance=`pair accuracy / kaggle score`, effort=`step`

## Distribution snapshot
- LLM: performance mean=0.303, median=0.276; duration mean=895.276, median=397.429; cost mean=1.409, median=0.274
- Human: performance mean=0.720, median=0.800; duration mean=242.975, median=198.431; ability mean=0.000, median=0.073
- Non-LLM: performance mean=0.113, median=0.060; primary effort mean=265704.200, median=217174.000; oracle mean=0.168, median=0.075

## Hypotheses entertained
- Shared ARC task difficulty should align across humans, LLMs, and TRM.
- Better performance should usually come with more resource use, but the sign of that tradeoff may differ by source.
- A small number of latent axes should summarize most of the efficiency variation.
- Generic efficiency features should still identify the source family.

## Cross-source alignment
The current data do not support a direct person-level latent correlation between model theta and human ability, because those estimates live on different rows and scales. The cleanest comparison is task-level alignment.
### Shared-task human vs LLM score
- human_solve_rate vs llm_mean_score: Pearson=0.440 (p=<0.001, CI [0.268, 0.581]); Spearman=0.428 (p=<0.001, CI [0.253, 0.584])
### Shared-task human vs LLM duration
- human_solve_rate vs llm_mean_duration_seconds: Pearson=-0.284 (p=0.002, CI [-0.430, -0.123]); Spearman=-0.332 (p=<0.001, CI [-0.481, -0.156])
- human_mean_duration_seconds vs llm_mean_duration_seconds: Pearson=0.149 (p=0.113, CI [-0.087, 0.360]); Spearman=0.112 (p=0.232, CI [-0.073, 0.298])
### Shared-task human vs LLM cost
- human_solve_rate vs llm_mean_cost: Pearson=-0.311 (p=<0.001, CI [-0.460, -0.147]); Spearman=-0.357 (p=<0.001, CI [-0.505, -0.186])
### Shared-task human vs TRM score
- human_solve_rate vs trm_best_task_score: Pearson=0.071 (p=0.449, CI [-0.102, 0.232]); Spearman=0.055 (p=0.562, CI [-0.127, 0.232])
### Public-eval human vs average model
- solve_rate vs lm_mean: Pearson=0.402 (p=<0.001, CI [0.223, 0.560]); Spearman=0.454 (p=<0.001, CI [0.270, 0.625])
### Public-eval human vs best single model
- solve_rate vs lm_best_single_model: Pearson=0.276 (p=0.003, CI [0.099, 0.453]); Spearman=0.276 (p=0.003, CI [0.097, 0.446])
- Shared-task alignment is materially stronger for LLM score than TRM score: delta Spearman=0.371 (95% bootstrap CI [0.119, 0.628])

## Within-source efficiency
### LLM performance vs thinking rank
- performance_rate vs thinking_rank: Spearman=0.805 (p=<0.001, CI [0.591, 0.920])
### LLM duration vs thinking rank
- avg_duration_per_task vs thinking_rank: Spearman=0.815 (p=<0.001, CI [0.619, 0.899])
### LLM performance vs duration
- performance_rate vs avg_duration_per_task: Spearman=0.672 (p=<0.001, CI [0.408, 0.828])
### LLM duration vs cost
- avg_duration_per_task vs avg_cost_per_task: Spearman=0.670 (p=<0.001, CI [0.296, 0.928])
### Human performance vs ability
- performance_rate vs ability: Spearman=0.873 (p=<0.001, CI [0.841, 0.898])
### Human performance vs duration
- performance_rate vs mean_duration_seconds: Spearman=-0.338 (p=<0.001, CI [-0.422, -0.251])
### Human performance vs outfit
- performance_rate vs outfit: Spearman=-0.691 (p=<0.001, CI [-0.748, -0.628])
### TRM performance vs step
- performance_rate vs primary_effort: Spearman=0.957 (p=<0.001, CI [0.752, 1.000])
### TRM performance vs oracle
- performance_rate vs oracle_rate: Spearman=0.985 (p=<0.001, CI [0.911, 1.000])

## Predictive models
- Best CV model for human_solve_rate: geometry_plus_llm_perf with R2=0.093, MAE=0.177, Pearson=0.370, Spearman=0.357
- Best CV model for human_mean_duration_seconds: geometry_only with R2=-0.183, MAE=96.941, Pearson=0.009, Spearman=0.074

Nested OLS comparisons:
- human_solve_rate: geometry_only -> geometry_plus_llm_perf delta R2=0.115, F=6.151, p=0.003
- human_solve_rate: geometry_plus_llm_perf -> geometry_plus_llm_perf_effort delta R2=0.059, F=1.621, p=0.177
- human_solve_rate: geometry_plus_llm_perf_effort -> geometry_plus_llm_plus_trm delta R2=0.007, F=0.724, p=0.397
- human_mean_duration_seconds: geometry_only -> geometry_plus_llm_perf delta R2=0.003, F=0.123, p=0.885
- human_mean_duration_seconds: geometry_plus_llm_perf -> geometry_plus_llm_perf_effort delta R2=0.072, F=1.836, p=0.131
- human_mean_duration_seconds: geometry_plus_llm_perf_effort -> geometry_plus_llm_plus_trm delta R2=0.006, F=0.561, p=0.456

## Latent structure
- llm: PC1 variance=0.390, PC2 variance=0.330
  - pc1 avg_duration_per_task: 0.607
  - pc1 performance_rate: 0.538
  - pc1 avg_cost_per_task: 0.505
  - pc1 avg_total_tokens_per_task: 0.243
- human: PC1 variance=0.492, PC2 variance=0.208
  - pc1 performance_rate: 0.571
  - pc1 ability: 0.537
  - pc1 outfit: -0.420
  - pc1 mean_duration_seconds: -0.339
- non_llm: PC1 variance=0.610, PC2 variance=0.293
  - pc1 oracle_rate: 0.515
  - pc1 secondary_effort: -0.503
  - pc1 performance_rate: 0.475
  - pc1 primary_effort: -0.460
- shared ARC-2 task PCA:
  - PC1 variance=0.395, PC2 variance=0.165
  - PC1 loadings:
    - llm_mean_cost: 0.522
    - llm_mean_duration_seconds: 0.513
    - llm_mean_total_tokens: 0.452
    - llm_mean_score: -0.372
    - human_solve_rate: -0.289
  - PC2 loadings:
    - human_solve_rate: 0.518
    - human_mean_duration_seconds: -0.498
    - human_mean_submissions: -0.493
    - llm_mean_total_tokens: 0.267
    - llm_mean_score: 0.253

## Weak or discarded analyses
- The full-feature source classifier is trivial, because raw telemetry makes the source family obvious.
- The shared 4-feature classifier is the informative one; it still separates sources but not perfectly.
- Geometry-only prediction of human solve rate is weak.
- Adding LLM effort or TRM score after LLM performance does not materially improve the shared-task prediction.
- Human duration is not explained well by geometry or LLM features.
- Compress ARC has only one row, so it is descriptive rather than inferential.
- VARC has only four rows, so correlation claims there are fragile.

## Figures
- C:/Users/cooki/Desktop/ARC-AGI/Psychometric Analysis/analysis/efficiency/figures/source_tradeoff.png
- C:/Users/cooki/Desktop/ARC-AGI/Psychometric Analysis/analysis/efficiency/figures/task_correlation_heatmap.png
- C:/Users/cooki/Desktop/ARC-AGI/Psychometric Analysis/analysis/efficiency/figures/shared_task_latent_map.png
- C:/Users/cooki/Desktop/ARC-AGI/Psychometric Analysis/analysis/efficiency/figures/trm_progression.png