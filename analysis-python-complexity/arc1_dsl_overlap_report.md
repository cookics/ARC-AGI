# ARC-1 Overlap-Regime Complexity Analysis

## What This Follow-Up Does

- Starts from the saved validated ARC-1 task join rather than rerunning solver validation.
- Uses pair-level LLM solve rates and smoothed pair-level difficulties, not just binary task solved flags.
- Builds shared-regime subsets where pooled GPT+Claude pair success is close to the human solve rate.

## Metric Inventory

- composite: 1 metrics
- dsl_primitive_usage: 152 metrics
- dsl_structure: 14 metrics
- dynamic_execution: 27 metrics
- python_static: 29 metrics
- task_shape: 5 metrics

## Full Set Vs Best Matched Subset

- Full set pooled pair-difficulty: best single metric = `prim_hmirror_count` with Pearson r = 0.404; PC1 = 0.335; human vs pooled pair-difficulty = 0.245.
- Best shared-regime pooled subset = `gap_le_0.30` with n = 91, human mean solve rate = 0.729, pooled GPT+Claude mean pair rate = 0.745, mean rate gap = 0.127.
- In that matched subset, best single metric = `max_dependency_depth` (dsl_structure) with pooled pair-difficulty Pearson r = 0.504.
- Using the same metric on humans gives Pearson r = 0.446; Williams p for pooled-vs-human difference on that metric = 0.2463.
- Human latent difficulty vs pooled pair-level difficulty jumps to Pearson r = 0.853 in the matched subset.

## Matched-Subset Target Snapshot

- Claude pair-level smoothed difficulty: PC1 = 0.325; best single = `max_dependency_depth` with Pearson r = 0.423.
- GPT pair-level smoothed difficulty: PC1 = 0.417; best single = `max_dependency_depth` with Pearson r = 0.450.
- Human latent difficulty: PC1 = 0.349; best single = `max_dependency_depth` with Pearson r = 0.446.
- Pooled GPT+Claude pair-level smoothed difficulty: PC1 = 0.434; best single = `max_dependency_depth` with Pearson r = 0.504.

## Latent-Style Alignment Across Gap Thresholds

- gap_le_0.10: n = 38, human mean = 0.807, pooled mean = 0.816, human vs pooled pair-difficulty = 0.942.
- gap_le_0.15: n = 49, human mean = 0.771, pooled mean = 0.781, human vs pooled pair-difficulty = 0.937.
- gap_le_0.20: n = 68, human mean = 0.765, pooled mean = 0.790, human vs pooled pair-difficulty = 0.894.
- gap_le_0.25: n = 80, human mean = 0.750, pooled mean = 0.778, human vs pooled pair-difficulty = 0.866.
- gap_le_0.30: n = 91, human mean = 0.729, pooled mean = 0.745, human vs pooled pair-difficulty = 0.853.
- gap_le_0.35: n = 116, human mean = 0.708, pooled mean = 0.679, human vs pooled pair-difficulty = 0.810.
- gap_le_0.40: n = 132, human mean = 0.702, pooled mean = 0.631, human vs pooled pair-difficulty = 0.792.
- gap_le_0.45: n = 148, human mean = 0.699, pooled mean = 0.606, human vs pooled pair-difficulty = 0.755.
- gap_le_0.50: n = 180, human mean = 0.708, pooled mean = 0.565, human vs pooled pair-difficulty = 0.674.
