# ARC-1 DSL Complexity vs Human and LLM Difficulty

- Validated DSL solvers: 400 tasks.
- Validation status counts: {'passed': 391, 'wrong_answer': 8, 'error': 1}
- Human latent difficulty used complete-participant HRC responses; complete vs all-task difficulty correlation = 0.977.

## Headline Correlations (DSL complexity PC1)

- Human latent difficulty (complete participants): Pearson r = 0.193, Spearman rho = 0.175, n = 391.
- GPT-4o failure rate: Pearson r = 0.331, Spearman rho = 0.294, n = 391.
- Claude 3.5 Sonnet failure rate: Pearson r = 0.281, Spearman rho = 0.248, n = 391.
- Pooled GPT+Claude smoothed difficulty: Pearson r = 0.338, Spearman rho = 0.297, n = 391.
- IceCuber failure rate: Pearson r = 0.415, Spearman rho = 0.403, n = 391.

## Best Single Complexity Metrics By Target

- Claude 3.5 Sonnet failure rate: best single metric = `dsl_distinct_function_count_dynamic` (Pearson 0.317, Spearman 0.307, n = 391).
- GPT-4o failure rate: best single metric = `mean_line_length` (Pearson 0.337, Spearman 0.311, n = 391).
- Human latent difficulty (complete participants): best single metric = `prim_branch_count` (Pearson 0.324, Spearman 0.195, n = 50).
- IceCuber failure rate: best single metric = `dsl_complexity_pc1` (Pearson 0.415, Spearman 0.403, n = 391).
- Pooled GPT+Claude smoothed difficulty: best single metric = `dsl_distinct_function_count_dynamic` (Pearson 0.362, Spearman 0.352, n = 391).

## Best Cross-Validated Complexity Models

- Claude 3.5 Sonnet failure rate: best model = `pls_2` (train Pearson 0.542, LOO Pearson 0.342, LOO Spearman 0.326, n = 391).
- GPT-4o failure rate: best model = `pls_2` (train Pearson 0.586, LOO Pearson 0.442, LOO Spearman 0.403, n = 391).
- Human latent difficulty (complete participants): best model = `pls_2` (train Pearson 0.565, LOO Pearson 0.375, LOO Spearman 0.378, n = 391).
- IceCuber failure rate: best model = `pls_1` (train Pearson 0.439, LOO Pearson 0.410, LOO Spearman 0.395, n = 391).
- Pooled GPT+Claude smoothed difficulty: best model = `pls_2` (train Pearson 0.599, LOO Pearson 0.448, LOO Spearman 0.396, n = 391).

## Human vs Other Correlation Differences

- GPT-4o failure rate: Pearson diff = -0.138, Williams p = 0.0252, Spearman diff = -0.119, bootstrap 95% CI = [-0.246, 0.002], bootstrap p = 0.057.
- Claude 3.5 Sonnet failure rate: Pearson diff = -0.088, Williams p = 0.1356, Spearman diff = -0.073, bootstrap 95% CI = [-0.190, 0.042], bootstrap p = 0.222.
- Pooled GPT+Claude smoothed difficulty: Pearson diff = -0.145, Williams p = 0.01387, Spearman diff = -0.122, bootstrap 95% CI = [-0.240, -0.008], bootstrap p = 0.034.
- IceCuber failure rate: Pearson diff = -0.223, Williams p = 0.000358, Spearman diff = -0.227, bootstrap 95% CI = [-0.341, -0.114], bootstrap p = 0.0005.
