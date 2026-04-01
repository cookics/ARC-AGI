# Latent Difficulty vs Solver Complexity

- Rows analyzed: `56`
- Complexity metrics in PCA: `19`

## Explained Variance

- PC1: `0.574` (cumulative `0.574`)
- PC2: `0.175` (cumulative `0.749`)
- PC3: `0.098` (cumulative `0.848`)
- PC4: `0.041` (cumulative `0.888`)
- PC5: `0.030` (cumulative `0.919`)

## Latent Difficulty Correlations

- Best single metric: `ast_node_count` with Pearson `0.666` and Spearman `0.637`
- PC1 score vs latent difficulty: Pearson `0.629`, Spearman `0.593`
- Best PCR model by LOO Pearson: `pcr_5pc` with train Pearson `0.752` and LOO Pearson `0.691`
- Best PLS model by LOO Pearson: `pls_7comp` with train Pearson `0.829` and LOO Pearson `0.671`
- Ridge model: train Pearson `0.862`, LOO Pearson `0.657`, alpha `0.100`

## Component Readings

- PC1 usually reads like overall solver size / structure complexity.
- PC2 usually separates runtime and grid-volume behavior from pure code-size effects.
- PC3 often picks up normalized execution intensity rather than raw solver length.

## Files

- `latent_complexity_pca_explained_variance.csv`
- `latent_complexity_pca_loadings.csv`
- `latent_complexity_pc_correlations.csv`
- `latent_complexity_pcr_models.csv`
- `latent_complexity_pls_models.csv`
- `latent_complexity_single_metric_baseline.csv`
- `latent_complexity_metric_correlations.csv`
- `complexity_with_latent_components.csv`
- `latent_complexity_pca_summary.json`