# Guide

## Main claim

The strongest benchmark-level conclusion from this package is:

`Language models mostly act uniformly across standard benchmarks, but the current factor models do not yet justify a strong claim that we have recovered the true latent structure of their abilities.`

## Main pieces of evidence

1. Correlation matrix:
   Most benchmark pairs are strongly positively correlated.

2. Variance concentration:
   The first component explains about two thirds of variance in the newest benchmark matrix.

3. Holdout benchmark prediction:
   A single common score built from the other benchmarks predicts most held-out benchmarks well.

4. Factor models:
   Better than pure single-g, but too unstable or assumption-sensitive to treat as final proof.

5. Robustness:
   The general result survives ordinary transformations and benchmark holdout checks.

6. Caveats:
   Effective sample size is smaller than the raw model count because of floor effects and family duplication.

## Recommended reading order

1. `paper/report.pdf`
2. `figures/benchmark_correlation_heatmap.png`
3. `figures/holdout_g_prediction.png`
4. `tables/holdout_g_prediction.csv`
5. `tables/web_fit_summary.csv`
6. `notes/completeness_review.md`

## Reproduction

Run:

```powershell
& "C:\Program Files\R\R-4.5.3\bin\Rscript.exe" scripts/reproduce_core_results.R
```

This regenerates the central benchmark-level outputs in `reproduced/`.

The script is intentionally lightweight and uses only base R. It reproduces the core correlation and held-out benchmark analyses without trying to rebuild every historical factor-analysis branch in the original project.
