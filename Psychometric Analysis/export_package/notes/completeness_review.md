# Completeness Review

This is the final "did we miss anything strong?" check for the benchmark-level question.

## Strong analyses worth keeping

- Correlation matrix / positive manifold
- One-factor variance concentration and loadings
- Held-out benchmark prediction from common benchmark ability
- Single-factor versus richer factor-model comparisons
- Transformation robustness
- Raw-versus-residual profile correlation contrast
- Intrinsic dimensionality
- Guttman / Mokken scalability
- Effective-N / duplication / floor-effect caveats

## Checked, but not central

- Hallucination-shape regression:
  The saved `HALLUCINATION_RESULTS.csv` is weak (`R^2` about `0.10` to `0.11`) and does not add much beyond the simpler conclusion that non-hallucination is the main benchmark outlier.

- Exact subfactor congruence across runs:
  `FACTOR_CONGRUENCE_N140.csv` is unstable and mostly useful as a caution against overclaiming the exact latent architecture.

- Human-data comparisons:
  Dropped on purpose because they are not central to the benchmark-level question in the final note.

## Final judgment

The strongest remaining benchmark-level result beyond the original factor analyses was the holdout benchmark prediction test. That is now included in the paper and should probably be treated as one of the headline findings.
