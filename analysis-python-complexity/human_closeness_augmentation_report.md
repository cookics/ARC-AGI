# Human Closeness Augmentation

## Setup

- Baseline model: `outcome ~ lm_mean`.
- Augmented model: `outcome ~ lm_mean + exact_any + padded cell accuracy + shape IoU + color IoU + component-size IoU + adjacency IoU`.
- By construction, the augmented model cannot fit worse in-sample because it nests the baseline.
- Human latent ease PC1 is the first principal component of standardized `solve_rate`, `-difficulty`, and `-log1p(duration)`.
- Leave-one-out (LOO) results are included as a basic out-of-sample check.

## Headline

- The latent human score improves from `R^2 = 0.122` to `R^2 = 0.200` (`delta = 0.078`, nested-model `p = 0.0249`).
- On LOO validation, the latent human score also improves slightly from `R^2 = 0.101` to `R^2 = 0.113`.
- Human time cost shows the strongest gain: `R^2 = 0.055` to `R^2 = 0.180` with `p = 0.0012`; LOO `R^2` rises from `0.030` to `0.081`.

## Interpretation

- Replacing pass/fail with a single soft metric was mostly disappointing.
- Letting closeness enter as an additive block works better: it preserves the original pass/fail signal and only uses the softer metrics when they explain residual human variation.
- The extra signal appears to be strongest for how long humans take, and weaker for raw solve rate or psychometric difficulty on their own.
