# Additional Human-Side Findings

## Human Difficulty Is Not Just Grid Size

- On well-sampled public-eval pairs (`attempts >= 8`), human difficulty correlates with mean human duration at `r = 0.391`.
- The same human difficulty signal is much weaker for raw input size (`input_cells`): `r = -0.096`.
- More train/test examples slightly raise human difficulty in this public-eval slice: `n_train_pairs r = 0.222`, `n_test_pairs r = 0.209`.

## Human Advantage Over Models Has Its Own Signature

- On well-sampled public-eval pairs, human-vs-average-model gap is larger on bigger boards (`input_cells r = 0.266`).
- The same gap is smaller when tasks expose more train/test examples (`n_train_pairs r = -0.268`, `n_test_pairs r = -0.366`).
- That pattern suggests extra examples may help models close the gap more than they help humans, while larger spatial layouts still favor humans.

## Pair-Level Heterogeneity Matters

- Among 44 public-eval tasks with multiple test pairs, the mean within-task difficulty range is `0.719` logits and the max is `2.498`.
- Mean within-task solve-rate range is `0.185`.
- Task fixed effects explain about `0.749` of pair-level human difficulty variance on the multi-pair subset.
- So task identity matters a lot, but test-pair choice still contributes substantial human-specific variance that task-level solver complexity cannot see.

## Residual Human vs Residual LLM Difficulty

- After controlling for LLM difficulty, residual human difficulty is more connected to human duration than to solver structure.
- After controlling for human difficulty, residual LLM difficulty still aligns with solver structure (`cyclomatic_complexity`, `complexity_pc1_score`) and runtime burden.
- That supports the idea that the two systems are not merely noisy versions of the same latent variable.