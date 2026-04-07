# Mini Paper: Closeness Signals, Human Fit, and Human-vs-LLM Similarity

## Abstract

- I revisited the closeness-to-solution idea using models that are more natural for the available data: grouped-binomial and weighted regressions on the human side, and partial-credit IRT as a sensitivity analysis on the LLM side.
- The human-side conclusion is robust: adding closeness features improves the fit to human outcomes under better-specified models.
- The human-vs-LLM overlap conclusion is more limited: humans and LLMs are moderately aligned on difficulty and solve rate, but the overlap set is too small to support many fine-grained superiority or equivalence claims.

## Data And Design

- Human main analyses use the full public-eval pair table: 161 task-pair rows across 115 tasks.
- Direct human-vs-LLM comparisons use the approved ARC-2 overlap table: 17 tasks.
- LLM partial-credit models use discretized closeness categories fit with graded-response and generalized partial-credit models.
- For similarity tests, I treated `|delta r| < 0.15` as a practically small difference and required the 90% bootstrap CI to fall fully inside that band.

## Human-Side Main Result

- Grouped-binomial solve model: pseudo-`R^2` increases from `0.127` to `0.188` (`p = 0.0018`).
- Weighted latent human-ease model: `R^2` increases from `0.131` to `0.221` (`p = 0.0097`).
- Weighted duration model: `R^2` increases from `0.044` to `0.183` (`p = 0.0005`).
- Interpretation: once the human outcomes are modeled in a way that respects sparse counts and unequal sampling, closeness features are not just numerically helpful; they are statistically supported.

## Direct Human-vs-LLM Alignment

- Human difficulty and LLM logit difficulty are moderately aligned: `r = 0.531`, 95% CI `[0.180, 0.781]`, permutation `p = 0.0293`.
- Human solve rate and LLM pass rate are similarly aligned: `r = 0.541`, 95% CI `[0.228, 0.777]`, permutation `p = 0.0260`.
- Human duration remains the least stable target: the strongest closeness-based predictor on the overlap is partial-credit graded difficulty with `r = 0.409`, 95% CI `[0.028, 0.780]`, permutation `p = 0.1025`, and it does not clear a superiority test against binary logit.

## Difference Versus Similarity

- `cyclomatic_complexity` is more LLM-like than human-like: human `r = 0.150`, LLM `r = 0.591`, delta `= 0.441`, 95% CI `[0.012, 0.906]`, `p = 0.0420`.
- `peak_memory_bytes` is more LLM-like than human-like: human `r = -0.570`, LLM `r = -0.159`, delta `= 0.411`, 95% CI `[0.011, 0.833]`, `p = 0.0432`.
- No human-vs-LLM complexity metric cleared the pre-registered practical-similarity rule `|delta r| < 0.15`. The non-significant cases are therefore inconclusive rather than demonstrably similar.
- No overlap outcome showed a significant advantage of partial-credit graded difficulty over binary logit difficulty. The largest numerical gain was for human duration (`delta r = 0.358`), but its 95% CI still crossed zero.

## Power

- `Human-LLM overlap tasks`: with `n = 17`, 80% power requires about `|r| >= 0.634`.
- `Full approved LLM tasks (n=55)`: with `n = 55`, 80% power requires about `|r| >= 0.370`.
- `Full approved LLM tasks (n=56)`: with `n = 56`, 80% power requires about `|r| >= 0.367`.
- `Human pair-level analysis`: with `n = 161`, 80% power requires about `|r| >= 0.219`.
- The overlap sample therefore only has conventional power for large effects. For example, the observed human-vs-LLM difficulty alignment (`r = 0.531`) would need about `n = 26` overlap tasks for 80% power, not `n = 17`.
- The same applies to the partial-credit duration signal: its observed overlap effect would need roughly `n = 45` tasks for 80% power.
- Difference tests between two correlated predictors are even less powered than single-correlation tests, so the overlap slice is suitable for strong directional signals but not for fine-grained equivalence claims.

## Bottom Line

- Main paper result: on the human side, closeness-to-solution adds statistically defensible signal once the model respects the data-generating structure.
- Sensitivity result: on the full LLM task set, partial-credit latent models do not beat the simpler binary logit/Rasch difficulty summaries on external complexity criteria.
- Human-vs-LLM comparison result: humans and LLMs are significantly aligned on overlap difficulty and solve rate, but they are not yet similar enough, with this sample, to support equivalence claims.
- The clearest human-vs-LLM differences are that cyclomatic complexity and memory burden behave more like LLM difficulty signals than human difficulty signals in the overlap slice.