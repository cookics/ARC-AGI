# Closeness Model Suite

## What Counts As Straightforward

- Human pair outcomes are sparse and heteroskedastic, so the straightforward models are grouped-binomial for solve counts and weighted regression for continuous item summaries.
- LLM response data support true latent-difficulty fits, so the straightforward sensitivity check is a partial-credit IRT model on discretized closeness scores, not just another correlation on averaged soft metrics.

## Human Side

- Grouped-binomial solve model: pseudo-`R^2` rises from `0.127` to `0.188` (`delta = 0.061`, `p = 0.0018`).
- Weighted latent human-ease model: `R^2` rises from `0.131` to `0.221` (`delta = 0.090`, `p = 0.0097`).
- Weighted duration model: `R^2` rises from `0.044` to `0.183` (`delta = 0.139`, `p = 0.0005`).

## LLM Side

- On the full approved task set, the best external-complexity fit is still `Binary logit difficulty` for `Code complexity PC1` with aligned `r = 0.641`.
- The best partial-credit model on the full task set is `Partial-credit graded location` for `Cyclomatic complexity` with aligned `r = 0.491`.
- Partial-credit regression add-ons do not generalize well on the full 55-56 task set: the nested LLM regressions show positive in-sample deltas but little or negative leave-one-out gain.

## Read

- Human conclusion: yes, there is a clean and more psychometrically sensible way to do this, and it helps.
- LLM conclusion: the proper partial-credit latent models are worth reporting as a sensitivity analysis, but they do not beat the simpler binary logit/Rasch difficulty on the main full-task complexity criteria.
- Best overlap-only partial-credit result: `Partial-credit graded location` reaches aligned `r = 0.748` on `Human overlap solve rate` over `n = 17` tasks.
- Exploratory exception: on the tiny 17-task human-overlap subset, partial-credit latent difficulty can look better, but those gains are not stable enough to treat as a main result.
