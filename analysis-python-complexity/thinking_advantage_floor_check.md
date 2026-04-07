# Thinking Advantage Floor Check

## Definition

- `thinking_advantage` = `pass_rate_thinking - pass_rate_standard`
- `thinking_logit_advantage` = smoothed log-odds difference between thinking-model and standard-model pass rates

## Floor-Effect Question

Question: does the negative correlation between thinking advantage and item difficulty happen only because hard items drive both model groups to zero, forcing the gap to disappear?

## Diagnostic Results

- Approved eval rows analyzed: `55`
- Items where both thinking and standard models had zero successes: `0`
- Items where thinking models had zero successes: `0`
- Items where standard models had zero successes but thinking models had at least one success: `21`

So the negative relationship is **not** being driven by a pile of all-zero items.

## Correlations

- All items:
  - `thinking_advantage` vs `logit_difficulty_all`: `r = -0.678`
  - `thinking_logit_advantage` vs `logit_difficulty_all`: `r = -0.669`
- Excluding any item where standard models had zero successes:
  - `thinking_advantage` vs `logit_difficulty_all`: `r = -0.524`
  - `thinking_logit_advantage` vs `logit_difficulty_all`: `r = -0.648`
- Restricting to items where both groups had interior pass rates (`0 < rate < 1` for both):
  - `thinking_advantage` vs `logit_difficulty_all`: `r = -0.831`
  - `thinking_logit_advantage` vs `logit_difficulty_all`: `r = -0.866`
- Restricting further to items where both groups were in the `0.1` to `0.9` range:
  - `thinking_advantage` vs `logit_difficulty_all`: `r = -0.927`
  - `thinking_logit_advantage` vs `logit_difficulty_all`: `r = -0.923`

These restrictions remove the raw floor/ceiling problem rather than creating it, and the negative pattern gets stronger.

## Alternative Modeling Structure

I also fit a grouped binomial-logit model at the item-by-model-type level:

`success ~ model_type + difficulty + model_type:difficulty`

Using `logit_difficulty_all` as the item difficulty covariate:

- Full set interaction coefficient: `-0.718`, `p = 5.7e-06`
- Excluding all items with zero standard-model successes: interaction coefficient `-1.043`, `p = 2.5e-10`

Interpretation: on the log-odds scale, the thinking-model advantage still declines as difficulty increases, even when the obvious floor cases are removed.

## Read

The simplest interpretation is not “hard items force both groups to zero.” It is:

- thinking models gain the most over standard models on medium-hard items
- on the very hardest items, both groups struggle and the extra gain from thinking shrinks
- that shrinking gap survives floor-resistant modeling, so it looks like a real pattern rather than a trivial bounded-scale artifact
