# Thinking Label Verification

## What Was Checked

The original `thinking_advantage` analysis grouped models using a name-based heuristic. That heuristic had plausible false positives and false negatives, so the label audit now distinguishes:

- `legacy_label`: original heuristic used in the earlier analysis
- `strict_label`: only explicit `thinking-*`, `deep-think`, or `reasoning` names count as Thinking
- `verified_label`: source-backed grouping for the ambiguous cases

See [thinking_label_audit.csv](C:/Users/cooki/Desktop/ARC-AGI/Python%20solutions/approved_only/thinking_label_audit.csv).

## Ambiguous Models and Verified Resolution

- `gemini-3-pro-preview` -> `Thinking`
  Reason: Google states Gemini 2.5 Pro is a thinking model and that thinking cannot be disabled for Pro. Our local model name is `gemini-3-pro-preview`, not `deep-think`, so this was ambiguous from the name alone, but provider-side framing supports treating Pro as a thinking model.
- `gpt-5-pro-2025-10-06` -> `Thinking`
  Reason: OpenAI's pro model docs describe the pro line as supporting reasoning effort settings and advanced reasoning behavior.
- `gpt-5-2-pro-2025-12-11-high` -> `Thinking`
- `gpt-5-2-pro-2025-12-11-medium` -> `Thinking`
  Reason: the `high` and `medium` suffixes are reasoning-effort style budget labels rather than plain base-model names.
- `QwQ-32B-Fireworks` -> `Thinking`
  Reason: Alibaba describes QwQ-32B as a reasoning model.

## What This Does to the Result

Using the provider-backed grouping, which corresponds to the `maximal` schema in [thinking_advantage_sensitivity.csv](C:/Users/cooki/Desktop/ARC-AGI/Python%20solutions/approved_only/thinking_advantage_sensitivity.csv):

- Raw `thinking_advantage` vs LLM difficulty: `r = -0.378`, permutation `p = 0.0045`
- Smoothed-logit `thinking_logit_advantage` vs LLM difficulty: `r = -0.294`, permutation `p = 0.0245`

But the floor diagnostics show why that is not the whole story:

- There are still `0` both-zero items.
- However, there are `35` items where the Standard group has zero successes while the Thinking group has at least one success.
- Restricting to the `21` items where the Standard group is nonzero changes the raw-gap correlation to `r = +0.280` with permutation `p = 0.2321`.
- Restricting further to the `19` items where both groups have interior pass rates makes the raw-gap correlation essentially `0` (`r = -0.009`, permutation `p = 0.9753`) and the logit-gap correlation only mildly negative (`r = -0.156`, permutation `p = 0.5138`).

## Read

The source-backed label audit suggests:

- the earlier negative `thinking_advantage` result is **not** purely an all-zero artifact
- but under the verified grouping it is **substantially driven by standard-group floor effects**
- so the scientifically careful claim is:
  the raw probability-gap version of `thinking_advantage` is not robust enough to support a strong substantive conclusion by itself

That is why the main statistical audit keeps the result in the table but treats it as a bounded-scale quantity that needs label sensitivity and floor diagnostics to interpret safely.
