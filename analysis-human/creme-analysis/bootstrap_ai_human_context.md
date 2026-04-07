# Bootstrap AI vs Human Context

This note puts the AI-vs-human item correlation in context by comparing it to a large reference distribution of human-vs-human split-half correlations.

## Setup

- Restrict to Public Eval task pairs with at least 8 human attempts overall.
- Randomly split human sessions into two halves 5,000 times and compute the item-level correlation each time.
- Bootstrap the AI-vs-human item correlation 8,000 times by resampling task pairs with replacement.

## Human reference

- Human split-half median Pearson correlation: 0.398
- Human split-half 95% interval: [0.275, 0.515]

## AI in context

```text
           series  observed_pearson  bootstrap_median  bootstrap_ci_lo  bootstrap_ci_hi  percentile_vs_human_split
    Average model             0.402             0.405            0.213            0.561                      0.525
Best single model             0.276             0.277            0.097            0.448                      0.026
  Per-pair oracle             0.343             0.345            0.168            0.502                      0.185
```

## Takeaway

- The average-model correlation sits right in the middle of the human split-half distribution, so that aggregate AI profile is genuinely tracking human difficulty structure on this ARC subset.
- The best single model sits much lower in the human split-half distribution, so a single frontier model is still not as human-like as one human subsample is to another.
- This is the cleanest context for the AI correlation: not just `is it above zero?`, but `is it as strong as a noisy human-vs-human benchmark?`
