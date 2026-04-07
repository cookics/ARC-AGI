# Stronger Construct Checks

This note pushes the ARC human-vs-model comparison one step further. It still cannot prove shared reasoning, but it can test whether the average-model result is mostly an ensemble phenomenon and whether model families cluster together more tightly than they cluster with humans.

## What we can and cannot test

- We can test item-level alignment, ensemble effects, and cross-family convergence because we have pair-level human correctness and pair-level model correctness on the same Public Eval items.
- We cannot directly test solution-path similarity or wrong-answer similarity, because the human testing file does not contain human grid outputs or action traces.

## Stronger results we do have

- The existing analyses already showed one important difference: the best single model by score (`5.2 xhigh`) is not especially human-aligned (0.276, 2.6th percentile vs the human split-half distribution).
- But if we cherry-pick for alignment instead of score, the most human-aligned single model is `Opus 16k` at 0.439, which lands at the 74.7th percentile of the human split-half distribution. That is interesting, but it is not the same as saying frontier models are generally human-like.
- Across all non-degenerate single models, the median human-correlation is only 0.223, well below the human split-half median of 0.398.

## Ensemble effect

- The observed average-model correlation is 0.402.
- Random one-model draws are usually much lower than that. As ensemble size grows, the median human-correlation rises steadily.
- By the largest tested ensemble size (30 models), the median random-ensemble correlation reaches 0.402, which is almost exactly at the human split-half median.
- That means the strong aggregate result can be reproduced by averaging over many imperfect, partially idiosyncratic models. This is a concrete non-`general intelligence` explanation for why the average-model profile looks so human-aligned.

## Cross-family convergence

- Model-family consensus profiles are fairly correlated with humans, but they are typically even more correlated with each other. The average human-to-family correlation is 0.287, whereas the average family-to-family correlation is 0.544.
- That pattern is what you would expect if there is a shared machine consensus about which items are broadly easy or hard, without needing to assume that the machines and humans are using the same cognitive process.

## Interpretation

- These checks strengthen the skeptical interpretation more than the `same construct` interpretation.
- The average-model benchmark looks real, but much of it is explainable as consensus smoothing across many models rather than a single model cleanly matching human cognition.
- The remaining open possibility is a shared latent difficulty axis that is richer than trivial grid-size cues but still much weaker than psychometric equivalence.

## Top single-model alignments

```text
                                  label        family  pair_accuracy  human_pearson  percentile_vs_human_split
                               Opus 16k   Claude Opus          0.164          0.439                      0.747
                               Opus 64k   Claude Opus          0.291          0.430                      0.696
                               Opus 32k   Claude Opus          0.200          0.351                      0.218
                             Flash high        Gemini          0.245          0.322                      0.116
                                Opus 8k   Claude Opus          0.082          0.321                      0.114
claude-sonnet-4-5-20250929-thinking-32k Claude Sonnet          0.082          0.307                      0.076
                                5.2 med           GPT          0.227          0.306                      0.074
                   gemini-3-pro-preview        Gemini          0.218          0.289                      0.043
                              5.2 xhigh           GPT          0.491          0.276                      0.026
                            5.2 pro med           GPT          0.300          0.273                      0.024
                           5.2 pro high           GPT          0.464          0.273                      0.024
 claude-sonnet-4-5-20250929-thinking-8k Claude Sonnet          0.036          0.246                      0.007
```

## Ensemble size summary

```text
 ensemble_size  n_draws  median_pearson  ci_lo  ci_hi  median_percentile_vs_human_split
             1     3000           0.222  0.022  0.439                             0.003
             2     3000           0.287  0.099  0.453                             0.039
             3     3000           0.319  0.143  0.463                             0.108
             5     3000           0.349  0.210  0.464                             0.207
             8     3000           0.371  0.265  0.464                             0.324
            12     3000           0.382  0.302  0.454                             0.397
            20     3000           0.396  0.344  0.441                             0.488
            30     3000           0.402  0.402  0.402                             0.525
```

## Family summary

```text
       family  n_models  human_pearson  human_spearman  mean_pair_accuracy
  Claude Opus         6          0.489           0.445               0.130
       Gemini         6          0.368           0.388               0.155
Claude Sonnet         5          0.311           0.302               0.042
          GPT        15          0.287           0.317               0.145
 Claude Haiku         5          0.239           0.229               0.015
         Grok         1          0.031           0.025               0.045
```
