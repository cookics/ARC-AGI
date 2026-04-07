# Non-LLM Psychometric Analysis

## Setup

- Main comparison space: ARC-AGI-2 Public Eval test pairs, because that is where the human attempt data and the stored LLM prediction corpus overlap.
- Primary threshold: at least 8 human attempts per test pair (`110` pairs).
- Main human benchmark: item-level solve rates, with human split-half correlations used as a reliability reference.
- Main non-LLM sources: TRM ARC-AGI-II evaluator submissions and VARC ARC-2 prediction dumps.
- CompressARC is included only as an ARC-1 sidecar, because it does not overlap the human ARC-2 set.

## Main Findings

- The strongest human-like profile is still the `LLM average`, with Pearson `0.402` and Spearman `0.454` on the well-sampled ARC-2 subset.
- That lands at the `51.6`th percentile of the human split-half distribution; the human split-half median is `0.400` with a 95% interval of `[0.276, 0.514]`.
- The best-score single LLM on this subset is `gpt-5-2-2025-12-11-thinking-xhigh` at pair accuracy `0.491` and human-correlation `0.276`.
- The most human-aligned single LLM is `claude-opus-4-5-20251101-thinking-16k` at Pearson `0.439` and pair accuracy `0.164`.
- The best-score TRM profile is `TRM 651522 pass@2` at pair accuracy `0.109`, but its human-correlation is only `0.113`.
- The most human-aligned TRM profile is `TRM 361957 pass@2` at Pearson `0.214`, still far below the LLM average and below many single LLMs.
- The best VARC profile is `VARC ARC-2_ViT pass@2` at pair accuracy `0.036` and human-correlation `0.158`.

## Interpretation

- On ARC-2, the current non-LLM systems do not reproduce the human difficulty structure nearly as well as the LLM consensus profile does.
- But they are not just trivial copies of the LLM average either: they solve a few human-easy, LLM-hard items and add some orthogonal signal.
- On the primary subset there are `22` human-easy / LLM-hard pairs, and the best non-LLM systems rescue `6` of them.
- TRM is especially interesting because human-alignment peaks in the middle of training (around step `289565`; peak Pearson `0.214`) and then drops as ARC score continues to rise. That suggests optimization is moving the model away from the human item profile, not toward it.
- So the cleanest answer from the current data is: humans and the LLM average share a stronger common difficulty axis than humans and these non-LLM systems do.

## Threshold Sensitivity

- I reran the main item-correlation comparison at thresholds of 2, 3, 5, and 8 human attempts per pair.
- The ranking is stable: the LLM average stays on top, while TRM and VARC remain much lower across thresholds.

```text
 threshold  n_pairs                                system  pair_accuracy  human_pearson  percentile_vs_human_split
         2      161                           LLM average          0.109          0.345                      0.785
         2      161     gpt-5-2-2025-12-11-thinking-xhigh          0.491          0.256                      0.276
         2      161 claude-opus-4-5-20251101-thinking-16k          0.174          0.348                      0.799
         2      161                     TRM 651522 pass@2          0.099          0.090                      0.001
         2      161                     TRM 361957 pass@2          0.050          0.164                      0.023
         2      161                 VARC ARC-2_ViT pass@2          0.043          0.091                      0.001
         3      159                           LLM average          0.109          0.351                      0.807
         3      159     gpt-5-2-2025-12-11-thinking-xhigh          0.484          0.242                      0.210
         3      159 claude-opus-4-5-20251101-thinking-16k          0.176          0.362                      0.852
         3      159                     TRM 651522 pass@2          0.101          0.098                      0.002
         3      159                     TRM 361957 pass@2          0.050          0.171                      0.030
         3      159                 VARC ARC-2_ViT pass@2          0.044          0.096                      0.002
         5      129                           LLM average          0.105          0.367                      0.776
         5      129     gpt-5-2-2025-12-11-thinking-xhigh          0.481          0.231                      0.089
         5      129 claude-opus-4-5-20251101-thinking-16k          0.163          0.403                      0.909
         5      129                     TRM 651522 pass@2          0.093          0.095                      0.000
         5      129                     TRM 361957 pass@2          0.039          0.199                      0.037
         5      129                 VARC ARC-2_ViT pass@2          0.039          0.142                      0.005
         8      110                           LLM average          0.108          0.402                      0.516
         8      110     gpt-5-2-2025-12-11-thinking-xhigh          0.491          0.276                      0.026
         8      110 claude-opus-4-5-20251101-thinking-16k          0.164          0.439                      0.748
         8      110                     TRM 651522 pass@2          0.109          0.113                      0.000
         8      110                     TRM 361957 pass@2          0.045          0.214                      0.002
         8      110                 VARC ARC-2_ViT pass@2          0.036          0.158                      0.000
```

## Feature Pattern

- The LLM average still tracks human-like structure after controlling for raw item features.
- The non-LLM systems are much less tied to the human difficulty gradient and show weaker or stranger feature sensitivities, especially around color-count cues.

```text
                           system feature_label  pearson
                            Human   Input cells    0.124
                            Human  Input colors    0.052
                            Human Output colors    0.097
                            Human Mean duration   -0.349
                      LLM average   Input cells   -0.209
                      LLM average  Input colors    0.127
                      LLM average Output colors    0.087
                      LLM average Mean duration   -0.217
gpt-5-2-2025-12-11-thinking-xhigh   Input cells   -0.204
gpt-5-2-2025-12-11-thinking-xhigh  Input colors    0.004
gpt-5-2-2025-12-11-thinking-xhigh Output colors   -0.022
gpt-5-2-2025-12-11-thinking-xhigh Mean duration   -0.209
                TRM 651522 pass@2   Input cells   -0.018
                TRM 651522 pass@2  Input colors   -0.296
                TRM 651522 pass@2 Output colors   -0.184
                TRM 651522 pass@2 Mean duration    0.028
                TRM 361957 pass@2   Input cells   -0.019
                TRM 361957 pass@2  Input colors   -0.222
                TRM 361957 pass@2 Output colors   -0.149
                TRM 361957 pass@2 Mean duration   -0.105
            VARC ARC-2_ViT pass@2   Input cells    0.022
            VARC ARC-2_ViT pass@2  Input colors   -0.164
            VARC ARC-2_ViT pass@2 Output colors   -0.041
            VARC ARC-2_ViT pass@2 Mean duration   -0.064
```

## ARC-1 Sidecar

- CompressARC is the one clearly valid non-LLM prediction artifact we have on ARC-1 rather than ARC-2.
- Stored ARC-1 scores: final top-1 `18.75%`, final top-2 `20.25%`, ranked-anywhere `34.25%`.
- It is useful as a bona fide non-LLM prediction archive, but it cannot answer the ARC-2 human-vs-LLM question directly.

## Bottom Line

- The best evidence here favors a mixed conclusion: humans and the LLM consensus are measurably similar in item difficulty structure on ARC-2, but the non-LLM systems we have do not currently share that similarity to the same degree.
- Non-LLM systems are not useless or redundant; they add some complementary successes. But in psychometric terms they look more like partial, idiosyncratic alternative solvers than human-like replicas.

## Selected Systems

```text
                               system  pair_accuracy  human_pearson  human_spearman  percentile_vs_human_split  corr_with_human_residual
                   Family Claude Opus          0.130          0.489           0.445                      0.937                     0.027
claude-opus-4-5-20251101-thinking-16k          0.164          0.439           0.426                      0.748                     0.050
                          LLM average          0.108          0.402           0.454                      0.516                    -0.210
                        Family Gemini          0.155          0.368           0.388                      0.305                    -0.177
                           Family GPT          0.145          0.287           0.317                      0.039                    -0.283
    gpt-5-2-2025-12-11-thinking-xhigh          0.491          0.276           0.276                      0.026                    -0.075
                    TRM 361957 pass@2          0.045          0.214           0.203                      0.002                     0.157
                VARC ARC-2_ViT pass@2          0.036          0.158           0.159                      0.000                     0.128
                    TRM 651522 pass@2          0.109          0.113           0.102                      0.000                     0.112
```
