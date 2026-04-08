# Results

## Reliability Context and Expected Magnitudes

Our primary ARC-2 analysis uses `110` public-evaluation test pairs with at least `8` human attempts each. Human split-half reliability on this subset is moderate rather than high (median Pearson `0.398`, 95% interval `[0.275, 0.515]`), which implies a Spearman-Brown full-length reliability of `0.569` and an approximate raw-correlation ceiling of `0.755` for any perfect latent predictor measured against these noisy item solve rates. That matters because raw correlations in the `0.2` to `0.4` range are not automatically trivial on this benchmark.

A second source of apparent weakness is score sparsity. For low-accuracy binary systems on ARC-2, the fixed-accuracy random-placement null already spans roughly `[-0.193, 0.191]` in raw correlation. In other words, a low raw correlation can still be meaningful if it is larger than what a system with the same number of wins would achieve by solving random items.

## Primary ARC-2 Null Hypotheses

We formalized `humans, LLMs, and non-LLMs are the same` as several narrower operational nulls. The table below reports the core tests, with Benjamini-Hochberg q-values computed within each hypothesis family.

```text
null_id                system  estimate  ci_lo  ci_hi  p_value  q_value       decision                                                                   note
     H1           LLM average     0.402  0.275  0.515    0.950    0.950 fail_to_reject  Two-sided empirical p-value against the human split-half distribution
     H1      Best-aligned LLM     0.439  0.275  0.515    0.506    0.607 fail_to_reject  Two-sided empirical p-value against the human split-half distribution
     H1        Best-score LLM     0.276  0.275  0.515    0.052    0.078 fail_to_reject  Two-sided empirical p-value against the human split-half distribution
     H1     TRM 361957 pass@2     0.214  0.275  0.515    0.004    0.008         reject  Two-sided empirical p-value against the human split-half distribution
     H1     TRM 651522 pass@2     0.113  0.275  0.515    0.000    0.001         reject  Two-sided empirical p-value against the human split-half distribution
     H1 VARC ARC-2_ViT pass@2     0.158  0.275  0.515    0.000    0.001         reject  Two-sided empirical p-value against the human split-half distribution
     H2           LLM average     0.458  0.257  0.602    0.000    0.001         reject One-sided bootstrap test on partial corr after simple feature controls
     H2     TRM 651522 pass@2     0.108 -0.063  0.283    0.086    0.086 fail_to_reject One-sided bootstrap test on partial corr after simple feature controls
     H2     TRM 361957 pass@2     0.219  0.087  0.353    0.001    0.001         reject One-sided bootstrap test on partial corr after simple feature controls
     H2 VARC ARC-2_ViT pass@2     0.157  0.054  0.258    0.001    0.001         reject One-sided bootstrap test on partial corr after simple feature controls
     H3     TRM 651522 pass@2     0.117 -0.070  0.282    0.112    0.112 fail_to_reject   One-sided bootstrap test on partial corr controlling the LLM average
     H3     TRM 361957 pass@2     0.186  0.042  0.305    0.009    0.028         reject   One-sided bootstrap test on partial corr controlling the LLM average
     H3 VARC ARC-2_ViT pass@2     0.145 -0.015  0.263    0.034    0.052         reject   One-sided bootstrap test on partial corr controlling the LLM average
```

The strongest ARC-2 result is that the human-equivalence null is not rejected for the LLM aggregate (`r = 0.402`, `p = 0.950`) or for the most human-aligned single LLM (`r = 0.439`, `p = 0.506`), but it is rejected for the best current non-LLM profiles: TRM mid-training (`r = 0.214`, `p = 0.0040`), TRM best-score (`r = 0.113`, `p = 0.0004`), and VARC (`r = 0.158`, `p = 0.0004`). The best-score LLM sits right on the boundary (`r = 0.276`, `p = 0.052`), so we treat that case as borderline rather than decisive.

The feature-only null is also too strong. After controlling for coarse task features, the LLM average remains aligned with humans (`partial r = 0.458`, `p < 0.001`), and so do the best-aligned TRM checkpoint and best VARC profile. That means their human-alignment is not reducible to simple size, color-count, or train-pair cues alone. However, the strongest residual null is more mixed: after controlling for the LLM average itself, the TRM best-score checkpoint does not retain clear extra human signal (`partial r = 0.117`, `p = 0.105`), whereas the mid-training TRM checkpoint (`partial r = 0.186`, `p = 0.009`) and VARC (`partial r = 0.145`, `p = 0.037`) do show weak residual alignment under regression control. We treat that residual evidence as suggestive rather than fully settled, because the more conservative subtraction-based check is less decisive.

## Low Correlations: What Is Expected, and What Is Not

The low absolute correlations are partly expected here for three reasons. First, the human benchmark itself is noisy. Second, the ARC-2 primary subset contains only `110` robustly sampled test pairs. Third, several non-LLM systems solve very few items, so their achievable item-profile signal is inherently sparse. The right question is therefore not `is the raw correlation numerically large?` but `is it larger than the two relevant nulls: human measurement noise and random item placement at the same accuracy?`

```text
      dataset                system  pair_accuracy  successes  observed_corr  null_ci_lo  null_ci_hi  p_value  q_value       decision
ARC-2 primary        Best-score LLM          0.491         54          0.276      -0.185       0.186    0.003    0.008         reject
ARC-2 primary      Best-aligned LLM          0.164         18          0.439      -0.189       0.186    0.000    0.001         reject
ARC-2 primary     TRM 361957 pass@2          0.045          5          0.214      -0.184       0.188    0.012    0.018         reject
ARC-2 primary     TRM 651522 pass@2          0.109         12          0.113      -0.193       0.191    0.122    0.122 fail_to_reject
ARC-2 primary VARC ARC-2_ViT pass@2          0.036          4          0.158      -0.181       0.184    0.053    0.063 fail_to_reject
ARC-1 sidecar      CompressARC top2          0.130         30          0.160      -0.135       0.126    0.005    0.011         reject
```

That fixed-accuracy null changes the interpretation substantially. The mid-training TRM checkpoint exceeds the same-accuracy random-placement null (`p = 0.012`), as does VARC, but only narrowly (`p = 0.053`). The later high-score TRM checkpoint does not (`p = 0.122`). So the `TRM 651522` profile is better at raw ARC scoring but not at placing its wins on specifically human-like items. This is exactly the kind of distinction that the raw accuracy table misses.

## Direct System-to-System Comparisons

```text
                             contrast  estimate  ci_lo  ci_hi  p_value       decision  q_value
      LLM average - TRM 361957 pass@2     0.188 -0.024  0.388    0.084 fail_to_reject    0.106
      LLM average - TRM 651522 pass@2     0.289  0.031  0.548    0.029         reject    0.049
  LLM average - VARC ARC-2_ViT pass@2     0.244  0.031  0.432    0.028         reject    0.049
 Best-aligned LLM - TRM 361957 pass@2     0.224  0.036  0.408    0.022         reject    0.049
TRM 361957 pass@2 - TRM 651522 pass@2     0.101 -0.041  0.246    0.177 fail_to_reject    0.177
```

Paired bootstrap comparison tests show that the LLM average is more human-aligned than the best-score TRM checkpoint (`delta r = 0.289`, `p = 0.029`) and more human-aligned than VARC (`delta r = 0.244`, `p = 0.028`). The comparison against the most human-aligned mid-training TRM checkpoint points in the same direction but does not clear the `0.05` threshold on this subset (`delta r = 0.188`, `p = 0.084`). This is one reason not to over-index on a simple winner-loser story: there is a weak residual non-LLM signal, but it is not as stable or as large as the LLM aggregate alignment.

The accuracy-matched LLM check reaches a similar conclusion.

```text
               system  system_accuracy  system_human_corr  matched_llm_count  matched_llm_corr_median  matched_llm_corr_min  matched_llm_corr_max                       matched_llm_best
    TRM 651522 pass@2            0.109              0.113                  7                    0.160                 0.050                 0.321   claude-opus-4-5-20251101-thinking-8k
    TRM 361957 pass@2            0.045              0.214                 12                    0.187                 0.022                 0.246 claude-sonnet-4-5-20250929-thinking-8k
VARC ARC-2_ViT pass@2            0.036              0.158                 12                    0.187                 0.022                 0.246 claude-sonnet-4-5-20250929-thinking-8k
```

The best-score TRM checkpoint falls below the median human-alignment of accuracy-matched weak LLMs. The best-aligned TRM checkpoint and VARC sit inside the weak-LLM band, not clearly above it. So current non-LLMs do not look like the LLM consensus, but neither do they look wholly alien to the low-performance end of the LLM distribution.

## Complementarity and Training Dynamics

```text
               system  total_solved_items  rescued_human_easy_llm_hard  expected_if_random  hypergeom_p_value  q_value       decision
    TRM 651522 pass@2                  12                            5                 2.4              0.061    0.061 fail_to_reject
VARC ARC-2_ViT pass@2                   4                            3                 0.8              0.025    0.037         reject
       TRM+VARC union                  13                            6                 2.6              0.022    0.037         reject
```

```text
                             test statistic  estimate  p_value       decision  q_value
      TRM pass@2 step vs accuracy  spearman     0.850    0.002         reject    0.007
    TRM pass@2 step vs human_corr  spearman    -0.085    0.827 fail_to_reject    0.827
TRM pass@2 accuracy vs human_corr  spearman    -0.265    0.451 fail_to_reject    0.676
```

Despite their weaker overall human alignment, the non-LLM systems are not redundant. The TRM+VARC union rescues `6` human-easy / LLM-hard ARC-2 items, which is more than expected by chance (`p = 0.022`). The trajectory analysis clarifies what is happening inside TRM: training step strongly predicts accuracy (`Spearman rho = 0.850`, `p = 0.002`), but not human alignment (`Spearman rho = -0.085`, `p = 0.827`). The accuracy-vs-alignment relation is negative but not significant (`rho = -0.265`, `p = 0.451`). In plain language, later checkpoints get better at the benchmark without reliably becoming more human-like.

## ARC-1 Sidecar

We do not have a dedicated ARC-1 human benchmark file, but we do have an ARC-1 sidecar through task reuse in the ARC-AGI-2 Public Train human data. That yields `230` single-pair ARC-1 evaluation tasks with a human split-half median of `0.372`.

```text
               system  pair_accuracy  human_pearson  percentile_vs_human_split  human_split_median  human_split_ci_lo  human_split_ci_hi  p_value       decision  q_value
     ARC1 LLM average          0.291          0.285                      0.028               0.372              0.282              0.459    0.057 fail_to_reject    0.061
ARC1 best-aligned LLM          0.539          0.286                      0.030               0.372              0.282              0.459    0.061 fail_to_reject    0.061
     CompressARC top2          0.130          0.160                      0.000               0.372              0.282              0.459    0.000         reject    0.001
```

```text
          system  system_accuracy  system_human_corr  matched_llm_count  matched_llm_corr_median  matched_llm_corr_min  matched_llm_corr_max                      matched_llm_best
CompressARC top2             0.13               0.16                  2                    0.139                 0.119                 0.159 claude-haiku-4-5-20251001-thinking-1k
```

On that ARC-1 overlap, CompressARC is genuinely non-random with respect to human difficulty (`r = 0.160`, fixed-accuracy `p = 0.005`), but it still falls well below the human split-half benchmark (`p < 0.001` against human-equivalence). So CompressARC is a useful real prediction artifact, but it does not overturn the broader ARC-2 pattern.

## Interpretation

Taken together, these tests support a mixed but fairly clear synthesis. We reject the strong null that the current non-LLM systems are psychometrically indistinguishable from humans on ARC-2. We also reject the claim that all observed alignment is just trivial task-feature matching. At the same time, we do not have grounds to claim that non-LLMs define a wholly separate and robust human-like axis beyond the LLM average. Some evidence for a small residual exists, especially for the mid-training TRM checkpoint and VARC, but it is weak, method-sensitive, and much smaller than the main LLM aggregate effect.

The most defensible take is therefore: human item difficulty on ARC-2 is better approximated by the LLM consensus than by the currently stored non-LLM systems, yet the non-LLM systems still contribute complementary successes and some nontrivial human-relevant structure. The important distinction is not `same` versus `completely different`; it is `closer to the human difficulty axis`, and on that measure the LLM aggregate is currently ahead.
