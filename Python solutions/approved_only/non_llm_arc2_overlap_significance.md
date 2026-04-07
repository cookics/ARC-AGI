# ARC-2 overlap significance check

This note focuses on the shared `17` approved ARC-2 eval tasks that have all three outcome types:

- human difficulty
- LLM difficulty
- non-LLM difficulty

The goal is to quantify the "non-LLM is in between humans and LLMs" pattern from the comparison chart.

## Individual correlations

### `cyclomatic_complexity`

| Outcome | `r` | `p` | 95% CI |
|---|---:|---:|---:|
| Human difficulty | `0.150` | `0.565` | `[-0.356, 0.588]` |
| LLM difficulty | `0.591` | `0.0125` | `[0.154, 0.834]` |
| Non-LLM difficulty | `0.356` | `0.161` | `[-0.151, 0.714]` |

### `complexity_pc1_score`

| Outcome | `r` | `p` | 95% CI |
|---|---:|---:|---:|
| Human difficulty | `0.135` | `0.604` | `[-0.369, 0.578]` |
| LLM difficulty | `0.535` | `0.0268` | `[0.074, 0.808]` |
| Non-LLM difficulty | `0.388` | `0.124` | `[-0.114, 0.732]` |

## Pairwise difference tests

Bootstrap tests compare whether a metric is more correlated with one outcome than another on the same `17` tasks.

### `cyclomatic_complexity`

| Comparison | Delta `r` | `p` | 95% bootstrap CI |
|---|---:|---:|---:|
| LLM minus human | `0.441` | `0.0392` | `[0.020, 0.911]` |
| LLM minus non-LLM | `0.235` | `0.298` | `[-0.203, 0.888]` |
| Non-LLM minus human | `0.206` | `0.569` | `[-0.503, 0.774]` |

### `complexity_pc1_score`

| Comparison | Delta `r` | `p` | 95% bootstrap CI |
|---|---:|---:|---:|
| LLM minus human | `0.400` | `0.103` | `[-0.095, 0.887]` |
| LLM minus non-LLM | `0.147` | `0.546` | `[-0.287, 0.940]` |
| Non-LLM minus human | `0.252` | `0.498` | `[-0.524, 0.784]` |

## Power reality check

With only `n = 17` tasks, the study is low-powered for moderate effect sizes.

- Approximate absolute correlation needed for `80%` power at two-sided `alpha = .05`: `|r| ~= 0.634`
- Approximate sample size needed for `80%` power:
  - `r = 0.30`: `n ~= 85`
  - `r = 0.35`: `n ~= 62`
  - `r = 0.40`: `n ~= 47`
  - `r = 0.50`: `n ~= 30`
  - `r = 0.60`: `n ~= 20`

## Bottom line

- The "LLM > human" difference is supported for `cyclomatic_complexity`.
- The "non-LLM is in between" pattern is directionally consistent, but not statistically distinguishable from either side with `17` tasks.
- The main limitation is sample size, not a lack of any signal at all.
