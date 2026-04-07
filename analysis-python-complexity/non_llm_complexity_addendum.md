# Non-LLM Complexity Addendum

## Scope

This addendum asks whether the approved-solver complexity measures that worked well for humans and LLMs also say anything useful about the non-LLM systems in this repo.

I kept the analysis separate from the existing write-up and treated it as a first pass with explicit hypothesis families and multiple-testing correction.

## Data Used

- ARC-1 approved eval overlap with solver complexity: `38` tasks
- ARC-2 approved eval overlap with solver complexity: `18` tasks
- ARC-2 approved tasks with both human and non-LLM difficulty: `17` tasks
- ARC-1 non-LLM profiles in the main task matrix: `11` profiles
- ARC-2 non-LLM profiles in the main task matrix: `28` profiles

Main ARC-1 profiles:
- `VARC ARC-1_Unet pass@1-4`
- `VARC ARC-1_ViT pass@1-4`
- `CompressARC final_pick_pass@1`
- `CompressARC final_pick_pass@2`
- `CompressARC ranked_candidate_pass@2`

Main ARC-2 profiles:
- `TRM` steps `72391` through `723914`, each at `pass@1` and `pass@2`
- `VARC ARC-2_Unet pass@1-4`
- `VARC ARC-2_ViT pass@1-4`

## Primary Hypotheses

- `N1`: ARC-2 non-LLM difficulty aligns with ARC-2 LLM difficulty.
- `N2`: ARC-2 non-LLM difficulty aligns with ARC-2 human difficulty.
- `N3`: Human difficulty is more aligned with LLM difficulty than with non-LLM difficulty on the shared ARC-2 overlap.
- `N4`: ARC-2 non-LLM difficulty positively tracks structural solver complexity.
- `N5`: Structural solver complexity is more strongly associated with LLM difficulty than with non-LLM difficulty.
- `N6`: After controlling LLM difficulty, ARC-2 non-LLM difficulty still retains structural-complexity signal.
- `N7`: For non-LLM difficulty, structural complexity is more informative than runtime intensity.
- `N8`: ARC-1 non-LLM difficulty positively tracks structural solver complexity.
- `N9`: CompressARC search-depth difficulty positively tracks structural solver complexity.

## How The Non-LLM Difficulty Axes Were Built

For each dataset, I built a binary task-by-profile matrix and then derived three item-difficulty summaries:

- smoothed logit difficulty from profile solve rate
- a PCA `PC1` item difficulty
- a penalized 1PL / Rasch-like item difficulty

The primary outcome in the key tests is the smoothed logit difficulty. The other two are robustness checks.

Sanity checks on those difficulty summaries:

- ARC-2 pass-rate vs logit difficulty: `r = -0.975`
- ARC-2 logit vs Rasch difficulty: `r = 1.000`
- ARC-2 logit vs PC1 difficulty: `r = 0.853`
- ARC-1 pass-rate vs logit difficulty: `r = -0.989`
- ARC-1 logit vs Rasch difficulty: `r = 0.999`
- ARC-1 logit vs PC1 difficulty: `r = 0.978`

Those are all high enough that the primary difficulty proxy is not behaving erratically.

## Statistical Framework

- Pearson correlations
- bootstrap 95% confidence intervals
- permutation p-values for direct correlation tests
- bootstrap difference-of-correlation tests
- BH-FDR q-values across the `9` key hypotheses

Null hypotheses followed the same style as the main write-up:

- alignment nulls: unrelated task ordering
- difference nulls: equal correlation strength
- residual nulls: no remaining association after controlling the shared axis

## Headline Results

Best structural metric on ARC-2 non-LLM difficulty:
- `Gzip bytes`, `r = 0.504`, `q = 0.217`

Best structural metric on ARC-1 non-LLM difficulty:
- `Gzip bytes`, `r = 0.366`, `q = 0.191`

Best pooled metric after within-dataset standardization:
- `Gzip bytes`, `r = 0.378`, `q = 0.062`

Key-test table:

```text
claim_id  estimate  ci_low  ci_high  p_value  q_value_bh  reject_fdr_0_05
      N1     0.113  -0.364    0.518    0.657       0.739            False
      N2     0.062  -0.381    0.440    0.818       0.818            False
      N3     0.469   0.109    0.864    0.518       0.721            False
      N4     0.352  -0.157    0.660    0.148       0.667            False
      N5     0.167  -0.170    0.742    0.561       0.721            False
      N6     0.346  -0.044    0.658    0.508       0.721            False
      N7     0.356  -0.066    0.817    0.495       0.721            False
      N8     0.255  -0.042    0.507    0.125       0.667            False
      N9     0.113  -0.211    0.399    0.499       0.721            False
```

## Supported Claims

- None of the key tests survived BH-FDR correction in this first pass.

## Unsupported Or Borderline Claims

- `N1`: estimate `0.113`, 95% CI `[-0.364, 0.518]`, `p = 0.657`, `q = 0.739`.
- `N2`: estimate `0.062`, 95% CI `[-0.381, 0.440]`, `p = 0.818`, `q = 0.818`.
- `N3`: estimate `0.469`, 95% CI `[0.109, 0.864]`, `p = 0.518`, `q = 0.721`.
- `N4`: estimate `0.352`, 95% CI `[-0.157, 0.660]`, `p = 0.148`, `q = 0.667`.
- `N5`: estimate `0.167`, 95% CI `[-0.170, 0.742]`, `p = 0.561`, `q = 0.721`.
- `N6`: estimate `0.346`, 95% CI `[-0.044, 0.658]`, `p = 0.508`, `q = 0.721`.
- `N7`: estimate `0.356`, 95% CI `[-0.066, 0.817]`, `p = 0.495`, `q = 0.721`.
- `N8`: estimate `0.255`, 95% CI `[-0.042, 0.507]`, `p = 0.125`, `q = 0.667`.
- `N9`: estimate `0.113`, 95% CI `[-0.211, 0.399]`, `p = 0.499`, `q = 0.721`.

## Interpretation

The cleanest through-line from this non-LLM pass is:

- non-LLM item difficulty is **not random** with respect to approved solver complexity
- but it is **weaker and less cleanly structured** than the LLM result
- the ARC-2 non-LLM profiles still line up with the LLM difficulty axis more than with the human axis
- once LLM difficulty is controlled, any extra non-LLM structural signal is much smaller and less secure

The pooled view is the clearest hint that there is a real but modest complexity signal: once ARC-1 and ARC-2 are each standardized onto their own non-LLM difficulty scale and then combined, the size/structure metrics become more consistently positive. That still does not clear the formal corrected key-test bar on its own, but it makes the split-dataset nulls look more like a low-power story than a directionless one.

That means the non-LLM systems do seem to feel some of the same task pressure captured by solver complexity, but not as strongly and not in as focused a way as the LLM profiles.

The ARC-1 sidecar is useful because it gives more overlap (`38` approved tasks instead of `18`), and it suggests the structural signal is not exclusive to ARC-2. But ARC-1 is still a sidecar because it does not directly overlap the main human ARC-2 psychometric setup.

One especially useful contrast is:

- On ARC-2, the earlier LLM-side structural result was much stronger than the non-LLM one.
- In this pass, the LLM-vs-non-LLM difference test for cyclomatic complexity is the direct version of that question.

So the most cautious conclusion is:

> Approved solver structure seems to track non-LLM difficulty somewhat, but the strongest and cleanest relationship still belongs to the LLM difficulty axis.

## Figures

![ARC-2 relationships](chart_non_llm_arc2_relationships.png)

![Human vs LLM vs Non-LLM comparison](chart_non_llm_complexity_comparison.png)

![ARC-1 sidecar](chart_non_llm_arc1_sidecar.png)

![TRM trajectory](chart_non_llm_trm_trajectory.png)

## Output Files

- `non_llm_arc1_task_profiles.csv`
- `non_llm_arc2_task_profiles.csv`
- `non_llm_task_outcomes.csv`
- `non_llm_complexity_metric_correlations.csv`
- `non_llm_complexity_key_tests.csv`
- `non_llm_complexity_summary.json`
- `non_llm_complexity_addendum.md`
- `non_llm_complexity_addendum.tex`
