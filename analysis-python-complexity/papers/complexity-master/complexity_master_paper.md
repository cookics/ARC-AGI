# ARC Solver Complexity Across Humans, LLMs, and Non-LLM Systems

## What This Paper Is

This paper consolidates the full analysis thread in this folder into one place. It starts from the approved ARC Python solvers, explains how they were fetched and validated, summarizes the complexity panel that was computed over those solvers, and then traces how those complexity measures relate to:

- latent human difficulty
- latent LLM difficulty
- thinking-vs-standard model differences
- non-LLM system difficulty

The main question is simple:

> When a task requires a more structurally complex correct Python solver, does that show up as greater task difficulty for humans, LLMs, or both?

The short answer is:

- yes for LLMs, strongly
- yes for humans, but much more weakly
- yes for non-LLM systems directionally, but not yet with enough power to be confident

The strongest significant contrast in the whole project is:

> structural solver complexity is more strongly associated with LLM difficulty than with human difficulty on the matched overlap tasks

That is the result that kept surviving the later scientific audit.

## Executive Summary

- We fetched `511` ARC `solution.py` files from `arc.huikang.dev` and validated them against the official ARC task data.
- Only `127 / 511` passed validation.
- The site's `approved` status explained almost all of that gap: `120 / 120` approved programs passed, while nearly all `submitted`, `attempted`, and `skipped` programs failed.
- We therefore restricted the main analysis to `120` validated approved Python solvers.
- Across those solvers, structural complexity measures like `ast_node_count`, `token_count`, `cyclomatic_complexity`, and `halstead_volume` clustered tightly together.
- The best single solver-complexity proxy for latent difficulty was `ast_node_count` (`r = 0.666`) in the original latent-scale analysis.
- A conservative supervised composite improved that only modestly (`PCR-5`, leave-one-out `r = 0.691`).
- On the expanded LLM analysis, `log1p(cyclomatic_complexity)` reached `r = 0.707` against simple LLM logit difficulty.
- Human and LLM difficulty share a real common axis on the approved overlap tasks:
  - human difficulty vs LLM difficulty: `r = 0.531`, `p = 0.0268`, `q = 0.0366`
  - human solve rate vs LLM pass rate: `r = 0.541`, `p = 0.0187`, `q = 0.0280`
- The strongest significant difference claim is:
  - cyclomatic complexity is more strongly associated with LLM difficulty than with human difficulty: delta-`r = 0.441`, `p = 0.0378`, `q = 0.0472`
- Residual analyses sharpen that story:
  - residual LLM difficulty after removing human difficulty still tracks cyclomatic complexity: `r = 0.603`, `p = 0.0117`, `q = 0.0194`
  - residual human difficulty after removing LLM difficulty tracks human duration only suggestively: `r = 0.470`, `p = 0.0505`, `q = 0.0583`
- Human pair-level data show a different signature from LLM difficulty:
  - human pair difficulty tracks human duration: `r = 0.391`, `p = 1.67e-4`, `q = 3.57e-4`
  - raw board size alone is weak for human difficulty: `r = -0.096`, `p = 0.311`
  - within-task human difficulty heterogeneity is substantial: mean range `0.719`, max `2.498`
- The flashy negative `thinking_advantage` result did not survive the full scientific audit cleanly. It looked strong under the legacy model grouping, but weakened materially after verified relabeling and floor-sensitive checks.
- The first-pass non-LLM results are interesting but underpowered:
  - their complexity correlations are directionally between human and LLM correlations on the shared `17` ARC-2 tasks
  - but the intermediate position is not yet statistically distinguishable from either side

## Data Assembly and Validation

### Task IDs and local ARC data

The local ARC mirror was checked against the public ARC-AGI-1 and ARC-AGI-2 repositories.

- ARC-AGI-1 matched the official `400` train and `400` eval task IDs exactly.
- ARC-AGI-2 matched the official `1000` train and `120` eval task IDs exactly.
- ARC-AGI-1 train and eval plus ARC-AGI-2 train reconstructed cleanly against the official public task content.
- ARC-AGI-2 eval had `6` task-content differences versus the current public repo:
  - `4a21e3da`
  - `abc82100`
  - `b6f77b65`
  - `d8e07eb2`
  - `f560132c`
  - `faa9f03d`

This mattered for provenance, but not for the basic task-ID fetch pipeline.

### Solver fetch and validation

We discovered `1,147` unique ARC task IDs across ARC-1 train/eval and ARC-2 train/eval and used the direct site endpoint for solutions:

- `https://arc.huikang.dev/solutions/<task_id>/solution.py`

Fetch outcome:

- `511` unique `solution.py` files on disk
- `636` tasks with no stored `solution.py`

Validation outcome on the official task data:

- passed: `127`
- wrong answer: `380`
- crash: `4`
- timeout: `0`

The site status labels explained the pattern almost perfectly:

- `approved`: `120 / 120` passed
- `submitted`: `1 / 236` passed
- `attempted`: `5 / 138` passed, `4` crashed
- `skipped`: `1 / 17` passed
- `correct`: `629` tasks had no stored `solution.py`

That gave a very clean methodological decision:

> Use only the `approved` solutions for the complexity study.

### Approved package

The final approved-only package contains `120` unique task IDs.

By dataset membership:

- ARC-AGI 1 train: `57`
- ARC-AGI 1 eval: `39`
- ARC-AGI 2 train: `100`
- ARC-AGI 2 eval: `20`

These overlap because ARC-2 train contains many ARC-1 tasks.

## Complexity Measures

### Static and structural metrics

The solver complexity panel included:

- `nonblank_lines`
- `token_count`
- `ast_node_count`
- `function_count`
- `call_count_static`
- `branch_node_count`
- `cyclomatic_complexity`
- `max_nesting_depth`
- `gzip_bytes`
- `halstead_volume`
- `halstead_effort`

### Dynamic and runtime metrics

The dynamic panel included:

- `opcode_count_dynamic`
- `branch_opcode_count_dynamic`
- `python_call_count_dynamic`
- `elapsed_ms_total`
- `elapsed_ms_per_test`
- `peak_memory_bytes`
- normalization by input-cell count where relevant

### First descriptive pass

On the approved solver set:

- nonblank LOC median: `55`
- nonblank LOC mean: `67.8`
- cyclomatic complexity median: `17.5`
- cyclomatic complexity mean: `23.9`
- Halstead volume median: `1623`
- maximum nesting depth median: `4`

The important conceptual result was not the raw medians, but that static structural metrics clustered very tightly while runtime measures formed a weaker, partially separate axis.

## PCA and Composite Complexity Models

PCA over the main complexity panel found more than one dimension:

- `PC1`: `57.4%` of variance
- `PC2`: `17.5%`
- `PC3`: `9.8%`

Interpretation:

- `PC1` reads as overall solver size and structural density
- `PC2` reads as runtime intensity and execution burden
- `PC3` reads more like grid volume and memory footprint

So the clean answer to the earlier question "are these all the same thing?" is:

- mostly, but not completely
- there is one dominant structural factor
- there is also a distinct runtime-style factor

The best single versus composite predictors were:

- best single metric: `ast_node_count`, `r = 0.666`
- unsupervised `PC1`: `r = 0.629`
- best conservative composite: `PCR-5`, leave-one-out `r = 0.691`

This is important because it says:

- a lot of the complexity panel is measuring the same underlying thing
- but a supervised combination can still extract a bit more signal
- the gain is real but modest, not revolutionary

## Human and LLM Psychometric Data

### LLM-side overlap

The main LLM psychometric join covered `56` approved eval rows:

- `38` ARC-1 eval
- `18` ARC-2 eval

LLM difficulty was summarized in several ways:

- earlier shared-model latent difficulty
- pooled all-model Rasch difficulty
- simple pass-rate and logit-difficulty summaries
- 2PL difficulty and discrimination

These difficulty summaries were extremely stable:

- previous latent difficulty vs pooled Rasch: `r = 0.991`
- pooled Rasch vs simple logit difficulty: `r = 0.970`

That means the LLM difficulty signal is not fragile to the exact latent-modeling choice.

### Human-side overlap

The human task-level overlap came from the public evaluation table and is much narrower.

The key limitation is:

- the human task table is almost entirely ARC-2 public eval
- only `6` human tasks belong to both ARC-1 eval and ARC-2 eval
- only `2` approved ARC-1 eval tasks overlap with the approved human-linked task set

So the fully matched `human + LLM + non-LLM + solver complexity` comparison is effectively an ARC-2 comparison on `17` independent tasks.

That is not a bug in the code. It is a data-overlap limit.

## Main LLM Difficulty Results

The strongest LLM-side result is that structural solver complexity tracks LLM difficulty strongly.

Best headline correlations:

- `log1p(cyclomatic_complexity)` vs LLM logit difficulty: `r = 0.707`
- `ast_node_count` vs pooled Rasch difficulty: `r = 0.653`
- `log1p(cyclomatic_complexity)` vs previous shared latent difficulty: `r = 0.691`

Structural code measures consistently outperformed raw runtime burden for LLM difficulty.

This suggests that, in ARC, LLM difficulty is tracking something closer to:

- structural rule complexity
- branching rule machinery
- compact description length of the successful solver

and less something like:

- total execution effort
- brute force runtime

## Human vs LLM: Shared Axis and Significant Difference

### Shared axis

Humans and LLMs are not living on unrelated difficulty scales.

Supported shared-axis results:

- `S1`: human difficulty aligns with LLM difficulty
  - `r = 0.531`
  - 95% CI `[0.178, 0.780]`
  - `p = 0.0268`
  - `q = 0.0366`
- `S2`: human solve rate aligns with LLM pass rate
  - `r = 0.541`
  - 95% CI `[0.226, 0.781]`
  - `p = 0.0187`
  - `q = 0.0280`

So there is a real common axis.

### The key significant difference

The most important result in the entire project is:

- `D1`: cyclomatic complexity is more strongly associated with LLM difficulty than with human difficulty
  - delta-`r = 0.441`
  - 95% CI `[0.024, 0.909]`
  - `p = 0.0378`
  - `q = 0.0472`

Raw correlations behind that difference:

- cyclomatic vs human difficulty: `r = 0.150`
- cyclomatic vs LLM difficulty: `r = 0.591`

That is the clearest significant evidence that human difficulty and LLM difficulty are not just noisy copies of the same thing.

An exploratory merged extension strengthens the same conclusion. If we pool the ARC-1 sidecar human tasks with ARC-2 eval human tasks, and pool ARC-1 eval with ARC-2 eval on the LLM side, then standardize within benchmark before correlating, `cyclomatic_complexity` is still only weakly associated with pooled human difficulty on the shared `n = 36` tasks (`r = 0.175`, permutation `p = 0.309`), but strongly associated with pooled LLM difficulty on that same shared pooled task set (`r = 0.649`, permutation `p = 1.00e-4`). The paired pooled difference remains significant: delta-`r = 0.474`, bootstrap `95%` CI `[0.190, 0.796]`, permutation `p = 0.0186`. On the full pooled ARC-1 + ARC-2 LLM sample (`n = 59`), the within-benchmark-standardized cyclomatic correlation is `r = 0.598`, permutation `p = 5.00e-5`. The unsupervised structural composite shows the same pattern: pooled human `r = 0.141`, `p = 0.417`; pooled matched LLM `r = 0.625`, `p = 1.50e-4`; pooled delta-`r = 0.484`, `p = 0.0163`.

I also ran the same benchmark-adjusted pooled test across the full 38-metric complexity panel. The broad pattern stays the same: only 2 of 38 pooled human correlations clear `p < 0.05`, 27 of 38 pooled LLM correlations clear `p < 0.05`, and 31 of 38 pooled delta tests are significant. The strongest pooled asymmetries are concentrated in memory, runtime, branching, and dynamic execution counts. The full metric table is in `analysis-latent-crossarc/tables/pooled_structure_significance_all.csv`.

### Residual split

Residual analyses sharpen the same story.

After removing the shared human/LLM axis:

- `D4`: residual LLM difficulty still tracks cyclomatic complexity
  - `r = 0.603`
  - 95% CI `[0.284, 0.816]`
  - `p = 0.0117`
  - `q = 0.0194`

But the analogous human-side result is only suggestive:

- `D3`: residual human difficulty vs mean human duration
  - `r = 0.470`
  - 95% CI `[0.067, 0.766]`
  - `p = 0.0505`
  - `q = 0.0583`

So the pattern is:

- shared axis exists
- LLM-specific residuals still look structural
- human-specific residuals look more time/search-like, but the task-level overlap is small

## Human-Side Findings Beyond the Overlap Table

The pair-level human data are where the human story becomes clearer.

Supported pair-level results:

- `H1`: human pair difficulty tracks mean human duration
  - `r = 0.391`
  - `p = 1.67e-4`
  - `q = 3.57e-4`
- `H2`: raw board size alone is weak for human difficulty
  - `r = -0.096`
  - `p = 0.311`
- `H3`: duration is more informative than raw board size
  - delta-`r = 0.487`
  - `p = 1.25e-4`
  - `q = 3.57e-4`
- `H4`: the human-over-LLM gap shrinks when public-eval tasks expose more test pairs
  - `r = -0.366`
  - `p = 3.33e-4`
  - `q = 6.25e-4`
- `H5`: task identity explains a large share of pair-level human difficulty variation on multi-pair tasks
  - `R^2 = 0.749`
  - `p = 7.93e-5`
  - `q = 3.57e-4`

Descriptive heterogeneity results:

- `44` public-eval tasks had multiple test pairs
- mean within-task difficulty range: `0.719`
- max within-task difficulty range: `2.498`

This matters conceptually because a single solver file is a task-level object, while human difficulty can vary meaningfully across test pairs within the same task.

That gives a plausible reason why human difficulty should not be expected to line up with solver structure as strongly as LLM difficulty does.

## Thinking Advantage: Initial Result, Audit, and Downgrade

### What thinking advantage was

The original definition was:

- `thinking_advantage = pass_rate_thinking - pass_rate_standard`

with a logit-gap analog as well.

### Legacy result

Under the original heuristic model grouping, the result looked strong:

- `T1`: thinking advantage declines with LLM difficulty
  - `r = -0.492`
  - `p = 1.67e-4`
  - `q = 3.57e-4`
- `T2`: grouped binomial GLM interaction is negative
  - coefficient `-0.718`
  - `p = 5.70e-6`
  - `q = 8.55e-5`

### Why it was downgraded

We then did two things:

1. audited ambiguous thinking-model labels against provider documentation
2. checked whether the result was mainly a floor artifact

Ambiguous models included:

- `gemini-3-pro-preview`
- `gpt-5-pro-2025-10-06`
- `gpt-5-2-pro-2025-12-11-high`
- `gpt-5-2-pro-2025-12-11-medium`
- `QwQ-32B-Fireworks`

Under the verified grouping:

- there were `35 / 56` items where the standard group had zero successes
- there were `0 / 56` items where both groups had zero successes

The result then weakened sharply on floor-resistant subsets:

- all rows: raw-gap `r = -0.378`
- standard-nonzero subset: raw-gap `r = 0.280`
- fully interior subset: raw-gap `r = -0.009`

So the right scientific read is:

- the original result was not pure nonsense
- but it is not robust enough after label verification and floor checks to treat as a stable substantive conclusion

This is actually a good sign for the pipeline. It means the audit did not merely preserve every exciting result.

## Non-LLM First Pass

### Data used

The non-LLM first pass used:

- `38` approved ARC-1 eval tasks
- `18` approved ARC-2 eval tasks
- `17` shared ARC-2 tasks with human, LLM, and non-LLM outcomes

The system profiles included:

- ARC-1 VARC and CompressARC profiles
- ARC-2 TRM, VARC, and related non-LLM profiles

### Main finding

The non-LLM results were directionally coherent but underpowered.

Best metric-level results:

- ARC-2 non-LLM difficulty vs `gzip_bytes`: `r = 0.504`, `q = 0.217`
- ARC-1 non-LLM difficulty vs `gzip_bytes`: `r = 0.366`, `q = 0.191`
- pooled within-dataset-standardized non-LLM difficulty vs `gzip_bytes`: `r = 0.378`, `q = 0.062`

But none of the pre-specified non-LLM key tests `N1` through `N9` survived BH-FDR correction.

### The interesting in-between pattern

On the shared `17` ARC-2 tasks:

- `cyclomatic_complexity`
  - human difficulty: `r = 0.150`, `p = 0.565`
  - non-LLM difficulty: `r = 0.356`, `p = 0.161`
  - LLM difficulty: `r = 0.591`, `p = 0.0125`
- `complexity_pc1_score`
  - human difficulty: `r = 0.135`, `p = 0.604`
  - non-LLM difficulty: `r = 0.388`, `p = 0.124`
  - LLM difficulty: `r = 0.535`, `p = 0.0268`

So non-LLM systems look visually intermediate.

But the paired difference tests say we should be careful:

- for cyclomatic complexity:
  - `LLM > human`: supported, `p = 0.039`
  - `LLM > non-LLM`: not supported, `p = 0.298`
  - `non-LLM > human`: not supported, `p = 0.569`
- for complexity PC1:
  - no pairwise difference was significant

### Power reality

This is mostly a power problem.

With `n = 17`, a two-sided `.05` correlation study needs about:

- `|r| ~= 0.63` for `80%` power

Rough sample sizes needed for `80%` power:

- `r = 0.30`: `n ~= 85`
- `r = 0.35`: `n ~= 62`
- `r = 0.40`: `n ~= 47`
- `r = 0.50`: `n ~= 30`

So the non-LLM "middle" pattern is plausible, but we cannot yet separate it cleanly from either side.

## Scientific Framework

### Estimators used

- Pearson correlations
- bootstrap `95%` confidence intervals
- permutation p-values for correlation tests
- BH-FDR correction for the main named claim families
- bootstrap differences of correlations for matched human-vs-LLM and related contrasts
- grouped binomial GLMs for thinking-vs-standard analyses
- OLS residualization for shared-vs-specific analyses

### Null hypotheses

- shared-axis null: two task orderings are unrelated
- difference-of-correlation null: a predictor is equally associated with two outcomes
- residual null: after removing the shared axis, the remaining residual is unrelated to the tested predictor
- pair-level human null: the pair-level human measure is unrelated to the tested pair feature
- thinking-gap null: thinking and standard model groups have the same difficulty slope

## Through-Line

The results do not feel random.

They line up into a coherent story:

1. The approved solver set is real and clean.
2. Structural complexity dominates the solver complexity panel.
3. LLM difficulty is extremely stable across different psychometric constructions.
4. Human and LLM difficulty share a real common axis.
5. The cleanest significant difference is that structural solver complexity is more LLM-linked than human-linked.
6. Residual LLM difficulty still looks structural.
7. Human difficulty looks more duration/search-like and more pair-heterogeneous.
8. The non-LLM systems seem intermediate, but current overlap is too small for a decisive claim.
9. The one especially flashy result, thinking advantage, became weaker when audited properly.

That pattern is exactly what a sane pipeline should look like: some results strengthen, some stay suggestive, and one attractive result gets downgraded.

## Full Hypothesis Checklist

This section lists the named hypotheses that were explicitly tested and what became of them.

### Shared human-LLM alignment

- `S1`: Human and LLM task difficulty are positively aligned on the approved ARC-2 overlap.
  Result: supported.
  `r = 0.531`, `p = 0.0268`, `q = 0.0366`.

- `S2`: Human solve rate aligns with average-model pass rate on the same overlap tasks.
  Result: supported.
  `r = 0.541`, `p = 0.0187`, `q = 0.0280`.

- `S3`: The earlier shared-model latent scale and the new pooled Rasch scale are effectively the same LLM difficulty axis.
  Result: strongly supported.
  `r = 0.991`, `p = 1.67e-4`, `q = 3.57e-4`.

- `S4`: Pooled Rasch difficulty and simple LLM logit difficulty are almost identical.
  Result: strongly supported.
  `r = 0.970`, `p = 1.67e-4`, `q = 3.57e-4`.

### Human vs LLM difference claims

- `D1`: Cyclomatic complexity is more strongly associated with LLM difficulty than with human difficulty.
  Result: supported.
  delta-`r = 0.441`, `p = 0.0378`, `q = 0.0472`.
  Exploratory pooled benchmark-standardized extension: same direction on the merged ARC-1 sidecar + ARC-2 task set.
  shared pooled human `r = 0.175`, `p = 0.309`; shared pooled LLM `r = 0.649`, `p = 1.00e-4`; pooled delta-`r = 0.474`, `p = 0.0186`.
  Full 38-metric benchmark-adjusted panel: 2 human correlations with `p < 0.05`, 27 LLM correlations with `p < 0.05`, and 31 pooled delta tests with `p < 0.05`.

- `D2`: Human duration is more strongly associated with human difficulty than with LLM difficulty.
  Result: suggestive, not FDR-supported.
  delta-`r = 0.374`, `p = 0.0820`, `q = 0.0879`.

- `D3`: Residual human difficulty still tracks human duration after removing shared LLM difficulty.
  Result: suggestive, not FDR-supported.
  `r = 0.470`, `p = 0.0505`, `q = 0.0583`.

- `D4`: Residual LLM difficulty still tracks solver structure after removing shared human difficulty.
  Result: supported.
  `r = 0.603`, `p = 0.0117`, `q = 0.0194`.

### Human pair-level claims

- `H1`: Human pair difficulty tracks human time cost.
  Result: supported.
  `r = 0.391`, `p = 1.67e-4`, `q = 3.57e-4`.

- `H2`: Raw board size alone is weak for human difficulty.
  Result: not supported as a positive predictor, which is the point.
  `r = -0.096`, `p = 0.311`.

- `H3`: Human duration is more informative than raw board size for human difficulty.
  Result: supported.
  delta-`r = 0.487`, `p = 1.25e-4`, `q = 3.57e-4`.

- `H4`: Human-over-LLM advantage shrinks when tasks expose more test pairs.
  Result: supported.
  `r = -0.366`, `p = 3.33e-4`, `q = 6.25e-4`.

- `H5`: Task identity explains a large share of pair-level human difficulty variance on multi-pair tasks.
  Result: supported.
  `R^2 = 0.749`, `p = 7.93e-5`, `q = 3.57e-4`.

### Thinking-advantage claims

- `T1`: Thinking advantage declines as approved-item difficulty rises under the legacy label schema.
  Result: statistically supported under the legacy schema, but downgraded after label audit and floor-sensitive checks.

- `T2`: Thinking-vs-standard success probability has a negative difficulty interaction under the legacy schema.
  Result: statistically supported under the legacy schema, but not treated as a final substantive claim after the verified-label audit.

### Non-LLM key tests

- `N1`: ARC-2 non-LLM difficulty aligns with ARC-2 LLM difficulty.
  Result: not supported.

- `N2`: ARC-2 non-LLM difficulty aligns with human difficulty.
  Result: not supported.

- `N3`: Human difficulty is more strongly aligned with LLM difficulty than with non-LLM difficulty.
  Result: not supported.

- `N4`: ARC-2 non-LLM difficulty positively tracks structural solver complexity.
  Result: directionally positive but not supported after correction.

- `N5`: Structural solver complexity is more strongly associated with LLM difficulty than with non-LLM difficulty.
  Result: not supported.

- `N6`: After controlling LLM difficulty, non-LLM difficulty still retains structural-complexity signal.
  Result: not supported.

- `N7`: For non-LLM difficulty, structural complexity is more informative than runtime intensity.
  Result: not supported.

- `N8`: ARC-1 non-LLM difficulty positively tracks structural solver complexity.
  Result: directionally positive but not supported after correction.

- `N9`: CompressARC search-depth difficulty positively tracks structural solver complexity.
  Result: not supported.

### Supplemental ARC-2 overlap-only checks

- On the `17` matched ARC-2 tasks, LLM correlations with structure are the only ones that are individually significant in the human/LLM/non-LLM side-by-side comparison.
- The "non-LLM sits in the middle" pattern is descriptively real-looking but underpowered.

## Limitations

- The approved solver set is only `120` tasks.
- Human task-level overlap with the approved set is only `17` independent tasks.
- Human public-eval data are mostly ARC-2, so ARC-1 does not add much independent human-overlap power.
- Many complexity metrics are highly collinear, so multiple large `r` values do not imply many independent discoveries.
- Observed solver size is not minimal description length; it is the size of one approved solver implementation.
- The thinking analysis depends on model grouping and bounded pass-rate scales, so it needed more auditing than the other analyses.

## Bottom Line

The strongest defensible statement from this project is:

> In approved ARC solvers, structural program complexity is a strong proxy for LLM task difficulty. Humans and LLMs share part of the same difficulty axis, but structural solver complexity is significantly more LLM-linked than human-linked, while human-specific difficulty looks more time/search-like and more pair-heterogeneous.

The best next step, if more certainty is needed, is not to add more complexity metrics. It is to increase the matched human-overlap sample or to find a second benchmark with the same low-boilerplate solver structure.

## Main Files

- Master hypothesis catalog: `../../complexity_master_hypotheses.csv`
- Earlier synthesis memo: `../synthesis-writeup/synthesis_writeup.md`
- Statistical audit: `../../statistical_hypothesis_report.md`
- Human vs LLM comparison: `../../human_llm_difference_report.md`
- Human-specific addendum: `../../human_additional_findings.md`
- LLM complexity report: `../../approved_llm_complexity_report.md`
- Non-LLM addendum: `../non-llm-addendum/non_llm_complexity_addendum.md`
- ARC-2 overlap significance note: `../../non_llm_arc2_overlap_significance.md`

