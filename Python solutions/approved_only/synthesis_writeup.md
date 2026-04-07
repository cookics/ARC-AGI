# ARC Approved-Solver Complexity, Human Difficulty, and LLM Difficulty

## Scope

This memo consolidates the work done in this folder on the `120` approved ARC Python solvers and the linked human/LLM psychometric data. It is meant to answer four questions:

1. What did we actually compute?
2. Which findings are statistically supported versus tentative?
3. What do the residual analyses imply?
4. Is there a coherent through-line, or are these disconnected correlations?

The short answer is that the results look focused rather than random:

- There is a real **shared difficulty axis** between humans and LLMs.
- The **LLM-specific** part of difficulty is much more aligned with **structural solver complexity**.
- The **human-specific** part of difficulty is more aligned with **time/search burden** and with **within-task heterogeneity across test pairs**.
- The earlier negative `thinking_advantage` story does **not** survive the label audit and floor diagnostics cleanly enough to treat as a stable substantive result.

## Data and Filtering

- Solver set: `120` approved `solution.py` files from `arc.huikang.dev`, validated as correct on the local ARC task mirrors.
- Complexity study: all `120` approved Python files.
- LLM psychometric overlap: `56` approved eval task rows (`38` ARC-1 eval, `18` ARC-2 eval).
- Human-vs-LLM task overlap: `17` approved ARC-2 eval tasks with both human and LLM task-level measures.
- Human pair-level analysis: `110` public-eval task-pair rows with at least `8` human attempts.
- Multi-pair heterogeneity analysis: `90` pair rows across `44` public-eval tasks with multiple test pairs.

## What Was Measured

### Solver complexity

- Text/size proxies: `nonblank_lines`, `token_count`, `gzip_bytes`
- Structural proxies: `ast_node_count`, `cyclomatic_complexity`, `branch_node_count`, `max_nesting_depth`
- Description-length style proxies: `halstead_volume`, `halstead_effort`
- Dynamic/runtime proxies: `opcode_count_dynamic`, `elapsed_ms_total`, `elapsed_ms_per_test`, normalized runtime measures
- Composites: PCA, PCR, PLS, and ridge-based summaries

### Human/LLM psychometrics

- Human task difficulty and solve rate from the human analysis workspace
- LLM task difficulty from:
  - the earlier shared-model latent scale
  - a pooled all-model Rasch fit
  - simple logit difficulty / fail-rate summaries
- Pair-level human measures: difficulty, duration, attempts, solve rate, train/test pair counts
- Group-gap measures: human minus LLM, thinking minus standard

## Statistical Framework

### Main estimators

- Pearson correlations for association claims
- Bootstrap 95% confidence intervals
- Permutation p-values for correlation tests
- Benjamini-Hochberg FDR q-values across the main hypothesis table
- Bootstrap differences of correlations for “human vs LLM” comparison claims
- Grouped binomial GLMs for `thinking_advantage`
- OLS residualization for “shared vs specific” analyses

### Null hypotheses

- Shared-axis null: human and LLM outcomes are unrelated across tasks.
- Difference-of-correlation null: a candidate predictor is equally associated with human and LLM difficulty.
- Residual null: after removing the shared human/LLM axis, the remaining residual difficulty is unrelated to the proposed predictor.
- Human pair-level null: human pair difficulty is unrelated to the pair feature being tested.
- Thinking-gap null: the thinking-vs-standard gap is unrelated to difficulty, or equivalently the grouped GLM interaction is zero.

## What We Did, in Order

1. Downloaded all available ARC `solution.py` files from the site and validated them against the official ARC data.
2. Found that only the `approved` subset was reliably correct, so the analysis was restricted to those `120` files.
3. Computed a broad solver-complexity panel and found that structural metrics clustered strongly together.
4. Ran PCA/composite analyses and found a dominant structural component plus secondary runtime and grid-size components.
5. Joined solver complexity to latent human difficulty and then to a larger LLM psychometric panel.
6. Compared human and LLM difficulty directly on the approved overlap tasks.
7. Ran residual analyses to separate shared difficulty from human-specific and LLM-specific residual structure.
8. Audited the `thinking` labels and re-ran the `thinking_advantage` checks under multiple schemas and floor-sensitive subsets.

## Early Complexity Result

The earliest robust signal was that **structural code size** correlates strongly with item difficulty in this benchmark family.

- Best single structural proxy on the earlier latent-scale analysis: `ast_node_count`, `r = 0.666`
- Best conservative composite from PCR: leave-one-out `r = 0.691`
- On the expanded LLM analysis, `log1p(cyclomatic_complexity)` reached `r = 0.707` against simple LLM logit difficulty

The practical interpretation is not “longer programs are universally harder problems.” It is narrower:

> In a tightly standardized single-task domain like ARC, the amount of structured solver machinery needed to express a correct rule tracks task difficulty surprisingly well.

This already suggested that ARC difficulty is more about **rule structure** than about **raw compute time**.

![PCA overview](chart_pca_overview.png)

![Prediction comparison](chart_prediction_comparison.png)

## Through-Line

The strongest through-line is:

1. Humans and LLMs share a meaningful difficulty axis.
2. Once that shared axis is removed, the leftover human difficulty and leftover LLM difficulty do **not** look the same.
3. LLM-specific residual difficulty still looks like **solver structure**.
4. Human-specific residual difficulty looks more like **search/time burden**, and pair-level human data show substantial within-task heterogeneity that task-level solver metrics cannot capture.

That is a coherent story. It is not what random data-mining usually looks like.

![Synthesis through-line](chart_synthesis_throughline.png)

## Strongest Supported Claims

These are the main results that survived bootstrap CIs, permutation or model-based p-values, and FDR correction.

| Claim | Estimate | 95% CI | p | q |
|---|---:|---:|---:|---:|
| Human difficulty aligns with LLM difficulty on approved ARC-2 overlap (`S1`) | `r = 0.531` | `[0.178, 0.780]` | `0.0268` | `0.0366` |
| Human solve rate aligns with LLM pass rate (`S2`) | `r = 0.541` | `[0.226, 0.781]` | `0.0187` | `0.0280` |
| Cyclomatic complexity is more LLM-linked than human-linked (`D1`) | `delta-r = 0.441` | `[0.024, 0.909]` | `0.0378` | `0.0472` |
| Residual LLM difficulty still tracks cyclomatic complexity after removing human difficulty (`D4`) | `r = 0.603` | `[0.284, 0.816]` | `0.0117` | `0.0194` |
| Human pair difficulty tracks mean duration (`H1`) | `r = 0.391` | `[0.238, 0.539]` | `1.67e-4` | `3.57e-4` |
| Human duration is more informative than raw board size for human difficulty (`H3`) | `delta-r = 0.487` | `[0.255, 0.713]` | `1.25e-4` | `3.57e-4` |
| Human-over-LLM advantage shrinks with more test pairs (`H4`) | `r = -0.366` | `[-0.539, -0.165]` | `3.33e-4` | `6.25e-4` |
| Task identity explains a large share of pair-level human difficulty variation (`H5`) | `R^2 = 0.749` | `NA` | `7.93e-5` | `3.57e-4` |

## Supported But Mostly Internal Checks

These were important sanity checks rather than novel theoretical claims.

| Claim | Estimate | 95% CI | p | q |
|---|---:|---:|---:|---:|
| Earlier shared-model latent scale and pooled Rasch are almost the same LLM difficulty axis (`S3`) | `r = 0.991` | `[0.984, 0.997]` | `1.67e-4` | `3.57e-4` |
| Pooled Rasch and simple LLM logit difficulty are nearly identical on the approved subset (`S4`) | `r = 0.970` | `[0.953, 0.981]` | `1.67e-4` | `3.57e-4` |

These checks matter because they show the LLM difficulty signal is not fragile to the exact latent-score construction.

## Residual Analyses

The residual analyses are what make the story sharper.

### Human residual after removing LLM difficulty

Model:

- `human_difficulty ~ beta0 + beta1 * llm_difficulty`
- residual = what is left of human difficulty after removing shared LLM difficulty

Result:

- residual human difficulty vs mean human duration: `r = 0.470`
- 95% CI `[0.067, 0.766]`
- `p = 0.0505`, `q = 0.0583`

Interpretation:

- This is suggestive but just misses the main corrected threshold.
- It points in a consistent direction: the **human-specific** part of difficulty looks more time/search-like.

### LLM residual after removing human difficulty

Model:

- `llm_difficulty ~ beta0 + beta1 * human_difficulty`
- residual = what is left of LLM difficulty after removing shared human difficulty

Result:

- residual LLM difficulty vs cyclomatic complexity: `r = 0.603`
- 95% CI `[0.284, 0.816]`
- `p = 0.0117`, `q = 0.0194`

Interpretation:

- This is the cleanest evidence that solver structure carries a real **LLM-specific** signal, not just a generic difficulty effect shared with humans.

## Human Pair-Level Structure

The pair-level human data fill in an important piece the task-level solver analysis cannot see.

- Human pair difficulty tracks mean duration strongly enough to be credible: `r = 0.391`
- Raw board size alone is weak for human difficulty: `r = -0.096`, `p = 0.311`
- Tasks with more test pairs reduce the human-over-LLM gap: `r = -0.366`
- Among multi-pair public-eval tasks, the mean within-task difficulty range is `0.719`, with a maximum of `2.498`

That last point matters: a single task can contain very different human difficulty across its test pairs, but the solver-complexity file is only one program per task. So we should not expect a task-level program metric to fully explain pair-level human behavior.

![Human pair-level structure](chart_synthesis_human_pair_level.png)

## Human–LLM Similarity Versus Difference

### Similarity

- Human and LLM task difficulty are positively aligned on the overlap tasks.
- Human solve rate and LLM pass rate are also positively aligned.
- Different LLM psychometric constructions collapse onto almost the same axis.

### Difference

- Structural solver complexity is much more tightly aligned with LLM difficulty than with human difficulty.
- Human time cost is more informative for human difficulty than raw grid size is.
- Human pair-level heterogeneity is substantial and cannot be captured by one task-level solver file.

So the data suggest a **shared core difficulty signal**, plus **different secondary burdens**:

- LLM burden: structural rule specification
- Human burden: search/time and pair-level variability

## Thinking-Advantage Audit

The original thinking result looked strong under the legacy grouping:

- raw `thinking_advantage` vs difficulty: `r = -0.492`, `p = 1.67e-4`
- legacy grouped-binomial GLM interaction: `-0.718`, `p = 5.70e-6`

But this was not the end of the story. We then audited the labels and found five low-certainty model assignments that materially mattered:

- `QwQ-32B-Fireworks`
- `gemini-3-pro-preview`
- `gpt-5-pro-2025-10-06`
- `gpt-5-2-pro-2025-12-11-high`
- `gpt-5-2-pro-2025-12-11-medium`

After using the source-backed verified grouping and checking the floor pattern:

- standard-group zero-success items: `35 / 56`
- thinking-group zero-success items: `0 / 56`
- both-zero items: `0 / 56`

Under the verified grouping:

- all rows: raw `r = -0.378`, logit-gap `r = -0.294`
- standard-nonzero subset: raw `r = 0.280`, logit-gap `r = 0.091`
- both-groups-interior subset: raw `r = -0.009`, logit-gap `r = -0.156`

Interpretation:

- The negative raw-gap result is **not** caused by all-zero items.
- But it is strongly influenced by tasks where the standard group is pinned at zero while the thinking group is not.
- Once those floor-sensitive cases are removed, the effect is no longer stable.

So the responsible conclusion is:

> Treat the negative `thinking_advantage` result as an audited, fragile finding rather than a settled substantive claim.

![Thinking-advantage audit](chart_synthesis_thinking_audit.png)

## What Did Not Clearly Survive

These are not “wrong,” but they are not strong enough to treat as confirmed main claims.

| Claim | Estimate | 95% CI | p | q | Interpretation |
|---|---:|---:|---:|---:|---|
| Human duration is more associated with human than LLM difficulty (`D2`) | `delta-r = 0.374` | `[-0.058, 0.803]` | `0.0820` | `0.0879` | Suggestive but not corrected-significant |
| Residual human difficulty tracks duration after removing LLM difficulty (`D3`) | `r = 0.470` | `[0.067, 0.766]` | `0.0505` | `0.0583` | Right direction, borderline |
| Raw board size predicts human pair difficulty (`H2`) | `r = -0.096` | `[-0.275, 0.086]` | `0.311` | `0.311` | No support |

## Why This Looks Focused Rather Than Random

If these were just random correlations, I would expect the effects to be scattered without a consistent geometry. Instead, they line up into a fairly coherent pattern:

1. Independent LLM difficulty constructions collapse onto one axis.
2. Human and LLM difficulty share a moderate common axis.
3. Structural solver metrics repeatedly favor the LLM side.
4. Human duration and pair-level heterogeneity repeatedly favor the human side.
5. The one flashy extra story, `thinking_advantage`, is precisely the result that weakened when the labeling and floor assumptions were stress-tested.

That last point actually raises confidence in the rest of the analysis. The pipeline did not simply preserve every attractive result; it also downgraded one when the diagnostics stopped supporting it.

## Practical Interpretation

The cleanest working model from everything in this folder is:

- **Shared task difficulty** exists for humans and LLMs in ARC.
- **LLM item difficulty** is especially sensitive to the amount of structured solver logic needed to specify the rule.
- **Human item difficulty** is especially sensitive to search/time burden and pair-level variation in how the task unfolds.

That is a narrower and stronger claim than “program length measures intelligence” or “solver size is a universal law.” It is specifically about a benchmark family with standardized single-task solvers and low boilerplate.

## Main Figure Set

![Key tests](chart_synthesis_stats_forest.png)

## Files Added in This Pass

- `plot_synthesis_story.py`
- `chart_synthesis_throughline.png`
- `chart_synthesis_human_pair_level.png`
- `chart_synthesis_thinking_audit.png`
- `chart_synthesis_stats_forest.png`
- `chart_synthesis_manifest.json`
- `synthesis_writeup.md`
- `synthesis_writeup.tex`

## Bottom Line

The most defensible statement from the full analysis is:

> In approved ARC solvers, structural program complexity is a strong proxy for LLM task difficulty, while human-specific difficulty looks more search/time-like and more sensitive to within-task heterogeneity. Humans and LLMs share part of the same difficulty axis, but not all of it.

