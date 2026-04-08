# Human vs LLM Solver-Complexity Comparison

## Scope

- Human source: `analysis-human/papers/human-testing/tables/public_eval_human_vs_models.csv`
- LLM source: `analysis-python-complexity/approved_llm_complexity_join.csv`
- Comparison scope: approved ARC-AGI-2 eval tasks that also appear in the public human testing table
- Overlap: 17 tasks total, 14 with at least 8 total human attempts

## Headline Findings

- Human and LLM difficulty are only moderately aligned on the overlap: `r = 0.531` for human difficulty vs LLM logit difficulty, and `r = 0.507` vs pooled Rasch difficulty.
- Human solve rate and LLM pass rate are similarly only moderately aligned: `r = 0.541`.
- Human item difficulty is tied to human time cost: `r = 0.425` with weighted mean human duration.
- LLM difficulty is almost unrelated to human time cost on the same tasks: `r = 0.050`.

## Thinking-Advantage Pattern

- Across all approved eval overlap items, `thinking_advantage` falls as LLM difficulty rises: Pearson `r = -0.678` against logit difficulty.
- A quadratic fit is clearly better than a straight line here: linear `R^2 = 0.460` vs quadratic `R^2 = 0.595`.
- Interpretation: thinking models gain the most on medium-hard items, but the gap compresses on the hardest items where both standard and thinking models often fail.

## Complexity Contrast

- `ast_node_count`: human difficulty `r = 0.233`, LLM difficulty `r = 0.537`.
- `token_count`: human difficulty `r = 0.203`, LLM difficulty `r = 0.524`.
- `cyclomatic_complexity`: human difficulty `r = 0.150`, LLM difficulty `r = 0.591`.
- `complexity_pc1_score`: human difficulty `r = 0.135`, LLM difficulty `r = 0.535`.
- Runtime burden does not explain human difficulty especially well either, but it matters more for human-vs-model gap and residual differences than structural size does.

## Working Hypotheses Supported by Current Data

- Structural solver complexity looks more like an LLM difficulty signal than a human difficulty signal in this overlap slice.
- Human difficulty appears to be closer to time-on-task and interactive search burden than to the amount of code needed in a final solver.
- The hardest model items are not the items with the biggest thinking-model gain; instead, thinking advantage seems to collapse on the hardest tasks.
- Human-vs-LLM differences are real but only moderately estimated here because the overlap is small and the human table is task-pair level before aggregation.
