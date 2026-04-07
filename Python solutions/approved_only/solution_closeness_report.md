# Solution-Closeness Analysis

## Scope

- LLM side: true partial-credit scoring from stored prediction grids.
- Human side: raw human wrong-answer grids are not present in the repo, so the human analysis asks whether softer LLM pair-level signals explain human pair outcomes better than exact-match rates do.

## Metric Families

- `exact_current`: existing attempt-1-first exact-match behavior used in the current tables.
- `exact_any`: exact match if either stored attempt solves the pair or task.
- `cell_accuracy_padded`: top-left-aligned cell agreement after padding to the larger canvas.
- `shape_iou`: overlap-over-union of output canvas shape.
- `color_iou`: multiset overlap of color counts.
- `component_size_iou`: overlap of connected-component size signatures by color.
- `adjacency_iou`: overlap of horizontal/vertical neighbor-pair signatures.
- `soft_composite`: mean of the five soft metrics above.

## High-Level Takeaways

- ARC-2 model-pair rows scored: `6279`.
- Approved task-model rows scored: `1614`.
- `attempt_2` already matters: pair-level exact improves over current scoring on `3.0%` of ARC-2 model-pair rows, and task-level exact improves on `5.3%` of approved model-task rows.
- Human-pair metric mean off-diagonal correlation: `0.518`.
- LLM-task metric mean off-diagonal correlation: `0.655`.
- Human-side raw `p < 0.05` improvements: `0` tests; FDR-significant: `0`.
- Human-side raw `p < 0.05` degradations: `1` tests.
- LLM-side raw `p < 0.05` improvements: `1` tests; FDR-significant: `0`.
- LLM-side raw `p < 0.05` degradations: `6` tests.

## Strongest Human-Side Improvements

- No human-side comparison cleared raw `p < 0.05`.

## Strongest LLM-Side Improvements

- `approved_arc2_overlap` | `Mean human duration`: `Exact match (either stored attempt)` beats baseline by `delta r = 0.120` (candidate `r = 0.171`, baseline `r = 0.050`, p `0.0130`, q `0.1407`).

## Largest Degradations

- `llm` | `Adjacency IoU` underperforms on `Cyclomatic complexity` by `delta r = -0.283` (candidate `r = 0.337`, baseline `r = 0.620`, p `0.0027`).
- `llm` | `Color multiset IoU` underperforms on `Cyclomatic complexity` by `delta r = -0.265` (candidate `r = 0.355`, baseline `r = 0.620`, p `0.0070`).
- `human` | `Shape IoU` underperforms on `Human solve rate` by `delta r = -0.240` (candidate `r = 0.105`, baseline `r = 0.345`, p `0.0418`).
- `llm` | `Padded cell accuracy` underperforms on `Cyclomatic complexity` by `delta r = -0.234` (candidate `r = 0.386`, baseline `r = 0.620`, p `0.0168`).
- `llm` | `Adjacency IoU` underperforms on `Complexity PC1` by `delta r = -0.223` (candidate `r = 0.417`, baseline `r = 0.641`, p `0.0143`).
- `llm` | `Padded cell accuracy` underperforms on `Complexity PC1` by `delta r = -0.206` (candidate `r = 0.435`, baseline `r = 0.641`, p `0.0323`).
- `llm` | `Color multiset IoU` underperforms on `Complexity PC1` by `delta r = -0.189` (candidate `r = 0.452`, baseline `r = 0.641`, p `0.0343`).

## Notes

- The human-side results are about better task-pair alignment with human outcomes, not direct partial-credit scoring of human submitted grids.
- `exact_any` isolates the effect of honoring the stored second attempt before any softer scoring is added.
- All improvement tests use paired bootstrap differences of correlations on the same sampled rows.