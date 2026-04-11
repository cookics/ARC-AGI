# Expanded Old-Style Metrics For ARC-1 DSL Solvers

## Headline

- Full set best old-style expanded metric for pooled GPT+Claude pair difficulty is `log1p_branch_opcode_count_dynamic` with Pearson r = 0.382.
- Shared-regime gap<=0.30 best old-style expanded metric for pooled GPT+Claude pair difficulty is `nonblank_lines` with Pearson r = 0.440.
- Expanded cyclomatic complexity is now non-degenerate across 73 distinct values.
- Expanded branch-opcode tracing is now non-degenerate across 382 distinct values.

## Biggest Surface -> Expanded Improvements For Pooled Pair Difficulty

- gap_le_0.30 / `opcode_per_input_cell`: surface r = 0.058, expanded r = 0.225, gain = 0.167.
- full_set / `opcode_per_input_cell`: surface r = -0.102, expanded r = 0.040, gain = 0.142.
- gap_le_0.30 / `elapsed_ms_per_input_cell`: surface r = 0.138, expanded r = 0.250, gain = 0.111.
- full_set / `call_count_static`: surface r = 0.262, expanded r = 0.333, gain = 0.071.
- full_set / `halstead_volume`: surface r = 0.258, expanded r = 0.328, gain = 0.069.
- full_set / `nonblank_lines`: surface r = 0.262, expanded r = 0.330, gain = 0.068.
- gap_le_0.30 / `nonblank_lines`: surface r = 0.373, expanded r = 0.440, gain = 0.067.
- full_set / `token_count`: surface r = 0.267, expanded r = 0.331, gain = 0.063.
- full_set / `ast_node_count`: surface r = 0.267, expanded r = 0.330, gain = 0.063.
- full_set / `halstead_effort`: surface r = 0.242, expanded r = 0.304, gain = 0.062.
