# Old Python Metric Set vs Current ARC-1 DSL Analysis

## Headline

- Old approved-Python analysis strongest headline was `log1p_cyclomatic_complexity` vs `logit_difficulty_all` at Pearson r = 0.707 on n = 56.
- In the current DSL analysis, 7 of the old 30 metrics are degenerate because the DSL representation removes or compresses their variation.

## Degenerate Old Metrics Under The DSL Representation

- `function_count`: unique values = 1, notes = direct metric
- `branch_node_count`: unique values = 1, notes = direct metric
- `cyclomatic_complexity`: unique values = 1, notes = direct metric
- `max_nesting_depth`: unique values = 1, notes = direct metric
- `branch_opcode_count_dynamic`: unique values = 1, notes = Mapped to bundle_branch_opcode_count_dynamic.
- `log1p_branch_opcode_count_dynamic`: unique values = 1, notes = Derived from mapped branch opcode count.
- `log1p_cyclomatic_complexity`: unique values = 1, notes = Derived from cyclomatic_complexity.

## Best Old-Compatible Metrics On The Current Full Set

- `log1p_python_call_count_dynamic`: pooled pair-difficulty r = 0.343, human latent r = 0.142, old best was latent_difficulty_prev_intersection22 at 0.398.
- `complexity_pc1_score`: pooled pair-difficulty r = 0.335, human latent r = 0.193, old best was logit_difficulty_all at 0.641.
- `log1p_ast_node_count`: pooled pair-difficulty r = 0.327, human latent r = 0.200, old best was latent_difficulty_prev_intersection22 at 0.658.
- `log1p_opcode_count_dynamic`: pooled pair-difficulty r = 0.312, human latent r = 0.268, old best was pc1_difficulty_z at 0.404.
- `log1p_elapsed_ms_per_test`: pooled pair-difficulty r = 0.285, human latent r = 0.146, old best was rasch_rmsea_x2 at 0.502.
- `gzip_bytes`: pooled pair-difficulty r = 0.284, human latent r = 0.206, old best was latent_difficulty_prev_intersection22 at 0.597.
- `log1p_elapsed_ms_total`: pooled pair-difficulty r = 0.277, human latent r = 0.145, old best was rasch_rmsea_x2 at 0.487.
- `token_count`: pooled pair-difficulty r = 0.267, human latent r = 0.214, old best was latent_difficulty_prev_intersection22 at 0.660.
- `ast_node_count`: pooled pair-difficulty r = 0.267, human latent r = 0.215, old best was latent_difficulty_prev_intersection22 at 0.666.
- `call_count_static`: pooled pair-difficulty r = 0.262, human latent r = 0.220, old best was two_pl_difficulty_all_models at 0.563.

## Best Old-Compatible Metrics On The Gap<=0.30 Shared-Regime Subset

- `input_cells_total`: pooled pair-difficulty r = 0.424, human latent r = 0.357.
- `output_cells_total`: pooled pair-difficulty r = 0.419, human latent r = 0.325.
- `complexity_pc1_score`: pooled pair-difficulty r = 0.414, human latent r = 0.319.
- `log1p_peak_memory_bytes`: pooled pair-difficulty r = 0.403, human latent r = 0.312.
- `log1p_ast_node_count`: pooled pair-difficulty r = 0.402, human latent r = 0.303.
- `halstead_effort`: pooled pair-difficulty r = 0.391, human latent r = 0.358.
- `log1p_elapsed_ms_per_test`: pooled pair-difficulty r = 0.391, human latent r = 0.301.
- `gzip_bytes`: pooled pair-difficulty r = 0.391, human latent r = 0.327.
- `token_count`: pooled pair-difficulty r = 0.391, human latent r = 0.339.
- `ast_node_count`: pooled pair-difficulty r = 0.389, human latent r = 0.338.
