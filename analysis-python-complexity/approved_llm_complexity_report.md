# LLM Difficulty vs Solver Complexity

- Overlap rows analyzed: `56`
- Unique overlap tasks: `56`

## Dataset Coverage

- arc_agi_1_eval: `24` models, `400` tasks, `372` variable tasks, PC1 explains `0.487` of model-response variance
- arc_agi_2_eval: `39` models, `120` tasks, `78` variable tasks, PC1 explains `0.426` of model-response variance

## Strongest Correlations By LLM Outcome

- abs_thinking_advantage: best metric is `halstead_effort` with Pearson `-0.497` and Spearman `-0.465`
- abs_thinking_logit_advantage: best metric is `ast_node_count` with Pearson `-0.496` and Spearman `-0.468`
- binary_entropy_bits: best metric is `halstead_effort` with Pearson `-0.291` and Spearman `-0.257`
- fail_rate_all: best metric is `log1p_cyclomatic_complexity` with Pearson `0.687` and Spearman `0.662`
- item_total_corr: best metric is `function_count` with Pearson `-0.449` and Spearman `-0.230`
- latent_difficulty_prev_intersection22: best metric is `log1p_cyclomatic_complexity` with Pearson `0.691` and Spearman `0.648`
- log1p_two_pl_discrimination: best metric is `log1p_ast_node_count` with Pearson `-0.327` and Spearman `-0.364`
- log1p_two_pl_info_theta0: best metric is `log1p_opcode_count_dynamic` with Pearson `-0.218` and Spearman `0.065`
- log1p_two_pl_max_info: best metric is `log1p_ast_node_count` with Pearson `-0.321` and Spearman `-0.364`
- logit_difficulty_all: best metric is `log1p_cyclomatic_complexity` with Pearson `0.707` and Spearman `0.661`
- pc1_difficulty_z: best metric is `log1p_branch_opcode_count_dynamic` with Pearson `0.424` and Spearman `0.481`
- pc1_discrimination: best metric is `function_count` with Pearson `-0.414` and Spearman `-0.254`
- rasch_abs_z_infit: best metric is `log1p_ast_node_count` with Pearson `-0.385` and Spearman `-0.357`
- rasch_abs_z_misfit: best metric is `log1p_ast_node_count` with Pearson `-0.368` and Spearman `-0.334`
- rasch_abs_z_outfit: best metric is `elapsed_ms_total` with Pearson `0.452` and Spearman `0.144`
- rasch_difficulty_all_models_pooled: best metric is `ast_node_count` with Pearson `0.653` and Spearman `0.623`
- rasch_infit: best metric is `input_cells_total` with Pearson `0.469` and Spearman `0.382`
- rasch_outfit: best metric is `elapsed_ms_total` with Pearson `0.587` and Spearman `0.306`
- rasch_rmsea_x2: best metric is `elapsed_ms_total` with Pearson `0.605` and Spearman `0.409`
- response_sd_all: best metric is `halstead_effort` with Pearson `-0.241` and Spearman `-0.270`
- thinking_advantage: best metric is `halstead_effort` with Pearson `-0.524` and Spearman `-0.491`
- thinking_logit_advantage: best metric is `halstead_effort` with Pearson `-0.545` and Spearman `-0.497`
- two_pl_difficulty_all_models: best metric is `ast_node_count` with Pearson `0.644` and Spearman `0.614`
- two_pl_discrimination: best metric is `log1p_ast_node_count` with Pearson `-0.321` and Spearman `-0.364`
- two_pl_info_theta0: best metric is `log1p_opcode_count_dynamic` with Pearson `-0.238` and Spearman `0.065`
- two_pl_max_info: best metric is `log1p_ast_node_count` with Pearson `-0.318` and Spearman `-0.364`

## Outcome Families

- difficulty: strongest result is `logit_difficulty_all` with `log1p_cyclomatic_complexity` at Pearson `0.707`
- discrimination_information: strongest result is `item_total_corr` with `function_count` at Pearson `-0.449`
- fit: strongest result is `rasch_rmsea_x2` with `elapsed_ms_total` at Pearson `0.605`
- group_gap: strongest result is `thinking_logit_advantage` with `halstead_effort` at Pearson `-0.545`

## Headline Stability By Dataset

- arc_agi_1_eval: `log1p_cyclomatic_complexity` vs `logit_difficulty_all` has Pearson `0.680` on `n=38`
- arc_agi_1_eval: `ast_node_count` vs `rasch_difficulty_all_models_pooled` has Pearson `0.534` on `n=37`
- arc_agi_1_eval: `halstead_effort` vs `thinking_logit_advantage` has Pearson `-0.200` on `n=38`
- arc_agi_1_eval: `elapsed_ms_total` vs `rasch_rmsea_x2` has Pearson `0.712` on `n=37`
- arc_agi_2_eval: `log1p_cyclomatic_complexity` vs `logit_difficulty_all` has Pearson `0.558` on `n=18`
- arc_agi_2_eval: `ast_node_count` vs `rasch_difficulty_all_models_pooled` has Pearson `0.547` on `n=18`
- arc_agi_2_eval: `halstead_effort` vs `thinking_logit_advantage` has Pearson `-0.596` on `n=18`
- arc_agi_2_eval: `elapsed_ms_total` vs `rasch_rmsea_x2` has Pearson `0.533` on `n=18`