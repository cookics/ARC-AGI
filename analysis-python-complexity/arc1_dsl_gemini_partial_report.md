# ARC-1 Gemini Partial Analysis

- Gemini completed-task subset size: 156
- Gemini solve rate on completed artifacts: 0.506
- Gemini pair mean rate on subset: 0.526
- Human mean solve rate on subset: 0.756
- 4-model pooled pair mean rate on subset: 0.216
- 5-model pooled pair mean rate on subset: 0.278

## Complexity PC1

- Human: r = 0.136
- 4-model pooled LLM: r = 0.396
- 5-model pooled LLM: r = 0.493
- Gemini alone: r = 0.412

## Locked Metric Comparisons

- gemini_completed_set / `complexity_pc1_score`: llm4_pair_difficulty r = 0.396, llm5_pair_difficulty r = 0.493, Williams p = 0.0002915.
- gemini_completed_set / `complexity_pc1_score`: human_difficulty_complete r = 0.136, llm5_pair_difficulty r = 0.493, Williams p = 0.0001432.
- gemini_completed_set / `log1p_cyclomatic_complexity`: llm4_pair_difficulty r = 0.329, llm5_pair_difficulty r = 0.406, Williams p = 0.005945.
- gemini_completed_set / `log1p_cyclomatic_complexity`: human_difficulty_complete r = 0.048, llm5_pair_difficulty r = 0.406, Williams p = 0.0002484.
- gemini_completed_set / `ast_node_count`: llm4_pair_difficulty r = 0.309, llm5_pair_difficulty r = 0.432, Williams p = 6.75e-06.
- gemini_completed_set / `ast_node_count`: human_difficulty_complete r = 0.163, llm5_pair_difficulty r = 0.432, Williams p = 0.004891.
- gemini_completed_set / `log1p_branch_opcode_count_dynamic`: llm4_pair_difficulty r = 0.477, llm5_pair_difficulty r = 0.544, Williams p = 0.01045.
- gemini_completed_set / `log1p_branch_opcode_count_dynamic`: human_difficulty_complete r = 0.120, llm5_pair_difficulty r = 0.544, Williams p = 4.198e-06.
- gemini_completed_gap_le_0.30 / `complexity_pc1_score`: llm4_pair_difficulty r = 0.369, llm5_pair_difficulty r = 0.429, Williams p = 0.1752.
- gemini_completed_gap_le_0.30 / `complexity_pc1_score`: human_difficulty_complete r = 0.334, llm5_pair_difficulty r = 0.429, Williams p = 0.338.
- gemini_completed_gap_le_0.30 / `log1p_cyclomatic_complexity`: llm4_pair_difficulty r = 0.273, llm5_pair_difficulty r = 0.317, Williams p = 0.3423.
- gemini_completed_gap_le_0.30 / `log1p_cyclomatic_complexity`: human_difficulty_complete r = 0.253, llm5_pair_difficulty r = 0.317, Williams p = 0.5293.
- gemini_completed_gap_le_0.30 / `ast_node_count`: llm4_pair_difficulty r = 0.330, llm5_pair_difficulty r = 0.402, Williams p = 0.109.
- gemini_completed_gap_le_0.30 / `ast_node_count`: human_difficulty_complete r = 0.271, llm5_pair_difficulty r = 0.402, Williams p = 0.1905.
- gemini_completed_gap_le_0.30 / `log1p_branch_opcode_count_dynamic`: llm4_pair_difficulty r = 0.434, llm5_pair_difficulty r = 0.482, Williams p = 0.2616.
- gemini_completed_gap_le_0.30 / `log1p_branch_opcode_count_dynamic`: human_difficulty_complete r = 0.354, llm5_pair_difficulty r = 0.482, Williams p = 0.1807.
