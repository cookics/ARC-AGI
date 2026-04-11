# ARC-1 Extra LLMs Analysis

## Added Models

- Nemotron coverage: 400 task artifacts, 391 validated tasks, solve rate on validated tasks = 0.123.
- Gemma coverage: 399 task artifacts, 390 validated tasks, solve rate on validated tasks = 0.223.

## Pooled Difficulty

- Two-model pooled pair mean rate = 0.263; four-model pooled pair mean rate = 0.221.
- Two-model effective model count = 1.234; four-model effective model count = 1.483.

## Locked Metric Comparison

- full_set / `log1p_cyclomatic_complexity`: pooled_pair_difficulty r = 0.322, llm4_pair_difficulty r = 0.379, Williams p = 0.0005856.
- full_set / `log1p_cyclomatic_complexity`: human_difficulty_complete r = 0.064, llm4_pair_difficulty r = 0.379, Williams p = 1.072e-07.
- full_set / `ast_node_count`: pooled_pair_difficulty r = 0.330, llm4_pair_difficulty r = 0.364, Williams p = 0.03741.
- full_set / `ast_node_count`: human_difficulty_complete r = 0.143, llm4_pair_difficulty r = 0.364, Williams p = 0.0001815.
- full_set / `complexity_pc1_score`: pooled_pair_difficulty r = 0.365, llm4_pair_difficulty r = 0.418, Williams p = 0.001042.
- full_set / `complexity_pc1_score`: human_difficulty_complete r = 0.125, llm4_pair_difficulty r = 0.418, Williams p = 4.715e-07.
- full_set / `log1p_branch_opcode_count_dynamic`: pooled_pair_difficulty r = 0.382, llm4_pair_difficulty r = 0.434, Williams p = 0.001186.
- full_set / `log1p_branch_opcode_count_dynamic`: human_difficulty_complete r = 0.118, llm4_pair_difficulty r = 0.434, Williams p = 4.926e-08.
- gap_le_0.30 / `log1p_cyclomatic_complexity`: pooled_pair_difficulty r = 0.367, llm4_pair_difficulty r = 0.462, Williams p = 0.05954.
- gap_le_0.30 / `log1p_cyclomatic_complexity`: human_difficulty_complete r = 0.391, llm4_pair_difficulty r = 0.462, Williams p = 0.244.
- gap_le_0.30 / `ast_node_count`: pooled_pair_difficulty r = 0.447, llm4_pair_difficulty r = 0.543, Williams p = 0.04505.
- gap_le_0.30 / `ast_node_count`: human_difficulty_complete r = 0.471, llm4_pair_difficulty r = 0.543, Williams p = 0.2156.
- gap_le_0.30 / `complexity_pc1_score`: pooled_pair_difficulty r = 0.402, llm4_pair_difficulty r = 0.502, Williams p = 0.04223.
- gap_le_0.30 / `complexity_pc1_score`: human_difficulty_complete r = 0.424, llm4_pair_difficulty r = 0.502, Williams p = 0.1905.
- gap_le_0.30 / `log1p_branch_opcode_count_dynamic`: pooled_pair_difficulty r = 0.367, llm4_pair_difficulty r = 0.450, Williams p = 0.1016.
- gap_le_0.30 / `log1p_branch_opcode_count_dynamic`: human_difficulty_complete r = 0.376, llm4_pair_difficulty r = 0.450, Williams p = 0.2316.

## Headline

- Full-set `log1p_cyclomatic_complexity`: two-model pooled r = 0.322, four-model pooled r = 0.379.
- Matched (`gap<=0.30`) `log1p_cyclomatic_complexity`: two-model pooled r = 0.367, four-model pooled r = 0.462.
- Human vs four-model pooled pair difficulty on `gap<=0.30` = 0.799.
