# CompressARC Curve-Based Difficulty Correlations

- Input difficulty table: C:\Users\cooki\Desktop\ARC-AGI\analysis-efficiency\arc_training_difficulty\arc_training_difficulty.csv
- Overlap with validated ARC-1 complexity table: 391 tasks.
- Solved-by-curve overlap used for the continuous analysis: 150 tasks (38.4%).
- `difficulty_score` is the normalized first step where the true solution first appears in the top-2 oracle picks.
- On the solved-only subset, `difficulty_score` and `first_hit_step` are perfectly rank-equivalent, so they give the same ordering.

## Locked Results

- `complexity_pc1_score`: r = 0.181, p = 0.0262, rho = 0.252.
- `dsl_complexity_pc1`: r = 0.255, p = 0.00162, rho = 0.328.

## Best Broad Metrics

metric,category,n,pearson_r,pearson_p,spearman_rho,spearman_p
temp_var_count,dsl_structure,150,0.2643900605777588,0.0010777842673767753,0.3144063698966792,8.917509649163911e-05
assignment_count,dsl_structure,150,0.2643900605777588,0.0010777842673767753,0.3144063698966792,8.917509649163911e-05
ast_call_count,dsl_structure,150,0.2643900605777588,0.0010777842673767753,0.3144063698966792,8.917509649163911e-05
source_line_count,python_static,150,0.2643900605777588,0.0010777842673767753,0.3144063698966792,8.917509649163911e-05
max_dependency_depth,dsl_structure,150,0.2634910552308088,0.0011225358424970468,0.2809322132179629,0.0004970889862332748


Interpretation: the solved-only continuous CompressARC difficulty signal is more complexity-linked than the final binary solved/not-solved label, but it is still only modest in size. The strongest broad single metrics are structural or dynamic counts rather than the global expanded complexity PC1.
