# CompressARC ARC-1 Training Complexity Correlations

- Source archive: https://raw.githubusercontent.com/iliao2345/CompressARC/master/results_for_the_blog_post/predictions_training.npz
- Overlap with validated ARC-1 DSL complexity table: 391 tasks.
- Final strict top-1 accuracy on the overlap: 121/391 = 30.9%.
- Final top-2 accuracy on the overlap: 140/391 = 35.8%.

## Complexity PC1

- Top-1 failure vs `complexity_pc1_score`: r = 0.036, p = 0.473.
- Top-2 failure vs `complexity_pc1_score`: r = 0.059, p = 0.247.

## Best Broad Metrics

metric,category,outcome,n,pearson_r,pearson_p,spearman_rho,spearman_p
geometry_op_count,dsl_structure,compressarc_top2_failure,391,0.20964055092903935,2.9327543659778033e-05,0.23941036749884087,1.6786995453115174e-06
bundle_opcode_count_dynamic,dynamic_execution,compressarc_top2_failure,391,0.18840854106011137,0.00017880340321431903,0.15953986351302485,0.0015514092018879793
solver_opcode_count_dynamic,dynamic_execution,compressarc_top2_failure,391,0.18840854106011137,0.00017880340321431903,0.15953986351302485,0.0015514092018879793
function_count,expanded_python_static,compressarc_top2_failure,391,0.18038350998530858,0.0003372298476647977,0.18631077260534726,0.00021159916930388193
nonblank_lines,expanded_python_static,compressarc_top2_failure,391,0.16903449947275437,0.0007910158917586662,0.1667926042619675,0.0009303906695800336


## First-Correct-Step Signals

metric,category,outcome,n,pearson_r,pearson_p,spearman_rho,spearman_p
assignment_count,dsl_structure,compressarc_top2_first_correct_step,150,0.2643900605777588,0.0010777842673767753,0.3144063698966792,8.917509649163911e-05
ast_call_count,dsl_structure,compressarc_top2_first_correct_step,150,0.2643900605777588,0.0010777842673767753,0.3144063698966792,8.917509649163911e-05
temp_var_count,dsl_structure,compressarc_top2_first_correct_step,150,0.2643900605777588,0.0010777842673767753,0.3144063698966792,8.917509649163911e-05
source_line_count,python_static,compressarc_top2_first_correct_step,150,0.2643900605777588,0.0010777842673767753,0.3144063698966792,8.917509649163911e-05
max_dependency_depth,dsl_structure,compressarc_top2_first_correct_step,150,0.2634910552308088,0.0011225358424970468,0.2809322132179629,0.0004970889862332748


Interpretation: the final binary CompressARC outcome has only a weak relationship with the existing global complexity PC1, but some broader structural and dynamic metrics show modest correlations around r ≈ 0.18 to 0.21. The archive's richer training-trajectory measure `compressarc_top2_first_correct_step` is somewhat more complexity-sensitive than the final solved/not-solved flag.
