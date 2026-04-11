# Old vs New Complexity Regimes

## Main read

- The old high correlation is real, but it came from a many-model item-difficulty signal on a small overlap sample.
- The new ARC-1 training GPT+Claude signal is much coarser, and the same complexity metric is significantly weaker on the full set.
- Matching the solve-rate regime helps a lot, but it still does not fully recover the old many-model effect.

## Same-metric comparison: `log1p_cyclomatic_complexity`

- new_arc1_train_full_two_model: new r = 0.322 (n=391), vs old overlap56 r = 0.707 (p = 0.0001856), vs old arc1_eval r = 0.680 (p = 0.004947).
- new_arc1_train_gap30_two_model: new r = 0.389 (n=96), vs old overlap56 r = 0.707 (p = 0.00623), vs old arc1_eval r = 0.680 (p = 0.03443).
- new_arc1_train_full_human: new r = 0.064 (n=391), vs old overlap56 r = 0.707 (p = 2.415e-08), vs old arc1_eval r = 0.680 (p = 1.435e-05).
- new_arc1_train_gap30_human: new r = 0.316 (n=96), vs old overlap56 r = 0.707 (p = 0.001282), vs old arc1_eval r = 0.680 (p = 0.01123).

## Within current ARC-1 training data

- full / log1p_cyclomatic_complexity: human r = 0.064, pooled pair r = 0.322, human-vs-pooled alignment = 0.245, Williams p = 1.774e-05.
- full / ast_node_count: human r = 0.143, pooled pair r = 0.330, human-vs-pooled alignment = 0.245, Williams p = 0.00175.
- full / complexity_pc1_score: human r = 0.125, pooled pair r = 0.365, human-vs-pooled alignment = 0.245, Williams p = 4.932e-05.
- gap30 / log1p_cyclomatic_complexity: human r = 0.316, pooled pair r = 0.389, human-vs-pooled alignment = 0.825, Williams p = 0.2.
- gap30 / ast_node_count: human r = 0.398, pooled pair r = 0.435, human-vs-pooled alignment = 0.825, Williams p = 0.4946.
- gap30 / complexity_pc1_score: human r = 0.338, pooled pair r = 0.430, human-vs-pooled alignment = 0.825, Williams p = 0.09825.