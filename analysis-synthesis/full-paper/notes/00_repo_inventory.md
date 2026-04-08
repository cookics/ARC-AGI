# Repo Inventory For The Full ARC Master Paper

## Scope Rule

This inventory is limited to the ARC analysis line that appears to belong to the main research narrative:

- LLM psychometrics
- human testing and human/LLM alignment
- non-LLM psychometrics
- Python-solver complexity and closeness work
- cross-ARC latent linkage
- efficiency analysis
- mechanistic paper synthesis from recent non-LLM ARC papers

## 1. Canonical Data Roots

### `data-human/`

Primary role:
- canonical ARC-AGI-2 human attempt logs

Key files:
- `data-human/README.md`
- `data-human/test_pair_attempts.csv`

### `data-llm/`

Primary role:
- canonical ARC task JSONs and public-eval LLM prediction corpora

Key files:
- `data-llm/ARC-AGI/README.md`
- `data-llm/ARC-AGI-2/readme.md`
- `data-llm/arc_agi_v1_public_eval/README.md`
- `data-llm/arc_agi_v2_public_eval/README.md`

### `data-non-llm/`

Primary role:
- raw and processed non-LLM artifacts

Key files:
- `data-non-llm/README.md`
- `data-non-llm/processed/compress_arc_predictions_evaluation_summary.json`
- `data-non-llm/processed/varc_predictions_summary.json`
- `data-non-llm/raw/TRM-ARC-AGI-II/README.md`

### `data-python-programs/`

Primary role:
- validated approved Python solver corpus used by the complexity line

Key files:
- `data-python-programs/approved_only_data/README.md`
- `data-python-programs/approved_only_data/summary.json`
- `data-python-programs/approved_only_data/approved_task_files.csv`
- `data-python-programs/approved_only_data/approved_task_index.json`

## 2. Analysis Workstreams

### `analysis-llm-psychometrics/`

Key manuscript and support files:
- `analysis-llm-psychometrics/README.md`
- `analysis-llm-psychometrics/papers/arc-psychometrics/paper.tex`
- `analysis-llm-psychometrics/export_package/README.md`
- `analysis-llm-psychometrics/export_package/notes/findings_manifest.md`
- `analysis-llm-psychometrics/export_package/paper/report.tex`

### `analysis-human/`

Key files:
- `analysis-human/papers/human-testing/human_testing_psychometric_report.md`
- `analysis-human/papers/human-testing/bootstrap_context_note.md`
- `analysis-human/papers/human-testing/latent_difficulty_alignment_note.md`
- `analysis-human/papers/human-testing/temporal_hypothesis_note.md`
- `analysis-human/creme-analysis/creme_thesis_synthesis.md`

### `analysis-non-llm/`

Key files:
- `analysis-non-llm/papers/psychometric/report.md`
- `analysis-non-llm/papers/psychometric/paper_results_section.md`
- `analysis-non-llm/papers/psychometric/hypothesis_synthesis.md`
- `analysis-non-llm/papers/arc-synthesis/README.md`
- `analysis-non-llm/papers/arc-synthesis/arc_synthesis.tex`

### `analysis-python-complexity/`

Key core files:
- `analysis-python-complexity/papers/complexity-master/complexity_master_paper.md`
- `analysis-python-complexity/papers/complexity-master/complexity_master_paper.tex`
- `analysis-python-complexity/latent_complexity_report.md`
- `analysis-python-complexity/approved_llm_complexity_report.md`
- `analysis-python-complexity/human_llm_difference_report.md`
- `analysis-python-complexity/human_additional_findings.md`
- `analysis-python-complexity/solution_closeness_report.md`
- `analysis-python-complexity/human_closeness_augmentation_report.md`
- `analysis-python-complexity/closeness_model_suite_report.md`
- `analysis-python-complexity/statistical_hypothesis_report.md`

### `analysis-latent-crossarc/`

Key files:
- `analysis-latent-crossarc/README.md`
- `analysis-latent-crossarc/report.md`
- `analysis-latent-crossarc/summary.json`
- `analysis-latent-crossarc/latent_split_half_report.md`
- `analysis-latent-crossarc/llm_sparsity_stress_report.md`

### `analysis-efficiency/`

Key files:
- `analysis-efficiency/papers/efficiency/report.md`
- `analysis-efficiency/papers/efficiency/paper.tex`
- `analysis-efficiency/papers/efficiency/analysis_results.json`

## 3. Literature Folder

### `papers-literature/`

Files:
- `Gao et al. - 2025 - Universal Reasoning Model.pdf`
- `Hu et al. - 2025 - ARC Is a Vision Problem!.pdf`
- `Jolicoeur-Martineau - 2025 - Less is More Recursive Reasoning with ...`
- `McGovern - 2025 - Test-time Adaptation of Tiny Recursive Models.pdf`

## 4. Immediate Master-Paper Use

This inventory implies the master paper needs at least the following major bodies:

1. Repository-wide setup and data provenance.
2. LLM psychometrics and broader benchmark-manifold context.
3. Human psychometrics and human-vs-model overlap.
4. Non-LLM psychometrics and complementarity.
5. Solver complexity as the main integrative engine.
6. Cross-ARC linkage and latent stability.
7. Efficiency and source-family tradeoffs.
8. External architectural synthesis and mechanistic interpretation.
9. Integrated discussion, limitations, and a forward research program.
