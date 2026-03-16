# Walkthrough: ARC-AGI Psychometric Analysis Paper

## Goal
Consolidate all existing ARC-AGI benchmark analyses into a single, self-contained LaTeX paper with coherent figures and data tables.

## What Was Done

### 1. Folder Structure
Created `Psychometric Analysis/` with organized subdirectories:
```
Psychometric Analysis/
├── paper.tex          ← LaTeX source
├── paper.pdf          ← Compiled paper (2.3 MB)
├── scripts/           ← All analysis code (7 files)
├── figures/           ← 8 figures (PNG + PDF, 16 files)
├── tables/            ← 4 CSV data tables
└── data/              ← (reserved for future use)
```

### 2. Master Python Script (`generate_all.py`)
Wrote a ~500-line Python script that replaces the functionality of all 4 R scripts:
- Loads raw JSON data from `arc_agi_v1_public_eval/` and `arc_agi_v2_public_eval/`
- Builds binary response matrices (25 models × 372 tasks)
- Runs PCA, Rasch ability estimation, and permutation test (200 iterations)
- Generates 7 coherent figures with identical styling
- Exports 4 CSV tables

### 3. Figures Generated

| Figure | Description |
|--------|-------------|
| `fig1_leaderboard` | Horizontal bar chart of model accuracy |
| `fig2_response_matrix_v1` | Guttman-sorted heatmap (V1) |
| `fig2b_response_matrix_v2` | Guttman-sorted heatmap (V2) |
| `fig3_scree_plot` | PCA scree plot (PC1 = 48.7%) |
| `fig4_cognitive_map` | PC1 vs PC2 scatter |
| `fig5_ability` | Rasch θ with AI-IQ labels |
| `fig6_dendrogram` | Ward clustering dendrogram |
| `fig7_difficulty_dist` | Task difficulty histogram |

### 4. Tables Exported

| Table | Contents |
|-------|----------|
| `table1_leaderboard.csv` | Model accuracy V1 + V2 |
| `table2_diagnostics.csv` | Rasch Outfit MSQ, P-values |
| `table3_iq_scores.csv` | Theta and AI-IQ scores |
| `table4_scale_metrics.csv` | Global scale metrics (H = 0.779) |

### 5. LaTeX Paper
Full paper with:
- Abstract, Introduction, Methods, Results (8 subsections), Discussion (4 subsections), Conclusion
- All 7 figures embedded
- 2 data tables (leaderboard + scale metrics)
- Complete mathematical notation for Rasch model and Loevinger's H

### 6. Compilation
- Installed MiKTeX via `winget`
- Compiled with `pdflatex` (2 passes, 0 errors)
- Final PDF: 2.3 MB, ~10 pages

## Key Findings Documented in Paper
1. **Unidimensional**: PC1 = 48.7%, Loevinger's H = 0.779
2. **Determinism problem**: Classical Rasch diagnostics fail on LLMs
3. **Thinking ≠ different**: Same latent dimension, just higher scores
4. **Guttman scalability**: Clear staircase in response matrix
5. **V2 is harder**: Max 32.5% vs 96.0% on V1

## Verification
- ✅ `generate_all.py` ran to completion (exit code 0)
- ✅ All 16 figure files generated (8 PNG + 8 PDF)
- ✅ All 4 CSV tables generated
- ✅ LaTeX compiled with 0 errors
- ✅ PDF produced at 2.3 MB
