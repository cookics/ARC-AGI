# ARC Papers Synthesis

This directory contains a LaTeX synthesis of recent architectural approaches to the Abstraction and Reasoning Corpus (ARC).

## 2026-03-16: Mechanistic Through Lines Synthesis
**Method:**
We analyzed four recent papers detailing ARC solvers (URM, TRM, VARC, and a compute-constrained test-time adaptation study by McGovern). We extracted the core architectural modifications and their corresponding ablation results to identify a shared "mechanistic through line."

**Findings:**
We found that despite their different motivations (e.g., biological plausibility, vision priors, parameter efficiency), all four successful "small model" architectures converge on the same fundamental loop: 
1. Use strong inductive bias (2D convolutions or spatial canvases).
2. Iterate a hypothesis operator (via recurrent weight-tying or gradient updates).
3. Select candidates at test time (via augmentation voting or consistency checking).

The synthesis paper `arc_synthesis.tex` documents this convergence and argues that the performance gains often attributed to "scale" in these small models is actually driven by test-time search and structural recurrence.

**Build Instructions:**
To compile the document into a PDF without path-length issues, use the included PowerShell script:
```powershell
.\build_paper.ps1
```
This script temporarily copies files to `C:\buildtmp_papers`, compiles with pdflatex and biber, and copies the resulting `arc_synthesis.pdf` back to this directory.
