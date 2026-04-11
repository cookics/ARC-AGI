from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SECTIONS_DIR = ROOT / "sections"
OUTPUT_DIR = ROOT / "output"
REFERENCES = ROOT / "references.bib"
MANIFEST = ROOT / "scaffold" / "01_master_manifest.json"

PREAMBLE = r"""
\documentclass[11pt,letterpaper]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{parskip}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{float}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage[numbers,sort&compress]{natbib}
\definecolor{darkslate}{HTML}{2C3E50}
\graphicspath{{../../../analysis-llm-psychometrics/figures/}{../../../analysis-human/papers/human-testing/figures/}{../../../analysis-human/creme-analysis/figures/}{../../../analysis-non-llm/papers/psychometric/figures/}{../../../analysis-latent-crossarc/figures/}{../../../analysis-efficiency/papers/efficiency/figures/}{../../../analysis-python-complexity/}}
\hypersetup{colorlinks=true,linkcolor=darkslate,citecolor=darkslate,urlcolor=darkslate}
"""

SECTION_SUMMARIES: dict[str, list[str]] = {
    "00_title_abstract.tex": [
        "One paper, not a stack of local reports.",
        "Shared ARC difficulty axis: real, but incomplete.",
        "Validated solver structure: strong for LLM difficulty, weak for human difficulty.",
        "Human burden: duration, search, pair heterogeneity.",
        "Current non-LLM systems: complementary signal, not broad human-like replication.",
        r"Mechanistic loop: strong prior + iterative refinement + candidate selection \cite{gao2025urm, hu2025varc, jolicoeur2025trm, mcgovern2025ttatrm}.",
    ],
    "01_introduction.tex": [
        "Why unify the repo: avoid fragmented claims and overread overlaps.",
        "Nine parts: data/methods; LLM; human; non-LLM; solver complexity; cross-ARC/efficiency; architecture; discussion.",
        "Core thesis: shared abstraction factor + source-specific burdens.",
        r"ARC motivation: severe sample constraints, pixel-perfect scoring, recent small-model ARC work \cite{chollet2019arc, arcprizefoundation2025, gao2025urm, hu2025varc, jolicoeur2025trm, mcgovern2025ttatrm}.",
    ],
    "02_repo_data_and_methods.tex": [
        "Repository-wide, partially overlapping data objects.",
        "Human sessions; LLM matrices; non-LLM artifacts; validated solvers; cross-ARC and efficiency summaries.",
        "Human analyses: sparse, opportunistic coverage; split-half reliability used as the ceiling.",
        "LLM analyses: deterministic test takers; permutation-based inference.",
        "Non-LLM analyses: low-score binary systems; threshold sensitivity, nulls, and complementarity checks.",
        "Solver complexity: validated corpus first; approved-only subset is decisive.",
        "Inference convention: bootstrap CIs, permutation tests, BH control, cross-validation, residualized comparisons.",
        "Centrality rule: canonical data, auditability, multiple estimators, interpretability, cross-workstream linkage.",
    ],
    "03_llm_psychometrics.tex": [
        "Goal: whether ARC is psychometrically usable on the model side.",
        r"Data: 24 frontier models; 372 informative ARC-1 items; 120 ARC-2 items; broader 203-model benchmark sidecar.",
        "Methods: exact-match scoring, PCA, Rasch-style estimation, Loevinger H, matrix-preserving permutation tests.",
        r"Results: strong staircase structure; Loevinger H = 0.779; PC1 = 48.7\% and PC2 = 8.8\%.",
        "Thinking variants stay mostly on the same latent dimension.",
        r"Broader benchmark addendum: PC1 = 66.4\%; held-out prediction mean $R^2 = 0.615$.",
        "Takeaway: ARC is structured enough to support the rest of the manuscript.",
    ],
    "04_human_psychometrics.tex": [
        "Goal: whether sparse ARC human logs are good enough to serve as a benchmark.",
        r"Data: 4,681 attempts; 509 sessions; 442 task IDs; 502 task-pair rows; 40.4\% task-pair coverage; 1.83\% matrix density.",
        "Methods: pair-level analysis, split-half simulation, latent estimation as reliability-anchored comparison.",
        "Results: noisy but usable; latent estimates stabilize better than raw means.",
        "Human vs average model: about 0.402, near the split-half center.",
        "Best single model is lower; per-pair oracle sits between.",
        "Sampling bias: attempted items are not representative; they are larger and harder in key ways.",
        "Takeaway: human difficulty is real, sparse, and not reducible to board size alone.",
    ],
    "05_non_llm_psychometrics.tex": [
        "Goal: do non-LLM systems look human-like, weaker than LLMs, or just different.",
        "Primary overlap: ARC-AGI-2 Public Eval; 110-pair high-attempt subset; TRM and VARC are the main non-LLM profiles.",
        "Methods: human split-half benchmark, threshold sensitivity, fixed-accuracy nulls, simple controls, matched weak LLMs, rescue analyses.",
        "Results: LLM aggregate remains the best human-profile match.",
        "TRM and VARC fall below human-equivalence on the primary overlap.",
        "Mid-training TRM can beat its same-accuracy random-placement null; later stronger-score checkpoints do not necessarily improve human-like ordering.",
        "Complementary rescue signal exists, but it is narrow.",
        "Takeaway: promising mechanism, weak psychometric human-likeness.",
    ],
    "06_solver_complexity.tex": [
        "Goal: does final solver complexity load equally on humans, LLMs, and non-LLM systems.",
        "Data: 511 fetched; 127 validated; 120 approved-and-validated retained.",
        "Methods: structural metrics, dynamic metrics, PCA, correlations, paired difference-of-correlation tests, residualization.",
        r"Results: PC1 = 57.4\% and behaves like overall solver size/density.",
        "Cyclomatic complexity predicts LLM difficulty strongly.",
        "Human and LLM share a real difficulty axis, but solver structure is much more LLM-linked than human-linked.",
        "Residual LLM difficulty still tracks solver structure.",
        "Takeaway: this is the integrative core of the repository.",
    ],
    "07_crossarc_efficiency.tex": [
        "Goal: widen the sparse overlap and compare score versus effort across source families.",
        "Cross-ARC: latent estimates are more stable than raw solve rates; ARC-1 and ARC-2 alignment persists under widening.",
        "Solver-structure asymmetry survives the wider overlap.",
        "Efficiency: score alignment is clearer than effort alignment.",
        "LLM performance rises with thinking rank and duration.",
        "Human performance rises with ability and falls with duration.",
        "Best human-solve model: geometry + LLM performance; human duration remains weakly predicted.",
        "Takeaway: shared difficulty, different burdens.",
    ],
    "08_architectural_synthesis.tex": [
        "Why the external literature belongs here: it gives a mechanistic explanation for the empirical pattern.",
        r"Common loop: strong inductive bias + iterative refinement + test-time candidate selection \cite{gao2025urm, hu2025varc, jolicoeur2025trm, mcgovern2025ttatrm}.",
        "URM: recurrent depth beats simply adding depth.",
        "TRM: latent scratchpad + test-time voting.",
        "VARC: vision prior + multi-view + test-time training.",
        "McGovern: adaptation under compute limits; frozen trunk is not enough.",
        "Takeaway: strong priors and structured test-time search, but not a claim of human-like solving.",
    ],
    "09_integrated_discussion.tex": [
        "Strongest thesis: shared difficulty axis, different burden structure.",
        "Central claims: LLM-side psychometric structure, sparse-but-usable human data, human-vs-average-model alignment, solver structure predicting LLM difficulty, and current non-LLM limits.",
        "Secondary claims: closeness addenda, pooled cross-ARC support, non-LLM middle pattern.",
        "Pressure points: tiny fully matched overlap, task-versus-pair mismatch, sampling bias, ARC-1 vs ARC-2 drift, exact-match lossiness.",
        "Avoid overclaims: not same reasons for humans and LLMs; not broadly human-like non-LLM systems; not a clean thinking-advantage flagship.",
        "Next steps: larger matched overlap, richer pair-level logging, careful cross-ARC expansion, test the architectural loop against human-like ordering.",
        "Purpose of the draft: make the repository legible as one argument.",
    ],
    "10_appendix_repo_coverage.tex": [
        "Coverage ledger: which workstreams map into the master manuscript.",
        "Claim ledger: retained strong, retained secondary, rejected, or downgraded.",
        "Main point: the narrative is narrower than the raw repository, by design.",
    ],
    "11_appendix_selected_results.tex": [
        "Appendix tables: extended leaderboard, threshold sensitivity, solver-complexity audit, efficiency ledger.",
        "Purpose: keep the manuscript close to the exported tables and CSVs.",
        "Use: traceability, not additional narrative.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the condensed master paper.")
    parser.add_argument("--compile", action="store_true", help="Compile the generated TeX with latexmk.")
    return parser.parse_args()


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def read_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def extract_blocks(text: str) -> list[str]:
    pattern = re.compile(
        r"\\begin\{(itemize|enumerate|table|figure|longtable)\}(?:\[[^\]]*\])?.*?\\end\{\1\}",
        re.DOTALL,
    )
    return [match.group(0).strip() for match in pattern.finditer(text)]


def heading_from_text(text: str) -> str | None:
    match = re.search(r"\\section\{([^}]*)\}", text)
    if match:
        return match.group(1)
    return None


def build_itemize(items: list[str]) -> str:
    lines = [r"\begin{itemize}"]
    for item in items:
        lines.append(r"\item " + item)
    lines.append(r"\end{itemize}")
    return "\n".join(lines)


def build_tex(manifest: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(PREAMBLE.strip())
    lines.append("")
    title = manifest["title"] + " (Condensed Version)"
    lines.append(r"\title{\textbf{" + latex_escape(title) + r"}}")
    lines.append(r"\author{" + latex_escape(manifest["author"]) + r"}")
    lines.append(r"\date{" + latex_escape(manifest["date"]) + r"}")
    lines.append(r"\begin{document}")
    lines.append(r"\maketitle")
    lines.append("")
    lines.append(r"\begin{abstract}")
    lines.append(build_itemize(SECTION_SUMMARIES["00_title_abstract.tex"]))
    lines.append(r"\end{abstract}")
    lines.append("")
    lines.append(r"\tableofcontents")
    lines.append(r"\clearpage")

    for section in manifest["sections"][1:]:
        path = SECTIONS_DIR / section["file"]
        text = path.read_text(encoding="utf-8")
        heading = heading_from_text(text) or section["label"]
        if section["file"] == "10_appendix_repo_coverage.tex":
            lines.append("")
            lines.append(r"\appendix")
        lines.append("")
        lines.append(r"\section{" + latex_escape(heading) + r"}")
        lines.append(build_itemize(SECTION_SUMMARIES[section["file"]]))
        blocks = extract_blocks(text)
        if blocks:
            lines.append("")
            lines.extend(blocks)
    lines.append("")
    lines.append(r"\bibliographystyle{plainnat}")
    lines.append(r"\bibliography{references}")
    lines.append(r"\end{document}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    manifest = read_manifest()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_tex = OUTPUT_DIR / "master_paper_condensed.tex"
    out_tex.write_text(build_tex(manifest), encoding="utf-8")
    shutil.copy2(REFERENCES, OUTPUT_DIR / "references.bib")
    if args.compile:
        subprocess.run(["latexmk", "-pdf", out_tex.name], cwd=OUTPUT_DIR, check=True)
    print(out_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
