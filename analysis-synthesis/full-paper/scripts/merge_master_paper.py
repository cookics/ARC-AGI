from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECTIONS_DIR = ROOT / "sections"
SCAFFOLD_DIR = ROOT / "scaffold"
OUTPUT_DIR = ROOT / "output"
REFERENCES = ROOT / "references.bib"
MANIFEST = SCAFFOLD_DIR / "01_master_manifest.json"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the full-paper section fragments into one master TeX file.")
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


def build_tex() -> str:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parts = [PREAMBLE.strip(), ""]
    parts.append(r"\title{\textbf{" + latex_escape(manifest["title"]) + r"}}")
    parts.append(r"\author{" + latex_escape(manifest["author"]) + r"}")
    parts.append(r"\date{" + latex_escape(manifest["date"]) + r"}")
    parts.append(r"\begin{document}")
    parts.append(r"\maketitle")
    for section in manifest["sections"]:
        parts.append("")
        parts.append("% ===== " + section["label"] + " =====")
        parts.append((SECTIONS_DIR / section["file"]).read_text(encoding="utf-8").strip())
    parts.append("")
    parts.append(r"\bibliographystyle{plainnat}")
    parts.append(r"\bibliography{references}")
    parts.append(r"\end{document}")
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_tex = OUTPUT_DIR / "master_paper.tex"
    out_tex.write_text(build_tex(), encoding="utf-8")
    shutil.copy2(REFERENCES, OUTPUT_DIR / "references.bib")
    if args.compile:
        subprocess.run(["latexmk", "-pdf", out_tex.name], cwd=OUTPUT_DIR, check=True)
    print(out_tex)


if __name__ == "__main__":
    main()
