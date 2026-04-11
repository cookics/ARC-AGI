from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent
JOIN_PATH = ANALYSIS_DIR / "arc1_dsl_extra_llms_join.csv"
OUTPUT_PATH = ANALYSIS_DIR / "chart_arc1_pc1_human_vs_llm_full391.png"


def zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return values * np.nan
    return (values - values.mean()) / std


def add_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, label: str) -> None:
    slope, intercept = np.polyfit(x, y, 1)
    order = np.argsort(x)
    ax.plot(x[order], intercept + slope * x[order], color=color, linewidth=2.2, label=label)


def main() -> None:
    df = pd.read_csv(JOIN_PATH)
    plot_df = df[["complexity_pc1_score", "human_difficulty_complete", "llm4_pair_difficulty"]].dropna()

    x = plot_df["complexity_pc1_score"]
    human_z = zscore(plot_df["human_difficulty_complete"])
    llm_z = zscore(plot_df["llm4_pair_difficulty"])

    human_r = float(x.corr(plot_df["human_difficulty_complete"], method="pearson"))
    llm_r = float(x.corr(plot_df["llm4_pair_difficulty"], method="pearson"))

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    ax.scatter(
        x,
        human_z,
        s=24,
        alpha=0.58,
        color="#1f77b4",
        edgecolors="none",
        label=f"Human latent difficulty (r={human_r:.3f})",
    )
    ax.scatter(
        x,
        llm_z,
        s=24,
        alpha=0.50,
        color="#d62728",
        edgecolors="none",
        label=f"Pooled 4-model LLM difficulty (r={llm_r:.3f})",
    )

    add_fit(ax, x.to_numpy(dtype=float), human_z.to_numpy(dtype=float), "#1f77b4", "Human fit")
    add_fit(ax, x.to_numpy(dtype=float), llm_z.to_numpy(dtype=float), "#d62728", "LLM fit")

    ax.set_title("ARC-1 Full Set (n=391): Complexity PC1 vs Human and LLM Difficulty", fontsize=13)
    ax.set_xlabel("Complexity PC1 score")
    ax.set_ylabel("Standardized difficulty (z-score within series)")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=9)

    fig.savefig(OUTPUT_PATH, dpi=200)


if __name__ == "__main__":
    main()
