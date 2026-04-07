from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"


def configure() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def color_for(name: str) -> str:
    if "LLM" in name:
        return "#F58518"
    if "VARC" in name:
        return "#54A24B"
    if "CompressARC" in name:
        return "#B279A2"
    return "#4C78A8"


def plot_alignment_band() -> None:
    residual = pd.read_csv(TABLES / "residual_alignment_tests.csv")
    summary = pd.read_csv(TABLES / "system_summary.csv")
    data_scope = pd.read_csv(TABLES / "data_scope_summary.csv")
    split = data_scope.loc[data_scope["dataset"] == "ARC-2 primary"].iloc[0]

    selected = [
        "Best-aligned LLM",
        "LLM average",
        "Best-score LLM",
        "TRM 361957 pass@2",
        "VARC ARC-2_ViT pass@2",
        "TRM 651522 pass@2",
    ]
    merged = residual.merge(summary[["system", "bootstrap_ci_lo", "bootstrap_ci_hi"]], on="system", how="left")
    merged = merged.set_index("system").loc[selected].reset_index()
    merged["y"] = np.arange(len(merged))[::-1]
    split_lo, split_hi = [float(part.strip()) for part in str(split["human_split_ci"]).strip("[]").split(",")]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvspan(split_lo, split_hi, color="#9FD0CB", alpha=0.20, label="Human split-half 95% interval")
    ax.axvline(split["human_split_median"], color="#1F3552", linestyle="--", linewidth=2.2, label="Human split-half median")

    for _, row in merged.iterrows():
        ax.hlines(row["y"], row["bootstrap_ci_lo"], row["bootstrap_ci_hi"], color=color_for(row["system"]), linewidth=3)
        ax.scatter(row["raw_human_corr"], row["y"], s=85, color=color_for(row["system"]), edgecolor="white", linewidth=0.9)

    ax.set_yticks(merged["y"])
    ax.set_yticklabels(merged["system"])
    ax.set_xlabel("Correlation with human solve rates")
    ax.set_title("Human alignment on ARC-2 relative to the human split-half benchmark")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIGURES / "fig09_alignment_band.png")
    plt.close(fig)


def plot_fixed_accuracy_null() -> None:
    fixed_df = pd.read_csv(TABLES / "fixed_accuracy_null_tests.csv")
    order = [
        "Best-aligned LLM",
        "Best-score LLM",
        "TRM 361957 pass@2",
        "TRM 651522 pass@2",
        "VARC ARC-2_ViT pass@2",
        "CompressARC top2",
    ]
    fixed_df = fixed_df.set_index("system").loc[order].reset_index()
    fixed_df["y"] = np.arange(len(fixed_df))[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for _, row in fixed_df.iterrows():
        ax.hlines(row["y"], row["null_ci_lo"], row["null_ci_hi"], color="#9A9A9A", linewidth=4)
        ax.scatter(row["observed_corr"], row["y"], s=90, color=color_for(row["system"]), edgecolor="white", linewidth=0.9, zorder=3)
        ax.text(row["null_ci_hi"] + 0.012, row["y"], f"p={row['p_value']:.3f}", va="center", fontsize=9)

    ax.axvline(0.0, color="#333333", linestyle=":", linewidth=1.5)
    ax.set_yticks(fixed_df["y"])
    ax.set_yticklabels(fixed_df["system"])
    ax.set_xlabel("Observed correlation vs fixed-accuracy random-placement null")
    ax.set_title("Low correlations are only meaningful when they exceed the same-accuracy null")
    fig.savefig(FIGURES / "fig10_fixed_accuracy_null.png")
    plt.close(fig)


def plot_residual_signal() -> None:
    hyp = pd.read_csv(TABLES / "hypothesis_test_summary.csv")
    residual = pd.read_csv(TABLES / "residual_alignment_tests.csv")

    h2 = hyp.loc[hyp["null_id"] == "H2"].copy()
    h2 = h2.set_index("system").loc[
        ["LLM average", "TRM 361957 pass@2", "VARC ARC-2_ViT pass@2", "TRM 651522 pass@2"]
    ].reset_index()
    h2["y"] = np.arange(len(h2))[::-1]

    h3 = residual.set_index("system").loc[
        ["TRM 361957 pass@2", "VARC ARC-2_ViT pass@2", "TRM 651522 pass@2"]
    ].reset_index()
    h3["y"] = np.arange(len(h3))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)

    for _, row in h2.iterrows():
        axes[0].hlines(row["y"], row["ci_lo"], row["ci_hi"], color=color_for(row["system"]), linewidth=3)
        axes[0].scatter(row["estimate"], row["y"], s=85, color=color_for(row["system"]), edgecolor="white", linewidth=0.9)
    axes[0].axvline(0.0, color="#333333", linestyle=":", linewidth=1.5)
    axes[0].set_yticks(h2["y"])
    axes[0].set_yticklabels(h2["system"])
    axes[0].set_title("After simple task-feature controls")
    axes[0].set_xlabel("Partial correlation with human solve rates")

    for _, row in h3.iterrows():
        axes[1].hlines(row["y"], row["partial_ci_lo"], row["partial_ci_hi"], color=color_for(row["system"]), linewidth=3)
        axes[1].scatter(row["partial_corr_given_llm_average"], row["y"], s=85, color=color_for(row["system"]), edgecolor="white", linewidth=0.9)
    axes[1].axvline(0.0, color="#333333", linestyle=":", linewidth=1.5)
    axes[1].set_yticks(h3["y"])
    axes[1].set_yticklabels(h3["system"])
    axes[1].set_title("After controlling for the LLM average")
    axes[1].set_xlabel("Partial correlation with human solve rates")

    fig.suptitle("Residual human-like structure in the non-LLM systems is real but modest")
    fig.savefig(FIGURES / "fig11_residual_signal.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure()
    plot_alignment_band()
    plot_fixed_accuracy_null()
    plot_residual_signal()
    print(f"Wrote paper figures to {FIGURES}")


if __name__ == "__main__":
    main()
