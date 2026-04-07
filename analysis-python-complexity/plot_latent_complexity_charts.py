from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent


def short_dataset_name(value: str) -> str:
    return {
        "arc_agi_1_eval": "ARC-1 eval",
        "arc_agi_2_eval": "ARC-2 eval",
    }.get(value, value)


def choose_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Label a few tasks that anchor the plot: easiest, hardest, and strong residuals.
    parts = [
        df.nsmallest(3, "latent_difficulty"),
        df.nlargest(3, "latent_difficulty"),
        df.nlargest(3, "prediction_abs_error"),
    ]
    labels = pd.concat(parts).drop_duplicates(subset=["dataset_key", "task_id"]).copy()
    labels["task_label"] = labels["dataset_key"].map(short_dataset_name) + ": " + labels["task_id"]
    return labels


def save_pca_overview(explained: pd.DataFrame, pc_corr: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    fig.patch.set_facecolor("#f7f3ea")

    top = explained.head(8).copy()
    axes[0].set_facecolor("#fffaf1")
    axes[0].bar(top["component"], top["explained_variance_ratio"], color="#d97706", edgecolor="#7c2d12")
    axes[0].plot(
        top["component"],
        top["cumulative_explained_variance_ratio"],
        color="#0f766e",
        marker="o",
        linewidth=2.5,
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Complexity PCA Scree", fontsize=16, weight="bold")
    axes[0].set_ylabel("Explained Variance")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].text(
        0.03,
        0.93,
        "PC1 alone explains 57.4%\nof metric variance",
        transform=axes[0].transAxes,
        fontsize=11,
        color="#7c2d12",
        va="top",
    )

    top_corr = pc_corr.head(6).copy()
    axes[1].set_facecolor("#fffaf1")
    colors = ["#0f766e" if value >= 0 else "#b91c1c" for value in top_corr["pearson_r_with_latent_difficulty"]]
    axes[1].bar(
        top_corr["component"],
        top_corr["pearson_r_with_latent_difficulty"],
        color=colors,
        edgecolor="#1f2937",
    )
    axes[1].axhline(0, color="#374151", linewidth=1)
    axes[1].set_ylim(-0.45, 0.75)
    axes[1].set_title("PC Correlation With Latent Difficulty", fontsize=16, weight="bold")
    axes[1].set_ylabel("Pearson r")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].text(
        0.03,
        0.93,
        "PC1 carries most of the signal.\nPC2 adds a separate runtime-style axis.",
        transform=axes[1].transAxes,
        fontsize=11,
        color="#1f2937",
        va="top",
    )

    fig.suptitle("Latent Difficulty vs Solver Complexity", fontsize=20, weight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_complexity_map(df: pd.DataFrame, labels: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 8), dpi=180)
    fig.patch.set_facecolor("#f7f3ea")
    ax.set_facecolor("#fffaf1")

    markers = {"arc_agi_1_eval": "o", "arc_agi_2_eval": "s"}
    cmap = sns.color_palette("crest", as_cmap=True)

    for dataset_key, marker in markers.items():
        subset = df[df["dataset_key"] == dataset_key]
        sc = ax.scatter(
            subset["PC1"],
            subset["PC2"],
            c=subset["latent_difficulty"],
            cmap=cmap,
            marker=marker,
            s=80,
            alpha=0.9,
            edgecolor="#1f2937",
            linewidth=0.6,
            label=short_dataset_name(dataset_key),
        )

    for _, row in labels.iterrows():
        ax.annotate(
            row["task_label"],
            (row["PC1"], row["PC2"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.18", fc="#fffaf1", ec="#d6d3d1", alpha=0.9),
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Latent difficulty", rotation=90)
    ax.axhline(0, color="#d6d3d1", linewidth=1)
    ax.axvline(0, color="#d6d3d1", linewidth=1)
    ax.set_title("Complexity Map: PC1 vs PC2", fontsize=18, weight="bold")
    ax.set_xlabel("PC1: overall solver size / structure")
    ax.set_ylabel("PC2: runtime intensity vs size")
    ax.legend(frameon=True, facecolor="#fffaf1", edgecolor="#d6d3d1")
    ax.text(
        0.02,
        0.98,
        "Rightward = larger, denser solvers\nUpward = more runtime-heavy behavior for their size",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def add_fit_line(ax, x: np.ndarray, y: np.ndarray, color: str) -> None:
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = coeffs[0] * x_line + coeffs[1]
    ax.plot(x_line, y_line, color=color, linewidth=2.5)


def save_prediction_comparison(df: pd.DataFrame, labels: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    fig.patch.set_facecolor("#f7f3ea")

    # Left: best single metric.
    ax = axes[0]
    ax.set_facecolor("#fffaf1")
    x = np.log1p(df["ast_node_count"].to_numpy(dtype=float))
    y = df["latent_difficulty"].to_numpy(dtype=float)
    ax.scatter(x, y, s=70, color="#d97706", edgecolor="#7c2d12", alpha=0.85)
    add_fit_line(ax, x, y, "#7c2d12")
    ax.set_title("Best Single Metric", fontsize=16, weight="bold")
    ax.set_xlabel("log(1 + AST node count)")
    ax.set_ylabel("Latent difficulty")
    ax.grid(alpha=0.2)
    ax.text(
        0.04,
        0.94,
        "Pearson r = 0.666\nSingle structural proxy",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color="#7c2d12",
    )

    # Right: PCR-5 leave-one-out prediction.
    ax = axes[1]
    ax.set_facecolor("#fffaf1")
    pred = df["complexity_best_pcr_loo_pred"].to_numpy(dtype=float)
    truth = df["latent_difficulty"].to_numpy(dtype=float)
    ax.scatter(pred, truth, s=70, color="#0f766e", edgecolor="#134e4a", alpha=0.85)
    min_v = min(pred.min(), truth.min())
    max_v = max(pred.max(), truth.max())
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="#374151", linewidth=2)
    ax.set_title("Best Composite Predictor", fontsize=16, weight="bold")
    ax.set_xlabel("LOO predicted latent difficulty (PCR-5)")
    ax.set_ylabel("Observed latent difficulty")
    ax.grid(alpha=0.2)
    ax.text(
        0.04,
        0.94,
        "LOO Pearson r = 0.691\nBeats the best single metric a bit",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color="#134e4a",
    )

    interesting = labels.head(4)
    for _, row in interesting.iterrows():
        ax.annotate(
            row["task_label"],
            (row["complexity_best_pcr_loo_pred"], row["latent_difficulty"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.5,
            color="#111827",
        )

    fig.suptitle("Single Proxy vs Composite Predictor", fontsize=20, weight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_loadings_panel(loadings: pd.DataFrame, output_path: Path) -> None:
    melted = (
        loadings[["PC1", "PC2", "PC3"]]
        .reset_index(names="metric")
        .melt(id_vars="metric", var_name="component", value_name="loading")
    )
    top_metrics = (
        melted.assign(abs_loading=lambda d: d["loading"].abs())
        .sort_values("abs_loading", ascending=False)
        .groupby("component")
        .head(6)
    )

    order = (
        top_metrics.groupby("metric")["loading"]
        .apply(lambda s: s.abs().max())
        .sort_values()
        .index
    )

    fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=180)
    fig.patch.set_facecolor("#f7f3ea")
    ax.set_facecolor("#fffaf1")

    palette = {"PC1": "#d97706", "PC2": "#0f766e", "PC3": "#b45309"}
    sns.barplot(
        data=top_metrics,
        y="metric",
        x="loading",
        hue="component",
        order=order,
        palette=palette,
        ax=ax,
    )
    ax.axvline(0, color="#374151", linewidth=1)
    ax.set_title("What The First Three PCs Are Measuring", fontsize=18, weight="bold")
    ax.set_xlabel("PCA loading")
    ax.set_ylabel("")
    ax.legend(title="", frameon=True, facecolor="#fffaf1", edgecolor="#d6d3d1")
    ax.text(
        0.02,
        0.98,
        "PC1 = structural size and branching\nPC2 = runtime intensity\nPC3 = grid volume and memory",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.family"] = "DejaVu Sans"

    df = pd.read_csv(BASE_DIR / "complexity_with_latent_components.csv")
    explained = pd.read_csv(BASE_DIR / "latent_complexity_pca_explained_variance.csv")
    pc_corr = pd.read_csv(BASE_DIR / "latent_complexity_pc_correlations.csv")
    loadings = pd.read_csv(BASE_DIR / "latent_complexity_pca_loadings.csv", index_col=0)

    df["prediction_abs_error"] = (
        df["complexity_best_pcr_loo_pred"] - df["latent_difficulty"]
    ).abs()
    labels = choose_labels(df)

    save_pca_overview(explained, pc_corr, BASE_DIR / "chart_pca_overview.png")
    save_complexity_map(df, labels, BASE_DIR / "chart_complexity_map.png")
    save_prediction_comparison(df, labels, BASE_DIR / "chart_prediction_comparison.png")
    save_loadings_panel(loadings, BASE_DIR / "chart_component_loadings.png")


if __name__ == "__main__":
    main()
