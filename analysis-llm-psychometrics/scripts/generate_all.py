"""
ARC-AGI Psychometric Analysis: Master Figure & Table Generator
==============================================================
Generates all publication-quality figures and CSV tables for the LaTeX paper.
All outputs go to ../figures/ and ../tables/ relative to this script.

Usage: python generate_all.py
Run from the scripts/ directory or from project root.
"""

import os
import sys
import json
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PROJECT_DIR)

FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
TABLES_DIR = os.path.join(PROJECT_DIR, "tables")

# Data paths — all inside Psychometric Analysis/data/
DATA_DIR = os.path.join(REPO_ROOT, "data-llm")
V1_PREDS = os.path.join(DATA_DIR, "arc_agi_v1_public_eval")
V1_TRUTH = os.path.join(DATA_DIR, "ARC-AGI", "data", "evaluation")
V2_PREDS = os.path.join(DATA_DIR, "arc_agi_v2_public_eval")
V2_TRUTH = os.path.join(DATA_DIR, "ARC-AGI-2", "data", "evaluation")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ============================================================================
# VISUAL STYLE (Coherent across all figures)
# ============================================================================

# Color palette
DARK_SLATE = "#2C3E50"
ALERT_RED = "#E31A1C"
THINKING_TEAL = "#1ABC9C"
STANDARD_CORAL = "#E74C3C"
ACCENT_BLUE = "#2980B9"
LIGHT_GRAY = "#BDC3C7"
BG_WHITE = "#FFFFFF"
TEXT_GRAY = "#555555"

# Shared style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': BG_WHITE,
    'axes.facecolor': BG_WHITE,
    'savefig.facecolor': BG_WHITE,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.figsize': (12, 8),
})

CAPTION = "Source: ARC-AGI Public Eval | By: @notcomplex_"

# ============================================================================
# DATA LOADING
# ============================================================================

def normalize_grid(grid):
    """Normalize a grid to a comparable string."""
    if not isinstance(grid, list) or len(grid) == 0:
        return "EMPTY"
    try:
        return ",".join(str(cell) for row in grid for cell in row)
    except:
        return "ERROR"


def load_truth(truth_dir):
    """Load ground truth for all tasks."""
    cache = {}
    for f in glob.glob(os.path.join(truth_dir, "*.json")):
        try:
            with open(f, 'r') as fh:
                data = json.load(fh)
            tid = os.path.basename(f)
            cache[tid] = [normalize_grid(pair['output']) for pair in data.get('test', [])]
        except:
            pass
    return cache


def build_response_matrix(preds_dir, truth_dir):
    """Build a binary response matrix: models × tasks."""
    truth_cache = load_truth(truth_dir)
    if not truth_cache:
        return None

    task_ids = sorted(truth_cache.keys())
    model_dirs = [d for d in glob.glob(os.path.join(preds_dir, "*")) if os.path.isdir(d)]

    data = {}
    for md in model_dirs:
        model_name = os.path.basename(md)
        if model_name.startswith('.'):
            continue  # skip .git etc.
        row = {tid: 0 for tid in task_ids}

        for pf in glob.glob(os.path.join(md, "*.json")):
            tid = os.path.basename(pf)
            if tid not in row:
                continue
            try:
                with open(pf, 'r') as fh:
                    pred_data = json.load(fh)
            except:
                continue

            true_outputs = truth_cache[tid]
            is_correct = True

            for j, true_out in enumerate(true_outputs):
                pred_entry = None
                # Strategy 1: metadata match
                for item in pred_data:
                    if item.get("metadata", {}).get("pair_index") == j:
                        pred_entry = item
                        break
                # Strategy 2: positional fallback
                if pred_entry is None and j < len(pred_data):
                    pred_entry = pred_data[j]

                if pred_entry is None:
                    is_correct = False
                    break

                # Check attempt_1 then attempt_2
                ans = None
                a1 = pred_entry.get("attempt_1") or {}
                ans = a1.get("answer")
                if not ans:
                    a2 = pred_entry.get("attempt_2") or {}
                    ans = a2.get("answer")

                if normalize_grid(ans) != true_out:
                    is_correct = False
                    break

            if is_correct:
                row[tid] = 1

        data[model_name] = row

    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reindex(columns=task_ids)
    return df


def classify_model(name):
    """Classify model as 'Thinking' or 'Standard'."""
    name_lower = name.lower()
    if "thinking-none" in name_lower:
        return "Standard"
    if any(kw in name_lower for kw in ["thinking", "deep", "reasoning"]):
        return "Thinking"
    if "gemini" in name_lower:
        return "Thinking"  # All Gemini models use thinking
    if "gpt-5-pro" in name_lower:
        return "Thinking"
    return "Standard"


# ============================================================================
# LEADERBOARD DATA (for correlation)
# ============================================================================

LEADERBOARD = {
    "gemini-3-deep-think-preview": {"V1": 87.5, "V2": 45.1},
    "gemini-3-pro-preview": {"V1": 75.0, "V2": 31.1},
    "gpt-5-pro-2025-10-06": {"V1": 70.2, "V2": 18.3},
    "gpt-5-1-2025-11-13-thinking-high": {"V1": 72.8, "V2": 17.6},
    "claude-sonnet-4-5-20250929-thinking-32k": {"V1": 63.7, "V2": 13.6},
    "gpt-5-1-2025-11-13-thinking-medium": {"V1": 57.7, "V2": 6.5},
    "claude-sonnet-4-5-20250929-thinking-16k": {"V1": 48.3, "V2": 6.9},
    "claude-haiku-4-5-20251001-thinking-32k": {"V1": 47.7, "V2": 4.0},
    "grok-4-fast-reasoning": {"V1": 48.5, "V2": 5.3},
    "claude-sonnet-4-5-20250929-thinking-8k": {"V1": 46.5, "V2": 6.9},
    "claude-haiku-4-5-20251001-thinking-16k": {"V1": 37.3, "V2": 2.8},
    "gpt-5-1-2025-11-13-thinking-low": {"V1": 33.2, "V2": 1.9},
    "claude-haiku-4-5-20251001-thinking-8k": {"V1": 25.5, "V2": 1.7},
    "claude-sonnet-4-5-20250929-thinking-1k": {"V1": 31.0, "V2": 5.8},
    "claude-sonnet-4-5-20250929": {"V1": 25.5, "V2": 3.8},
    "claude-haiku-4-5-20251001": {"V1": 14.3, "V2": 1.3},
    "claude-haiku-4-5-20251001-thinking-1k": {"V1": 16.8, "V2": 1.3},
    "qwen3-235b-a22b-instruct-2507": {"V1": None, "V2": None},
    "QwQ-32B-Fireworks": {"V1": None, "V2": None},
    "gpt-4-5-2025-02-27": {"V1": 10.3, "V2": 0.8},
    "gpt-4-1-2025-04-14": {"V1": 5.5, "V2": 0.4},
    "gpt-5-1-2025-11-13-thinking-none": {"V1": 5.8, "V2": 0.4},
    "gpt-4-1-mini-2025-04-14": {"V1": 3.5, "V2": 0.0},
    "gpt-4-1-nano-2025-04-14": {"V1": 0.0, "V2": 0.0},
}

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def rasch_diagnostics(mat):
    """Compute Rasch outfit MSQ and Loevinger's H via permutation."""
    scores = mat.mean(axis=1).values
    p_rates = mat.mean(axis=0).values
    m = mat.values

    # Rasch expected probabilities
    theta = stats.norm.ppf(np.clip(scores, 0.01, 0.99))
    beta = -stats.norm.ppf(np.clip(p_rates, 0.01, 0.99))
    P = 1.0 / (1.0 + np.exp(-(theta[:, None] - beta[None, :])))
    W = P * (1 - P)
    W = np.maximum(W, 1e-6)
    Z = (m - P) / np.sqrt(W)
    outfit = np.mean(Z**2, axis=1)

    # Loevinger's H
    cov_mat = np.cov(m.T)
    obs_c = np.sum(cov_mat[np.tril_indices_from(cov_mat, -1)])
    max_c = 0
    p = m.mean(axis=0)
    n_items = len(p)
    for i in range(n_items - 1):
        for j in range(i + 1, n_items):
            max_c += min(p[i], p[j]) - p[i] * p[j]
    H = obs_c / max_c if max_c != 0 else 0

    return outfit, H


def permutation_test(mat, n_perms=200):
    """Permutation test with fixed marginals (swap algorithm)."""
    m = mat.values.copy()
    obs_outfit, obs_H = rasch_diagnostics(mat)

    n_models, n_items = m.shape

    sim_outfits = np.zeros((n_models, n_perms))
    sim_H = np.zeros(n_perms)

    for perm in range(n_perms):
        if (perm + 1) % 50 == 0:
            print(f"  Permutation {perm+1}/{n_perms}...")
        # Swap algorithm: randomly swap 2x2 checkerboard submatrices
        perm_m = m.copy()
        n_swaps = n_models * n_items  # number of attempted swaps
        for _ in range(n_swaps):
            r1, r2 = np.random.randint(0, n_models, 2)
            c1, c2 = np.random.randint(0, n_items, 2)
            if r1 == r2 or c1 == c2:
                continue
            # Check for checkerboard pattern
            if perm_m[r1, c1] != perm_m[r2, c2] and perm_m[r1, c2] != perm_m[r2, c1]:
                if perm_m[r1, c1] != perm_m[r1, c2]:
                    # Swap
                    perm_m[r1, c1], perm_m[r2, c1] = perm_m[r2, c1], perm_m[r1, c1]
                    perm_m[r1, c2], perm_m[r2, c2] = perm_m[r2, c2], perm_m[r1, c2]

        perm_df = pd.DataFrame(perm_m, columns=mat.columns, index=mat.index)
        p_outfit, p_H = rasch_diagnostics(perm_df)
        sim_outfits[:, perm] = p_outfit
        sim_H[perm] = p_H

    # P-values
    p_misfit = np.array([(np.sum(sim_outfits[i, :] >= obs_outfit[i]) + 1) / (n_perms + 1)
                         for i in range(n_models)])
    p_consistency = np.array([(np.sum(sim_outfits[i, :] <= obs_outfit[i]) + 1) / (n_perms + 1)
                               for i in range(n_models)])
    p_H_global = (np.sum(sim_H >= obs_H) + 1) / (n_perms + 1)

    return obs_outfit, obs_H, p_misfit, p_consistency, p_H_global


def pca_analysis(mat):
    """Run PCA and return scores + variance explained."""
    from sklearn.decomposition import PCA
    pca = PCA()
    scores = pca.fit_transform(mat.values)
    var_explained = pca.explained_variance_ratio_
    return scores, var_explained, pca


def irt_rasch_ability(mat):
    """Estimate Rasch ability (theta) via MLE-like approximation."""
    scores = mat.mean(axis=1).values
    # Logit transform as theta proxy
    theta = np.log(np.clip(scores, 0.001, 0.999) / (1 - np.clip(scores, 0.001, 0.999)))
    # Scale to IQ-like (mean=100, sd=15)
    theta_z = (theta - np.mean(theta)) / np.std(theta) if np.std(theta) > 0 else theta
    iq = 100 + theta_z * 15
    return theta, iq


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def fig_leaderboard_bar(df_v1, df_v2):
    """Figure 1: Model leaderboard bar chart (V1 + V2 side by side)."""
    acc_v1 = df_v1.mean(axis=1).sort_values(ascending=False) * 100
    acc_v2 = df_v2.mean(axis=1).reindex(acc_v1.index) * 100 if df_v2 is not None else None

    fig, ax = plt.subplots(figsize=(14, 10))
    y = np.arange(len(acc_v1))
    bar_height = 0.35

    types = [classify_model(m) for m in acc_v1.index]
    colors_v1 = [THINKING_TEAL if t == "Thinking" else STANDARD_CORAL for t in types]

    bars1 = ax.barh(y - bar_height/2, acc_v1.values, bar_height, color=colors_v1,
                    edgecolor='white', linewidth=0.5, label='ARC-AGI-1 (400 tasks)')

    if acc_v2 is not None:
        colors_v2 = [THINKING_TEAL if t == "Thinking" else STANDARD_CORAL for t in types]
        # Make V2 bars lighter
        colors_v2_light = []
        for c in colors_v2:
            colors_v2_light.append(c + "80")  # won't work as hex, use alpha
        bars2 = ax.barh(y + bar_height/2, acc_v2.values, bar_height,
                        color=colors_v1, alpha=0.4,
                        edgecolor='white', linewidth=0.5, label='ARC-AGI-2 (120 tasks)')

    ax.set_yticks(y)
    ax.set_yticklabels(acc_v1.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("ARC-AGI Model Leaderboard", fontsize=20, fontweight='bold', color=DARK_SLATE)
    ax.set_xlim(0, 105)

    # Add value labels
    for i, (v1_val, name) in enumerate(zip(acc_v1.values, acc_v1.index)):
        ax.text(v1_val + 1, i - bar_height/2, f"{v1_val:.1f}%", va='center', fontsize=8,
                fontweight='bold', color=DARK_SLATE)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=THINKING_TEAL, label='Thinking'),
        Patch(facecolor=STANDARD_CORAL, label='Standard'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    fig.text(0.5, -0.02, CAPTION, ha='center', fontsize=9, color=TEXT_GRAY)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_leaderboard.png"), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_leaderboard.pdf"))
    plt.close()
    print("  [OK] fig1_leaderboard")


def fig_response_matrix(df, title, filename):
    """Figure 2: Guttman-sorted pass/fail heatmap."""
    # Sort: tasks by difficulty (easy→hard), models by ability (best→worst)
    task_order = df.sum(axis=0).sort_values(ascending=False).index
    model_order = df.sum(axis=1).sort_values(ascending=False).index
    sorted_df = df.loc[model_order, task_order]

    fig, ax = plt.subplots(figsize=(16, 10))
    cmap = ListedColormap([STANDARD_CORAL, THINKING_TEAL])
    sns.heatmap(sorted_df, cmap=cmap, cbar=False, linewidths=0, ax=ax)

    ax.set_title(title, fontsize=20, fontweight='bold', color=DARK_SLATE, pad=15)
    ax.set_xlabel("Tasks (Easy → Hard)", fontsize=13, color=DARK_SLATE)
    ax.set_ylabel("Models (Best → Worst)", fontsize=13, color=DARK_SLATE)
    ax.set_xticks([])
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=THINKING_TEAL, label='Solved'),
        Patch(facecolor=STANDARD_CORAL, label='Failed'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11)
    fig.text(0.5, -0.01, CAPTION, ha='center', fontsize=9, color=TEXT_GRAY)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename + ".png"), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, filename + ".pdf"))
    plt.close()
    print(f"  [OK] {filename}")


def fig_scree_plot(var_explained):
    """Figure 3: PCA scree plot."""
    n_show = min(20, len(var_explained))
    cum_var = np.cumsum(var_explained)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(1, n_show + 1)

    ax.bar(x, var_explained[:n_show] * 100, color=DARK_SLATE, alpha=0.85,
           edgecolor='white', linewidth=0.5, label='Individual')
    ax2 = ax.twinx()
    ax2.plot(x, cum_var[:n_show] * 100, color=ALERT_RED, marker='o', linewidth=2.5,
             markersize=7, label='Cumulative')
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=13, color=ALERT_RED)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(ALERT_RED)
    ax2.tick_params(axis='y', colors=ALERT_RED)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Dimensionality: Is ARC-AGI One Thing or Many?",
                 fontsize=18, fontweight='bold', color=DARK_SLATE)

    # Annotate PC1
    ax.annotate(f"PC1 = {var_explained[0]*100:.1f}%",
                xy=(1, var_explained[0]*100), xytext=(3, var_explained[0]*100 + 3),
                fontsize=12, fontweight='bold', color=ALERT_RED,
                arrowprops=dict(arrowstyle='->', color=ALERT_RED))

    ax.set_xticks(x)
    fig.text(0.5, -0.02, CAPTION, ha='center', fontsize=9, color=TEXT_GRAY)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_scree_plot.png"), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_scree_plot.pdf"))
    plt.close()
    print("  [OK] fig3_scree_plot")


def fig_cognitive_map(pca_scores, model_names, var_explained):
    """Figure 4: PCA biplot (PC1 vs PC2)."""
    fig, ax = plt.subplots(figsize=(14, 10))

    types = [classify_model(m) for m in model_names]
    colors = [THINKING_TEAL if t == "Thinking" else STANDARD_CORAL for t in types]

    ax.scatter(pca_scores[:, 0], pca_scores[:, 1], c=colors, s=120,
              edgecolors='white', linewidth=1.5, zorder=3)

    # Label all points
    for i, name in enumerate(model_names):
        # Shorten names for readability
        short = name.replace("claude-", "c-").replace("haiku-4-5-20251001", "H4.5")
        short = short.replace("sonnet-4-5-20250929", "S4.5").replace("2025-", "")
        short = short.replace("gpt-", "g").replace("thinking-", "t-")
        short = short.replace("-preview", "").replace("qwen3-235b-a22b-instruct-2507", "Qwen3-235B")
        short = short.replace("QwQ-32B-Fireworks", "QwQ-32B")
        short = short.replace("grok-4-fast-reasoning", "Grok-4")
        short = short.replace("gemini-3-deep-think", "Gem3-Deep")
        short = short.replace("gemini-3-pro", "Gem3-Pro")

        ax.annotate(short, (pca_scores[i, 0], pca_scores[i, 1]),
                    fontsize=7.5, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points',
                    color=DARK_SLATE, alpha=0.85)

    ax.axhline(0, color=LIGHT_GRAY, linestyle='--', linewidth=0.8)
    ax.axvline(0, color=LIGHT_GRAY, linestyle='--', linewidth=0.8)

    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% var)")
    ax.set_title("The AI Cognitive Map",
                 fontsize=20, fontweight='bold', color=DARK_SLATE)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=THINKING_TEAL, label='Thinking'),
        Patch(facecolor=STANDARD_CORAL, label='Standard'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    fig.text(0.5, -0.02, CAPTION, ha='center', fontsize=9, color=TEXT_GRAY)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_cognitive_map.png"), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_cognitive_map.pdf"))
    plt.close()
    print("  [OK] fig4_cognitive_map")


def fig_ability_plot(model_names, theta, iq):
    """Figure 5: Latent ability (theta) bar chart."""
    df_plot = pd.DataFrame({
        'Model': model_names,
        'Theta': theta,
        'IQ': iq,
        'Type': [classify_model(m) for m in model_names]
    }).sort_values('Theta', ascending=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = [THINKING_TEAL if t == "Thinking" else STANDARD_CORAL for t in df_plot['Type']]

    bars = ax.barh(range(len(df_plot)), df_plot['Theta'].values, color=colors,
                   edgecolor='white', linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['Model'].values, fontsize=9)
    ax.axvline(0, color=LIGHT_GRAY, linestyle='-', linewidth=1)
    ax.set_xlabel("Latent Ability (θ)")
    ax.set_title("Rasch Ability Estimates", fontsize=20, fontweight='bold', color=DARK_SLATE)

    # Add IQ labels
    for i, (t_val, iq_val) in enumerate(zip(df_plot['Theta'].values, df_plot['IQ'].values)):
        offset = 0.1 if t_val >= 0 else -0.1
        ha = 'left' if t_val >= 0 else 'right'
        ax.text(t_val + offset, i, f"IQ:{iq_val:.0f}", va='center', ha=ha,
                fontsize=7.5, fontweight='bold', color=DARK_SLATE)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=THINKING_TEAL, label='Thinking'),
        Patch(facecolor=STANDARD_CORAL, label='Standard'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    fig.text(0.5, -0.02, CAPTION, ha='center', fontsize=9, color=TEXT_GRAY)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_ability.png"), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_ability.pdf"))
    plt.close()
    print("  [OK] fig5_ability")


def fig_dendrogram(df):
    """Figure 6: Hierarchical cluster dendrogram."""
    dist = pdist(df.values, metric='jaccard')
    Z = linkage(dist, method='ward')

    fig, ax = plt.subplots(figsize=(14, 7))
    dend = dendrogram(Z, labels=df.index.tolist(), ax=ax, leaf_rotation=45,
                      leaf_font_size=9, color_threshold=0.7 * max(Z[:, 2]),
                      above_threshold_color=DARK_SLATE)

    ax.set_title("Model Family Tree (Ward Clustering)",
                 fontsize=18, fontweight='bold', color=DARK_SLATE)
    ax.set_ylabel("Distance", fontsize=13)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', rotation=45)
    fig.text(0.5, -0.05, CAPTION, ha='center', fontsize=9, color=TEXT_GRAY)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_dendrogram.png"), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_dendrogram.pdf"))
    plt.close()
    print("  [OK] fig6_dendrogram")


def fig_difficulty_distribution(df):
    """Figure 7: Task difficulty distribution."""
    task_difficulty = (1 - df.mean(axis=0)) * 100  # % failure rate

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(task_difficulty.values, bins=30, color=DARK_SLATE, alpha=0.85,
            edgecolor='white', linewidth=0.5)

    ax.axvline(task_difficulty.median(), color=ALERT_RED, linestyle='--', linewidth=2,
               label=f'Median: {task_difficulty.median():.1f}%')

    ax.set_xlabel("Task Difficulty (% of models that failed)")
    ax.set_ylabel("Number of Tasks")
    ax.set_title("Distribution of Task Difficulty",
                 fontsize=18, fontweight='bold', color=DARK_SLATE)
    ax.legend(fontsize=12)
    fig.text(0.5, -0.02, CAPTION, ha='center', fontsize=9, color=TEXT_GRAY)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig7_difficulty_dist.png"), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, "fig7_difficulty_dist.pdf"))
    plt.close()
    print("  [OK] fig7_difficulty_dist")


# ============================================================================
# TABLE GENERATION
# ============================================================================

def export_tables(df_v1, df_v2, obs_outfit, obs_H, p_misfit, p_consistency, p_H_global, theta, iq):
    """Export all analysis results as CSV tables."""
    models = df_v1.index.tolist()

    # Table 1: Main leaderboard
    t1 = pd.DataFrame({
        'Model': models,
        'Type': [classify_model(m) for m in models],
        'V1_Accuracy': (df_v1.mean(axis=1) * 100).round(1).values,
        'V1_Solved': df_v1.sum(axis=1).astype(int).values,
        'V1_Total': df_v1.shape[1],
    })
    if df_v2 is not None:
        v2_models = df_v2.index.tolist()
        t1['V2_Accuracy'] = t1['Model'].map(
            lambda m: round(df_v2.loc[m].mean() * 100, 1) if m in v2_models else np.nan
        )
        t1['V2_Solved'] = t1['Model'].map(
            lambda m: int(df_v2.loc[m].sum()) if m in v2_models else np.nan
        )
    t1 = t1.sort_values('V1_Accuracy', ascending=False)
    t1.to_csv(os.path.join(TABLES_DIR, "table1_leaderboard.csv"), index=False)
    print("  [OK] table1_leaderboard.csv")

    # Table 2: Rasch diagnostics
    t2 = pd.DataFrame({
        'Model': models,
        'Score': df_v1.mean(axis=1).round(4).values,
        'Outfit_MSQ': np.round(obs_outfit, 4),
        'P_Misfit': np.round(p_misfit, 4),
        'P_Consistency': np.round(p_consistency, 4),
    }).sort_values('Score', ascending=False)
    t2.to_csv(os.path.join(TABLES_DIR, "table2_diagnostics.csv"), index=False)
    print("  [OK] table2_diagnostics.csv")

    # Table 3: IQ scores
    t3 = pd.DataFrame({
        'Model': models,
        'Type': [classify_model(m) for m in models],
        'Raw_Accuracy': (df_v1.mean(axis=1) * 100).round(1).values,
        'Theta': np.round(theta, 3),
        'AI_IQ': np.round(iq, 0).astype(int),
    }).sort_values('AI_IQ', ascending=False)
    t3.to_csv(os.path.join(TABLES_DIR, "table3_iq_scores.csv"), index=False)
    print("  [OK] table3_iq_scores.csv")

    # Table 4: Global scale metrics
    t4 = pd.DataFrame({
        'Metric': ["Loevinger's H", 'P-value (Scale Cohesion)', 'N Models', 'N Tasks (with variance)',
                    'Permutation Iterations'],
        'Value': [f"{obs_H:.3f}", f"{p_H_global:.4f}", len(models),
                  df_v1.shape[1], 200]
    })
    t4.to_csv(os.path.join(TABLES_DIR, "table4_scale_metrics.csv"), index=False)
    print("  [OK] table4_scale_metrics.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("ARC-AGI PSYCHOMETRIC ANALYSIS: GENERATING ALL OUTPUTS")
    print("=" * 60)

    # --- Load Data ---
    print("\n[1/4] Loading response matrices...")
    df_v1 = build_response_matrix(V1_PREDS, V1_TRUTH)
    if df_v1 is None:
        print("FATAL: Could not build V1 response matrix. Check data paths.")
        sys.exit(1)

    # Filter out .git and zero-variance items
    df_v1 = df_v1.loc[~df_v1.index.str.startswith('.')]
    item_means = df_v1.mean(axis=0)
    df_v1_clean = df_v1.loc[:, (item_means > 0) & (item_means < 1)]
    print(f"  V1: {df_v1.shape[0]} models × {df_v1.shape[1]} tasks ({df_v1_clean.shape[1]} with variance)")

    df_v2 = build_response_matrix(V2_PREDS, V2_TRUTH)
    if df_v2 is not None:
        df_v2 = df_v2.loc[~df_v2.index.str.startswith('.')]
        print(f"  V2: {df_v2.shape[0]} models × {df_v2.shape[1]} tasks")

    # --- Run Analyses ---
    print("\n[2/4] Running psychometric analyses...")
    print("  PCA...")
    pca_scores, var_explained, pca_model = pca_analysis(df_v1_clean)

    print("  Rasch ability estimation...")
    theta, iq = irt_rasch_ability(df_v1_clean)

    print("  Permutation test (this takes a few minutes)...")
    obs_outfit, obs_H, p_misfit, p_consistency, p_H_global = permutation_test(df_v1_clean, n_perms=200)

    # --- Generate Figures ---
    print("\n[3/4] Generating figures...")
    fig_leaderboard_bar(df_v1, df_v2)
    fig_response_matrix(df_v1, "ARC-AGI-1: Response Matrix (Guttman Sorted)", "fig2_response_matrix_v1")
    if df_v2 is not None:
        fig_response_matrix(df_v2, "ARC-AGI-2: Response Matrix (Guttman Sorted)", "fig2b_response_matrix_v2")
    fig_scree_plot(var_explained)
    fig_cognitive_map(pca_scores, df_v1_clean.index.tolist(), var_explained)
    fig_ability_plot(df_v1_clean.index.tolist(), theta, iq)
    fig_dendrogram(df_v1_clean)
    fig_difficulty_distribution(df_v1)

    # --- Export Tables ---
    print("\n[4/4] Exporting tables...")
    export_tables(df_v1_clean, df_v2, obs_outfit, obs_H, p_misfit, p_consistency, p_H_global, theta, iq)

    print("\n" + "=" * 60)
    print("DONE. All outputs saved to:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables:  {TABLES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
