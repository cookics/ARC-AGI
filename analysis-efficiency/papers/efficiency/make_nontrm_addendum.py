from __future__ import annotations

from pathlib import Path
from itertools import combinations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"


def latex_escape(text: str) -> str:
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    out = str(text)
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return "--"
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and abs(numeric - round(numeric)) < 1e-9:
            return str(int(round(numeric)))
        return f"{numeric:.{digits}f}"
    return latex_escape(str(value))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(TABLES / "non_llm_runs.csv")
    return df.loc[df["subtype"].isin(["compress_arc", "varc"])].copy()


def prepare_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["oracle_gap"] = work["oracle_rate"] - work["performance_rate"]
    work["selection_efficiency"] = work["performance_rate"] / work["oracle_rate"].replace(0, np.nan)
    work["pass_gain_1_to_4"] = work["pass4_rate"].fillna(work["performance_rate"]) - work["pass1_rate"]
    return work[
        [
            "label",
            "subtype",
            "performance_rate",
            "oracle_rate",
            "oracle_gap",
            "selection_efficiency",
            "pass1_rate",
            "pass2_rate",
            "pass3_rate",
            "pass4_rate",
            "pass_gain_1_to_4",
        ]
    ].copy()


def exact_two_group_pvalue(values: pd.Series, groups: pd.Series, positive_group: str) -> float:
    clean = pd.DataFrame({"value": values, "group": groups}).dropna()
    if clean["group"].nunique() != 2:
        return float("nan")
    group_levels = list(clean["group"].unique())
    counts = clean["group"].value_counts().tolist()
    if sorted(counts) != [2, 2]:
        return float("nan")
    observed = clean.loc[clean["group"] == positive_group, "value"].mean() - clean.loc[
        clean["group"] != positive_group, "value"
    ].mean()
    vals = clean["value"].to_numpy()
    indices = list(range(len(vals)))
    diffs = []
    for left in combinations(indices, counts[0]):
        left = set(left)
        right = [idx for idx in indices if idx not in left]
        left_mean = vals[list(left)].mean()
        right_mean = vals[right].mean()
        diffs.append(left_mean - right_mean)
    diffs = np.asarray(diffs)
    return float(np.mean(np.abs(diffs) >= abs(observed) - 1e-12))


def build_split_stats(summary: pd.DataFrame) -> pd.DataFrame:
    varc = summary.loc[summary["subtype"] == "varc"].copy()
    metrics = [
        "performance_rate",
        "oracle_rate",
        "selection_efficiency",
        "pass_gain_1_to_4",
    ]
    rows: list[dict[str, object]] = []
    for metric in metrics:
        arc1 = varc.loc[varc["label"].str.contains("ARC-1"), metric]
        arc2 = varc.loc[varc["label"].str.contains("ARC-2"), metric]
        rows.append(
            {
                "metric": metric,
                "arc1_mean": arc1.mean(),
                "arc2_mean": arc2.mean(),
                "difference": arc1.mean() - arc2.mean(),
                "exact_p_value": exact_two_group_pvalue(
                    varc[metric],
                    varc["label"].str.extract(r"(ARC-\d)")[0],
                    "ARC-1",
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_ceiling(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    palette = {"compress_arc": "#D1495B", "varc": "#1D7874"}

    ax = axes[0]
    for subtype, group in summary.groupby("subtype"):
        ax.scatter(
            group["oracle_rate"],
            group["performance_rate"],
            s=95 if subtype == "compress_arc" else 70,
            marker="D" if subtype == "compress_arc" else "o",
            color=palette.get(subtype, "#444444"),
            alpha=0.9,
            label="Compress ARC" if subtype == "compress_arc" else "VARC",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["label"].replace("ARC-", "A"),
                (row["oracle_rate"], row["performance_rate"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
    lim = max(summary["oracle_rate"].max(), summary["performance_rate"].max()) * 1.08
    ax.plot([0, lim], [0, lim], linestyle="--", color="#666666", linewidth=1, label="oracle = final")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("oracle rate")
    ax.set_ylabel("final performance rate")
    ax.set_title("Final selection vs oracle candidate ceiling")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    order = summary.copy()
    order["order"] = order["subtype"].map({"compress_arc": 0, "varc": 1})
    order = order.sort_values(["order", "oracle_rate"])
    colors = {"compress_arc": "#D1495B", "varc": "#1D7874"}
    bars = ax.bar(
        np.arange(len(order)),
        order["selection_efficiency"],
        color=[colors[sub] for sub in order["subtype"]],
        alpha=0.88,
    )
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order["label"], rotation=28, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("selection efficiency")
    ax.set_title("How much oracle coverage reaches the final answer?")
    ax.grid(True, axis="y", alpha=0.2)
    for bar, value in zip(bars, order["selection_efficiency"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = FIGURES / "nontrm_ceiling_and_selection.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_pass_curves(summary: pd.DataFrame) -> Path:
    varc = summary.loc[summary["subtype"] == "varc"].copy()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    style_by_split = {"ARC-1": ("#4C78A8", "-"), "ARC-2": ("#F58518", "--")}
    marker_by_label = {"Unet": "o", "ViT": "s"}
    x = np.arange(1, 5)
    for _, row in varc.iterrows():
        split = "ARC-1" if "ARC-1" in row["label"] else "ARC-2"
        arch = "ViT" if "ViT" in row["label"] else "Unet"
        color, linestyle = style_by_split[split]
        y = [row["pass1_rate"], row["pass2_rate"], row["pass3_rate"], row["pass4_rate"]]
        ax.plot(
            x,
            y,
            marker=marker_by_label[arch],
            linewidth=2.2,
            linestyle=linestyle,
            color=color,
            label=row["label"],
        )
        ax.scatter(x[-1], y[-1], color=color, s=35, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"pass@{k}" for k in x])
    ax.set_ylim(0, max(0.5, float(varc[["pass4_rate", "performance_rate"]].max().max()) * 1.12))
    ax.set_ylabel("solve rate")
    ax.set_title("VARC shows diminishing returns across successive passes")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    out = FIGURES / "nontrm_varc_pass_curves.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def build_tex(summary: pd.DataFrame, fig1: Path, fig2: Path) -> str:
    fig1_rel = fig1.relative_to(ROOT).as_posix()
    fig2_rel = fig2.relative_to(ROOT).as_posix()
    lines: list[str] = []
    lines.append(r"\section{Non-TRM addendum}")
    lines.append(
        "This addendum strips TRM out of the non-LLM story and keeps only Compress ARC and VARC. "
        "The useful efficiency proxies here are final performance, oracle candidate coverage, selection efficiency, and the pass@k progression. "
        "Because Compress ARC has one run and VARC has four, this is descriptive rather than a formal inferential comparison."
    )
    lines.append("")
    lines.append(r"\subsection{Hypotheses}")
    lines.append(r"\begin{itemize}")
    lines.append(r"\item Final performance is limited partly by selection, not only candidate generation.")
    lines.append(r"\item ARC-1 should be more selection-efficient than ARC-2.")
    lines.append(r"\item VARC gains should diminish after pass@2.")
    lines.append(r"\end{itemize}")
    lines.append("")
    lines.append(r"\subsection{Not pursued}")
    lines.append(r"\begin{itemize}")
    lines.append(
        r"\item We did not pool TRM into this addendum, because step count is not on the same scale as candidate-count or iteration proxies."
    )
    lines.append(r"\item We did not treat Compress ARC as inferential evidence, because it contributes only a single run.")
    lines.append(
        r"\item We did not force one unified non-LLM efficiency score across all subtypes, because the telemetry is structurally different across Compress ARC, VARC, and TRM."
    )
    lines.append(r"\end{itemize}")
    lines.append("")
    lines.append(r"\subsection{Summary table}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Non-TRM efficiency summary.}")
    lines.append(r"\label{tab:nontrm_summary}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"run & performance & oracle & gap & selection & pass1 & pass4 \\")
    lines.append(r"\midrule")
    for _, row in summary.iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(row["label"]),
                    fmt(row["performance_rate"]),
                    fmt(row["oracle_rate"]),
                    fmt(row["oracle_gap"]),
                    fmt(row["selection_efficiency"]),
                    fmt(row["pass1_rate"]),
                    fmt(row["pass4_rate"]),
                ]
            )
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\subsection{ARC-1 versus ARC-2 within VARC}")
    lines.append(
        "Because VARC has two ARC-1 and two ARC-2 runs, we can do an exact two-group permutation check on the split effect. "
        "This is still descriptive in spirit, but it gives a clean sanity check for whether ARC-1 is systematically easier to exploit."
    )
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Exact split comparison for the VARC runs.}")
    lines.append(r"\label{tab:varc_split}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"metric & ARC-1 mean & ARC-2 mean & difference & exact p \\")
    lines.append(r"\midrule")
    split_stats = build_split_stats(summary)
    for _, row in split_stats.iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(row["metric"]),
                    fmt(row["arc1_mean"]),
                    fmt(row["arc2_mean"]),
                    fmt(row["difference"]),
                    fmt(row["exact_p_value"], digits=4),
                ]
            )
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\subsection{Takeaway}")
    lines.append(
        "The main pattern is that oracle coverage is consistently larger than final performance, which points to selection as an important bottleneck. "
        "Compress ARC and ARC-1 Unet sit around 0.59 selection efficiency, ARC-1 ViT is a bit better at about 0.64, and ARC-2 drops to roughly 0.33--0.36. "
        "That makes the ARC-2 VARC runs look less like a search-depth problem and more like a harder candidate-ranking problem."
    )
    lines.append("")
    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\includegraphics[width=0.96\linewidth]{{{fig1_rel}}}")
    lines.append(r"\caption{Non-TRM candidate-ceiling view. Left: final performance versus oracle candidate coverage. Right: selection efficiency, defined as final performance divided by oracle rate.}")
    lines.append(r"\label{fig:nontrm_ceiling}")
    lines.append(r"\end{figure}")
    lines.append("")
    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\includegraphics[width=0.86\linewidth]{{{fig2_rel}}}")
    lines.append(r"\caption{VARC pass@k curves. Gains are front-loaded and flatten quickly, especially on ARC-2.}")
    lines.append(r"\label{fig:nontrm_passk}")
    lines.append(r"\end{figure}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = prepare_summary(load_data())
    summary.to_csv(TABLES / "non_trm_addendum_summary.csv", index=False)
    fig1 = plot_ceiling(summary)
    fig2 = plot_pass_curves(summary)
    tex = build_tex(summary, fig1, fig2)
    (ROOT / "non_trm_addendum.tex").write_text(tex, encoding="utf-8")
    print(f"Wrote {ROOT / 'non_trm_addendum.tex'}")
    print(f"Wrote {fig1}")
    print(f"Wrote {fig2}")
    print(f"Wrote {TABLES / 'non_trm_addendum_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
