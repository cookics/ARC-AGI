from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "coverage-audit"
ARC2_HUMAN_CSV = ROOT / "data-human" / "test_pair_attempts.csv"
ARC1_HRC_SUMMARY_CSV = ROOT / "HR data" / "data" / "summary_data.csv"
ARC1_HRC_ACTION_CSV = ROOT / "HR data" / "data" / "data.csv"
ARC1_HRC_SURVEY_DIR = ROOT / "HR data" / "survey"
ARC1_TRAIN_DIR = ROOT / "data-llm" / "ARC-AGI" / "data" / "training"
ARC1_EVAL_DIR = ROOT / "data-llm" / "ARC-AGI" / "data" / "evaluation"
ARC2_TRAIN_DIR = ROOT / "data-llm" / "ARC-AGI-2" / "data" / "training"
ARC2_EVAL_DIR = ROOT / "data-llm" / "ARC-AGI-2" / "data" / "evaluation"

LIGHT_GRAY = "#D8DEE9"
FAIL_RED = "#E31A1C"
THINKING_TEAL = "#1ABC9C"
TEXT_GRAY = "#555555"
TITLE_GRAY = "#2C3E50"


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "text.color": TEXT_GRAY,
            "axes.labelcolor": TEXT_GRAY,
            "xtick.color": TEXT_GRAY,
            "ytick.color": TEXT_GRAY,
        }
    )


def count_task_and_pair_universe(folder: Path) -> tuple[set[str], set[str]]:
    task_ids: set[str] = set()
    pair_ids: set[str] = set()
    for json_path in sorted(folder.glob("*.json")):
        task_ids.add(json_path.stem)
        obj = json.loads(json_path.read_text(encoding="utf-8"))
        for test_index, _pair in enumerate(obj.get("test", [])):
            pair_ids.add(f"{json_path.stem}__{test_index}")
    return task_ids, pair_ids


def load_arc2_human_attempts() -> pd.DataFrame:
    df = pd.read_csv(ARC2_HUMAN_CSV)
    df["solved"] = (df["correct_submissions"] > 0).astype(int)
    df["task_pair_id"] = df["task_ID"] + "__" + df["test_index"].astype(str)
    return df


def load_arc1_hrc_responses() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "exp_name",
        "task_type",
        "hashed_id",
        "joint_id_task",
        "task_name",
        "task_number",
        "attempt_number",
        "num_actions",
        "solved",
        "complete",
    ]
    raw = pd.read_csv(ARC1_HRC_SUMMARY_CSV, usecols=cols)
    raw["task_id"] = raw["task_name"].astype(str).str.replace(".json", "", regex=False)
    grouped = (
        raw.groupby(["hashed_id", "task_type", "task_id"], as_index=False)
        .agg(
            solved=("solved", "max"),
            complete=("complete", "max"),
            max_attempt_number=("attempt_number", "max"),
            total_summary_rows=("task_id", "size"),
            max_num_actions=("num_actions", "max"),
            exp_name=("exp_name", "first"),
        )
        .sort_values(["task_type", "hashed_id", "task_id"])
        .reset_index(drop=True)
    )
    return raw, grouped


def fit_person_item_model(
    df: pd.DataFrame,
    person_col: str,
    item_col: str,
    outcome_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    design = encoder.fit_transform(df[[person_col, item_col]])
    y = df[outcome_col].astype(int).to_numpy()

    model = LogisticRegression(
        C=2.0,
        solver="saga",
        max_iter=8000,
        fit_intercept=True,
        random_state=0,
    )
    model.fit(design, y)

    feature_names = pd.Index(encoder.get_feature_names_out([person_col, item_col]))
    coefficients = pd.Series(model.coef_[0], index=feature_names)

    person_values = coefficients[feature_names.str.startswith(f"{person_col}_")]
    person_values.index = person_values.index.str.replace(f"{person_col}_", "", regex=False)
    person_values = person_values - person_values.mean()

    item_ease = coefficients[feature_names.str.startswith(f"{item_col}_")]
    item_ease.index = item_ease.index.str.replace(f"{item_col}_", "", regex=False)
    item_difficulty = -(item_ease - item_ease.mean())

    person_df = pd.DataFrame({person_col: person_values.index, "ability": person_values.values})
    item_df = pd.DataFrame({item_col: item_difficulty.index, "difficulty": item_difficulty.values})
    return person_df, item_df


def build_coverage_summary(
    arc1_hrc_grouped: pd.DataFrame,
    arc2_human: pd.DataFrame,
    arc1_train_tasks: set[str],
    arc1_eval_tasks: set[str],
    arc2_train_tasks: set[str],
    arc2_eval_tasks: set[str],
    arc2_train_pairs: set[str],
    arc2_eval_pairs: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_row(
        dataset: str,
        split: str,
        unit: str,
        respondent_unit: str,
        respondent_count: int | float,
        response_rows: int | float,
        attempted_units: int,
        total_units: int,
        solve_rate: float | None,
        mean_items_per_respondent: float | None,
        median_items_per_respondent: float | None,
        note: str = "",
    ) -> None:
        density = np.nan
        if respondent_count and total_units:
            density = response_rows / (respondent_count * total_units)
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "unit": unit,
                "respondent_unit": respondent_unit,
                "respondent_count": respondent_count,
                "response_rows": response_rows,
                "attempted_units": attempted_units,
                "total_units": total_units,
                "coverage": attempted_units / total_units if total_units else np.nan,
                "matrix_density": density,
                "solve_rate": solve_rate,
                "mean_items_per_respondent": mean_items_per_respondent,
                "median_items_per_respondent": median_items_per_respondent,
                "note": note,
            }
        )

    for split, universe in [("training", arc1_train_tasks), ("evaluation", arc1_eval_tasks)]:
        sub = arc1_hrc_grouped.loc[arc1_hrc_grouped["task_type"] == split].copy()
        by_person = sub.groupby("hashed_id")["task_id"].nunique()
        add_row(
            dataset="ARC1_HRC",
            split=split,
            unit="task_id",
            respondent_unit="participant",
            respondent_count=sub["hashed_id"].nunique(),
            response_rows=len(sub),
            attempted_units=sub["task_id"].nunique(),
            total_units=len(universe),
            solve_rate=float(sub["solved"].mean()),
            mean_items_per_respondent=float(by_person.mean()),
            median_items_per_respondent=float(by_person.median()),
            note=f"complete_participants={sub.loc[sub['complete'], 'hashed_id'].nunique()}",
        )

    by_person_arc1 = arc1_hrc_grouped.groupby("hashed_id")["task_id"].nunique()
    add_row(
        dataset="ARC1_HRC",
        split="overall",
        unit="task_id",
        respondent_unit="participant",
        respondent_count=arc1_hrc_grouped["hashed_id"].nunique(),
        response_rows=len(arc1_hrc_grouped),
        attempted_units=arc1_hrc_grouped["task_id"].nunique(),
        total_units=len(arc1_train_tasks | arc1_eval_tasks),
        solve_rate=float(arc1_hrc_grouped["solved"].mean()),
        mean_items_per_respondent=float(by_person_arc1.mean()),
        median_items_per_respondent=float(by_person_arc1.median()),
        note=f"complete_participants={arc1_hrc_grouped.loc[arc1_hrc_grouped['complete'], 'hashed_id'].nunique()}",
    )

    for task_set_label, task_universe, pair_universe in [
        ("Public Train", arc2_train_tasks, arc2_train_pairs),
        ("Public Eval", arc2_eval_tasks, arc2_eval_pairs),
    ]:
        sub = arc2_human.loc[arc2_human["task_set"] == task_set_label].copy()
        by_session_tasks = sub.groupby("session_ID")["task_ID"].nunique()
        by_session_pairs = sub.groupby("session_ID")["task_pair_id"].nunique()
        add_row(
            dataset="ARC2_ARCPrize",
            split=task_set_label,
            unit="task_id",
            respondent_unit="session",
            respondent_count=sub["session_ID"].nunique(),
            response_rows=len(sub),
            attempted_units=sub["task_ID"].nunique(),
            total_units=len(task_universe),
            solve_rate=float(sub["solved"].mean()),
            mean_items_per_respondent=float(by_session_tasks.mean()),
            median_items_per_respondent=float(by_session_tasks.median()),
            note="task-level coverage from pair-attempt log",
        )
        add_row(
            dataset="ARC2_ARCPrize",
            split=task_set_label,
            unit="task_pair_id",
            respondent_unit="session",
            respondent_count=sub["session_ID"].nunique(),
            response_rows=len(sub),
            attempted_units=sub["task_pair_id"].nunique(),
            total_units=len(pair_universe),
            solve_rate=float(sub["solved"].mean()),
            mean_items_per_respondent=float(by_session_pairs.mean()),
            median_items_per_respondent=float(by_session_pairs.median()),
            note="pair-level official denominator",
        )

    by_session_tasks_arc2 = arc2_human.groupby("session_ID")["task_ID"].nunique()
    by_session_pairs_arc2 = arc2_human.groupby("session_ID")["task_pair_id"].nunique()
    add_row(
        dataset="ARC2_ARCPrize",
        split="overall",
        unit="task_id",
        respondent_unit="session",
        respondent_count=arc2_human["session_ID"].nunique(),
        response_rows=len(arc2_human),
        attempted_units=arc2_human["task_ID"].nunique(),
        total_units=len(arc2_train_tasks | arc2_eval_tasks),
        solve_rate=float(arc2_human["solved"].mean()),
        mean_items_per_respondent=float(by_session_tasks_arc2.mean()),
        median_items_per_respondent=float(by_session_tasks_arc2.median()),
        note="Public Train + Public Eval task IDs",
    )
    add_row(
        dataset="ARC2_ARCPrize",
        split="overall",
        unit="task_pair_id",
        respondent_unit="session",
        respondent_count=arc2_human["session_ID"].nunique(),
        response_rows=len(arc2_human),
        attempted_units=arc2_human["task_pair_id"].nunique(),
        total_units=len(arc2_train_pairs | arc2_eval_pairs),
        solve_rate=float(arc2_human["solved"].mean()),
        mean_items_per_respondent=float(by_session_pairs_arc2.mean()),
        median_items_per_respondent=float(by_session_pairs_arc2.median()),
        note="Public Train + Public Eval task pairs",
    )

    combined_task_attempted = arc1_hrc_grouped["task_id"].nunique() + arc2_human["task_ID"].nunique()
    combined_task_total = len(arc1_train_tasks | arc1_eval_tasks) + len(arc2_train_tasks | arc2_eval_tasks)
    rows.append(
        {
            "dataset": "Combined",
            "split": "overall",
            "unit": "benchmark_tagged_task_id",
            "respondent_unit": "not_applicable",
            "respondent_count": np.nan,
            "response_rows": np.nan,
            "attempted_units": combined_task_attempted,
            "total_units": combined_task_total,
            "coverage": combined_task_attempted / combined_task_total,
            "matrix_density": np.nan,
            "solve_rate": np.nan,
            "mean_items_per_respondent": np.nan,
            "median_items_per_respondent": np.nan,
            "note": "ARC1 task IDs + ARC2 task IDs counted within benchmark",
        }
    )

    combined_item_attempted = arc1_hrc_grouped["task_id"].nunique() + arc2_human["task_pair_id"].nunique()
    combined_item_total = len(arc1_train_tasks | arc1_eval_tasks) + len(arc2_train_pairs | arc2_eval_pairs)
    rows.append(
        {
            "dataset": "Combined",
            "split": "overall",
            "unit": "benchmark_tagged_item",
            "respondent_unit": "not_applicable",
            "respondent_count": np.nan,
            "response_rows": np.nan,
            "attempted_units": combined_item_attempted,
            "total_units": combined_item_total,
            "coverage": combined_item_attempted / combined_item_total,
            "matrix_density": np.nan,
            "solve_rate": np.nan,
            "mean_items_per_respondent": np.nan,
            "median_items_per_respondent": np.nan,
            "note": "ARC1 tasks + ARC2 task pairs",
        }
    )

    return pd.DataFrame(rows)


def build_metadata_catalog() -> pd.DataFrame:
    entries = [
        {
            "dataset": "ARC2_ARCPrize",
            "table_name": "test_pair_attempts.csv",
            "row_unit": "session-task-pair attempt",
            "column_name": "task_ID",
            "description": "ARC-AGI-2 task identifier.",
            "join_key": "task_ID",
            "per_solve": True,
        },
        {
            "dataset": "ARC2_ARCPrize",
            "table_name": "test_pair_attempts.csv",
            "row_unit": "session-task-pair attempt",
            "column_name": "task_set",
            "description": "Public Train or Public Eval.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC2_ARCPrize",
            "table_name": "test_pair_attempts.csv",
            "row_unit": "session-task-pair attempt",
            "column_name": "test_index",
            "description": "Index of the test pair within the task.",
            "join_key": "task_ID + test_index",
            "per_solve": True,
        },
        {
            "dataset": "ARC2_ARCPrize",
            "table_name": "test_pair_attempts.csv",
            "row_unit": "session-task-pair attempt",
            "column_name": "session_ID",
            "description": "Anonymous session identifier.",
            "join_key": "session_ID",
            "per_solve": True,
        },
        {
            "dataset": "ARC2_ARCPrize",
            "table_name": "test_pair_attempts.csv",
            "row_unit": "session-task-pair attempt",
            "column_name": "start_time_seconds",
            "description": "Seconds from session start when the attempt began.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC2_ARCPrize",
            "table_name": "test_pair_attempts.csv",
            "row_unit": "session-task-pair attempt",
            "column_name": "duration_seconds",
            "description": "Time spent on the attempt.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC2_ARCPrize",
            "table_name": "test_pair_attempts.csv",
            "row_unit": "session-task-pair attempt",
            "column_name": "submissions",
            "description": "Number of submissions made during the attempt.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC2_ARCPrize",
            "table_name": "test_pair_attempts.csv",
            "row_unit": "session-task-pair attempt",
            "column_name": "correct_submissions",
            "description": "Count of correct submissions for the attempt.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "summary_data.csv",
            "row_unit": "participant-task attempt summary",
            "column_name": "exp_name",
            "description": "Internal experiment version identifier.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "summary_data.csv",
            "row_unit": "participant-task attempt summary",
            "column_name": "task_type",
            "description": "training or evaluation split.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "summary_data.csv",
            "row_unit": "participant-task attempt summary",
            "column_name": "hashed_id",
            "description": "Anonymous participant identifier.",
            "join_key": "hashed_id",
            "per_solve": True,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "summary_data.csv",
            "row_unit": "participant-task attempt summary",
            "column_name": "joint_id_task",
            "description": "Participant-task composite identifier.",
            "join_key": "joint_id_task",
            "per_solve": True,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "summary_data.csv",
            "row_unit": "participant-task attempt summary",
            "column_name": "task_name",
            "description": "Task file name such as 32597951.json.",
            "join_key": "task_name",
            "per_solve": True,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "summary_data.csv",
            "row_unit": "participant-task attempt summary",
            "column_name": "task_number",
            "description": "Order of the task within a participant run.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "summary_data.csv",
            "row_unit": "participant-task attempt summary",
            "column_name": "attempt_number",
            "description": "Attempt counter for repeated tries on the same task.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "summary_data.csv",
            "row_unit": "participant-task attempt summary",
            "column_name": "num_actions",
            "description": "Number of actions taken up to submission.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "summary_data.csv",
            "row_unit": "participant-task attempt summary",
            "column_name": "solved",
            "description": "Whether the attempt solved the task.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "summary_data.csv",
            "row_unit": "participant-task attempt summary",
            "column_name": "test_output_grid",
            "description": "Ground-truth output grid in string form.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "summary_data.csv",
            "row_unit": "participant-task attempt summary",
            "column_name": "first_written_solution",
            "description": "First solution grid the participant wrote.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "summary_data.csv",
            "row_unit": "participant-task attempt summary",
            "column_name": "last_written_solution",
            "description": "Last written solution at the end of the attempt.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "summary_data.csv",
            "row_unit": "participant-task attempt summary",
            "column_name": "complete",
            "description": "Whether the participant completed the experiment protocol.",
            "join_key": "",
            "per_solve": True,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "data.csv",
            "row_unit": "participant-task-attempt action",
            "column_name": "time",
            "description": "Timestamp for each fine-grained interface action.",
            "join_key": "",
            "per_solve": False,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "data.csv",
            "row_unit": "participant-task-attempt action",
            "column_name": "action",
            "description": "Action verb taken by the participant.",
            "join_key": "",
            "per_solve": False,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "data.csv",
            "row_unit": "participant-task-attempt action",
            "column_name": "action_x/action_y",
            "description": "Coordinates for the interface action.",
            "join_key": "",
            "per_solve": False,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "data.csv",
            "row_unit": "participant-task-attempt action",
            "column_name": "selected_symbol/selected_tool",
            "description": "Currently selected color and tool.",
            "join_key": "",
            "per_solve": False,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "data.csv",
            "row_unit": "participant-task-attempt action",
            "column_name": "test_input_grid/test_output_grid",
            "description": "Input and target grids, plus their dimensions.",
            "join_key": "",
            "per_solve": False,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "survey/*.csv",
            "row_unit": "participant-level sidecar",
            "column_name": "age/gender/race/education_level/normal_vision/color_blind/fluent_english",
            "description": "Joinable participant demographics keyed by hashed_id.",
            "join_key": "hashed_id",
            "per_solve": False,
        },
        {
            "dataset": "ARC1_HRC",
            "table_name": "survey/*.csv",
            "row_unit": "participant-level sidecar",
            "column_name": "feedback/withdraw/withdraw_reason/withdraw_comment",
            "description": "Joinable qualitative feedback and withdrawal metadata keyed by hashed_id.",
            "join_key": "hashed_id",
            "per_solve": False,
        },
    ]
    return pd.DataFrame(entries)


def plot_arc1_hrc_response_matrices(
    arc1_hrc_grouped: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    split_titles = {
        "training": "ARC-1 HRC Training",
        "evaluation": "ARC-1 HRC Evaluation",
    }
    split_universes = {
        "training": set(p.stem for p in ARC1_TRAIN_DIR.glob("*.json")),
        "evaluation": set(p.stem for p in ARC1_EVAL_DIR.glob("*.json")),
    }
    cmap = ListedColormap([LIGHT_GRAY, FAIL_RED, THINKING_TEAL])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), sharey=False)
    diagnostics: list[dict[str, object]] = []

    for ax, split in zip(axes, ["training", "evaluation"]):
        sub = arc1_hrc_grouped.loc[arc1_hrc_grouped["task_type"] == split, ["hashed_id", "task_id", "solved"]].copy()
        person_df, item_df = fit_person_item_model(sub, "hashed_id", "task_id", "solved")
        ordered_people = person_df.sort_values("ability")["hashed_id"].tolist()
        ordered_items = item_df.sort_values("difficulty")["task_id"].tolist()

        matrix = (
            sub.pivot_table(index="hashed_id", columns="task_id", values="solved", aggfunc="max")
            .reindex(index=ordered_people, columns=ordered_items)
            .to_numpy(dtype=float)
        )
        matrix = np.where(np.isnan(matrix), -1, matrix)
        ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, origin="lower")
        ax.set_title(split_titles[split], loc="left", color=TITLE_GRAY)
        ax.set_xlabel("Tasks: Easier to Harder")
        ax.set_xticks([])
        ax.set_yticks([])
        if split == "training":
            ax.set_ylabel("Participants: Lower to Higher Ability")

        diagnostics.append(
            {
                "split": split,
                "participants": len(ordered_people),
                "tasks": len(ordered_items),
                "matrix_density": len(sub) / (len(ordered_people) * len(split_universes[split])),
                "solve_rate": float(sub["solved"].mean()),
            }
        )

    legend_handles = [
        Patch(facecolor=LIGHT_GRAY, edgecolor=LIGHT_GRAY, label="Not Attempted"),
        Patch(facecolor=FAIL_RED, edgecolor=FAIL_RED, label="Failed"),
        Patch(facecolor=THINKING_TEAL, edgecolor=THINKING_TEAL, label="Solved"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", frameon=False)
    fig.suptitle(
        "ARC-1 HRC Participant-by-Task Response Matrices\nSorted within split by latent ability and task difficulty",
        x=0.01,
        ha="left",
        color=TITLE_GRAY,
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Training and evaluation are separate participant pools in HRC, so the latent ordering is fit within split rather than forced onto one global scale.",
        fontsize=10,
        color=TEXT_GRAY,
    )
    fig.savefig(out_path)
    plt.close(fig)
    return pd.DataFrame(diagnostics)


def write_report(
    coverage_summary: pd.DataFrame,
    metadata_catalog: pd.DataFrame,
    arc1_plot_diag: pd.DataFrame,
) -> None:
    def row(dataset: str, split: str, unit: str) -> pd.Series:
        return coverage_summary.loc[
            (coverage_summary["dataset"] == dataset)
            & (coverage_summary["split"] == split)
            & (coverage_summary["unit"] == unit)
        ].iloc[0]

    arc1_overall = row("ARC1_HRC", "overall", "task_id")
    arc2_task_overall = row("ARC2_ARCPrize", "overall", "task_id")
    arc2_pair_overall = row("ARC2_ARCPrize", "overall", "task_pair_id")
    combined_tasks = row("Combined", "overall", "benchmark_tagged_task_id")
    combined_items = row("Combined", "overall", "benchmark_tagged_item")

    lines = [
        "# Human Coverage Audit",
        "",
        "## Headline Coverage",
        "",
        f"- ARC-1 HRC covers all official ARC-1 task IDs: {int(arc1_overall['attempted_units'])} / {int(arc1_overall['total_units'])} = {arc1_overall['coverage']:.1%}.",
        f"- ARC-2 ARC Prize human data covers {int(arc2_task_overall['attempted_units'])} / {int(arc2_task_overall['total_units'])} task IDs = {arc2_task_overall['coverage']:.1%}.",
        f"- ARC-2 pair-level coverage is {int(arc2_pair_overall['attempted_units'])} / {int(arc2_pair_overall['total_units'])} = {arc2_pair_overall['coverage']:.1%}.",
        f"- Combined benchmark-tagged task coverage is {int(combined_tasks['attempted_units'])} / {int(combined_tasks['total_units'])} = {combined_tasks['coverage']:.1%}.",
        f"- Combined benchmark-tagged item coverage, counting ARC-1 tasks plus ARC-2 task pairs, is {int(combined_items['attempted_units'])} / {int(combined_items['total_units'])} = {combined_items['coverage']:.1%}.",
        "",
        "## Metadata",
        "",
        f"- ARC-2 per-attempt table contributes {int((metadata_catalog['dataset'] == 'ARC2_ARCPrize').sum())} documented fields.",
        f"- ARC-1 HRC contributes {int((metadata_catalog['dataset'] == 'ARC1_HRC').sum())} documented field groups across summary rows, action logs, and survey sidecars.",
        "- The ARC-1 figure is split into train and evaluation panels because HRC uses disjoint participant pools for those two splits.",
        "",
        "## ARC-1 Plot Diagnostics",
        "",
        arc1_plot_diag.to_csv(index=False),
    ]
    (OUT_DIR / "arc_human_coverage_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_out_dir()
    configure_style()

    arc1_train_tasks, _arc1_train_pairs = count_task_and_pair_universe(ARC1_TRAIN_DIR)
    arc1_eval_tasks, _arc1_eval_pairs = count_task_and_pair_universe(ARC1_EVAL_DIR)
    arc2_train_tasks, arc2_train_pairs = count_task_and_pair_universe(ARC2_TRAIN_DIR)
    arc2_eval_tasks, arc2_eval_pairs = count_task_and_pair_universe(ARC2_EVAL_DIR)

    arc2_human = load_arc2_human_attempts()
    _arc1_hrc_raw, arc1_hrc_grouped = load_arc1_hrc_responses()

    coverage_summary = build_coverage_summary(
        arc1_hrc_grouped=arc1_hrc_grouped,
        arc2_human=arc2_human,
        arc1_train_tasks=arc1_train_tasks,
        arc1_eval_tasks=arc1_eval_tasks,
        arc2_train_tasks=arc2_train_tasks,
        arc2_eval_tasks=arc2_eval_tasks,
        arc2_train_pairs=arc2_train_pairs,
        arc2_eval_pairs=arc2_eval_pairs,
    )
    metadata_catalog = build_metadata_catalog()
    arc1_plot_diag = plot_arc1_hrc_response_matrices(
        arc1_hrc_grouped=arc1_hrc_grouped,
        out_path=OUT_DIR / "chart_arc1_hrc_response_matrix_train_eval.png",
    )

    coverage_summary.to_csv(OUT_DIR / "arc_human_coverage_summary.csv", index=False)
    metadata_catalog.to_csv(OUT_DIR / "arc_human_metadata_catalog.csv", index=False)
    arc1_plot_diag.to_csv(OUT_DIR / "arc1_hrc_response_matrix_diagnostics.csv", index=False)
    write_report(coverage_summary, metadata_catalog, arc1_plot_diag)


if __name__ == "__main__":
    main()
