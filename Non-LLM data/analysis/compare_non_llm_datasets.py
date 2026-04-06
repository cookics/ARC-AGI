from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
PROCESSED_DIR = ROOT / "processed"


@dataclass(frozen=True)
class DatasetRecord:
    collection: str
    dataset: str
    path: str
    split: str
    kind: str
    item_count: int | None
    notes: str | None = None


def iter_json_files(base: Path) -> Iterable[Path]:
    for path in sorted(base.rglob("*.json")):
        if path.is_file():
            yield path


def infer_source(path: Path) -> str:
    parts = path.relative_to(RAW_DIR).parts
    return parts[0] if parts else "unknown"


def infer_dataset(path: Path) -> str:
    parts = path.relative_to(RAW_DIR).parts
    if len(parts) >= 2:
        return "/".join(parts[:-1])
    return parts[0] if parts else "unknown"


def infer_split(filename: str) -> str:
    lower = filename.lower()
    match = re.search(r"arc-agi-(1-5|1|2)", lower)
    if match:
        token = match.group(1)
        return "1.5" if token == "1-5" else token
    if "training" in lower:
        return "training"
    if "evaluation" in lower:
        return "evaluation"
    if "test" in lower:
        return "test"
    if "concept" in lower:
        return "concept"
    if "1-5" in lower or "1_5" in lower:
        return "1.5"
    if "_1_" in f"_{lower}_":
        return "1"
    if "_2_" in f"_{lower}_":
        return "2"
    return "unknown"


def infer_kind(filename: str) -> str:
    lower = filename.lower()
    if "challenges" in lower:
        return "challenges"
    if "solutions" in lower:
        return "solutions"
    if "submission" in lower:
        return "submission"
    if lower.endswith(".json"):
        return "json"
    return "unknown"


def count_top_level_items(data: Any) -> int | None:
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return len(data)
    return None


def build_inventory() -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for path in iter_json_files(RAW_DIR):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            item_count = count_top_level_items(payload)
            notes = None
        except Exception as exc:  # noqa: BLE001
            item_count = None
            notes = f"unreadable json: {exc}"

        records.append(
            DatasetRecord(
                collection=infer_source(path),
                dataset=infer_dataset(path),
                path=str(path.relative_to(ROOT)).replace("\\", "/"),
                split=infer_split(path.name),
                kind=infer_kind(path.name),
                item_count=item_count,
                notes=notes,
            )
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a normalized inventory for the imported non-LLM datasets."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "non_llm_dataset_inventory.json",
        help="Where to write the inventory JSON.",
    )
    args = parser.parse_args()

    records = build_inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([asdict(record) for record in records], indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
