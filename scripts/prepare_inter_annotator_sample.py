#!/usr/bin/env python3
"""Create a 30-row human inter-annotator worksheet from the validation split."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "openai_developer_community_labeled.csv"
OUTPUT_PATH = ROOT / "data" / "inter_annotator_sample.csv"


def main() -> None:
    with SOURCE_PATH.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == "validation"]

    fieldnames = ["id", "text", "labeler_a_label", "labeler_b_label", "notes"]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "labeler_a_label": row["label"],
                    "labeler_b_label": "",
                    "notes": "",
                }
            )
    print(f"wrote {OUTPUT_PATH} with {len(rows)} rows")


if __name__ == "__main__":
    main()
