#!/usr/bin/env python3
"""Analyze inter-annotator agreement for TakeMeter labels.

Fill `labeler_b_label` in data/inter_annotator_sample.csv with an independent
human label for each row, then run this script to compute percentage agreement
and Cohen's kappa. The repository does not claim this stretch point until that
second human annotation is present.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "inter_annotator_sample.csv"
OUTPUT_PATH = ROOT / "results" / "inter_annotator_agreement.json"
VALID_LABELS = {"actionable", "underspecified", "opinion_or_request"}


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    total = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / total
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum((counts_a[label] / total) * (counts_b[label] / total) for label in VALID_LABELS)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def main() -> None:
    with INPUT_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    missing = [row["id"] for row in rows if not row.get("labeler_b_label", "").strip()]
    invalid = [
        row["id"]
        for row in rows
        if row.get("labeler_b_label", "").strip()
        and row["labeler_b_label"].strip() not in VALID_LABELS
    ]
    if missing or invalid:
        raise SystemExit(
            "Inter-annotator analysis is not ready. "
            f"Missing labeler_b_label for {len(missing)} rows; invalid labels for {len(invalid)} rows."
        )

    labels_a = [row["labeler_a_label"].strip() for row in rows]
    labels_b = [row["labeler_b_label"].strip() for row in rows]
    correct = sum(a == b for a, b in zip(labels_a, labels_b))
    output = {
        "example_count": len(rows),
        "percentage_agreement": round(correct / len(rows), 4),
        "cohen_kappa": round(cohen_kappa(labels_a, labels_b), 4),
        "labeler_a_distribution": dict(Counter(labels_a)),
        "labeler_b_distribution": dict(Counter(labels_b)),
        "disagreements": [
            {
                "id": row["id"],
                "text": row["text"],
                "labeler_a_label": row["labeler_a_label"],
                "labeler_b_label": row["labeler_b_label"],
            }
            for row in rows
            if row["labeler_a_label"] != row["labeler_b_label"]
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
