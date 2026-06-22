#!/usr/bin/env python3
"""Export sample TakeMeter predictions with confidence scores.

Milestone 6 asks for 3-5 examples run through the fine-tuned model with
predicted labels and confidence. The original Colab checkpoint was not
committed because model weights are large, so this script reruns the same
small DistilBERT training setup on the committed split and exports a compact
sample table for the README.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "openai_developer_community_labeled.csv"
OUTPUT_PATH = ROOT / "results" / "milestone6_sample_classifications.json"

MODEL_NAME = "distilbert-base-uncased"
LABEL_MAP = {
    "actionable": 0,
    "underspecified": 1,
    "opinion_or_request": 2,
}
ID_TO_LABEL = {value: key for key, value in LABEL_MAP.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["label_id"] = LABEL_MAP[row["label"]]
    return rows


class TextDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.encodings = tokenizer(
            [row["text"] for row in rows],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor([row["label_id"] for row in rows], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, dataloader: DataLoader, device: torch.device) -> tuple[float, list[int], list[float]]:
    model.eval()
    predictions: list[int] = []
    confidences: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            confidence, predicted = probabilities.max(dim=-1)
            predictions.extend(predicted.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    return accuracy_score(labels, predictions), predictions, confidences


def shorten(text: str, max_chars: int = 260) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def build_samples(rows: list[dict[str, str]], predictions: list[int], confidences: list[float]) -> list[dict[str, object]]:
    enriched = []
    for row, prediction, confidence in zip(rows, predictions, confidences):
        true_label = row["label"]
        predicted_label = ID_TO_LABEL[prediction]
        enriched.append(
            {
                "id": row["id"],
                "text": row["text"],
                "short_text": shorten(row["text"]),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": round(float(confidence), 3),
                "correct": true_label == predicted_label,
                "source_url": row["source_url"],
            }
        )

    correct = [row for row in enriched if row["correct"]]
    wrong = [row for row in enriched if not row["correct"]]
    correct.sort(key=lambda row: (-row["confidence"], row["id"]))
    wrong.sort(key=lambda row: (-row["confidence"], row["id"]))

    selected = correct[:3] + wrong[:2]
    selected_ids = {row["id"] for row in selected}
    if len(selected) < 5:
        for row in enriched:
            if row["id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["id"])
            if len(selected) == 5:
                break
    selected.sort(key=lambda row: row["id"])
    return selected[:5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    rows = load_rows(DATA_PATH)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = TextDataset(train_rows, tokenizer, args.max_length)
    validation_dataset = TextDataset(validation_rows, tokenizer, args.max_length)
    test_dataset = TextDataset(test_rows, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

    device = choose_device()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_MAP))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, total_steps),
        num_training_steps=total_steps,
    )

    best_validation_accuracy = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += float(outputs.loss.detach().cpu())

        validation_accuracy, _, _ = evaluate(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(running_loss / max(len(train_loader), 1), 4),
                "validation_accuracy": round(float(validation_accuracy), 4),
            }
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        print(f"epoch {epoch}: validation_accuracy={validation_accuracy:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_accuracy, test_predictions, test_confidences = evaluate(model, test_loader, device)
    true_ids = [row["label_id"] for row in test_rows]
    report = classification_report(
        true_ids,
        test_predictions,
        labels=list(ID_TO_LABEL.keys()),
        target_names=[ID_TO_LABEL[index] for index in range(len(ID_TO_LABEL))],
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(true_ids, test_predictions, labels=list(ID_TO_LABEL.keys())).tolist()

    output = {
        "milestone": 6,
        "created_at": "2026-06-22",
        "purpose": "Sample classifications with confidence scores for the Milestone 6 README and demo.",
        "note": "Generated by rerunning the committed train/validation/test split with the same DistilBERT setup; large model weights are intentionally not committed.",
        "model": {
            "base_model": MODEL_NAME,
            "seed": args.seed,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "max_length": args.max_length,
            "device": str(device),
        },
        "training_history": history,
        "test_metrics_for_this_sample_export_run": {
            "accuracy": round(float(test_accuracy), 4),
            "per_class": {
                label: {
                    "precision": round(float(report[label]["precision"]), 3),
                    "recall": round(float(report[label]["recall"]), 3),
                    "f1": round(float(report[label]["f1-score"]), 3),
                    "support": int(report[label]["support"]),
                }
                for label in LABEL_MAP
            },
            "confusion_matrix": {
                "labels": [ID_TO_LABEL[index] for index in range(len(ID_TO_LABEL))],
                "rows": "true_label",
                "columns": "predicted_label",
                "matrix": matrix,
            },
        },
        "sample_classifications": build_samples(test_rows, test_predictions, test_confidences),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    print(json.dumps(output["sample_classifications"], indent=2))


if __name__ == "__main__":
    main()
