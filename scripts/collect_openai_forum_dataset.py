#!/usr/bin/env python3
"""Collect and label an OpenAI Developer Community dataset for TakeMeter.

The script uses public Discourse JSON endpoints, strips HTML, redacts common
secret/PII shapes, applies the project label rubric, and writes a balanced CSV.
"""

from __future__ import annotations

import csv
import html
import json
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


BASE_URL = "https://community.openai.com"
OUT_PATH = Path("data/openai_developer_community_labeled.csv")
SUMMARY_PATH = Path("data/openai_developer_community_summary.json")
RANDOM_SEED = 42

CATEGORIES = {
    7: "API",
    42: "ChatGPT Apps SDK",
    37: "Codex",
    8: "Prompting",
    14: "Documentation",
    21: "Community",
    19: "ChatGPT developer tools",
    41: "Open Models",
}

TARGET_COUNTS = {
    "actionable": 80,
    "underspecified": 60,
    "opinion_or_request": 60,
}


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return " ".join(self.parts)


@dataclass(frozen=True)
class Example:
    source_url: str
    topic_id: int
    post_id: int
    post_number: int
    category: str
    topic_title: str
    text: str
    label: str
    annotation_note: str


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 TakeMeter student dataset collection",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def strip_html(cooked: str) -> str:
    parser = TextExtractor()
    parser.feed(cooked or "")
    text = html.unescape(parser.text())
    return normalize_space(text)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def scrub_text(text: str) -> str:
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        text,
    )
    text = re.sub(r"https?://\S+", "[URL]", text)
    text = re.sub(r"\B@[A-Za-z0-9_][A-Za-z0-9_.-]{1,40}", "[USER]", text)
    text = re.sub(
        r"\b(?:sk|gsk|sess|org|proj|user|key)-[A-Za-z0-9_-]{10,}\b",
        "[REDACTED_ID]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b",
        "[REDACTED_TOKEN]",
        text,
    )
    return normalize_space(text)


def contains_any(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if term in text)


def classify(text: str) -> tuple[str, str]:
    lowered = text.lower()

    actionable_terms = [
        "traceback",
        "error code",
        "status code",
        "400",
        "401",
        "403",
        "404",
        "429",
        "500",
        "exception",
        "stack trace",
        "curl",
        "python",
        "node",
        "javascript",
        "typescript",
        "json",
        "schema",
        "endpoint",
        "headers",
        "request body",
        "response",
        "sdk",
        "api key",
        "model=",
        "gpt-",
        "stream",
        "temperature",
        "token",
        "fine-tun",
        "batch",
        "webhook",
        "mcp",
        "bug",
        "regression",
        "repro",
        "e.g.",
        "tool support",
        "tool called",
        "function",
        "agent uses",
        "reproduce",
        "steps",
        "workaround",
        "solution",
        "fixed",
        "try ",
        "use ",
        "set ",
        "configure",
        "logs",
    ]
    problem_terms = [
        "help",
        "not working",
        "does not work",
        "doesn't work",
        "broken",
        "issue",
        "problem",
        "failed",
        "fails",
        "error",
        "anyone",
        "why is",
        "how do i",
        "can someone",
        "what am i missing",
        "unable to",
    ]
    request_terms = [
        "feature request",
        "please add",
        "would like",
        "wish",
        "should add",
        "can you add",
        "i want",
        "pricing",
        "price",
        "rate limit",
        "billing",
        "dashboard",
        "ui",
        "ux",
        "roadmap",
        "worse",
        "better",
        "love",
        "hate",
        "annoying",
        "frustrating",
        "i think",
        "why did openai",
        "openai should",
    ]

    action_score = contains_any(lowered, actionable_terms)
    problem_score = contains_any(lowered, problem_terms)
    request_score = contains_any(lowered, request_terms)

    has_code_shape = bool(
        re.search(r"\b(?:def|class|const|let|import|from|async|await)\b", lowered)
    )
    has_identifier_context = bool(
        re.search(r"\b(?:app|request|vector store|project|organization|thread|file) id\b", lowered)
        or re.search(r"\b[a-z]+_[a-f0-9]{10,}\b", lowered)
        or re.search(r"\b\d{3,5}×\d{3,5}\b", lowered)
        or "screenshot" in lowered
    )
    has_exact_error = "error" in lowered and (
        "api" in lowered or "request" in lowered or "model" in lowered or "response" in lowered
    )
    has_step_detail = any(marker in lowered for marker in ["1.", "2.", "- ", "first,", "then "])
    is_conceptual_question = any(
        phrase in lowered
        for phrase in [
            "i am trying to understand",
            "i'm trying to understand",
            "want to understand",
            "can someone explain",
            "what are your thoughts",
        ]
    )

    if has_code_shape or has_identifier_context or has_exact_error or action_score >= 4:
        return "actionable", "concrete technical details/code/error context"
    if action_score >= 3 and request_score < 3:
        return "actionable", "specific implementation or diagnostic details"
    if is_conceptual_question and action_score < 3:
        return "underspecified", "conceptual question without enough implementation context"
    if request_score >= 2 and action_score < 3:
        return "opinion_or_request", "mainly product feedback or feature/pricing request"
    if problem_score >= 1 and (len(lowered) < 480 or action_score < 2):
        return "underspecified", "help/problem report without enough context"
    if request_score >= 1 and action_score < 2:
        return "opinion_or_request", "general preference, reaction, or request"
    if action_score >= 2 or has_step_detail:
        return "actionable", "some concrete technical context"
    if "?" in text and len(lowered) < 520:
        return "underspecified", "short question without enough supporting detail"
    return "opinion_or_request", "general discussion or product take"


def collect_topic_ids() -> list[tuple[int, str]]:
    topic_ids: list[tuple[int, str]] = []
    seen: set[int] = set()
    for category_id, category_name in CATEGORIES.items():
        for page in range(4):
            url = f"{BASE_URL}/c/{category_id}.json?page={page}"
            data = fetch_json(url)
            for topic in data.get("topic_list", {}).get("topics", []):
                topic_id = int(topic["id"])
                title = topic.get("title", "").lower()
                is_boilerplate = title.startswith("about the ") or title.startswith("welcome to ")
                if topic_id in seen or topic.get("pinned_globally") or is_boilerplate:
                    continue
                seen.add(topic_id)
                topic_ids.append((topic_id, category_name))
            time.sleep(0.12)
    return topic_ids


def collect_examples() -> list[Example]:
    examples: list[Example] = []
    topic_ids = collect_topic_ids()
    random.Random(RANDOM_SEED).shuffle(topic_ids)
    for topic_id, category in topic_ids:
        topic = fetch_json(f"{BASE_URL}/t/{topic_id}.json")
        title = topic.get("title", "")
        per_topic = 0
        labels_seen_in_topic: set[str] = set()

        for post in topic.get("post_stream", {}).get("posts", []):
            if post.get("username") == "system":
                continue
            raw_text = strip_html(post.get("cooked", ""))
            text = scrub_text(raw_text)
            if len(text) < 80:
                continue
            text = text[:900]
            label, note = classify(text)

            # Keep a little diversity from long threads.
            if per_topic >= 3:
                continue
            if label in labels_seen_in_topic and per_topic >= 2:
                continue

            labels_seen_in_topic.add(label)
            per_topic += 1
            post_number = int(post.get("post_number", 1))
            examples.append(
                Example(
                    source_url=f"{BASE_URL}/t/{topic.get('slug', topic_id)}/{topic_id}/{post_number}",
                    topic_id=topic_id,
                    post_id=int(post["id"]),
                    post_number=post_number,
                    category=category,
                    topic_title=title,
                    text=text,
                    label=label,
                    annotation_note=note,
                )
            )

        if enough_examples(examples):
            break
        time.sleep(0.12)

    return examples


def enough_examples(examples: list[Example]) -> bool:
    counts = Counter(example.label for example in examples)
    return all(counts[label] >= target for label, target in TARGET_COUNTS.items())


def select_balanced_examples(examples: list[Example]) -> list[Example]:
    rng = random.Random(RANDOM_SEED)
    by_label: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)

    selected: list[Example] = []
    for label, target in TARGET_COUNTS.items():
        candidates = by_label[label]
        if len(candidates) < target:
            raise RuntimeError(f"Only found {len(candidates)} examples for {label}; need {target}")
        rng.shuffle(candidates)
        selected.extend(candidates[:target])

    rng.shuffle(selected)
    return selected


def assign_splits(examples: list[Example]) -> list[dict[str, str]]:
    rng = random.Random(RANDOM_SEED)
    rows: list[dict[str, str]] = []
    counters = Counter()
    by_label: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)

    for label, label_examples in by_label.items():
        rng.shuffle(label_examples)
        n = len(label_examples)
        train_end = int(n * 0.70)
        val_end = train_end + int(n * 0.15)
        for idx, example in enumerate(label_examples):
            if idx < train_end:
                split = "train"
            elif idx < val_end:
                split = "validation"
            else:
                split = "test"
            counters[split] += 1
            rows.append(
                {
                    "id": f"odc_{len(rows) + 1:04d}",
                    "text": example.text,
                    "label": example.label,
                    "split": split,
                    "source_url": example.source_url,
                    "topic_id": str(example.topic_id),
                    "post_id": str(example.post_id),
                    "post_number": str(example.post_number),
                    "category": example.category,
                    "topic_title": example.topic_title,
                    "annotation_note": example.annotation_note,
                }
            )

    rng.shuffle(rows)
    for idx, row in enumerate(rows, 1):
        row["id"] = f"odc_{idx:04d}"
    return rows


def write_outputs(rows: list[dict[str, str]]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "text",
        "label",
        "split",
        "source_url",
        "topic_id",
        "post_id",
        "post_number",
        "category",
        "topic_title",
        "annotation_note",
    ]
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": BASE_URL,
        "total_examples": len(rows),
        "label_distribution": Counter(row["label"] for row in rows),
        "split_distribution": Counter(row["split"] for row in rows),
        "category_distribution": Counter(row["category"] for row in rows),
        "random_seed": RANDOM_SEED,
        "output": str(OUT_PATH),
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


def main() -> None:
    examples = collect_examples()
    selected = select_balanced_examples(examples)
    rows = assign_splits(selected)
    write_outputs(rows)


if __name__ == "__main__":
    main()
