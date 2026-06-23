#!/usr/bin/env python3
"""Small local TakeMeter interface.

The app trains a lightweight local classifier on the committed training split
at startup and serves a browser UI that accepts a new post and returns a label
plus confidence. It is intentionally small so it can run without committing a
large DistilBERT checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "openai_developer_community_labeled.csv"
LABELS = ["actionable", "underspecified", "opinion_or_request"]


class TakeMeterModel:
    def __init__(self) -> None:
        self.mode = "keyword"
        self.vectorizer = None
        self.model = None
        self._train()

    def _load_train_rows(self) -> tuple[list[str], list[str]]:
        with DATA_PATH.open(newline="", encoding="utf-8") as file:
            rows = [row for row in csv.DictReader(file) if row["split"] == "train"]
        return [row["text"] for row in rows], [row["label"] for row in rows]

    def _train(self) -> None:
        texts, labels = self._load_train_rows()
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
            features = self.vectorizer.fit_transform(texts)
            self.model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            self.model.fit(features, labels)
            self.mode = "tfidf_logistic_regression"
        except Exception:
            self.mode = "keyword_fallback"

    def predict(self, text: str) -> dict[str, object]:
        text = " ".join(text.split())
        if not text:
            return {
                "label": "underspecified",
                "confidence": 0.0,
                "probabilities": {label: 0.0 for label in LABELS},
                "mode": self.mode,
            }
        if self.mode == "tfidf_logistic_regression" and self.vectorizer is not None and self.model is not None:
            features = self.vectorizer.transform([text])
            probabilities = self.model.predict_proba(features)[0]
            classes = list(self.model.classes_)
            scores = {label: 0.0 for label in LABELS}
            for label, probability in zip(classes, probabilities):
                scores[label] = float(probability)
        else:
            scores = self._keyword_scores(text)

        label = max(scores, key=scores.get)
        return {
            "label": label,
            "confidence": round(scores[label], 3),
            "probabilities": {key: round(value, 3) for key, value in scores.items()},
            "mode": self.mode,
        }

    def _keyword_scores(self, text: str) -> dict[str, float]:
        lowered = text.lower()
        weights = {
            "actionable": ["error", "code", "sdk", "api", "model", "endpoint", "json", "mcp", "steps", "fix", "bug"],
            "underspecified": ["help", "not working", "same issue", "problem", "failed", "anyone", "why"],
            "opinion_or_request": ["should", "wish", "prefer", "feedback", "pricing", "better", "request", "future"],
        }
        raw_scores = []
        for label in LABELS:
            score = 1.0 + sum(1.0 for term in weights[label] if term in lowered)
            raw_scores.append(score)
        exps = [math.exp(score) for score in raw_scores]
        total = sum(exps)
        return {label: value / total for label, value in zip(LABELS, exps)}


MODEL = TakeMeterModel()


def render_page(result: dict[str, object] | None = None, text: str = "") -> bytes:
    escaped_text = html.escape(text)
    result_markup = ""
    if result:
        probabilities = result["probabilities"]
        bars = "\n".join(
            f"""
            <div class="meter-row">
              <span>{html.escape(label)}</span>
              <div class="meter"><div style="width: {int(probabilities[label] * 100)}%"></div></div>
              <strong>{probabilities[label]:.3f}</strong>
            </div>
            """
            for label in LABELS
        )
        result_markup = f"""
        <section class="result" aria-live="polite">
          <div>
            <p class="eyebrow">Prediction</p>
            <h2>{html.escape(result["label"])}</h2>
            <p class="confidence">Confidence {result["confidence"]:.3f}</p>
          </div>
          <div class="meters">{bars}</div>
          <p class="mode">Model mode: {html.escape(result["mode"])}</p>
        </section>
        """

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TakeMeter Interface</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #182026;
      --muted: #5d6872;
      --line: #d9e0e6;
      --panel: #ffffff;
      --page: #f5f7f8;
      --accent: #186f65;
      --accent-2: #3454d1;
      --accent-3: #b65a21;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--page);
    }}
    main {{
      width: min(980px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 32px 0;
    }}
    header {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 20px;
    }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: 28px; font-weight: 750; }}
    .status {{ color: var(--muted); font-size: 14px; }}
    form, .result {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 12px 24px rgba(24, 32, 38, 0.06);
    }}
    form {{ padding: 18px; }}
    label {{ display: block; font-weight: 700; margin-bottom: 10px; }}
    textarea {{
      width: 100%;
      min-height: 220px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
      font: inherit;
      line-height: 1.45;
      background: #fbfcfd;
      color: var(--ink);
    }}
    .actions {{
      display: flex;
      justify-content: flex-end;
      margin-top: 14px;
    }}
    button {{
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 750;
      padding: 11px 16px;
      cursor: pointer;
    }}
    button:hover {{ background: #12564f; }}
    .result {{
      margin-top: 18px;
      padding: 18px;
      display: grid;
      grid-template-columns: minmax(180px, 260px) 1fr;
      gap: 20px;
      align-items: start;
    }}
    .eyebrow {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
      font-weight: 750;
      margin-bottom: 8px;
    }}
    .result h2 {{ font-size: 24px; }}
    .confidence {{ color: var(--accent-3); margin-top: 8px; font-weight: 750; }}
    .meters {{ display: grid; gap: 12px; }}
    .meter-row {{
      display: grid;
      grid-template-columns: minmax(150px, 210px) 1fr 56px;
      align-items: center;
      gap: 10px;
      font-size: 14px;
    }}
    .meter {{
      height: 12px;
      background: #e8edf1;
      border-radius: 6px;
      overflow: hidden;
    }}
    .meter div {{
      height: 100%;
      background: var(--accent-2);
    }}
    .mode {{
      grid-column: 1 / -1;
      color: var(--muted);
      font-size: 13px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }}
    @media (max-width: 720px) {{
      header, .result {{ display: block; }}
      .status {{ margin-top: 8px; }}
      .meters {{ margin-top: 18px; }}
      .meter-row {{ grid-template-columns: 1fr; gap: 6px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>TakeMeter Interface</h1>
      </div>
      <p class="status">OpenAI Developer Community classifier</p>
    </header>
    <form method="post" action="/classify">
      <label for="post">Post text</label>
      <textarea id="post" name="post" placeholder="Paste a community post here">{escaped_text}</textarea>
      <div class="actions">
        <button type="submit">Classify &gt;</button>
      </div>
    </form>
    {result_markup}
  </main>
</body>
</html>"""
    return page.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(404)
            return
        self._send_html(render_page())

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        if self.path == "/api/classify":
            payload = json.loads(body or "{}")
            result = MODEL.predict(str(payload.get("post", "")))
            self._send_json(result)
            return
        if self.path == "/classify":
            text = parse_qs(body).get("post", [""])[0]
            result = MODEL.predict(text)
            self._send_html(render_page(result, text))
            return
        self.send_error(404)

    def log_message(self, format: str, *args) -> None:
        return

    def _send_html(self, payload: bytes) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"TakeMeter interface running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
