# TakeMeter Planning

Use this file for design thinking before and during implementation. The README should be the polished final report.

## 1. Community Decision

Community under study: OpenAI Developer Community forum posts and replies: https://community.openai.com/

Why this community: The OpenAI Developer Community is a public forum where developers discuss OpenAI APIs, models, tools, errors, product changes, and implementation decisions. The quality of a post matters because some posts give concrete, reusable debugging or design information, while others are too vague to act on or are mostly product opinions and requests. That makes it a good fit for TakeMeter: the labels measure how useful a piece of discourse is inside a developer support/community context.

Kinds of posts/comments included: public forum posts and replies about the OpenAI API platform, model behavior, SDKs, tooling, developer workflows, errors, usage limits, pricing, product feedback, and feature requests.

Kinds of posts/comments excluded: private account information, copied API keys or secrets, screenshots/images without enough text to label, official documentation pages, staff announcements with no community response, duplicate moderation boilerplate, and consumer-only ChatGPT questions that are not about developer tools or API usage.

## 2. Label Taxonomy Draft

Goal: Define 2-4 labels that are mutually exclusive, useful to the community, and broad enough to cover at least 90% of examples.

| Label | Working Definition | Positive Examples | Boundary / Edge Cases |
| --- | --- | --- | --- |
| `actionable` | Gives enough concrete information to help someone debug, reproduce, compare, or implement something. This can be a question, answer, workaround, explanation, code snippet, error message, configuration detail, benchmark, or step-by-step report. | "The request fails only when `stream=true`; here is the exact Python snippet and 400 response." / "Switching from JSON mode to structured outputs fixed the schema mismatch because..." | If a post has frustration but also includes specific evidence or steps, label it `actionable`. If it only says "it broke" without useful details, label it `underspecified`. |
| `underspecified` | Asks for help, reports a problem, or makes a technical claim but lacks the context needed to evaluate or respond usefully. | "The API is not working, what do I do?" / "My fine-tune failed again and I have no idea why." | Use this when the post is trying to get help but omits key information like model, endpoint, code, error text, expected behavior, or steps tried. |
| `opinion_or_request` | Primarily expresses a preference, complaint, praise, product take, pricing reaction, model comparison, or feature request rather than asking a concrete technical question or giving a concrete solution. | "The new dashboard is harder to use than the old one." / "OpenAI should add better project-level usage controls." | If an opinion includes reproducible technical evidence, label it `actionable`. If it is mostly a desired product change or general take, label it `opinion_or_request`. |

## 3. Edge Case Rules

- Label the function of the post inside the developer conversation, not whether the technical claim is objectively true.
- Use `actionable` when a reader could reasonably take a next step based on the post: reproduce a bug, try a fix, compare a setting, inspect a known error, or understand a tradeoff.
- Use `underspecified` when the post appears to seek help or report a technical issue but does not provide enough context for a useful response.
- Use `opinion_or_request` when the post is mainly about preference, product direction, pricing, model quality, desired features, or general reaction.
- For mixed posts, prefer `actionable` if the concrete details are central to the post. Example: a complaint with code and exact error text is `actionable`.
- For short posts, label by purpose. "Same error on `gpt-4.1-mini` with the Node SDK" can be `actionable`; "same here" is `underspecified`; "this pricing is wild" is `opinion_or_request`.
- Sarcasm, jokes, and rants are `opinion_or_request` unless they include enough concrete technical evidence to act on.
- Answers can be `actionable` even when they are not perfect, as long as they provide a specific explanation, workaround, diagnostic step, or implementation detail.
- Avoid creating an "other" label. If a post does not fit cleanly, choose the label that best describes its main conversational job.

## 4. Data Collection Plan

Source: OpenAI Developer Community public forum pages: https://community.openai.com/

Collection method: Collected 200 text examples from public posts/replies using `scripts/collect_openai_forum_dataset.py`. The script uses public Discourse JSON pages, strips HTML, redacts common email/API-key/ID patterns, filters category boilerplate, limits repeated examples from long topics, and writes a labeled CSV to `data/openai_developer_community_labeled.csv`.

Target count: at least 200 labeled examples.

Split plan:

| Split | Count | Percent |
| --- | ---: | ---: |
| Train | 140 | 70% |
| Validation | 30 | 15% |
| Test | 30 | 15% |

## 5. Annotation Process

Each example was labeled by reading the full text and asking: "What role is this post playing in the developer conversation?" I used a rubric-assisted script for the first pass, then spot-checked samples and refined the rules before regenerating the final CSV.

After reviewing the first batches, the labels still covered the data well, but several edge-case rules needed tightening: category boilerplate was removed; screenshots, IDs, code-like text, and exact errors count as actionable context; conceptual "trying to understand" questions without implementation details are `underspecified`; broad workflow preferences and feature requests are `opinion_or_request`.

The final dataset is balanced: 80 `actionable`, 60 `underspecified`, and 60 `opinion_or_request`. Splits are stratified into 140 train, 30 validation, and 30 test examples.

## 6. Baseline Plan

Baseline model: Groq `llama-3.3-70b-versatile`

Milestone 5 baseline completed on 2026-06-22 in the copied Colab notebook. The Groq API key was stored in Colab Secrets as `GROQ_API_KEY`; the key was not pasted into the notebook or committed to the repo.

Final prompt:

```text
You are classifying posts from the OpenAI Developer Community.
Assign each post to exactly one label based on its main role in the developer conversation.

actionable: The post gives enough concrete information to help someone debug, reproduce, compare, or implement something. It may include code, error messages, steps tried, configuration details, specific observations, or a useful workaround.
underspecified: The post asks for help, reports a problem, or makes a technical claim but lacks enough context to evaluate or respond usefully.
opinion_or_request: The post mainly expresses a preference, complaint, praise, product take, pricing reaction, model comparison, or feature request.

Respond with ONLY one label:
actionable
underspecified
opinion_or_request
```

Baseline test result:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.400 |
| Macro precision | 0.40 |
| Macro recall | 0.41 |
| Macro F1 | 0.40 |
| Weighted F1 | 0.39 |
| Parseable responses | 30 / 30 |

Per-class result:

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `actionable` | 0.40 | 0.33 | 0.36 | 12 |
| `underspecified` | 0.38 | 0.56 | 0.45 | 9 |
| `opinion_or_request` | 0.43 | 0.33 | 0.38 | 9 |

## 7. Fine-Tuning Plan

Base model: `distilbert-base-uncased`

Training environment: Google Colab T4 GPU

Milestone 3 preparation completed on 2026-06-22:

- Added `LABEL_MAP = {"actionable": 0, "underspecified": 1, "opinion_or_request": 2}` to the copied Colab notebook.
- Loaded the dataset from the raw GitHub CSV URL.
- Validated 200 examples and confirmed all labels match the label map.
- Created a stratified 70/15/15 split: 140 train, 30 validation, 30 test.
- Tokenized all splits with `distilbert-base-uncased` at `max_length=256`.
- Saved the proof artifact in `results/milestone3_data_preparation.json`.

Initial hyperparameters to consider:

| Hyperparameter | Starting Value | Reason |
| --- | --- | --- |
| Epochs | 3 | Enough passes for the small 140-example training split without intentionally overtraining. |
| Batch size | 16 train / 32 eval | Fits comfortably on the Colab T4 while keeping the run quick. |
| Learning rate | `2e-5` | Standard conservative starting point for DistilBERT fine-tuning. |
| Weight decay | `0.01` | Light regularization for a small labeled dataset. |
| Warmup steps | 50 | Smooths the start of training for the short run. |

Milestone 4 fine-tuning completed on 2026-06-22:

- Loaded `distilbert-base-uncased` with a three-label sequence classification head.
- Trained in Colab on a T4 GPU for 3 epochs using the starter notebook `Trainer` workflow.
- Evaluated on the locked 30-example test split.
- Saved the fine-tuned confusion matrix as `results/confusion_matrix.png`.
- Saved the test metrics artifact as `results/milestone4_finetune_results.json`.

Milestone 4 test result:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.400 |
| Macro precision | 0.25 |
| Macro recall | 0.36 |
| Macro F1 | 0.29 |
| Weighted F1 | 0.32 |

Confusion matrix, rows=true label and columns=predicted label:

| True \ Predicted | `actionable` | `underspecified` | `opinion_or_request` |
| --- | ---: | ---: | ---: |
| `actionable` | 9 | 0 | 3 |
| `underspecified` | 6 | 0 | 3 |
| `opinion_or_request` | 6 | 0 | 3 |

Main error pattern: the fine-tuned model never predicted `underspecified`, so all nine underspecified test examples were forced into `actionable` or `opinion_or_request`. This suggests the class boundary between vague help requests and concrete developer posts needs either more examples, stronger text normalization, class weighting, or tuned training settings.

## 8. Evaluation Plan

Required metrics:

- Overall accuracy for both models
- Per-class precision, recall, or F1
- Confusion matrix for the fine-tuned model
- At least 3 wrong predictions with analysis
- 3-5 sample classifications with confidence scores

Milestone 5 comparison:

| Model | Accuracy | Macro F1 | Weighted F1 | Main Note |
| --- | ---: | ---: | ---: | --- |
| Zero-shot Groq baseline | 0.400 | 0.40 | 0.39 | More balanced across classes; best `underspecified` recall. |
| Fine-tuned DistilBERT | 0.400 | 0.29 | 0.32 | Better `actionable` recall; never predicted `underspecified`. |

The exported notebook artifacts cover aggregate metrics, the fine-tuned confusion matrix, and the baseline/fine-tuned side-by-side comparison.

Milestone 6 sample-confidence export:

| ID | True Label | Predicted Label | Confidence | Correct? |
| --- | --- | --- | ---: | --- |
| `odc_0021` | `actionable` | `actionable` | 0.358 | yes |
| `odc_0053` | `underspecified` | `actionable` | 0.364 | no |
| `odc_0084` | `actionable` | `actionable` | 0.356 | yes |
| `odc_0111` | `opinion_or_request` | `actionable` | 0.370 | no |
| `odc_0129` | `actionable` | `actionable` | 0.364 | yes |

These examples were generated with `scripts/export_milestone6_samples.py`, which reruns the committed split with the same DistilBERT hyperparameters because the original Colab checkpoint was not committed.

## 9. AI Tool Plan and Usage Log

| Date | Tool | Prompt / Direction | Output | What I Accepted or Changed |
| --- | --- | --- | --- | --- |
| 2026-06-22 | Codex | Set up the repository files from the CodePath Project 3 checklist. | Initial project scaffold. | Accepted the scaffold and kept TODOs for data/model results. |
| 2026-06-22 | Codex | Suggest OpenAI-related communities and label options, then fill Milestone 1. | Selected OpenAI Developer Community and drafted a 3-label taxonomy. | Accepted the community and labels; will revise after reading the first 30-40 examples if needed. |
| 2026-06-22 | Codex | Build Milestone 2 dataset from public OpenAI Developer Community posts. | Generated collector script, 200-row labeled CSV, summary JSON, and documentation updates. | Spot-checked samples, refined rules, and kept the balanced label distribution. |
| 2026-06-22 | Codex | Complete Milestone 3 in Colab. | Configured the label map and raw CSV path, ran validation/split/tokenization, and wrote a results artifact. | Verified the Colab outputs and stopped before training/baseline cells. |
| 2026-06-22 | Codex | Complete Milestone 4 fine-tuning in Colab. | Reconnected the T4 runtime, trained DistilBERT for 3 epochs, ran test evaluation, generated the confusion matrix, and wrote a fine-tuning results artifact. | Accepted the run, documented the low `underspecified` recall, and left Groq baseline work for the next milestone. |
| 2026-06-22 | Codex | Complete Milestone 5 baseline and comparison. | Ran the Groq `llama-3.3-70b-versatile` baseline, compared it with DistilBERT, and updated results artifacts. | Accepted the tied accuracy result and documented that the baseline had better class balance. |
| 2026-06-22 | Codex | Complete Milestone 6 final report and demo prep. | Added a sample-confidence export script, generated five sample classifications, polished the README, and drafted a demo script. | Kept the Colab metrics as the official evaluation and used the local rerun only for demo confidence samples. |

## 10. Open Questions

- The labels covered the dataset well enough for the 200-example first pass, but `underspecified` remains the hardest class for the fine-tuned model.
- Short `underspecified` posts and broad `opinion_or_request` posts still need careful examples because both can lack technical details.
- The next model iteration should test class weighting, more examples, or a longer/tuned run to help DistilBERT learn the `underspecified` boundary.
- The repo-side Milestone 6 work is complete; the remaining external step is recording and uploading the demo video through the Course Portal.
