# TakeMeter

CodePath AI201 Project 3: a fine-tuned text classifier for evaluating discourse quality in an online community.

## Project Status

Milestones 1-5 complete. Community choice, label taxonomy, dataset collection, Colab dataset validation, stratified split, tokenization, DistilBERT fine-tuning, Groq zero-shot baseline evaluation, final comparison artifacts, and the fine-tuned confusion matrix are done. Demo polish is the main remaining project task.

## Community Choice

This project studies the [OpenAI Developer Community](https://community.openai.com/), a public forum where developers discuss OpenAI APIs, models, SDKs, tooling, errors, usage limits, pricing, product feedback, and implementation decisions.

This community is a good fit for TakeMeter because discourse quality has practical consequences: some posts give concrete debugging context or reusable solutions, while others are too vague to act on or are mainly product opinions and feature requests. The classifier will measure the role and usefulness of a post inside a developer conversation.

## Label Taxonomy

The classifier uses three mutually exclusive labels. The examples below are illustrative examples of the label boundaries, not final dataset rows.

| Label | Definition | Example 1 | Example 2 |
| --- | --- | --- | --- |
| `actionable` | Gives enough concrete information to help someone debug, reproduce, compare, or implement something. This can include code, error messages, configuration details, specific observations, or a workaround. | "The request fails only when `stream=true`; here is the exact Python snippet and 400 response." | "Switching from JSON mode to structured outputs fixed the schema mismatch because..." |
| `underspecified` | Asks for help, reports a problem, or makes a technical claim but lacks enough context to evaluate or respond usefully. | "The API is not working, what do I do?" | "My fine-tune failed again and I have no idea why." |
| `opinion_or_request` | Primarily expresses a preference, complaint, praise, product take, pricing reaction, model comparison, or feature request rather than a concrete technical question or solution. | "The new dashboard is harder to use than the old one." | "OpenAI should add better project-level usage controls." |

## Data Collection

Examples come from public [OpenAI Developer Community](https://community.openai.com/) forum posts and replies. The labeled dataset is in [`data/openai_developer_community_labeled.csv`](data/openai_developer_community_labeled.csv).

The dataset includes posts about API usage, model behavior, SDKs, tooling, developer workflows, errors, usage limits, pricing, and product feedback. I excluded private account information, copied API keys or secrets, screenshots/images without enough text to label, official documentation pages, category boilerplate, duplicate moderation boilerplate, and consumer-only ChatGPT questions that are not about developer tools or API usage.

Collection was done with [`scripts/collect_openai_forum_dataset.py`](scripts/collect_openai_forum_dataset.py), which uses public Discourse JSON pages, strips HTML, redacts common email/API-key/ID patterns, caps repeated examples from long topics, and creates a balanced 200-row dataset.

## Labeling Process

Each example was labeled by identifying its main role in the developer conversation: actionable technical contribution, underspecified help request/problem report, or opinion/product request.

I used a rubric-assisted labeling script, then spot-checked samples and refined the rules. The main refinements were: category boilerplate was filtered out; screenshots, IDs, code-like text, and exact errors counted as actionable context; conceptual "trying to understand" questions without implementation detail counted as `underspecified`; broad workflow preferences and feature requests counted as `opinion_or_request`.

For mixed posts, I prefer `actionable` if the concrete details are central to the post. Short replies are labeled by purpose: a short reply with a model name or exact error can still be `actionable`, while "same here" is `underspecified` and broad reactions are `opinion_or_request`.

### Label Distribution

| Label | Count | Percent |
| --- | ---: | ---: |
| `actionable` | 80 | 40% |
| `underspecified` | 60 | 30% |
| `opinion_or_request` | 60 | 30% |

### Dataset Split

| Split | Count | Percent |
| --- | ---: | ---: |
| Train | 140 | 70% |
| Validation | 30 | 15% |
| Test | 30 | 15% |

### Difficult Examples

| Text | Final Label | Why It Was Difficult | Decision Rule |
| --- | --- | --- | --- |
| `odc_0019`: asks about `gpt-oss-20b` returning reasoning instead of one-line JSON | `actionable` | It is phrased as a help request, but it gives a model, expected output, observed behavior, and failure pattern. | Concrete technical details beat the fact that it asks for help. |
| `odc_0069`: asks whether Codex Ask helps with game-logic bugs | `underspecified` | It mentions a real workflow, but there is no code, exact bug, error, or reproduction path. | A broad applicability question without technical context is underspecified. |
| `odc_0151`: describes using one model for planning and Codex for execution | `opinion_or_request` | It references an actual workflow, but the main point is personal preference and tool comparison. | Workflow opinions without reproducible details stay `opinion_or_request`. |

## Fine-Tuning Approach

Base model: `distilbert-base-uncased`

Copied Colab notebook: https://colab.research.google.com/drive/15G3bG4fVFzDTiwTDjYdOUVYpOihvv3i5

Milestone 3 prepared the training data in Colab:

- Label map: `actionable -> 0`, `underspecified -> 1`, `opinion_or_request -> 2`
- Dataset source: raw GitHub CSV at `data/openai_developer_community_labeled.csv`
- Validation: 200 examples loaded; all labels matched `LABEL_MAP`
- Split: 140 train, 30 validation, 30 test with stratification by `label_id`
- Tokenizer: `distilbert-base-uncased`, `max_length=256`

The Milestone 3 validation artifact is [`results/milestone3_data_preparation.json`](results/milestone3_data_preparation.json).

Milestone 4 fine-tuned `distilbert-base-uncased` in Colab on a T4 GPU:

- Epochs: 3
- Train batch size: 16
- Evaluation batch size: 32
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Warmup steps: 50
- Best checkpoint selected by validation accuracy

I kept the starter notebook's conservative DistilBERT settings because the dataset is small: 140 training examples, 30 validation examples, and 30 test examples. A `2e-5` learning rate and 3 epochs are reasonable first-pass hyperparameters for avoiding extreme overfitting while still letting the classifier move away from the base model.

The Milestone 4 fine-tuning artifact is [`results/milestone4_finetune_results.json`](results/milestone4_finetune_results.json), and the fine-tuned confusion matrix is [`results/confusion_matrix.png`](results/confusion_matrix.png).

## Zero-Shot Baseline

Baseline model: Groq `llama-3.3-70b-versatile`

Final baseline prompt:

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

I ran this prompt in the copied Colab notebook against the same 30-example test split used for the fine-tuned model. The Groq API key was stored as the Colab secret `GROQ_API_KEY` with notebook access enabled; no secret is stored in this repository.

All 30 baseline responses were parseable as one of the three valid labels. The baseline results are saved in [`results/milestone5_baseline_results.json`](results/milestone5_baseline_results.json).

## Evaluation Report

Milestone 5 compares the Groq zero-shot baseline with the fine-tuned DistilBERT classifier on the same 30-example test split. Both models reached 0.400 accuracy, but they made different kinds of mistakes.

### Metrics

| Model | Accuracy | Notes |
| --- | ---: | --- |
| Zero-shot Groq baseline | 0.400 | 12 correct / 30 test examples; all 30 responses were parseable. |
| Fine-tuned DistilBERT | 0.400 | 12 correct / 30 test examples; never predicted `underspecified`. |

Accuracy tied, so the fine-tuning run did not improve the headline score over the baseline. The baseline was more balanced across labels and recovered some `underspecified` examples, while the fine-tuned model learned a stronger `actionable` signal but collapsed away from the `underspecified` class.

### Per-Class Metrics

| Model | Label | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| Zero-shot Groq baseline | `actionable` | 0.40 | 0.33 | 0.36 |
| Zero-shot Groq baseline | `underspecified` | 0.38 | 0.56 | 0.45 |
| Zero-shot Groq baseline | `opinion_or_request` | 0.43 | 0.33 | 0.38 |
| Fine-tuned DistilBERT | `actionable` | 0.43 | 0.75 | 0.55 |
| Fine-tuned DistilBERT | `underspecified` | 0.00 | 0.00 | 0.00 |
| Fine-tuned DistilBERT | `opinion_or_request` | 0.33 | 0.33 | 0.33 |

The baseline macro F1 was 0.40 and weighted F1 was 0.39. The fine-tuned model's macro F1 was 0.29 and weighted F1 was 0.32. Even though accuracy tied, the baseline had better class balance on this small test set.

### Confusion Matrix

Rows are true labels. Columns are predicted labels.

| True \ Predicted | `actionable` | `underspecified` | `opinion_or_request` |
| --- | ---: | ---: | ---: |
| `actionable` | 9 | 0 | 3 |
| `underspecified` | 6 | 0 | 3 |
| `opinion_or_request` | 6 | 0 | 3 |

![Fine-tuned confusion matrix](results/confusion_matrix.png)

### Wrong Predictions

| Text | True Label | Predicted Label | Why It Failed |
| --- | --- | --- | --- |
| "Livestream ... Sam Altman ... Really excited." | `actionable` | `opinion_or_request` | The short announcement-like text reads like community reaction even though the event details make it actionable. |
| "Seems the prompt caching is broken..." | `underspecified` | `opinion_or_request` | It complains about a technical issue but lacks enough reproduction details; the model treated the complaint tone as the main signal. |
| "I'm new to using AI for my small business..." | `opinion_or_request` | `actionable` | The post asks for practical direction, so the model over-weighted the help-seeking wording instead of the broad request style. |

### Sample Classifications

The Milestone 5 notebook exported aggregate metrics and model comparison artifacts, but it did not persist per-example confidence scores. I am not inventing confidence values here; the wrong-prediction table above is the human-readable sample analysis for this milestone. A final polish pass should add a small exported table of examples with model confidence if the demo rubric requires it.

## Reflection

The label taxonomy was meant to separate conversational function: concrete technical usefulness, vague help-seeking, and product opinion or feature request. The zero-shot baseline followed those definitions fairly evenly, especially for `underspecified`, because the prompt explicitly described missing context as the key signal.

The fine-tuned DistilBERT model learned some useful surface cues for `actionable`, such as model names, technical terms, and implementation-like language. Its main failure was the `underspecified` boundary: on this run, it never predicted that class. That suggests the small training set did not give the model enough stable examples of vague technical help requests, or the wording overlap between `underspecified` and the other labels was too high for a first-pass DistilBERT run.

## Spec Reflection

The project spec helped by forcing a clean baseline-versus-fine-tuned comparison on the same test split. Without that constraint, the 0.400 fine-tuned accuracy might look like the only result; with the baseline, it is clearer that the fine-tuned model did not beat a carefully prompted LLM and needs more data or training iteration.

One implementation detail I handled carefully was secret management. Instead of pasting the Groq API key into the notebook or repo, I used Colab Secrets with notebook access enabled. The tradeoff is that the repo documents the baseline results and artifacts, but it does not contain any runnable secret.

## AI Usage

| Task | What I Asked AI To Do | What It Produced | What I Changed |
| --- | --- | --- | --- |
| Project setup | Scaffold the repository and documentation sections from the CodePath checklist. | Initial README, planning file, folders, and git setup. | Accepted the scaffold and kept TODOs for data/model results. |
| Milestone 1 planning | Suggest OpenAI-related communities and label options, then fill Milestone 1. | Selected OpenAI Developer Community and drafted the 3-label taxonomy. | Accepted the community and labels; will revise after reading the first 30-40 examples if needed. |
| Milestone 2 data | Collect public forum examples and create a labeled dataset using the Milestone 1 taxonomy. | A reproducible collector script, 200-row CSV, balanced label distribution, and split metadata. | Refined the labeling rules after spot checks; filtered boilerplate and redacted common sensitive patterns. |
| Milestone 3 preparation | Configure the Colab notebook for the project dataset and run the validation, split, and tokenization cells. | Label map, GitHub CSV path, validated data counts, stratified split, tokenizer output, and a JSON proof artifact. | Verified Colab outputs and kept fine-tuning/baseline cells for later milestones. |
| Milestone 4 fine-tuning | Reconnect Colab, train DistilBERT, evaluate on the test split, and document the result. | A completed fine-tuning run, test metrics, confusion matrix, and repo artifacts. | Kept the first-pass run, documented the `underspecified` failure mode, and left Groq baseline comparison for the next milestone. |
| Milestone 5 baseline | Run the Groq zero-shot baseline, compare it with the fine-tuned model, and update the report artifacts. | Baseline metrics, side-by-side comparison, and final evaluation JSON files. | Documented the tie in accuracy and highlighted that the baseline had better class balance. |

## Repository Structure

```text
.
├── README.md
├── planning.md
├── data/
│   └── openai_developer_community_labeled.csv
├── notebooks/
├── results/
│   ├── confusion_matrix.png
│   ├── evaluation_results.json
│   ├── milestone3_data_preparation.json
│   ├── milestone4_finetune_results.json
│   └── milestone5_baseline_results.json
├── scripts/
│   └── collect_openai_forum_dataset.py
└── src/
```

## How to Run

Use the copied Colab notebook:
https://colab.research.google.com/drive/15G3bG4fVFzDTiwTDjYdOUVYpOihvv3i5

For Milestones 3-5 reproduction:

1. Open the copied Colab notebook and use a T4 GPU runtime.
2. Confirm the Colab secret `GROQ_API_KEY` exists with notebook access enabled for later baseline cells.
3. Run the setup/import cells.
4. Use the `LABEL_MAP` shown above.
5. Set `CSV_PATH` to the raw GitHub CSV URL.
6. Run Sections 1-2 through tokenization.
7. Run Section 3 to load and fine-tune `distilbert-base-uncased`.
8. Run Section 4 to evaluate on the test set and save `confusion_matrix.png`.
9. Run the Groq baseline section with `llama-3.3-70b-versatile`.
10. Run the comparison/export section to generate `evaluation_results.json` and the final side-by-side metrics.
