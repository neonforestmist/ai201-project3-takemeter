# TakeMeter

CodePath AI201 Project 3: a fine-tuned text classifier for evaluating discourse quality in an online community.

## Project Status

Milestones 1-3 complete. Community choice, label taxonomy, dataset collection, Colab dataset validation, stratified split, and tokenization are done. Fine-tuning, baseline evaluation, final report, and demo are still in progress.

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

TODO: After training, describe the final training setup and at least one hyperparameter decision.

## Zero-Shot Baseline

Baseline model: Groq `llama-3.3-70b-versatile`

Draft baseline prompt:

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

TODO: Update the prompt if label definitions change after annotation, then explain how baseline results were collected.

## Evaluation Report

TODO: Report accuracy and per-class metrics for both the zero-shot baseline and fine-tuned model.

### Metrics

| Model | Accuracy | Notes |
| --- | ---: | --- |
| Zero-shot baseline | TODO | TODO |
| Fine-tuned classifier | TODO | TODO |

### Per-Class Metrics

| Model | Label | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| TODO | TODO | TODO | TODO | TODO |

### Confusion Matrix

TODO: Write the fine-tuned model confusion matrix as a markdown table here.

### Wrong Predictions

| Text | True Label | Predicted Label | Why It Failed |
| --- | --- | --- | --- |
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |

### Sample Classifications

| Text | Predicted Label | Confidence | Notes |
| --- | --- | ---: | --- |
| TODO | TODO | TODO | TODO |

## Reflection

TODO: Explain what the model seemed to learn compared with what the label taxonomy was intended to capture.

## Spec Reflection

TODO: Describe one way the project spec helped guide the implementation.

TODO: Describe one way the implementation diverged from the spec and why.

## AI Usage

| Task | What I Asked AI To Do | What It Produced | What I Changed |
| --- | --- | --- | --- |
| Project setup | Scaffold the repository and documentation sections from the CodePath checklist. | Initial README, planning file, folders, and git setup. | Accepted the scaffold and kept TODOs for data/model results. |
| Milestone 1 planning | Suggest OpenAI-related communities and label options, then fill Milestone 1. | Selected OpenAI Developer Community and drafted the 3-label taxonomy. | Accepted the community and labels; will revise after reading the first 30-40 examples if needed. |
| Milestone 2 data | Collect public forum examples and create a labeled dataset using the Milestone 1 taxonomy. | A reproducible collector script, 200-row CSV, balanced label distribution, and split metadata. | Refined the labeling rules after spot checks; filtered boilerplate and redacted common sensitive patterns. |
| Milestone 3 preparation | Configure the Colab notebook for the project dataset and run the validation, split, and tokenization cells. | Label map, GitHub CSV path, validated data counts, stratified split, tokenizer output, and a JSON proof artifact. | Verified Colab outputs and kept fine-tuning/baseline cells for later milestones. |

## Repository Structure

```text
.
├── README.md
├── planning.md
├── data/
│   └── openai_developer_community_labeled.csv
├── notebooks/
├── results/
│   └── milestone3_data_preparation.json
├── scripts/
│   └── collect_openai_forum_dataset.py
└── src/
```

## How to Run

Use the copied Colab notebook:
https://colab.research.google.com/drive/15G3bG4fVFzDTiwTDjYdOUVYpOihvv3i5

For Milestone 3 reproduction:

1. Open the copied Colab notebook and use a T4 GPU runtime.
2. Confirm the Colab secret `GROQ_API_KEY` exists with notebook access enabled for later baseline cells.
3. Run the setup/import cells.
4. Use the `LABEL_MAP` shown above.
5. Set `CSV_PATH` to the raw GitHub CSV URL.
6. Run Sections 1-2 through tokenization.

TODO: Add the training and evaluation steps after the fine-tuning and baseline runs are complete.
