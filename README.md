# TakeMeter

CodePath AI201 Project 3: a fine-tuned text classifier for evaluating discourse quality in an online community.

## Project Status

Milestone 1 complete. Community choice and label taxonomy are defined. Dataset collection, model results, evaluation, and demo are still in progress.

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

Examples will come from public [OpenAI Developer Community](https://community.openai.com/) forum posts and replies.

The dataset will include posts about API usage, model behavior, SDKs, tooling, developer workflows, errors, usage limits, pricing, and product feedback. I will exclude private account information, copied API keys or secrets, screenshots/images without enough text to label, official documentation pages, staff announcements without community discussion, duplicate moderation boilerplate, and consumer-only ChatGPT questions that are not about developer tools or API usage.

TODO: Add the final labeled dataset to `data/` as CSV or JSON.

## Labeling Process

Each example will be labeled by reading the text and identifying its main role in the developer conversation: actionable technical contribution, underspecified help request/problem report, or opinion/product request.

For mixed posts, I will prefer `actionable` if the concrete details are central to the post. Short replies will be labeled by purpose: a short reply with a model name or exact error can still be `actionable`, while "same here" is `underspecified` and broad reactions are `opinion_or_request`.

### Label Distribution

| Label | Count | Percent |
| --- | ---: | ---: |
| `actionable` | TODO | TODO |
| `underspecified` | TODO | TODO |
| `opinion_or_request` | TODO | TODO |

### Difficult Examples

| Text | Final Label | Why It Was Difficult | Decision Rule |
| --- | --- | --- | --- |
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |

## Fine-Tuning Approach

Base model: `distilbert-base-uncased`

Copied Colab notebook: https://colab.research.google.com/drive/15G3bG4fVFzDTiwTDjYdOUVYpOihvv3i5

TODO: Describe the training setup, dataset split, and at least one hyperparameter decision.

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

## Repository Structure

```text
.
├── README.md
├── planning.md
├── data/
├── notebooks/
├── results/
└── src/
```

## How to Run

Use the copied Colab notebook:
https://colab.research.google.com/drive/15G3bG4fVFzDTiwTDjYdOUVYpOihvv3i5

TODO: Describe the steps needed to reproduce training and evaluation after the dataset and labels are finalized.
