# TakeMeter

CodePath AI201 Project 3: a fine-tuned text classifier for evaluating discourse quality in an online community.

## Project Status

Setup complete. Community choice, label taxonomy, dataset, model results, and demo are still in progress.

## Community Choice

TODO: Name the online community you are studying.

TODO: Explain why this community has meaningful differences in post or comment quality that are worth classifying.

## Label Taxonomy

TODO: Define 2-4 mutually exclusive labels.

| Label | Definition | Example 1 | Example 2 |
| --- | --- | --- | --- |
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |

## Data Collection

TODO: Describe where the examples came from.

TODO: Describe how posts or comments were selected and cleaned.

TODO: Add the final labeled dataset to `data/` as CSV or JSON.

## Labeling Process

TODO: Explain how each example was labeled.

TODO: Document edge cases and hard calls.

### Label Distribution

| Label | Count | Percent |
| --- | ---: | ---: |
| TODO | 0 | 0% |

### Difficult Examples

| Text | Final Label | Why It Was Difficult | Decision Rule |
| --- | --- | --- | --- |
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |

## Fine-Tuning Approach

Base model: `distilbert-base-uncased`

TODO: Describe the training setup, dataset split, and at least one hyperparameter decision.

## Zero-Shot Baseline

Baseline model: Groq `llama-3.3-70b-versatile`

TODO: Include the exact prompt used for zero-shot classification and explain how the baseline results were collected.

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
| Project setup | Scaffold the repository and documentation sections from the CodePath checklist. | Initial README, planning file, folders, and git setup. | TODO: Update after reviewing. |
| TODO | TODO | TODO | TODO |

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

TODO: Link your copied Colab notebook and describe the steps needed to reproduce training and evaluation.
