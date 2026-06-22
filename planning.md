# TakeMeter Planning

Use this file for design thinking before and during implementation. The README should be the polished final report.

## 1. Community Decision

Community under study: TODO

Why this community: TODO

Kinds of posts/comments included: TODO

Kinds of posts/comments excluded: TODO

## 2. Label Taxonomy Draft

Goal: Define 2-4 labels that are mutually exclusive, useful to the community, and broad enough to cover at least 90% of examples.

| Label | Working Definition | Positive Examples | Boundary / Edge Cases |
| --- | --- | --- | --- |
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |

## 3. Edge Case Rules

TODO: Add rules for sarcasm, short comments, jokes, low-effort posts, mixed-quality posts, and posts that seem to fit more than one label.

## 4. Data Collection Plan

Source: TODO

Collection method: TODO

Target count: at least 200 labeled examples.

Split plan:

| Split | Count | Percent |
| --- | ---: | ---: |
| Train | TODO | TODO |
| Validation | TODO | TODO |
| Test | TODO | TODO |

## 5. Annotation Process

TODO: Describe how examples will be labeled and reviewed for consistency.

TODO: After labeling the first 30-40 examples, note whether the labels needed revision.

## 6. Baseline Plan

Baseline model: Groq `llama-3.3-70b-versatile`

Prompt draft:

```text
TODO
```

## 7. Fine-Tuning Plan

Base model: `distilbert-base-uncased`

Training environment: Google Colab T4 GPU

Initial hyperparameters to consider:

| Hyperparameter | Starting Value | Reason |
| --- | --- | --- |
| Epochs | TODO | TODO |
| Batch size | TODO | TODO |
| Learning rate | TODO | TODO |

## 8. Evaluation Plan

Required metrics:

- Overall accuracy for both models
- Per-class precision, recall, or F1
- Confusion matrix for the fine-tuned model
- At least 3 wrong predictions with analysis
- 3-5 sample classifications with confidence scores

## 9. AI Tool Plan and Usage Log

| Date | Tool | Prompt / Direction | Output | What I Accepted or Changed |
| --- | --- | --- | --- | --- |
| 2026-06-22 | Codex | Set up the repository files from the CodePath Project 3 checklist. | Initial project scaffold. | TODO: Review and edit. |

## 10. Open Questions

- TODO: Which community is best for clean but meaningful labels?
- TODO: Which 2-4 labels will cover most examples without becoming vague?
- TODO: How will difficult examples be documented during annotation?
