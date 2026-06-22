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

Prompt draft:

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
| 2026-06-22 | Codex | Set up the repository files from the CodePath Project 3 checklist. | Initial project scaffold. | Accepted the scaffold and kept TODOs for data/model results. |
| 2026-06-22 | Codex | Suggest OpenAI-related communities and label options, then fill Milestone 1. | Selected OpenAI Developer Community and drafted a 3-label taxonomy. | Accepted the community and labels; will revise after reading the first 30-40 examples if needed. |
| 2026-06-22 | Codex | Build Milestone 2 dataset from public OpenAI Developer Community posts. | Generated collector script, 200-row labeled CSV, summary JSON, and documentation updates. | Spot-checked samples, refined rules, and kept the balanced label distribution. |

## 10. Open Questions

- After reading the first 30-40 examples, do the labels cover at least 90% of posts without feeling forced?
- Are `underspecified` and `opinion_or_request` easy enough to separate in short posts?
- Which source categories on the OpenAI Developer Community produce the best balance across the three labels?
