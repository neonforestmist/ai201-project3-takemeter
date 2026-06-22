# Data

The Milestone 2 labeled dataset is:

- `openai_developer_community_labeled.csv`

It contains 200 public OpenAI Developer Community examples with labels and train/validation/test splits.

Columns:

| Column | Purpose |
| --- | --- |
| `id` | Stable row id |
| `text` | Scrubbed post or reply text |
| `label` | Final human label |
| `split` | `train`, `validation`, or `test` |
| `source_url` | Public forum URL for traceability |
| `topic_id` | Discourse topic id |
| `post_id` | Discourse post id |
| `post_number` | Post number within the topic |
| `category` | Forum category |
| `topic_title` | Source topic title |
| `annotation_note` | Short note from the labeling rubric |
