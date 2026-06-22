# TakeMeter Demo Script

Use this for the required 3-5 minute Milestone 6 video.

## 1. Open With The Project

"This project is TakeMeter, a text classifier for the OpenAI Developer Community. The goal is to classify forum posts by their role in developer discourse: `actionable`, `underspecified`, or `opinion_or_request`."

Show:

- `README.md`
- The label taxonomy table
- The dataset distribution table

## 2. Show The Model And Samples

"I fine-tuned `distilbert-base-uncased` on 140 training examples, used 30 validation examples, and evaluated on a locked 30-example test set. For the demo, I exported five test-set predictions with confidence scores."

Show the README's Sample Classifications table.

Correct prediction to narrate:

- `odc_0021` was predicted as `actionable` with confidence `0.358`.
- Explanation: "This is reasonable because the post names a specific MCP pagination behavior, points to the specification, describes what ChatGPT is doing wrong, and asks for an implementation update. Even though the confidence is low, the label matches the concrete technical evidence in the text."

Incorrect prediction to narrate:

- `odc_0111` was predicted as `actionable` with confidence `0.370`, but the true label was `opinion_or_request`.
- Explanation: "The model was probably pulled toward `actionable` by the structured prompting steps and technical-sounding language. The project label treats the post as `opinion_or_request` because its main role is a broad method proposal and product-style take, not a reproducible bug report or concrete support answer."

## 3. Walk Through Evaluation

"The zero-shot Groq baseline and the fine-tuned DistilBERT model tied on accuracy at `0.400`. The baseline had better balance across the three labels, while DistilBERT had stronger `actionable` recall but failed on `underspecified`."

Show:

- Metrics table
- Per-class metrics table
- Confusion matrix markdown table

Main failure pattern:

"The fine-tuned model never predicted `underspecified` in the original Colab evaluation. That means the hardest boundary is between vague help requests and posts that merely sound technical or opinionated."

## 4. Close With Reflection

"The project taught me that label design matters as much as training. The model learned surface cues like technical terms and structured wording, but it did not reliably learn the intended discourse function. The next improvement would be more `underspecified` examples, clearer boundary examples, and possibly class weighting."

End by showing:

- `results/evaluation_results.json`
- `results/milestone6_sample_classifications.json`
- GitHub repository URL
