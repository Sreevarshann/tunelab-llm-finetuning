# Evaluation methodology

- **Same conditions for both models.** Base and tuned evaluations use the same split, the same
  system prompt (the one the adapter was trained with), greedy decoding and `max_tokens=96`. The only
  difference is whether the adapter file is loaded.
- **Strict parsing.** A prediction counts as JSON only if the whole reply is a JSON object, optionally
  inside a code fence. Chatty text around JSON is not repaired.
- **Metrics (JSON tasks).** `json_valid_pct`, `schema_exact_pct` (exactly the expected keys),
  `all_fields_correct_pct` (the headline) and per-field accuracy. **Text tasks:** exact match and
  token F1. Both report teacher-forced loss on the split's reference answers.
- **Baselines are shared** per dataset, split and base-model revision, and computed once.
- **Every number shown is computed by a job in this repo.** Per-example outputs are stored
  (`var/jobs/<eval>/examples.json`) and browsable side by side in the UI.

Limitations: one task, one synthetic dataset and one base model; single seeds; no confidence
intervals. With n = 120 per split, one example is 0.83 percentage points.
