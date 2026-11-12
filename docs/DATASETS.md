# Datasets

**Format:** JSONL, one `{"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}` per
line, ending with a non-empty assistant reply (the training target). Validation reports problems
by line number. Limits: 20,000 rows, 8,000 characters per conversation, 20 MB per upload.

**Splits:** a single file is split 80/10/10 into train / valid / test by a hash of each row's
content (deterministic: re-uploading the same rows gives the same splits). Alternatively, upload
explicit `train` + `valid` (+ `test`). Only `train` is used for gradient updates. `valid` is used
for loss reporting during training and test splits only for evaluation.

**Scoring** is detected automatically: if every assistant reply is a JSON object with the same keys,
per-field accuracy, JSON validity, exact schema and all-fields-correct are computed. Otherwise the
metrics are normalised exact match and token F1. Teacher-forced loss is always reported.

## Built-in synthetic task: support-ticket triage

`backend/src/tunelab/tasks/ticket_triage.py` renders tickets from sampled labels:
`category` (5), `priority` (4), `product` (5 fictional products), `order_id` (or null) and
`refund_requested`. Because the text is generated from the label, ground truth is exact. The data is
**synthetic** and labelled so everywhere. Real support tickets are messier, so results on this task
say nothing about real-world triage accuracy.

Templates are split: 6 of 8 ticket bodies per category, 3 of 4 urgency phrasings and 3 of 4 refund
phrasings are "seen". The `test_heldout` split uses **only the unseen phrasings**, so it measures
whether the adapter learned the mapping or memorised wording. Default sizes (seed 7): 600 train,
80 valid, 120 test, 120 test_heldout, with no duplicate input across splits.
