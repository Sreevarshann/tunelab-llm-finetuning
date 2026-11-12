# Results (2026-11-12)

All numbers come from jobs run on this machine and exported by `backend/scripts/export_results.py`
to [`eval/results.json`](../eval/results.json). Nothing was edited by hand.

**Setup.** Apple M4 (16 GB), macOS 26.6, MLX 0.32.4, mlx-lm 0.32.2, Metal GPU. Base model
`mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `a5339a4` (494M parameters). Synthetic ticket-triage dataset,
seed 7: 600 train / 80 valid / 120 test / 120 test with held-out phrasings. Greedy decoding, the same
system prompt for base and tuned models, single seed per run.

## Headline: all five fields correct

| Model | Trainable params | Steps | Train time | Test | Held-out phrasings |
|---|---|---|---|---|---|
| Base (no adapter) | — | — | — | 8.3% | 5.0% |
| LoRA r8, lr 1e-4, 16 blocks | 2.93M (0.59%) | 300 | 9.7 min | **100.0%** | **63.3%** |
| LoRA r8, lr 1e-4, 16 blocks | 2.93M | 100 | 4.4 min | 100.0% | 50.0% |
| LoRA r16, dropout 0.1 | 5.87M | 300 | 10.1 min | 99.2% | 40.0% |

n = 120 per split (one example = 0.83 points). JSON validity and exact schema were 100% for every
model, so the base model already follows the format. It gets the labels wrong.

## Per-field accuracy

| Field | Base test | Base held-out | r8·300 test | r8·300 held-out | r8·100 held-out | r16 held-out |
|---|---|---|---|---|---|---|
| category | 29.2 | 30.8 | 100.0 | 94.2 | 100.0 | 100.0 |
| priority | 35.0 | 46.7 | 100.0 | **69.2** | **50.0** | **40.0** |
| product | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| order_id | 95.8 | 98.3 | 100.0 | 100.0 | 100.0 | 99.2 |
| refund_requested | 66.7 | 55.8 | 100.0 | 100.0 | 100.0 | 100.0 |

Teacher-forced loss on reference answers: base 1.139 (test) / 1.116 (held-out); r8·300 0.0001 / 0.042.

## What this shows

- **The fine-tune is real and large in-distribution.** With the same prompt, the frozen model reaches 8.3%
  and the adapter 100%. Validation loss fell from 1.126 to 0.00007, and `lora_b`'s norm went from 0 to 4.58.
- **Generalisation is partial, and priority is the weak field.** On unseen phrasings, category, refund
  and order id transfer almost perfectly. Priority does not, because in this task it is signalled *only* by an
  urgency sentence, and the held-out split uses urgency sentences never seen in training. Priority confusion
  for r8·300 on the held-out split (reference → prediction):
  - `high → urgent` 24 of 26: the unseen "This is a serious problem for me." is read as urgent.
  - `urgent → high` 8 of 27.
  - `medium` 36 of 39 and `low` 28 of 28 correct.

  The base model gets "high" right (26/26) but labels 31 of 39 medium tickets and all 27 urgent tickets "high".
- **More capacity did not help here.** Rank 16 with dropout fit the training data equally well but scored
  lowest on held-out priority (40%, below the base model's 46.7%). That is consistent with memorising the
  seen urgency phrases rather than learning severity. One seed per configuration, so these differences
  between adapters are indicative only.
- **Training longer helped generalisation** (100 → 300 steps: 50.0% → 63.3% held-out) even though
  in-distribution accuracy was already 100% at 100 steps.

## Limits

Synthetic data with templated phrasing (real tickets are messier); one task, one base model, single
seeds, no confidence intervals. The natural next experiment is more paraphrase diversity for urgency
in the *training* templates, keeping the held-out phrasings unseen, plus multiple seeds.
