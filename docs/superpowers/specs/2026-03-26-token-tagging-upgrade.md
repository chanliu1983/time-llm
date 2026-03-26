# Design: Token-Tagging Upgrade (MiniLM + Rule Builder)

**Date:** 2026-03-26  
**Status:** Implemented  
**Goal:** Replace lossy pooled multi-head parsing with token-level tagging so weekday/time spans are preserved and converted into rules deterministically.

---

## Why this change

The previous architecture pooled the whole sentence into one vector, then predicted all fields (`weekday`, `start`, `end`, etc.) with independent heads. That loses token-level structure and makes it hard to generalize to different phrasing.

This task is fundamentally structured entity extraction. Token classification is a better fit:

- detects where DAY/START/END actually appear in text
- handles varied sentence order more naturally
- separates semantic encoding (MiniLM) from deterministic rule assembly
- improves debuggability (you can inspect predicted tags directly)

---

## What changed

## 1) New tag vocabulary

Implemented in [tagging_utils.py](file:///Users/leiliu/Documents/time-llm/tagging_utils.py):

- `O`
- `B-DAY`, `I-DAY`
- `B-START`, `I-START`
- `B-END`, `I-END`
- `B-POLARITY`

Also added:

- `TAG2ID`, `ID2TAG`, `NUM_TAGS`
- day/time normalization helpers
- char-label creation and token alignment using offset mappings

## 2) New model: token classifier

In [train_time_logicformer.py](file:///Users/leiliu/Documents/time-llm/train_time_logicformer.py):

- `TimeLogicTagger` now uses:
  - `AutoModel.from_pretrained(backbone)`
  - linear token classifier to `NUM_TAGS`
- output shape is `(B, seq_len, NUM_TAGS)`
- sentence pooling heads were removed

`TimeLogicFormer` is kept as an alias of `TimeLogicTagger` for compatibility.

## 3) Dataset changed to token labels

`RuleDataset` now:

- tokenizes with `return_offsets_mapping=True`
- builds char-level labels from structured rules
- aligns char labels to token labels
- returns:
  - `input_ids`
  - `attention_mask`
  - `labels` (token tag IDs, `-100` for ignored positions)

## 4) Loss function changed

Training loss is now token-level CE:

- `nn.CrossEntropyLoss(ignore_index=-100)`
- computed over flattened `(B*seq)` tokens

## 5) New deterministic rule builder

Added [rule_builder.py](file:///Users/leiliu/Documents/time-llm/rule_builder.py):

- `extract_spans(tokens, tags)`
- `build_rules(spans)`
- `build_rules_from_tags(tokens, tags)`

Inference path:

`logits -> tag IDs -> BIO tags -> spans -> normalized rules JSON`

## 6) Inference consumers updated

Updated to new decode signature:

- [simple_ui_server.py](file:///Users/leiliu/Documents/time-llm/simple_ui_server.py)
- [evaluate_probe_set.py](file:///Users/leiliu/Documents/time-llm/evaluate_probe_set.py)

## 7) Data generator option added

In [generate_synthetic_blacklist_data.py](file:///Users/leiliu/Documents/time-llm/generate_synthetic_blacklist_data.py):

- new flag: `--emit-token-labels`
- emits `token_supervision` (`tokens`, `tags`) for inspection/debugging

## 8) Tests updated

Updated [tests/test_model.py](file:///Users/leiliu/Documents/time-llm/tests/test_model.py) for token-classification behavior and decode flow.

---

## New training script usage

## Train

```bash
python3 train_time_logicformer.py \
  --mode train \
  --data synthetic_blacklist_ultra.jsonl \
  --device mps \
  --backbone sentence-transformers/all-MiniLM-L6-v2 \
  --epochs 10 \
  --batch-size 64 \
  --max-len 128 \
  --lr 2e-4 \
  --encoder-lr 2e-5 \
  --out-dir artifacts_token_tagger
```

## Evaluate

```bash
python3 train_time_logicformer.py \
  --mode eval \
  --eval-data eval_balanced_ultra.jsonl \
  --checkpoint artifacts_token_tagger/best_model.pt \
  --device mps
```

## Run paraphrase probe comparison

```bash
python3 evaluate_probe_set.py \
  --old-checkpoint artifacts_mps_large/best_model.pt \
  --new-checkpoint artifacts_token_tagger/best_model.pt
```

## Generate data with token-level supervision metadata

```bash
python3 generate_synthetic_blacklist_data.py \
  --mode balanced \
  --num-samples 5000 \
  --emit-token-labels \
  --output synthetic_with_token_labels.jsonl
```

---

## Notes and limits

- Current rule builder pairs spans by order (`DAY[i]`, `START[i]`, `END[i]`).
- This is intentionally lightweight; relation modeling is a future step.
- If needed, next upgrade path is:
  - CRF over tags
  - explicit span-linking head for DAY↔TIME pairing
  - richer polarity/condition tags
