# Design: MiniLM Backbone for TimeLogicFormer

**Date:** 2026-03-26
**Status:** Approved
**Goal:** Improve accuracy on noisy natural language by replacing the character-level encoder with a pretrained MiniLM backbone.

---

## Problem

The current TimeLogicFormer fails across all error modes — wrong weekday, wrong time values, wrong rule count, and collapse on unusual phrasings. Root cause: the character-level tokenizer and from-scratch Transformer must learn semantic meaning (e.g. `"Mon"` = `"Monday"` = `"monday"`) entirely from synthetic data. This is a fundamental capacity limitation, not a data volume problem.

---

## Architecture

### What changes

| Component | Before | After |
|---|---|---|
| Tokenizer | `CharTokenizer` (char-level, vocab 7000) | HuggingFace `AutoTokenizer` from `all-MiniLM-L6-v2` (BPE, ~30K vocab) |
| Encoder | From-scratch `TransformerEncoder` (4 layers, d_model=256) | Pretrained `AutoModel` from `all-MiniLM-L6-v2` (6 layers, hidden=384, ~22MB) |
| Positional encoding | `SinusoidalPositionalEncoding` | Removed — baked into pretrained model |
| Pooling | Masked mean pool | Mean pool of last hidden states (same behaviour) |
| Classification heads | Linear heads off 256-dim | Same heads off 384-dim |

### What stays the same

- 7 classification heads: `count`, `weekday`, `start_h`, `start_m`, `end_h`, `end_m`, `polarity` — one set per rule slot
- `compute_loss` — cross-entropy per head, active-rule masking, label smoothing
- `RuleDataset` structure — only the tokenizer call swaps
- `evaluate_effectiveness` — unchanged metrics
- `simple_ui_server.py` inference — only tokenizer load swaps

### Dual learning rate

The pretrained encoder uses a lower LR to preserve learned representations:

- `--encoder-lr` (default `2e-5`) — applied to the pretrained backbone parameters
- `--lr` (default `2e-4`) — applied to the classification head parameters

Implemented via two `param_groups` in the `AdamW` optimizer.

### New CLI flags

| Flag | Default | Description |
|---|---|---|
| `--backbone` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model ID |
| `--encoder-lr` | `2e-5` | LR for pretrained encoder layers |

Removed (unused with pretrained backbone): `--hidden-dim`, `--num-layers`, `--num-heads`, `--ff-dim`, `--vocab-size`.

### Checkpoint format

Saves `model_state_dict` + `config` dict. Config stores `backbone` (model ID string) instead of layer/dim params. All consumers (`evaluate_effectiveness`, `simple_ui_server.py`) reconstruct model from `config["backbone"]`.

---

## Data improvements

Targeted additions to `generate_synthetic_blacklist_data.py` only — no structural changes.

### Time format variants (add to `diversify_text`)

- 12h clock: `"8am"`, `"8:00am"`, `"8 AM"`, `"8 o'clock"`
- Casual ranges: `"8-10am"`
- Named periods: `"overnight"` → `00:00–06:00`, `"business hours"` → `09:00–17:00`

### Day name variants (expand `random_weekday_alias`)

- Short forms with period: `"Mon."`, `"Sat."`
- Alternate abbrevs: `"Tues"`, `"Thurs"`, `"Weds"`
- Lowercase: `"monday"`, `"tue"`
- Ordinal phrasing: `"every Monday"`, `"on Mondays"`

Generation scale and command unchanged (80K samples, balanced mode).

---

## Training

### Recommended command

```bash
python3 train_time_logicformer.py \
  --mode train \
  --data synthetic_blacklist_ultra.jsonl \
  --device mps \
  --epochs 10 \
  --batch-size 64 \
  --lr 2e-4 \
  --encoder-lr 2e-5 \
  --out-dir artifacts_minilm
```

Fewer epochs (10 vs 20) — pretrained model converges faster.
Smaller batch size (64 vs 128) — MiniLM hidden states use more memory than char embeddings.

### Dependencies

```
transformers
huggingface_hub
```

Model (~22MB) downloads on first run and caches in `~/.cache/huggingface/`.

---

## Files changed

| File | Change |
|---|---|
| `train_time_logicformer.py` | Replace `CharTokenizer` + from-scratch encoder with HuggingFace backbone; add `--backbone` and `--encoder-lr` flags; dual-LR optimizer |
| `generate_synthetic_blacklist_data.py` | Expand `diversify_text` time formats; expand `random_weekday_alias` day variants |
| `simple_ui_server.py` | Swap CharTokenizer load for HuggingFace tokenizer load |
| `evaluate_probe_set.py` | Same tokenizer swap as `simple_ui_server.py` |

---

## Out of scope

- Seq2seq / autoregressive JSON generation
- Retrieval-augmented or LLM API inference
- Changes to evaluation metrics or UI layout
