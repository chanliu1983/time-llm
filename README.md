# TimeLogicFormer: Natural Language Time-Rule Parsing

## What we are trying to do

This project converts natural language scheduling restrictions into a strict JSON blacklist format.

Given an input sentence like:

`Forbid access on Tuesday from 03:00 to 04:00 and block Monday all day.`

the target output is:

```json
{
  "forbidden": [
    {"weekday": 1, "start": "00:00", "end": "23:59"},
    {"weekday": 2, "start": "03:00", "end": "04:00"}
  ]
}
```

The goal is robust understanding across:

- weekday lists/ranges/exceptions
- time ranges and full-day constraints
- varied language styles (formal, casual, noisy, abbreviated)
- multiple forbidden windows in one sentence

---

## How we are doing it

We use a compact Transformer encoder model (`TimeLogicFormer`) trained on synthetic but diverse data.

### 1) Data generation

Script: `generate_synthetic_blacklist_data.py`

It creates JSONL samples with:

- sentence text (`text`)
- strict structured label (`label.forbidden`)

Generation includes:

- multiple linguistic templates
- weekday aliases (`Mon`, `Tue`, ...)
- text perturbations (spacing/case/punctuation variation)
- minor typo/noise injection for robustness

### 2) Model training

Script: `train_time_logicformer.py`

Model characteristics:

- character-level tokenizer
- Transformer encoder
- multi-head outputs for:
  - rule count
  - weekday
  - start/end hour+minute
  - polarity

Training setup includes:

- larger network defaults
- warmup + cosine LR schedule
- gradient clipping
- label smoothing
- Apple Metal (`mps`) support

### 3) Evaluation

Two evaluation paths:

- dataset-level metrics via `train_time_logicformer.py --mode eval`
- paraphrase probe comparison via `evaluate_probe_set.py`

Main reported metrics include:

- count accuracy
- exact match (ordered and set)
- rule-level micro precision/recall/F1
- aligned field accuracy (weekday/start/end)

### 4) Local UI testing

Script: `simple_ui_server.py`

Provides:

- web UI at `/`
- prediction API at `/predict`
- metadata at `/meta`

Inference pipeline:

- MiniLM token tagging (`DAY/START/END/ALLDAY/POLARITY`)
- deterministic rule builder
- strict JSON output

---

## Repository scripts

- `generate_synthetic_blacklist_data.py` — build synthetic datasets
- `train_time_logicformer.py` — train/evaluate model checkpoints
- `evaluate_probe_set.py` — compare checkpoints on hard paraphrase probes
- `simple_ui_server.py` — run interactive local demo UI

---

## Quick start

### 1) Generate data

```bash
python3 generate_synthetic_blacklist_data.py \
  --mode balanced \
  --num-samples 80000 \
  --seed 4040 \
  --output synthetic_blacklist_ultra.jsonl
```

### 2) Train on Metal (MPS)

```bash
python3 train_time_logicformer.py \
  --mode train \
  --data synthetic_blacklist_ultra.jsonl \
  --device mps \
  --epochs 20 \
  --batch-size 128 \
  --lr 2e-4 \
  --out-dir artifacts_mps_ultra
```

### 3) Evaluate

```bash
python3 train_time_logicformer.py \
  --mode eval \
  --eval-data eval_balanced_ultra.jsonl \
  --checkpoint artifacts_mps_ultra/best_model.pt \
  --device mps
```

### 4) Launch UI

```bash
python3 simple_ui_server.py \
  --checkpoint artifacts_mps_ultra/best_model.pt \
  --device mps \
  --host 0.0.0.0 \
  --port 8008
```

Open: `http://127.0.0.1:8008/`

---

## Current direction

The project uses a two-stage structure: model tagging for semantic understanding, then deterministic rule building for stability and valid JSON.
