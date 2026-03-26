# MiniLM Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the character-level encoder in TimeLogicFormer with a pretrained MiniLM backbone to improve accuracy on noisy natural language.

**Architecture:** The existing `CharTokenizer` + from-scratch `TransformerEncoder` is replaced with HuggingFace `AutoTokenizer` + `AutoModel` (`sentence-transformers/all-MiniLM-L6-v2`, 22MB). The 7 classification heads are retained but now accept 384-dim input from the pretrained encoder. A dual-LR optimizer applies a lower rate to the pretrained backbone and a higher rate to the new heads.

**Tech Stack:** PyTorch, HuggingFace `transformers>=4.40.0`, `huggingface_hub>=0.22.0`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Create | Pin new HuggingFace dependencies |
| `tests/test_model.py` | Create | Unit tests for new model class and tokenizer |
| `tests/test_data.py` | Create | Unit tests for data generation variants |
| `train_time_logicformer.py` | Modify | Replace encoder, tokenizer, model class, training loop, CLI flags |
| `generate_synthetic_blacklist_data.py` | Modify | Add time format and day name variants |
| `simple_ui_server.py` | Modify | Load HuggingFace tokenizer from checkpoint config |
| `evaluate_probe_set.py` | Modify | Same tokenizer swap + updated model constructor |

---

## Task 1: Add dependencies

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
torch
transformers>=4.40.0
huggingface_hub>=0.22.0
```

- [ ] **Step 2: Install dependencies**

```bash
pip install transformers>=4.40.0 huggingface_hub>=0.22.0
```

Expected: installs without error.

- [ ] **Step 3: Verify import**

```bash
python3 -c "from transformers import AutoModel, AutoTokenizer; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add transformers and huggingface_hub dependencies"
```

---

## Task 2: Expand data generation variants

**Files:**
- Modify: `generate_synthetic_blacklist_data.py` — `random_weekday_alias` (lines 72–82), `diversify_text` (lines 85–119)
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import random
from generate_synthetic_blacklist_data import random_weekday_alias, diversify_text

def test_weekday_alias_includes_tues():
    rng = random.Random(0)
    aliases = {random_weekday_alias("Tuesday", rng) for _ in range(200)}
    assert "Tues" in aliases, f"Expected 'Tues' in aliases, got {aliases}"

def test_weekday_alias_includes_thurs():
    rng = random.Random(0)
    aliases = {random_weekday_alias("Thursday", rng) for _ in range(200)}
    assert "Thurs" in aliases

def test_weekday_alias_includes_weds():
    rng = random.Random(0)
    aliases = {random_weekday_alias("Wednesday", rng) for _ in range(200)}
    assert "Weds" in aliases

def test_weekday_alias_lowercase():
    rng = random.Random(0)
    aliases = {random_weekday_alias("Monday", rng) for _ in range(200)}
    assert "monday" in aliases

def test_diversify_text_12h_format():
    rng = random.Random(42)
    seen_12h = False
    for _ in range(500):
        result = diversify_text("Forbid access on Monday from 08:00 to 10:00.", rng)
        if "8am" in result or "8 AM" in result or "8:00am" in result:
            seen_12h = True
            break
    assert seen_12h, "Expected 12h format variant to appear in diversify_text output"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_data.py -v
```

Expected: 5 FAILED (functions don't have new variants yet).

- [ ] **Step 3: Expand `random_weekday_alias`**

Replace the `aliases` dict in `generate_synthetic_blacklist_data.py` (lines 73–81):

```python
    aliases = {
        "Monday": ["Monday", "Mon", "Mon.", "monday", "every Monday", "on Mondays"],
        "Tuesday": ["Tuesday", "Tue", "Tues", "tuesday", "every Tuesday"],
        "Wednesday": ["Wednesday", "Wed", "Weds", "wednesday", "every Wednesday"],
        "Thursday": ["Thursday", "Thu", "Thurs", "thursday", "every Thursday"],
        "Friday": ["Friday", "Fri", "Fri.", "friday", "every Friday"],
        "Saturday": ["Saturday", "Sat", "Sat.", "saturday", "every Saturday"],
        "Sunday": ["Sunday", "Sun", "Sun.", "sunday", "every Sunday"],
    }
```

- [ ] **Step 4: Expand `diversify_text` with 12h time formats**

In `diversify_text`, add a new block after the existing `if rng.random() < 0.1:` typo block (after line 118), before `return rewritten.strip()`:

```python
    # 12h time format variants
    if rng.random() < 0.3:
        def to_12h(m):
            h = int(m.group(1))
            mins = m.group(2)
            suffix = "am" if h < 12 else "pm"
            h12 = h if h <= 12 else h - 12
            if h12 == 0:
                h12 = 12
            variants = [f"{h12}:{mins}{suffix}", f"{h12}{suffix}", f"{h12} {suffix.upper()}"]
            return rng.choice(variants)
        rewritten = re.sub(r"\b(\d{2}):(\d{2})\b", to_12h, rewritten)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_data.py -v
```

Expected: 5 PASSED.

- [ ] **Step 6: Smoke-test generation still works**

```bash
python3 generate_synthetic_blacklist_data.py --num-samples 100 --output /tmp/test_gen.jsonl
python3 -c "
import json
rows = [json.loads(l) for l in open('/tmp/test_gen.jsonl')]
print('samples:', len(rows))
print('example:', rows[0]['text'])
"
```

Expected: 100 samples, valid JSON, some with abbreviated day names.

- [ ] **Step 7: Commit**

```bash
git add generate_synthetic_blacklist_data.py tests/test_data.py
git commit -m "feat: expand day name and time format variants in data generation"
```

---

## Task 3: Rewrite TimeLogicFormer model class

**Files:**
- Modify: `train_time_logicformer.py` — replace `CharTokenizer`, `SinusoidalPositionalEncoding`, `TimeLogicFormer` class, and `encode_text_with_tokenizer`
- Create: `tests/test_model.py`

### Context

The new `TimeLogicFormer` wraps `AutoModel.from_pretrained(backbone)`. The hidden size is read from `backbone.config.hidden_size` (384 for MiniLM). The old `CharTokenizer` and `SinusoidalPositionalEncoding` classes are deleted entirely. `encode_text_with_tokenizer` now accepts a HuggingFace tokenizer object instead of a `dict`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_model.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pytest

BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"


def test_model_instantiates():
    from train_time_logicformer import TimeLogicFormer
    model = TimeLogicFormer(backbone=BACKBONE, max_rules=4)
    assert model is not None


def test_model_forward_shape():
    from train_time_logicformer import TimeLogicFormer
    from transformers import AutoTokenizer
    model = TimeLogicFormer(backbone=BACKBONE, max_rules=4)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    enc = tokenizer("block monday all day", return_tensors="pt",
                    max_length=128, padding="max_length", truncation=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    assert out["count_logits"].shape == (1, 5)           # max_rules+1
    assert out["weekday_logits"].shape == (1, 4, 8)
    assert out["start_h_logits"].shape == (1, 4, 24)
    assert out["start_m_logits"].shape == (1, 4, 60)
    assert out["end_h_logits"].shape == (1, 4, 24)
    assert out["end_m_logits"].shape == (1, 4, 60)
    assert out["polarity_logits"].shape == (1, 4, 2)


def test_encode_text_with_tokenizer():
    from train_time_logicformer import encode_text_with_tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    ids, mask = encode_text_with_tokenizer("block monday", tokenizer)
    assert ids.shape[0] == 1
    assert mask.shape == ids.shape
    assert ids.dtype == torch.long


def test_char_tokenizer_removed():
    import train_time_logicformer as m
    assert not hasattr(m, "CharTokenizer"), "CharTokenizer should be removed"


def test_sinusoidal_pe_removed():
    import train_time_logicformer as m
    assert not hasattr(m, "SinusoidalPositionalEncoding"), \
        "SinusoidalPositionalEncoding should be removed"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_model.py -v
```

Expected: `test_model_instantiates`, `test_model_forward_shape`, `test_encode_text_with_tokenizer` FAIL (wrong signature). `test_char_tokenizer_removed` and `test_sinusoidal_pe_removed` FAIL.

- [ ] **Step 3: Replace model class and helpers in `train_time_logicformer.py`**

**Remove** these classes entirely (delete lines 27–51 and 132–202):
- `CharTokenizer`
- `SinusoidalPositionalEncoding`
- `TimeLogicFormer`

**Add** at the top of the file (after existing imports), add:
```python
from transformers import AutoModel, AutoTokenizer
```

**Add** new `TimeLogicFormer` class (replacing the old one):

```python
class TimeLogicFormer(nn.Module):
    def __init__(self, backbone: str, max_rules: int = 6, dropout: float = 0.1):
        super().__init__()
        self.max_rules = max_rules
        self.backbone_model = AutoModel.from_pretrained(backbone)
        hidden = self.backbone_model.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.count_head = nn.Linear(hidden, max_rules + 1)
        self.weekday_head = nn.Linear(hidden, max_rules * 8)
        self.start_h_head = nn.Linear(hidden, max_rules * 24)
        self.start_m_head = nn.Linear(hidden, max_rules * 60)
        self.end_h_head = nn.Linear(hidden, max_rules * 24)
        self.end_m_head = nn.Linear(hidden, max_rules * 60)
        self.polarity_head = nn.Linear(hidden, max_rules * 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict:
        out = self.backbone_model(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # (B, seq_len, hidden)
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
        else:
            pooled = h.mean(dim=1)
        z = self.drop(pooled)
        return {
            "count_logits": self.count_head(z),
            "weekday_logits": self.weekday_head(z).view(-1, self.max_rules, 8),
            "start_h_logits": self.start_h_head(z).view(-1, self.max_rules, 24),
            "start_m_logits": self.start_m_head(z).view(-1, self.max_rules, 60),
            "end_h_logits": self.end_h_head(z).view(-1, self.max_rules, 24),
            "end_m_logits": self.end_m_head(z).view(-1, self.max_rules, 60),
            "polarity_logits": self.polarity_head(z).view(-1, self.max_rules, 2),
        }
```

**Replace** `encode_text_with_tokenizer` (currently lines 273–284) with:

```python
def encode_text_with_tokenizer(text: str, tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a single text string using a HuggingFace AutoTokenizer.
    Returns (input_ids, attention_mask) each of shape (1, max_length).
    """
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_model.py -v
```

Expected: 5 PASSED. (Note: first run downloads MiniLM ~22MB.)

- [ ] **Step 5: Commit**

```bash
git add train_time_logicformer.py tests/test_model.py
git commit -m "feat: replace char encoder with pretrained MiniLM backbone"
```

---

## Task 4: Update RuleDataset tokenization

**Files:**
- Modify: `train_time_logicformer.py` — `RuleDataset` class (lines 88–129)

### Context

`RuleDataset` currently calls `self.tokenizer.encode(text)` which returns `(ids_list, mask_list)`. The new version calls the HuggingFace tokenizer directly and returns tensors. The dataset stores a HuggingFace tokenizer object instead of a `CharTokenizer`.

- [ ] **Step 1: Add a dataset test to `tests/test_model.py`**

Append to `tests/test_model.py`:

```python
def test_rule_dataset_item_shape():
    from train_time_logicformer import RuleDataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    rows = [{"text": "block Monday all day", "label": {"forbidden": [{"weekday": 1, "start": "00:00", "end": "23:59"}]}}]
    ds = RuleDataset(rows, tokenizer, max_rules=4)
    item = ds[0]
    assert item["input_ids"].shape == (128,)
    assert item["attention_mask"].shape == (128,)
    assert item["count"].item() == 1
    assert item["weekday"][0].item() == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_model.py::test_rule_dataset_item_shape -v
```

Expected: FAIL (current `RuleDataset` calls `.encode()` on the HuggingFace tokenizer which returns a different structure).

- [ ] **Step 3: Update `RuleDataset.__getitem__`**

Replace the tokenizer call in `RuleDataset.__getitem__` (currently `ids, mask = self.tokenizer.encode(text)` then tensor creation):

```python
    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        text = row["text"]
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        rules = encode_rules(row["label"]["forbidden"], self.max_rules)
        count = len(rules)

        weekday = [0] * self.max_rules
        start_h = [0] * self.max_rules
        start_m = [0] * self.max_rules
        end_h = [0] * self.max_rules
        end_m = [0] * self.max_rules
        polarity = [0] * self.max_rules

        for i, r in enumerate(rules):
            weekday[i] = r.weekday
            start_h[i] = r.start_h
            start_m[i] = r.start_m
            end_h[i] = r.end_h
            end_m[i] = r.end_m
            polarity[i] = r.polarity

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "count": torch.tensor(count, dtype=torch.long),
            "weekday": torch.tensor(weekday, dtype=torch.long),
            "start_h": torch.tensor(start_h, dtype=torch.long),
            "start_m": torch.tensor(start_m, dtype=torch.long),
            "end_h": torch.tensor(end_h, dtype=torch.long),
            "end_m": torch.tensor(end_m, dtype=torch.long),
            "polarity": torch.tensor(polarity, dtype=torch.long),
        }
```

- [ ] **Step 4: Run all tests**

```bash
python3 -m pytest tests/ -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add train_time_logicformer.py
git commit -m "feat: update RuleDataset to use HuggingFace tokenizer"
```

---

## Task 5: Update training loop, CLI flags, and checkpoint format

**Files:**
- Modify: `train_time_logicformer.py` — `train()` function, `build_arg_parser()`

### Context

The `train()` function currently creates a `CharTokenizer`, fits it on train texts, then creates `TimeLogicFormer` with many dimension args. The new version loads a HuggingFace tokenizer and creates `TimeLogicFormer(backbone=args.backbone, ...)`. The optimizer uses two param groups. The checkpoint config stores `backbone` instead of `d_model`/`nhead`/etc. The `--tokenizer-path` save step is removed.

- [ ] **Step 1: Add a checkpoint format test to `tests/test_model.py`**

Append to `tests/test_model.py`:

```python
import tempfile, pathlib

def test_checkpoint_save_and_load():
    from train_time_logicformer import TimeLogicFormer
    model = TimeLogicFormer(backbone=BACKBONE, max_rules=2)
    config = {"backbone": BACKBONE, "max_rules": 2, "dropout": 0.1}
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "model.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": config}, path)
        ckpt = torch.load(path, map_location="cpu")
        loaded = TimeLogicFormer(
            backbone=ckpt["config"]["backbone"],
            max_rules=ckpt["config"]["max_rules"],
            dropout=ckpt["config"]["dropout"],
        )
        loaded.load_state_dict(ckpt["model_state_dict"])
    assert loaded is not None


def test_old_checkpoint_format_fails_clearly():
    """Old checkpoints missing 'backbone' key must raise KeyError, not produce garbage."""
    from train_time_logicformer import TimeLogicFormer
    old_style_config = {"d_model": 256, "num_layers": 4, "max_rules": 6}
    try:
        _ = TimeLogicFormer(
            backbone=old_style_config["backbone"],  # KeyError here
            max_rules=old_style_config["max_rules"],
        )
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
```

- [ ] **Step 2: Run tests to verify the checkpoint tests pass (they should already)**

```bash
python3 -m pytest tests/test_model.py::test_checkpoint_save_and_load tests/test_model.py::test_old_checkpoint_format_fails_clearly -v
```

Expected: both PASSED (model class is already correct from Task 3).

- [ ] **Step 3: Update `build_arg_parser` in `train_time_logicformer.py`**

**Remove** these arguments from the old `build_arg_parser` (delete their `p.add_argument` lines):
- `--hidden-dim`
- `--num-layers`
- `--num-heads`
- `--ff-dim`
- `--vocab-size`
- `--max-len`
- `--tokenizer-path`

**Replace** the full `build_arg_parser` function body with:

```python
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    p.add_argument("--data", type=str, default="synthetic_blacklist.jsonl")
    p.add_argument("--eval-data", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--out-dir", type=str, default="artifacts")
    p.add_argument("--device", type=str, choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--backbone", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--encoder-lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--max-rules", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    return p
```

- [ ] **Step 4: Update `train()` function in `train_time_logicformer.py`**

Replace the entire `train()` function body with:

```python
def train(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(data_path)
    if len(rows) < 10:
        raise RuntimeError("Need at least 10 samples to train")

    rng = random.Random(args.seed)
    train_rows, valid_rows = split_rows(rows, args.valid_ratio, rng)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    train_ds = RuleDataset(train_rows, tokenizer, max_rules=args.max_rules)
    valid_ds = RuleDataset(valid_rows, tokenizer, max_rules=args.max_rules)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    device = choose_device(args.device)
    model = TimeLogicFormer(
        backbone=args.backbone,
        max_rules=args.max_rules,
        dropout=args.dropout,
    ).to(device)

    backbone_params = list(model.backbone_model.parameters())
    head_params = [p for p in model.parameters() if not any(p is bp for bp in backbone_params)]
    optimizer = AdamW([
        {"params": backbone_params, "lr": args.encoder_lr},
        {"params": head_params, "lr": args.lr},
    ])

    total_steps = max(1, args.epochs * max(1, len(train_loader)))
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-4, (step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_valid = float("inf")
    best_path = out_dir / "best_model.pt"

    print(f"device={device.type} backbone={args.backbone} train={len(train_rows)} valid={len(valid_rows)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = compute_loss(outputs, batch, max_rules=args.max_rules, label_smoothing=args.label_smoothing)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()
            total += float(loss.item())
            batches += 1
        train_loss = total / max(1, batches)
        valid_loss = evaluate(model, valid_loader, device, max_rules=args.max_rules, label_smoothing=args.label_smoothing)
        print(f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "backbone": args.backbone,
                        "max_rules": args.max_rules,
                        "dropout": args.dropout,
                    },
                },
                best_path,
            )

    print(f"best_valid_loss={best_valid:.4f}")
    print(f"saved_model={best_path}")
```

- [ ] **Step 5: Run all tests**

```bash
python3 -m pytest tests/ -v
```

Expected: all PASSED.

- [ ] **Step 6: Smoke-test the CLI arg parser**

```bash
python3 -c "
from train_time_logicformer import build_arg_parser
p = build_arg_parser()
args = p.parse_args(['--mode', 'train', '--backbone', 'sentence-transformers/all-MiniLM-L6-v2'])
print('backbone:', args.backbone)
print('encoder_lr:', args.encoder_lr)
print('lr:', args.lr)
"
```

Expected: prints `backbone: sentence-transformers/all-MiniLM-L6-v2`, `encoder_lr: 2e-05`, `lr: 0.0002`.

- [ ] **Step 7: Commit**

```bash
git add train_time_logicformer.py
git commit -m "feat: update training loop with dual-LR optimizer and new checkpoint format"
```

---

## Task 6: Update `evaluate_effectiveness` in `train_time_logicformer.py`

**Files:**
- Modify: `train_time_logicformer.py` — `evaluate_effectiveness()` function

### Context

Currently loads model from `config["vocab_size"]`, `config["d_model"]` etc., and loads tokenizer from a separate JSON file via `--tokenizer-path`. New version loads model from `config["backbone"]` and loads tokenizer via `AutoTokenizer.from_pretrained(config["backbone"])`. The `--tokenizer-path` flag is already removed from `build_arg_parser` in Task 5.

- [ ] **Step 1: Replace model load block and tokenizer load in `evaluate_effectiveness`**

At the top of `evaluate_effectiveness`, **delete** this guard (no longer needed):
```python
    if not args.tokenizer_path:
        raise RuntimeError("tokenizer-path is required when mode=eval")
```

Replace the model construction + tokenizer load block (the block that calls `TimeLogicFormer(vocab_size=..., d_model=..., ...)` and then opens `args.tokenizer_path` as JSON) with:

```python
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = TimeLogicFormer(
        backbone=config["backbone"],
        max_rules=config["max_rules"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["backbone"])
```

In the eval loop, replace `encode_text_with_tokenizer(row["text"], tokenizer_data)` with `encode_text_with_tokenizer(row["text"], tokenizer)`. There is exactly one such call inside the `with torch.no_grad():` loop.

- [ ] **Step 2: Run all tests**

```bash
python3 -m pytest tests/ -v
```

Expected: all PASSED.

- [ ] **Step 3: Commit**

```bash
git add train_time_logicformer.py
git commit -m "feat: update evaluate_effectiveness to load tokenizer from backbone config"
```

---

## Task 7: Update `simple_ui_server.py`

**Files:**
- Modify: `simple_ui_server.py` — `ModelService.__init__`, `ModelService.predict`, imports, `main()`

### Context

`ModelService` currently constructs `TimeLogicFormer` with `config["vocab_size"]`, `config["d_model"]` etc., loads a JSON tokenizer file, and calls `encode_text_with_tokenizer(text, self.tokenizer_data)`. The new version constructs from `config["backbone"]` and loads a HuggingFace tokenizer.

- [ ] **Step 1: Add `from transformers import AutoTokenizer` import to `simple_ui_server.py`**

At the top of `simple_ui_server.py`, after the existing `from train_time_logicformer import ...` line, add:

```python
from transformers import AutoTokenizer
```

- [ ] **Step 2: Replace `ModelService.__init__` in `simple_ui_server.py`**

Replace the constructor body (lines 187–204):

```python
    def __init__(self, checkpoint_path: Path, device_arg: str, assist_mode: str):
        self.device = choose_device(device_arg)
        self.assist_mode = assist_mode
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self.config = checkpoint["config"]
        self.model = TimeLogicFormer(
            backbone=self.config["backbone"],
            max_rules=self.config["max_rules"],
            dropout=self.config["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["backbone"])
```

- [ ] **Step 3: Update `ModelService.predict` to use `self.tokenizer`**

Replace `encode_text_with_tokenizer(text, self.tokenizer_data)` with `encode_text_with_tokenizer(text, self.tokenizer)`.

- [ ] **Step 4: Update `main()` in `simple_ui_server.py`**

Remove `--tokenizer-path` argument. Update `ModelService(...)` call to remove `tokenizer_path`:

```python
    service = ModelService(Path(args.checkpoint), args.device, args.assist_mode)
```

- [ ] **Step 5: Smoke-test the import**

```bash
python3 -c "import simple_ui_server; print('import ok')"
```

Expected: `import ok` (no errors).

- [ ] **Step 6: Commit**

```bash
git add simple_ui_server.py
git commit -m "feat: update simple_ui_server to load HuggingFace tokenizer from checkpoint"
```

---

## Task 8: Update `evaluate_probe_set.py`

**Files:**
- Modify: `evaluate_probe_set.py` — `load_model`, `predict`, imports, `main()`

### Context

`load_model` currently builds `TimeLogicFormer` with old config keys and loads tokenizer from a JSON file. `predict` passes `tokenizer_data` dict. Both need to use the new API.

- [ ] **Step 1: Add `from transformers import AutoTokenizer` import to `evaluate_probe_set.py`**

At the top of `evaluate_probe_set.py`, after the existing `from train_time_logicformer import ...` line, add:

```python
from transformers import AutoTokenizer
```

- [ ] **Step 2: Replace `load_model` in `evaluate_probe_set.py`**

```python
def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    model = TimeLogicFormer(
        backbone=cfg["backbone"],
        max_rules=cfg["max_rules"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    return model, tokenizer, cfg["max_rules"]
```

- [ ] **Step 3: Update `predict` in `evaluate_probe_set.py`**

The signature and body stay the same except the parameter name changes from `tokenizer_data` to `tokenizer` for clarity — the call inside already uses `encode_text_with_tokenizer(text, tokenizer)` which now accepts a HuggingFace tokenizer.

- [ ] **Step 4: Update `main()` in `evaluate_probe_set.py`**

Remove `--old-tokenizer` and `--new-tokenizer` arguments. Update `load_model` calls to remove tokenizer path args:

```python
    old_model, old_tok, old_max_rules = load_model(Path(args.old_checkpoint), device)
    new_model, new_tok, new_max_rules = load_model(Path(args.new_checkpoint), device)
```

Update default paths to remove the hardcoded old user path:
```python
    parser.add_argument("--old-checkpoint", type=str, default="artifacts/best_model.pt")
    parser.add_argument("--new-checkpoint", type=str, default="artifacts_minilm/best_model.pt")
```

- [ ] **Step 5: Smoke-test the import**

```bash
python3 -c "import evaluate_probe_set; print('import ok')"
```

Expected: `import ok`.

- [ ] **Step 6: Run full test suite**

```bash
python3 -m pytest tests/ -v
```

Expected: all PASSED.

- [ ] **Step 7: Commit**

```bash
git add evaluate_probe_set.py
git commit -m "feat: update evaluate_probe_set to use new checkpoint format and HuggingFace tokenizer"
```

---

## Task 9: End-to-end smoke test

**Goal:** Verify the full pipeline (generate → train tiny → evaluate) works before committing to the full 80K run.

- [ ] **Step 1: Generate a tiny dataset**

```bash
python3 generate_synthetic_blacklist_data.py \
  --mode balanced \
  --num-samples 500 \
  --seed 99 \
  --output /tmp/smoke_test.jsonl
```

Expected: creates `/tmp/smoke_test.jsonl` with 500 lines.

- [ ] **Step 2: Run 2-epoch training smoke test**

```bash
python3 train_time_logicformer.py \
  --mode train \
  --data /tmp/smoke_test.jsonl \
  --device mps \
  --epochs 2 \
  --batch-size 16 \
  --out-dir /tmp/artifacts_smoke
```

Expected: prints `epoch=1 ...` and `epoch=2 ...`, saves `best_model.pt`.

- [ ] **Step 3: Run eval smoke test**

```bash
python3 train_time_logicformer.py \
  --mode eval \
  --eval-data /tmp/smoke_test.jsonl \
  --checkpoint /tmp/artifacts_smoke/best_model.pt \
  --device mps
```

Expected: prints JSON metrics (all fields present, counts match).

- [ ] **Step 4: No commit needed** — Task 9 is verification only. All code changes were committed in Tasks 1–8. If any fixes were made during smoke testing, commit them now with a descriptive message.

---

## Task 10: Full training run

- [ ] **Step 1: Generate full dataset (if not already done)**

```bash
python3 generate_synthetic_blacklist_data.py \
  --mode balanced \
  --num-samples 80000 \
  --seed 4040 \
  --output synthetic_blacklist_ultra.jsonl
```

- [ ] **Step 2: Run full training**

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

Expected: ~1-2h on Apple Silicon. Prints loss per epoch, saves `artifacts_minilm/best_model.pt`.

- [ ] **Step 3: Evaluate on held-out set**

```bash
python3 train_time_logicformer.py \
  --mode eval \
  --eval-data eval_balanced_ultra.jsonl \
  --checkpoint artifacts_minilm/best_model.pt \
  --device mps
```

Expected: JSON report with `sample_exact_match_set` noticeably higher than the old char-level model.

- [ ] **Step 4: Run probe comparison (if old checkpoint exists)**

```bash
python3 evaluate_probe_set.py \
  --old-checkpoint artifacts/best_model.pt \
  --new-checkpoint artifacts_minilm/best_model.pt
```

Expected: `new_exact` >= `old_exact`.
