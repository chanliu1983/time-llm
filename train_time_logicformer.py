import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class EncodedRule:
    weekday: int
    start_h: int
    start_m: int
    end_h: int
    end_m: int
    polarity: int


class CharTokenizer:
    def __init__(self, vocab_size: int = 5000, max_len: int = 80):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_id = 0
        self.unk_id = 1
        self.char2id = {"<pad>": 0, "<unk>": 1}

    def fit(self, texts: List[str]) -> None:
        freq = {}
        for text in texts:
            for ch in text:
                freq[ch] = freq.get(ch, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for i, (ch, _) in enumerate(ranked[: self.vocab_size - 2], start=2):
            self.char2id[ch] = i

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        ids = [self.char2id.get(ch, self.unk_id) for ch in text[: self.max_len]]
        mask = [1] * len(ids)
        if len(ids) < self.max_len:
            pad_n = self.max_len - len(ids)
            ids.extend([self.pad_id] * pad_n)
            mask.extend([0] * pad_n)
        return ids, mask


def parse_hm(hm: str) -> tuple[int, int]:
    h, m = hm.split(":")
    return int(h), int(m)


def encode_rules(rules: list[dict], max_rules: int) -> list[EncodedRule]:
    encoded = []
    for item in rules[:max_rules]:
        sh, sm = parse_hm(item["start"])
        eh, em = parse_hm(item["end"])
        encoded.append(
            EncodedRule(
                weekday=int(item["weekday"]),
                start_h=sh,
                start_m=sm,
                end_h=eh,
                end_m=em,
                polarity=1,
            )
        )
    return encoded


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class RuleDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer: CharTokenizer, max_rules: int = 4):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_rules = max_rules

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        text = row["text"]
        ids, mask = self.tokenizer.encode(text)
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
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "count": torch.tensor(count, dtype=torch.long),
            "weekday": torch.tensor(weekday, dtype=torch.long),
            "start_h": torch.tensor(start_h, dtype=torch.long),
            "start_m": torch.tensor(start_m, dtype=torch.long),
            "end_h": torch.tensor(end_h, dtype=torch.long),
            "end_m": torch.tensor(end_m, dtype=torch.long),
            "polarity": torch.tensor(polarity, dtype=torch.long),
        }


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TimeLogicFormer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 128,
        nhead: int = 2,
        num_layers: int = 2,
        dim_ff: int = 256,
        max_len: int = 80,
        dropout: float = 0.1,
        max_rules: int = 4,
    ):
        super().__init__()
        self.max_rules = max_rules
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.shared = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.count_head = nn.Linear(d_model, max_rules + 1)
        self.weekday_head = nn.Linear(d_model, max_rules * 8)
        self.start_h_head = nn.Linear(d_model, max_rules * 24)
        self.start_m_head = nn.Linear(d_model, max_rules * 60)
        self.end_h_head = nn.Linear(d_model, max_rules * 24)
        self.end_m_head = nn.Linear(d_model, max_rules * 60)
        self.polarity_head = nn.Linear(d_model, max_rules * 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict:
        x = self.embedding(input_ids)
        x = self.pos(x)
        pad_mask = None
        if attention_mask is not None:
            pad_mask = ~attention_mask.bool()
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        if attention_mask is None:
            pooled = h.mean(dim=1)
        else:
            m = attention_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
        z = self.shared(self.norm(pooled))
        return {
            "count_logits": self.count_head(z),
            "weekday_logits": self.weekday_head(z).view(-1, self.max_rules, 8),
            "start_h_logits": self.start_h_head(z).view(-1, self.max_rules, 24),
            "start_m_logits": self.start_m_head(z).view(-1, self.max_rules, 60),
            "end_h_logits": self.end_h_head(z).view(-1, self.max_rules, 24),
            "end_m_logits": self.end_m_head(z).view(-1, self.max_rules, 60),
            "polarity_logits": self.polarity_head(z).view(-1, self.max_rules, 2),
        }


def compute_loss(outputs: dict, batch: dict, max_rules: int, label_smoothing: float = 0.0) -> torch.Tensor:
    loss = F.cross_entropy(outputs["count_logits"], batch["count"], label_smoothing=label_smoothing)
    count = batch["count"]
    for r in range(max_rules):
        active = count > r
        if active.any():
            idx = active.nonzero(as_tuple=True)[0]
            loss = loss + F.cross_entropy(outputs["weekday_logits"][idx, r], batch["weekday"][idx, r], label_smoothing=label_smoothing)
            loss = loss + F.cross_entropy(outputs["start_h_logits"][idx, r], batch["start_h"][idx, r], label_smoothing=label_smoothing)
            loss = loss + F.cross_entropy(outputs["start_m_logits"][idx, r], batch["start_m"][idx, r], label_smoothing=label_smoothing)
            loss = loss + F.cross_entropy(outputs["end_h_logits"][idx, r], batch["end_h"][idx, r], label_smoothing=label_smoothing)
            loss = loss + F.cross_entropy(outputs["end_m_logits"][idx, r], batch["end_m"][idx, r], label_smoothing=label_smoothing)
            loss = loss + F.cross_entropy(outputs["polarity_logits"][idx, r], batch["polarity"][idx, r], label_smoothing=label_smoothing)
    return loss


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available")
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_rows(rows: list[dict], valid_ratio: float, rng: random.Random) -> tuple[list[dict], list[dict]]:
    rows_copy = list(rows)
    rng.shuffle(rows_copy)
    valid_size = max(1, int(len(rows_copy) * valid_ratio))
    valid = rows_copy[:valid_size]
    train = rows_copy[valid_size:]
    if not train:
        train = valid
    return train, valid


def evaluate(
    model: TimeLogicFormer,
    loader: DataLoader,
    device: torch.device,
    max_rules: int,
    label_smoothing: float = 0.0,
) -> float:
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = compute_loss(outputs, batch, max_rules=max_rules, label_smoothing=label_smoothing)
            total += float(loss.item())
            batches += 1
    model.train()
    if batches == 0:
        return 0.0
    return total / batches


def encode_text_with_tokenizer(text: str, tokenizer_data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    char2id = tokenizer_data["char2id"]
    max_len = int(tokenizer_data["max_len"])
    pad_id = int(char2id["<pad>"])
    unk_id = int(char2id["<unk>"])
    ids = [char2id.get(ch, unk_id) for ch in text[:max_len]]
    attention_mask = [1] * len(ids)
    if len(ids) < max_len:
        pad_len = max_len - len(ids)
        ids.extend([pad_id] * pad_len)
        attention_mask.extend([0] * pad_len)
    return torch.tensor([ids], dtype=torch.long), torch.tensor([attention_mask], dtype=torch.long)


def decode_rules(outputs: dict, max_rules: int) -> list[dict]:
    count = int(outputs["count_logits"].argmax(dim=-1).item())
    count = min(max(count, 0), max_rules)
    rules = []
    for idx in range(count):
        weekday = int(outputs["weekday_logits"][0, idx].argmax(dim=-1).item())
        weekday = min(max(weekday, 1), 7)
        start_h = int(outputs["start_h_logits"][0, idx].argmax(dim=-1).item())
        start_m = int(outputs["start_m_logits"][0, idx].argmax(dim=-1).item())
        end_h = int(outputs["end_h_logits"][0, idx].argmax(dim=-1).item())
        end_m = int(outputs["end_m_logits"][0, idx].argmax(dim=-1).item())
        rules.append({"weekday": weekday, "start": f"{start_h:02d}:{start_m:02d}", "end": f"{end_h:02d}:{end_m:02d}"})
    return rules


def to_rule_tuple(rule: dict) -> tuple[int, str, str]:
    return int(rule["weekday"]), str(rule["start"]), str(rule["end"])


def evaluate_effectiveness(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise RuntimeError("checkpoint is required when mode=eval")
    if not args.tokenizer_path:
        raise RuntimeError("tokenizer-path is required when mode=eval")

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = TimeLogicFormer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        max_len=config["max_len"],
        dropout=config["dropout"],
        max_rules=config["max_rules"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(args.tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    rows = read_jsonl(Path(args.eval_data))
    if not rows:
        raise RuntimeError("evaluation data is empty")

    sample_count = len(rows)
    count_ok = 0
    count_abs_error = 0
    sample_exact_ordered = 0
    sample_exact_set = 0
    tp = 0
    fp = 0
    fn = 0
    field_total = 0
    weekday_ok = 0
    start_ok = 0
    end_ok = 0

    with torch.no_grad():
        for row in rows:
            true_rules = row["label"]["forbidden"]
            input_ids, attention_mask = encode_text_with_tokenizer(row["text"], tokenizer_data)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask)
            pred_rules = decode_rules(outputs, max_rules=config["max_rules"])

            if len(pred_rules) == len(true_rules):
                count_ok += 1
            count_abs_error += abs(len(pred_rules) - len(true_rules))

            if pred_rules == true_rules:
                sample_exact_ordered += 1

            true_set = set(to_rule_tuple(x) for x in true_rules)
            pred_set = set(to_rule_tuple(x) for x in pred_rules)
            if true_set == pred_set:
                sample_exact_set += 1

            tp += len(true_set & pred_set)
            fp += len(pred_set - true_set)
            fn += len(true_set - pred_set)

            aligned = min(len(true_rules), len(pred_rules))
            for i in range(aligned):
                field_total += 1
                if int(pred_rules[i]["weekday"]) == int(true_rules[i]["weekday"]):
                    weekday_ok += 1
                if str(pred_rules[i]["start"]) == str(true_rules[i]["start"]):
                    start_ok += 1
                if str(pred_rules[i]["end"]) == str(true_rules[i]["end"]):
                    end_ok += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    report = {
        "device": str(device),
        "samples": sample_count,
        "count_accuracy": round(count_ok / sample_count, 4),
        "count_mae": round(count_abs_error / sample_count, 4),
        "sample_exact_match_ordered": round(sample_exact_ordered / sample_count, 4),
        "sample_exact_match_set": round(sample_exact_set / sample_count, 4),
        "rule_micro_precision": round(precision, 4),
        "rule_micro_recall": round(recall, 4),
        "rule_micro_f1": round(f1, 4),
        "weekday_accuracy_aligned": round(weekday_ok / field_total, 4) if field_total else 0.0,
        "start_accuracy_aligned": round(start_ok / field_total, 4) if field_total else 0.0,
        "end_accuracy_aligned": round(end_ok / field_total, 4) if field_total else 0.0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def train(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(data_path)
    if len(rows) < 10:
        raise RuntimeError("Need at least 10 samples to train")

    rng = random.Random(args.seed)
    train_rows, valid_rows = split_rows(rows, args.valid_ratio, rng)

    tokenizer = CharTokenizer(vocab_size=args.vocab_size, max_len=args.max_len)
    tokenizer.fit([x["text"] for x in train_rows])

    train_ds = RuleDataset(train_rows, tokenizer, max_rules=args.max_rules)
    valid_ds = RuleDataset(valid_rows, tokenizer, max_rules=args.max_rules)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    device = choose_device(args.device)
    model = TimeLogicFormer(
        vocab_size=args.vocab_size,
        d_model=args.hidden_dim,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        dim_ff=args.ff_dim,
        max_len=args.max_len,
        dropout=args.dropout,
        max_rules=args.max_rules,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
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
    tokenizer_path = out_dir / "tokenizer.json"

    print(f"device={device.type} train_samples={len(train_rows)} valid_samples={len(valid_rows)}")

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
        valid_loss = evaluate(
            model,
            valid_loader,
            device,
            max_rules=args.max_rules,
            label_smoothing=args.label_smoothing,
        )
        print(f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": args.vocab_size,
                        "d_model": args.hidden_dim,
                        "nhead": args.num_heads,
                        "num_layers": args.num_layers,
                        "dim_ff": args.ff_dim,
                        "max_len": args.max_len,
                        "dropout": args.dropout,
                        "max_rules": args.max_rules,
                    },
                },
                best_path,
            )

    with tokenizer_path.open("w", encoding="utf-8") as f:
        json.dump({"max_len": args.max_len, "char2id": tokenizer.char2id}, f, ensure_ascii=False)

    print(f"best_valid_loss={best_valid:.4f}")
    print(f"saved_model={best_path}")
    print(f"saved_tokenizer={tokenizer_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    p.add_argument("--data", type=str, default="synthetic_blacklist.jsonl")
    p.add_argument("--eval-data", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--tokenizer-path", type=str, default="")
    p.add_argument("--out-dir", type=str, default="artifacts")
    p.add_argument("--device", type=str, choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--vocab-size", type=int, default=7000)
    p.add_argument("--max-len", type=int, default=120)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-rules", type=int, default=6)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.mode == "eval":
        evaluate_effectiveness(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
