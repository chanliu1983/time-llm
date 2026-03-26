import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from rule_builder import build_rules_from_tags
from tagging_utils import ID2TAG, NUM_TAGS, align_char_labels_to_tokens, build_char_labels


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class RuleDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_len: int = 128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        text = row["text"]
        rules = row["label"]["forbidden"]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offsets = encoding["offset_mapping"].squeeze(0).tolist()
        char_labels = build_char_labels(text, rules)
        labels = align_char_labels_to_tokens(offsets, char_labels, attention_mask.tolist())
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class TimeLogicTagger(nn.Module):
    def __init__(self, backbone: str, dropout: float = 0.1):
        super().__init__()
        self.backbone_model = AutoModel.from_pretrained(backbone)
        hidden = self.backbone_model.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, NUM_TAGS)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.backbone_model(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state)
        return self.classifier(h)


TimeLogicFormer = TimeLogicTagger


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(logits.view(-1, NUM_TAGS), labels.view(-1))


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


def evaluate(model: TimeLogicTagger, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = compute_loss(logits, batch["labels"])
            total += float(loss.item())
            batches += 1
    model.train()
    if batches == 0:
        return 0.0
    return total / batches


def encode_text_with_tokenizer(text: str, tokenizer, max_len: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"], encoding["offset_mapping"]


def decode_rules(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer, max_rules: int = 6) -> list[dict]:
    pred_ids = logits.argmax(dim=-1)[0].detach().cpu().tolist()
    ids = input_ids[0].detach().cpu().tolist()
    mask = attention_mask[0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    tags = []
    clean_tokens = []
    for tok, tid, m in zip(tokens, pred_ids, mask):
        if m == 0:
            continue
        if tok in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
            continue
        clean_tokens.append(tok)
        tags.append(ID2TAG.get(tid, "O"))
    rules = build_rules_from_tags(clean_tokens, tags)
    return rules[:max_rules]


def to_rule_tuple(rule: dict) -> tuple[int, str, str]:
    return int(rule["weekday"]), str(rule["start"]), str(rule["end"])


def evaluate_effectiveness(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise RuntimeError("checkpoint is required when mode=eval")

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = TimeLogicTagger(
        backbone=config["backbone"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["backbone"], use_fast=True)
    max_len = int(config.get("max_len", 128))

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
            input_ids, attention_mask, _ = encode_text_with_tokenizer(row["text"], tokenizer, max_len=max_len)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask)
            pred_rules = decode_rules(logits, input_ids, attention_mask, tokenizer, max_rules=config["max_rules"])

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

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    train_ds = RuleDataset(train_rows, tokenizer, max_len=args.max_len)
    valid_ds = RuleDataset(valid_rows, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    device = choose_device(args.device)
    model = TimeLogicTagger(
        backbone=args.backbone,
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
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = compute_loss(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()
            total += float(loss.item())
            batches += 1
        train_loss = total / max(1, batches)
        valid_loss = evaluate(model, valid_loader, device)
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
                        "max_len": args.max_len,
                        "num_tags": NUM_TAGS,
                    },
                },
                best_path,
            )

    print(f"best_valid_loss={best_valid:.4f}")
    print(f"saved_model={best_path}")


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
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--encoder-lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--max-rules", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
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
