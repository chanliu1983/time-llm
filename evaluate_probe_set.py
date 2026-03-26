import json
import argparse
from pathlib import Path

import torch

from train_time_logicformer import TimeLogicFormer, decode_rules, encode_text_with_tokenizer


def load_model(checkpoint_path: Path, tokenizer_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    model = TimeLogicFormer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"],
        max_len=cfg["max_len"],
        dropout=cfg["dropout"],
        max_rules=cfg["max_rules"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with tokenizer_path.open("r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
    return model, tokenizer_data, cfg["max_rules"]


def predict(model, tokenizer_data, max_rules: int, text: str, device: torch.device):
    with torch.no_grad():
        input_ids, attention_mask = encode_text_with_tokenizer(text, tokenizer_data)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(input_ids, attention_mask)
        return decode_rules(outputs, max_rules=max_rules)


def norm(rules):
    return sorted(rules, key=lambda x: (x["weekday"], x["start"], x["end"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-checkpoint", type=str, default="/Users/leiliu/Documents/time-llm/artifacts_mps_large/best_model.pt")
    parser.add_argument("--old-tokenizer", type=str, default="/Users/leiliu/Documents/time-llm/artifacts_mps_large/tokenizer.json")
    parser.add_argument("--new-checkpoint", type=str, default="/Users/leiliu/Documents/time-llm/artifacts_mps_diverse/best_model.pt")
    parser.add_argument("--new-tokenizer", type=str, default="/Users/leiliu/Documents/time-llm/artifacts_mps_diverse/tokenizer.json")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    old_model, old_tok, old_max_rules = load_model(
        Path(args.old_checkpoint),
        Path(args.old_tokenizer),
        device,
    )
    new_model, new_tok, new_max_rules = load_model(
        Path(args.new_checkpoint),
        Path(args.new_tokenizer),
        device,
    )

    probes = [
        (
            "please lock monday for the entire day and do not allow fri 03:30-04:00",
            [{"weekday": 1, "start": "00:00", "end": "23:59"}, {"weekday": 5, "start": "03:30", "end": "04:00"}],
        ),
        (
            "tuesday should be blocked from 09:00 to 10:15",
            [{"weekday": 2, "start": "09:00", "end": "10:15"}],
        ),
        (
            "ban access on wed plus sat between 14:00 and 16:00",
            [{"weekday": 3, "start": "14:00", "end": "16:00"}, {"weekday": 6, "start": "14:00", "end": "16:00"}],
        ),
        (
            "allow thursday except from 05:00 to 06:00",
            [{"weekday": 4, "start": "05:00", "end": "06:00"}],
        ),
        (
            "disallow from monday to wednesday from 11:00 to 12:00 except tuesday",
            [{"weekday": 1, "start": "11:00", "end": "12:00"}, {"weekday": 3, "start": "11:00", "end": "12:00"}],
        ),
        (
            "friday all day blocked and also friday from 12:00 to 15:00 forbidden",
            [{"weekday": 5, "start": "00:00", "end": "23:59"}, {"weekday": 5, "start": "12:00", "end": "15:00"}],
        ),
    ]

    old_exact = 0
    new_exact = 0
    sample_outputs = []

    for text, label in probes:
        old_pred = norm(predict(old_model, old_tok, old_max_rules, text, device))
        new_pred = norm(predict(new_model, new_tok, new_max_rules, text, device))
        gold = norm(label)
        old_exact += int(old_pred == gold)
        new_exact += int(new_pred == gold)
        sample_outputs.append({"text": text, "target": gold, "old_pred": old_pred, "new_pred": new_pred})

    result = {"probe_samples": len(probes), "old_exact": old_exact, "new_exact": new_exact, "samples": sample_outputs}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
