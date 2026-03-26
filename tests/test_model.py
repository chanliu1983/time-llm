import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"


def test_model_instantiates():
    from train_time_logicformer import TimeLogicTagger
    model = TimeLogicTagger(backbone=BACKBONE)
    assert model is not None


def test_model_forward_shape():
    from train_time_logicformer import TimeLogicTagger
    from tagging_utils import NUM_TAGS
    from transformers import AutoTokenizer
    model = TimeLogicTagger(backbone=BACKBONE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    enc = tokenizer("block monday all day", return_tensors="pt",
                    max_length=128, padding="max_length", truncation=True)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    assert out.shape == (1, 128, NUM_TAGS)


def test_encode_text_with_tokenizer():
    from train_time_logicformer import encode_text_with_tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    ids, mask, offsets = encode_text_with_tokenizer("block monday", tokenizer)
    assert ids.shape[0] == 1
    assert mask.shape == ids.shape
    assert ids.dtype == torch.long
    assert offsets.shape[0] == 1


def test_rule_dataset_item_shape():
    from train_time_logicformer import RuleDataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    rows = [{"text": "block Monday all day", "label": {"forbidden": [{"weekday": 1, "start": "00:00", "end": "23:59"}]}}]
    ds = RuleDataset(rows, tokenizer, max_len=128)
    item = ds[0]
    assert item["input_ids"].shape == (128,)
    assert item["attention_mask"].shape == (128,)
    assert item["labels"].shape == (128,)
    assert (item["labels"] >= -100).all()


import tempfile, pathlib

def test_checkpoint_save_and_load():
    from train_time_logicformer import TimeLogicTagger
    model = TimeLogicTagger(backbone=BACKBONE)
    config = {"backbone": BACKBONE, "max_rules": 6, "dropout": 0.1, "max_len": 128, "num_tags": 8}
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "model.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": config}, path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        loaded = TimeLogicTagger(
            backbone=ckpt["config"]["backbone"],
            dropout=ckpt["config"]["dropout"],
        )
        loaded.load_state_dict(ckpt["model_state_dict"])
    assert loaded is not None


def test_decode_rules_runs():
    from train_time_logicformer import TimeLogicTagger, decode_rules
    from transformers import AutoTokenizer
    model = TimeLogicTagger(backbone=BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    enc = tokenizer("block monday from 08:00 to 10:00", return_tensors="pt", max_length=64, padding="max_length", truncation=True)
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
    rules = decode_rules(logits, enc["input_ids"], enc["attention_mask"], tokenizer, max_rules=6)
    assert isinstance(rules, list)
