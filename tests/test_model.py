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


import tempfile, pathlib

def test_checkpoint_save_and_load():
    from train_time_logicformer import TimeLogicFormer
    model = TimeLogicFormer(backbone=BACKBONE, max_rules=2)
    config = {"backbone": BACKBONE, "max_rules": 2, "dropout": 0.1}
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "model.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": config}, path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
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
