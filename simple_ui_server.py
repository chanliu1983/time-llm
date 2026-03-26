import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

from train_time_logicformer import TimeLogicFormer, choose_device, decode_rules, encode_text_with_tokenizer
from transformers import AutoTokenizer


HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TimeLogicFormer Demo</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px; margin: 32px auto; padding: 0 16px; }
    h1 { margin-bottom: 8px; }
    p { color: #444; }
    textarea { width: 100%; min-height: 120px; padding: 12px; font-size: 16px; }
    button { margin-top: 12px; padding: 10px 16px; font-size: 15px; cursor: pointer; }
    pre { background: #111; color: #eee; padding: 14px; border-radius: 8px; overflow: auto; }
    .row { display: flex; gap: 10px; align-items: center; margin-top: 8px; }
    .badge { padding: 2px 8px; border-radius: 999px; background: #f0f0f0; font-size: 12px; }
  </style>
</head>
<body>
  <h1>TimeLogicFormer UI</h1>
  <p>Enter a sentence and get strict blacklist JSON output.</p>
  <div class="row">
    <span class="badge" id="device">device: loading</span>
    <span class="badge" id="model">model: loading</span>
  </div>
  <textarea id="text">Forbid access on Tuesday from 03:00 to 04:00 and block Wednesday all day.</textarea>
  <br />
  <button onclick="runPredict()">Predict</button>
  <pre id="out">{"forbidden":[]}</pre>
  <script>
    async function fetchMeta() {
      const res = await fetch('/meta');
      const data = await res.json();
      document.getElementById('device').innerText = 'device: ' + data.device;
      document.getElementById('model').innerText = 'max_rules: ' + data.max_rules;
    }
    async function runPredict() {
      const text = document.getElementById('text').value;
      const res = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text})
      });
      const data = await res.json();
      document.getElementById('out').innerText = JSON.stringify(data, null, 2);
    }
    fetchMeta();
  </script>
</body>
</html>
"""


WEEKDAY_TO_ID = {
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
    "sunday": 7,
}


def normalize_time(raw: str) -> str | None:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", raw.strip())
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def find_weekdays(text: str) -> list[int]:
    found = set()
    lower = text.lower()
    for name, idx in WEEKDAY_TO_ID.items():
        if re.search(rf"\b{name}\b", lower):
            found.add(idx)
    return sorted(found)


def expand_day_range(start_day: int, end_day: int) -> list[int]:
    if start_day <= end_day:
        return list(range(start_day, end_day + 1))
    return list(range(start_day, 8)) + list(range(1, end_day + 1))


def parse_weekdays(clause: str) -> list[int]:
    lower = clause.lower()
    weekdays = set(find_weekdays(lower))
    range_match = re.search(
        r"\bfrom\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+to\s+"
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        lower,
    )
    if range_match:
        start_day = WEEKDAY_TO_ID[range_match.group(1)]
        end_day = WEEKDAY_TO_ID[range_match.group(2)]
        weekdays.update(expand_day_range(start_day, end_day))
    except_match = re.search(r"\bexcept\b(.+)$", lower)
    if except_match:
        excluded = set(find_weekdays(except_match.group(1)))
        weekdays = weekdays - excluded
    return sorted(weekdays)


def parse_time_windows(clause: str) -> list[tuple[str, str]]:
    lower = clause.lower()
    if "all day" in lower or "whole day" in lower or "entire day" in lower:
        return [("00:00", "23:59")]
    patterns = [
        r"\bfrom\s*(\d{1,2}:\d{2})\s*(?:to|-)\s*(\d{1,2}:\d{2})\b",
        r"\bbetween\s*(\d{1,2}:\d{2})\s*and\s*(\d{1,2}:\d{2})\b",
        r"\bexcept\s+from\s*(\d{1,2}:\d{2})\s*(?:to|-)\s*(\d{1,2}:\d{2})\b",
    ]
    windows = []
    for pattern in patterns:
        for m in re.finditer(pattern, lower):
            start = normalize_time(m.group(1))
            end = normalize_time(m.group(2))
            if start and end:
                windows.append((start, end))
    dedup = sorted(set(windows))
    return dedup


def clause_is_forbidden(clause: str) -> bool:
    lower = clause.lower()
    if any(token in lower for token in ["forbid", "disallow", "block", "ban", "prohibit", "prohibited", "not allowed", "no access"]):
        return True
    if "allow" in lower and "except" in lower:
        return True
    return False


def parse_rules_deterministic(text: str) -> list[dict]:
    normalized = text.replace("+", " and ")
    clauses = re.split(
        r"[.;]|\b(?:and|also)\s+(?=(?:forbid|disallow|block|ban|prohibit|prohibited|not allowed|no access))",
        normalized,
        flags=re.IGNORECASE,
    )
    rules = []
    for raw_clause in clauses:
        clause = raw_clause.strip()
        if not clause:
            continue
        if not clause_is_forbidden(clause):
            continue
        weekdays = parse_weekdays(clause)
        windows = parse_time_windows(clause)
        if not weekdays or not windows:
            continue
        for start, end in windows:
            for day in weekdays:
                rules.append({"weekday": day, "start": start, "end": end})
    unique = {(r["weekday"], r["start"], r["end"]): r for r in rules}
    ordered = [unique[k] for k in sorted(unique.keys())]
    return ordered


def merge_rules(parsed_rules: list[dict], model_rules: list[dict]) -> list[dict]:
    if parsed_rules:
        unique = {(r["weekday"], r["start"], r["end"]): r for r in parsed_rules}
        return [unique[k] for k in sorted(unique.keys())]
    unique = {(r["weekday"], r["start"], r["end"]): r for r in model_rules}
    return [unique[k] for k in sorted(unique.keys())]


class ModelService:
    def __init__(self, checkpoint_path: Path, device_arg: str):
        self.device = choose_device(device_arg)
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.model = TimeLogicFormer(
            backbone=self.config["backbone"],
            dropout=self.config["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["backbone"])

    def predict(self, text: str) -> dict:
        with torch.no_grad():
            input_ids, attention_mask, _ = encode_text_with_tokenizer(
                text, self.tokenizer, max_len=int(self.config.get("max_len", 128))
            )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            logits = self.model(input_ids, attention_mask)
            model_rules = decode_rules(
                logits,
                input_ids,
                attention_mask,
                self.tokenizer,
                max_rules=self.config["max_rules"],
            )
        return {"forbidden": merge_rules([], model_rules), "source": "model"}

    def meta(self) -> dict:
        return {"device": str(self.device), "max_rules": int(self.config["max_rules"])}


def make_handler(service: ModelService):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj: dict, status: int = 200):
            payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_html(self, html: str):
            payload = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/":
                self._send_html(HTML_PAGE)
                return
            if self.path == "/meta":
                self._send_json(service.meta())
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self):
            if self.path != "/predict":
                self._send_json({"error": "not found"}, status=404)
                return
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len)
            try:
                data = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"error": "invalid json"}, status=400)
                return
            text = str(data.get("text", "")).strip()
            if not text:
                self._send_json({"error": "text is required"}, status=400)
                return
            result = service.predict(text)
            self._send_json(result)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/Users/leiliu/Documents/time-llm/artifacts_mps_large/best_model.pt")
    parser.add_argument("--device", type=str, choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()

    service = ModelService(Path(args.checkpoint), args.device)
    handler = make_handler(service)
    server = HTTPServer((args.host, args.port), handler)
    print(f"http://{args.host}:{args.port}")
    print(json.dumps(service.meta(), ensure_ascii=False))
    server.serve_forever()


if __name__ == "__main__":
    main()
