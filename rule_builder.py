from typing import Dict, List, Tuple

from tagging_utils import parse_day, parse_time


def extract_spans(tokens: List[str], tags: List[str]) -> List[Dict[str, object]]:
    spans: List[Dict[str, object]] = []
    current: Dict[str, object] | None = None

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current is not None:
                spans.append(current)
            current = {"type": tag[2:], "tokens": [token]}
        elif tag.startswith("I-") and current is not None and current["type"] == tag[2:]:
            current["tokens"].append(token)
        else:
            if current is not None:
                spans.append(current)
                current = None

    if current is not None:
        spans.append(current)
    return spans


def _join_wordpieces(tokens: List[str]) -> str:
    parts: List[str] = []
    for tok in tokens:
        if tok.startswith("##") and parts:
            parts[-1] = parts[-1] + tok[2:]
        else:
            parts.append(tok)
    return " ".join(parts).replace(" .", ".").strip()


def build_rules(spans: List[Dict[str, object]]) -> List[Dict[str, object]]:
    days: List[int] = []
    starts: List[str] = []
    ends: List[str] = []
    polarity = 1

    for span in spans:
        span_type = str(span["type"])
        raw_text = _join_wordpieces(list(span["tokens"]))
        if span_type == "DAY":
            parsed = parse_day(raw_text)
            if parsed is not None:
                days.append(parsed)
        elif span_type == "START":
            parsed = parse_time(raw_text)
            if parsed is not None:
                starts.append(parsed)
        elif span_type == "END":
            parsed = parse_time(raw_text)
            if parsed is not None:
                ends.append(parsed)
        elif span_type == "POLARITY":
            polarity = 1

    n = min(len(days), len(starts), len(ends))
    rules = [{"weekday": days[i], "start": starts[i], "end": ends[i], "polarity": polarity} for i in range(n)]
    unique = {(r["weekday"], r["start"], r["end"], r["polarity"]): r for r in rules}
    return [unique[k] for k in sorted(unique.keys())]


def build_rules_from_tags(tokens: List[str], tags: List[str]) -> List[Dict[str, object]]:
    spans = extract_spans(tokens, tags)
    return build_rules(spans)
