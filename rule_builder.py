from typing import Dict, List

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
    polarity = 1
    active_days: List[int] = []
    pending_starts: List[str] = []
    rules: List[Dict[str, object]] = []

    def add_rule(day: int, start: str, end: str):
        rules.append({"weekday": day, "start": start, "end": end, "polarity": polarity})

    for span in spans:
        span_type = str(span["type"])
        raw_text = _join_wordpieces(list(span["tokens"]))

        if span_type == "POLARITY":
            polarity = 1
            continue

        if span_type == "DAY":
            parsed_day = parse_day(raw_text)
            if parsed_day is not None and parsed_day not in active_days:
                active_days.append(parsed_day)
            continue

        if span_type == "ALLDAY":
            if active_days:
                for day in active_days:
                    add_rule(day, "00:00", "23:59")
                pending_starts.clear()
            continue

        if span_type == "START":
            parsed_start = parse_time(raw_text)
            if parsed_start is not None:
                pending_starts.append(parsed_start)
            continue

        if span_type == "END":
            parsed_end = parse_time(raw_text)
            if parsed_end is None or not active_days:
                continue
            if pending_starts:
                start_time = pending_starts.pop(0)
            else:
                start_time = "00:00"
            for day in active_days:
                add_rule(day, start_time, parsed_end)
            continue

    if active_days and pending_starts:
        for start_time in pending_starts:
            for day in active_days:
                add_rule(day, start_time, "23:59")
    elif active_days and not rules:
        for day in active_days:
            add_rule(day, "00:00", "23:59")

    unique = {(r["weekday"], r["start"], r["end"], r["polarity"]): r for r in rules}
    return [unique[k] for k in sorted(unique.keys())]


def build_rules_from_tags(tokens: List[str], tags: List[str]) -> List[Dict[str, object]]:
    spans = extract_spans(tokens, tags)
    return build_rules(spans)
