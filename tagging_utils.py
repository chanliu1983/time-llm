import re
from typing import Dict, List, Tuple


TAG2ID = {
    "O": 0,
    "B-DAY": 1,
    "I-DAY": 2,
    "B-START": 3,
    "I-START": 4,
    "B-END": 5,
    "I-END": 6,
    "B-POLARITY": 7,
}

ID2TAG = {v: k for k, v in TAG2ID.items()}
NUM_TAGS = len(TAG2ID)

WEEKDAY_ALIASES: Dict[int, List[str]] = {
    1: ["monday", "mon", "mon.", "mondays", "every monday", "on mondays"],
    2: ["tuesday", "tue", "tues", "tue.", "tuesdays", "every tuesday", "on tuesdays"],
    3: ["wednesday", "wed", "weds", "wed.", "wednesdays", "every wednesday", "on wednesdays"],
    4: ["thursday", "thu", "thur", "thurs", "thu.", "thursdays", "every thursday", "on thursdays"],
    5: ["friday", "fri", "fri.", "fridays", "every friday", "on fridays"],
    6: ["saturday", "sat", "sat.", "saturdays", "every saturday", "on saturdays"],
    7: ["sunday", "sun", "sun.", "sundays", "every sunday", "on sundays"],
}

POLARITY_TERMS = [
    "forbid",
    "forbidden",
    "disallow",
    "block",
    "ban",
    "prohibit",
    "prohibited",
    "not allowed",
    "no access",
]


def parse_day(text: str) -> int | None:
    lower = text.lower().strip()
    for day, aliases in WEEKDAY_ALIASES.items():
        if lower in aliases:
            return day
    token = re.sub(r"[^a-z]", "", lower)
    for day, aliases in WEEKDAY_ALIASES.items():
        for alias in aliases:
            if token == re.sub(r"[^a-z]", "", alias):
                return day
    return None


def to_hhmm_from_12h(text: str) -> str | None:
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*([ap]m)", text.strip().lower())
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or "00")
    ampm = match.group(3)
    if hour < 1 or hour > 12 or minute < 0 or minute > 59:
        return None
    if ampm == "am":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12
    return f"{hour:02d}:{minute:02d}"


def parse_time(text: str) -> str | None:
    raw = text.strip().lower()
    match24 = re.fullmatch(r"(\d{1,2}):(\d{2})", raw)
    if match24:
        h = int(match24.group(1))
        m = int(match24.group(2))
        if 0 <= h <= 23 and 0 <= m <= 59:
            return f"{h:02d}:{m:02d}"
    converted = to_hhmm_from_12h(raw.replace(" ", ""))
    if converted:
        return converted
    converted = to_hhmm_from_12h(raw)
    if converted:
        return converted
    return None


def time_variants(hhmm: str) -> List[str]:
    h, m = hhmm.split(":")
    hour = int(h)
    minute = int(m)
    variants = {f"{hour:02d}:{minute:02d}", f"{hour}:{minute:02d}"}
    suffix = "am" if hour < 12 else "pm"
    h12 = hour if 1 <= hour <= 12 else (12 if hour in (0, 12) else hour - 12)
    variants.add(f"{h12}{suffix}")
    variants.add(f"{h12} {suffix}")
    variants.add(f"{h12}:{minute:02d}{suffix}")
    variants.add(f"{h12}:{minute:02d} {suffix}")
    variants.add(f"{h12} {suffix.upper()}")
    return sorted(variants, key=len, reverse=True)


def find_non_overlapping(text: str, candidates: List[str], used: List[Tuple[int, int]]) -> Tuple[int, int] | None:
    lower = text.lower()
    for cand in candidates:
        for m in re.finditer(re.escape(cand.lower()), lower):
            s, e = m.span()
            if not any(not (e <= us or s >= ue) for us, ue in used):
                return s, e
    return None


def build_char_labels(text: str, rules: List[dict]) -> List[str]:
    labels = ["O"] * len(text)
    used: List[Tuple[int, int]] = []

    for term in POLARITY_TERMS:
        for m in re.finditer(re.escape(term), text.lower()):
            s, e = m.span()
            if s < e:
                labels[s] = "B-POLARITY"
                for i in range(s + 1, e):
                    labels[i] = "I-END"

    for rule in rules:
        day = int(rule["weekday"])
        day_match = find_non_overlapping(text, WEEKDAY_ALIASES.get(day, []), used)
        if day_match:
            s, e = day_match
            labels[s] = "B-DAY"
            for i in range(s + 1, e):
                labels[i] = "I-DAY"
            used.append((s, e))

        start = str(rule["start"])
        start_match = find_non_overlapping(text, time_variants(start), used)
        if start_match:
            s, e = start_match
            labels[s] = "B-START"
            for i in range(s + 1, e):
                labels[i] = "I-START"
            used.append((s, e))

        end = str(rule["end"])
        end_match = find_non_overlapping(text, time_variants(end), used)
        if end_match:
            s, e = end_match
            labels[s] = "B-END"
            for i in range(s + 1, e):
                labels[i] = "I-END"
            used.append((s, e))

    return labels


def align_char_labels_to_tokens(
    offsets: List[Tuple[int, int]],
    char_labels: List[str],
    attention_mask: List[int],
) -> List[int]:
    out = []
    prev_type = "O"
    for i, (s, e) in enumerate(offsets):
        if i >= len(attention_mask) or attention_mask[i] == 0 or s == e:
            out.append(-100)
            prev_type = "O"
            continue
        segment = char_labels[s:e] if s < len(char_labels) else []
        tag = "O"
        for c in segment:
            if c != "O":
                tag = c
                break
        if tag.startswith("I-"):
            t = tag[2:]
            if prev_type == t:
                tag = f"I-{t}"
            else:
                tag = f"B-{t}"
        elif tag.startswith("B-"):
            t = tag[2:]
            if prev_type == t:
                tag = f"I-{t}"
        prev_type = tag[2:] if tag != "O" else "O"
        out.append(TAG2ID.get(tag, 0))
    return out
