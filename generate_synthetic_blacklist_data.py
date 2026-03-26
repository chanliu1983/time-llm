import argparse
import json
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List

from tagging_utils import build_char_labels

WEEKDAYS = [
    (1, "Monday"),
    (2, "Tuesday"),
    (3, "Wednesday"),
    (4, "Thursday"),
    (5, "Friday"),
    (6, "Saturday"),
    (7, "Sunday"),
]


@dataclass(frozen=True)
class Rule:
    weekday: int
    start: str
    end: str


def two_digits(v: int) -> str:
    return f"{v:02d}"


def fmt_time(hour: int, minute: int) -> str:
    return f"{two_digits(hour)}:{two_digits(minute)}"


def random_time_window(rng: random.Random) -> tuple[str, str]:
    start_hour = rng.randint(0, 22)
    start_minute = rng.choice([0, 15, 30, 45])
    max_end_hour = min(23, start_hour + rng.randint(1, 4))
    end_hour = rng.randint(start_hour, max_end_hour)
    end_minute = rng.choice([0, 15, 30, 45])
    if end_hour == start_hour and end_minute <= start_minute:
        end_minute = min(59, start_minute + 15)
    start = fmt_time(start_hour, start_minute)
    end = fmt_time(end_hour, end_minute)
    return start, end


def all_day_window() -> tuple[str, str]:
    return "00:00", "23:59"


def pick_weekday_name(day: int) -> str:
    for d, n in WEEKDAYS:
        if d == day:
            return n
    raise ValueError("Invalid weekday")


def join_day_names(days: List[int]) -> str:
    names = [pick_weekday_name(d) for d in days]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def rule_to_json(rule: Rule) -> dict:
    return {"weekday": rule.weekday, "start": rule.start, "end": rule.end}


def random_weekday_alias(name: str, rng: random.Random) -> str:
    aliases = {
        "Monday": ["Monday", "Mon", "Mon.", "monday", "every Monday", "on Mondays"],
        "Tuesday": ["Tuesday", "Tue", "Tues", "tuesday", "every Tuesday"],
        "Wednesday": ["Wednesday", "Wed", "Weds", "wednesday", "every Wednesday"],
        "Thursday": ["Thursday", "Thu", "Thurs", "thursday", "every Thursday"],
        "Friday": ["Friday", "Fri", "Fri.", "friday", "every Friday"],
        "Saturday": ["Saturday", "Sat", "Sat.", "saturday", "every Saturday"],
        "Sunday": ["Sunday", "Sun", "Sun.", "sunday", "every Sunday"],
    }
    return rng.choice(aliases.get(name, [name]))


def diversify_text(text: str, rng: random.Random) -> str:
    rewritten = text
    replacements = [
        ("Forbid", rng.choice(["Forbid", "Block", "Disallow", "Please block"])),
        ("forbid", rng.choice(["forbid", "block", "disallow"])),
        ("Disallow", rng.choice(["Disallow", "Forbid", "Block"])),
        ("Allow access on", rng.choice(["Access is allowed on", "You can allow on", "Allow on"])),
        ("between", rng.choice(["between", "from"])),
        (" and ", rng.choice([" and ", " plus ", " also ", ", "])),
    ]
    for old, new in replacements:
        rewritten = rewritten.replace(old, new)
    rewritten = rewritten.replace(" to ", rng.choice([" to ", " - ", " until "]))
    rewritten = rewritten.replace(" from ", rng.choice([" from ", " during ", " between "]))
    for _, weekday_name in WEEKDAYS:
        rewritten = rewritten.replace(weekday_name, random_weekday_alias(weekday_name, rng))
    if rng.random() < 0.5:
        rewritten = re.sub(r"\b0(\d):", r"\1:", rewritten)
    if rng.random() < 0.35:
        rewritten = rewritten.replace(",", rng.choice([",", " ,", " ;"]))
    if rng.random() < 0.3:
        rewritten = rewritten.replace(" and ", rng.choice([" and ", " + ", " & ", " also "]))
    if rng.random() < 0.25:
        rewritten = rewritten.replace("all day", rng.choice(["all day", "entire day", "whole day"]))
    if rng.random() < 0.2:
        rewritten = rng.choice(["please ", "policy: ", "rule says "]) + rewritten
    if rng.random() < 0.35:
        rewritten = rewritten.lower()
    if rng.random() < 0.25:
        rewritten = rewritten.rstrip(".")
    if rng.random() < 0.2:
        rewritten = "  " + rewritten + "  "
    if rng.random() < 0.1:
        rewritten = rewritten.replace("forbid", "forbbid").replace("disallow", "disalow")
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
        rewritten = re.sub(r"\b(\d{1,2}):(\d{2})\b", to_12h, rewritten)
    return rewritten.strip()


def pattern_single_time(rng: random.Random) -> tuple[str, List[Rule]]:
    day = rng.randint(1, 7)
    start, end = random_time_window(rng)
    verb = rng.choice(["forbid", "disallow", "ban", "not allowed"])
    day_name = pick_weekday_name(day)
    if verb == "not allowed":
        text = f"Access is not allowed on {day_name} from {start} to {end}."
    else:
        text = f"{verb.capitalize()} access on {day_name} from {start} to {end}."
    return text, [Rule(day, start, end)]


def pattern_single_full_day(rng: random.Random) -> tuple[str, List[Rule]]:
    day = rng.randint(1, 7)
    start, end = all_day_window()
    phrasing = rng.choice(
        [
            f"No access all day on {pick_weekday_name(day)}.",
            f"Block {pick_weekday_name(day)} for the whole day.",
            f"Access is prohibited on {pick_weekday_name(day)} all day.",
        ]
    )
    return phrasing, [Rule(day, start, end)]


def pattern_multi_list_same_time(rng: random.Random) -> tuple[str, List[Rule]]:
    count = rng.randint(2, 4)
    days = sorted(rng.sample(range(1, 8), count))
    start, end = random_time_window(rng)
    day_text = join_day_names(days)
    text = f"Forbid access on {day_text} between {start} and {end}."
    rules = [Rule(d, start, end) for d in days]
    return text, rules


def pattern_range_with_exception(rng: random.Random) -> tuple[str, List[Rule]]:
    start_day = rng.randint(1, 5)
    end_day = rng.randint(start_day + 1, 7)
    all_days = list(range(start_day, end_day + 1))
    exclude = rng.choice(all_days)
    start, end = random_time_window(rng)
    text = (
        f"Disallow from {pick_weekday_name(start_day)} to {pick_weekday_name(end_day)} "
        f"from {start} to {end}, except {pick_weekday_name(exclude)}."
    )
    rules = [Rule(d, start, end) for d in all_days if d != exclude]
    return text, rules


def pattern_negation_allow_forbidden(rng: random.Random) -> tuple[str, List[Rule]]:
    day = rng.randint(1, 7)
    start, end = random_time_window(rng)
    text = (
        f"Allow access on {pick_weekday_name(day)} except from {start} to {end}."
    )
    return text, [Rule(day, start, end)]


def pattern_mixed_two_windows(rng: random.Random) -> tuple[str, List[Rule]]:
    day1, day2 = rng.sample(range(1, 8), 2)
    s1, e1 = random_time_window(rng)
    if rng.random() < 0.45:
        s2, e2 = all_day_window()
        clause2 = f"and block {pick_weekday_name(day2)} all day"
    else:
        s2, e2 = random_time_window(rng)
        clause2 = f"and disallow {pick_weekday_name(day2)} from {s2} to {e2}"
    text = f"Forbid {pick_weekday_name(day1)} from {s1} to {e1}, {clause2}."
    rules = [Rule(day1, s1, e1), Rule(day2, s2, e2)]
    return text, rules


def pattern_three_segments(rng: random.Random) -> tuple[str, List[Rule]]:
    days = rng.sample(range(1, 8), 3)
    s1, e1 = random_time_window(rng)
    s2, e2 = random_time_window(rng)
    s3, e3 = all_day_window() if rng.random() < 0.5 else random_time_window(rng)
    text = (
        f"Disallow {pick_weekday_name(days[0])} from {s1} to {e1}, "
        f"also forbid {pick_weekday_name(days[1])} from {s2} to {e2}, "
        f"and block {pick_weekday_name(days[2])} {'all day' if (s3, e3)==('00:00','23:59') else f'from {s3} to {e3}'}."
    )
    return text, [Rule(days[0], s1, e1), Rule(days[1], s2, e2), Rule(days[2], s3, e3)]


PATTERN_MAP: Dict[str, Callable[[random.Random], tuple[str, List[Rule]]]] = {
    "single_time": pattern_single_time,
    "single_full_day": pattern_single_full_day,
    "multi_list_same_time": pattern_multi_list_same_time,
    "range_with_exception": pattern_range_with_exception,
    "negation_allow_forbidden": pattern_negation_allow_forbidden,
    "mixed_two_windows": pattern_mixed_two_windows,
    "three_segments": pattern_three_segments,
}

DEFAULT_RATIOS: Dict[str, float] = {
    "single_time": 0.2,
    "single_full_day": 0.2,
    "multi_list_same_time": 0.2,
    "range_with_exception": 0.2,
    "negation_allow_forbidden": 0.1,
    "mixed_two_windows": 0.1,
    "three_segments": 0.1,
}


def generate_sample(rng: random.Random, pattern_name: str | None = None) -> dict:
    if pattern_name is None:
        pattern = rng.choice(list(PATTERN_MAP.values()))
    else:
        pattern = PATTERN_MAP[pattern_name]
    text, rules = pattern(rng)
    if rng.random() < 0.85:
        text = diversify_text(text, rng)
    payload = {"text": text, "label": {"forbidden": [rule_to_json(r) for r in rules]}}
    return payload


def normalize(sample: dict) -> dict:
    forbidden = sample["label"]["forbidden"]
    forbidden = sorted(forbidden, key=lambda x: (x["weekday"], x["start"], x["end"]))
    return {"text": sample["text"], "label": {"forbidden": forbidden}}


def build_token_supervision(text: str, forbidden: List[dict]) -> dict:
    tokens = []
    spans = []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    char_labels = build_char_labels(text, forbidden)
    tags = []
    prev = "O"
    for s, e in spans:
        tag = "O"
        for c in char_labels[s:e]:
            if c != "O":
                tag = c
                break
        if tag.startswith("I-"):
            base = tag[2:]
            tag = f"I-{base}" if prev == base else f"B-{base}"
        prev = tag[2:] if tag != "O" else "O"
        tags.append(tag)
    return {"tokens": tokens, "tags": tags}


def parse_ratios(ratios_text: str) -> Dict[str, float]:
    if not ratios_text.strip():
        ratios = dict(DEFAULT_RATIOS)
        total = sum(ratios.values())
        return {k: v / total for k, v in ratios.items()}
    ratios: Dict[str, float] = {}
    for item in ratios_text.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid ratio item: {chunk}")
        key, value = chunk.split("=", 1)
        key = key.strip()
        if key not in PATTERN_MAP:
            raise ValueError(f"Unknown pattern key: {key}")
        parsed = float(value.strip())
        if parsed < 0:
            raise ValueError(f"Negative ratio for pattern: {key}")
        ratios[key] = parsed
    if not ratios:
        raise ValueError("No valid ratios provided")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Sum of ratios must be positive")
    for key in PATTERN_MAP:
        ratios.setdefault(key, 0.0)
    return {k: v / total for k, v in ratios.items()}


def allocate_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    raw = {k: ratios[k] * total for k in PATTERN_MAP}
    counts = {k: int(raw[k]) for k in PATTERN_MAP}
    assigned = sum(counts.values())
    remaining = total - assigned
    if remaining > 0:
        ranking = sorted(PATTERN_MAP.keys(), key=lambda k: raw[k] - counts[k], reverse=True)
        for i in range(remaining):
            counts[ranking[i % len(ranking)]] += 1
    return counts


def build_pattern_schedule(num_samples: int, rng: random.Random, ratios: Dict[str, float]) -> List[str]:
    counts = allocate_counts(num_samples, ratios)
    schedule: List[str] = []
    for name in PATTERN_MAP:
        schedule.extend([name] * counts[name])
    rng.shuffle(schedule)
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="synthetic_blacklist.jsonl")
    parser.add_argument("--mode", type=str, choices=["random", "balanced"], default="random")
    parser.add_argument("--ratios", type=str, default="")
    parser.add_argument("--emit-token-labels", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ratios = parse_ratios(args.ratios)

    with open(args.output, "w", encoding="utf-8") as f:
        if args.mode == "balanced":
            schedule = build_pattern_schedule(args.num_samples, rng, ratios)
            for name in schedule:
                sample = normalize(generate_sample(rng, pattern_name=name))
                if args.emit_token_labels:
                    sample["token_supervision"] = build_token_supervision(sample["text"], sample["label"]["forbidden"])
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            for _ in range(args.num_samples):
                sample = normalize(generate_sample(rng))
                if args.emit_token_labels:
                    sample["token_supervision"] = build_token_supervision(sample["text"], sample["label"]["forbidden"])
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
