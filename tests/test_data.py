import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import random
from generate_synthetic_blacklist_data import random_weekday_alias, diversify_text

def test_weekday_alias_includes_tues():
    rng = random.Random(0)
    aliases = {random_weekday_alias("Tuesday", rng) for _ in range(200)}
    assert "Tues" in aliases, f"Expected 'Tues' in aliases, got {aliases}"

def test_weekday_alias_includes_thurs():
    rng = random.Random(0)
    aliases = {random_weekday_alias("Thursday", rng) for _ in range(200)}
    assert "Thurs" in aliases

def test_weekday_alias_includes_weds():
    rng = random.Random(0)
    aliases = {random_weekday_alias("Wednesday", rng) for _ in range(200)}
    assert "Weds" in aliases

def test_weekday_alias_lowercase():
    rng = random.Random(0)
    aliases = {random_weekday_alias("Monday", rng) for _ in range(200)}
    assert "monday" in aliases

def test_diversify_text_12h_format():
    rng = random.Random(42)
    seen_12h = False
    for _ in range(500):
        result = diversify_text("Forbid access on Monday from 08:00 to 10:00.", rng)
        if "8am" in result or "8 AM" in result or "8:00am" in result:
            seen_12h = True
            break
    assert seen_12h, "Expected 12h format variant to appear in diversify_text output"
