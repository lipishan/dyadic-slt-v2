"""Guards on the frozen split file (plan section 0 and pitfall 19)."""
import csv
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT = os.path.join(ROOT, "splits", "dyadic_split_v1.csv")
META = os.path.join(ROOT, "splits", "dyadic_split_v1.meta.json")


def rows():
    with open(SPLIT, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_each_conversation_exactly_once_and_valid_labels():
    r = rows()
    counts = Counter(x["video"] for x in r)
    assert all(n == 1 for n in counts.values()), [v for v, n in counts.items() if n > 1]
    assert {x["split"] for x in r} == {"train", "valid", "test"}


def test_no_no_sub_duplicates():
    vids = {x["video"] for x in rows()}
    assert not any(v.endswith("_no_sub") for v in vids)


def test_frozen_counts_match_meta():
    meta = json.load(open(META, encoding="utf-8"))
    c = Counter(x["split"] for x in rows())
    for line in meta["report"][1:4]:
        name, rest = line.split(":", 1)
        assert c[name.strip()] == int(rest.split()[0])
    assert c["test"] >= meta["min_test"]
    assert all(x["unseen_guest_test"] == "True" for x in rows() if x["split"] == "test") or \
        "Unseen-guest" in meta["report"][-1]
