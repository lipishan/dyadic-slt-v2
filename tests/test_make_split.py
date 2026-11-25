"""Tests for data/make_split.py on made-up data: 60 conversations, host 'Alex' in all,
one guest ('Gina') in two conversations."""
import csv
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "data", "make_split.py")
TURN_HEAD = ["video", "line", "turn_id", "signer", "confidence", "likely_backchannel", "words", "duration"]


def make(tmp, n=60):
    turns, names, inv = [], {}, []
    for i in range(n):
        v = f"V{i:02d}"
        guest = "Gina" if i in (3, 17) else f"Guest{i}"
        for k in range(10 + i % 7):
            who = "Alex" if k % 2 == 0 else guest
            turns.append([v, k + 1, k + 1, who, "plausible", "False", 50 if k % 5 == 0 else 10, 5.0])
        turns.append([v, 99, 99, "Alex and " + guest, "plausible", "False", 10, 2.0])   # several people
        for who in ("Alex", guest):
            names.setdefault(who, set()).add(v)
        inv.append([v, "both" if i % 3 else "ann2_only"])
    with open(tmp / "turns.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(TURN_HEAD); w.writerows(turns)
    with open(tmp / "names.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["person_id", "name", "videos"])
        for nm, vs in names.items():
            w.writerow([nm, nm, " ".join(sorted(vs))])
    with open(tmp / "inv.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video", "kind"]); w.writerows(inv)


def run(tmp, *extra):
    return subprocess.run([sys.executable, SCRIPT, "--turns", str(tmp / "turns.csv"), "--names",
                           str(tmp / "names.csv"), "--inventory", str(tmp / "inv.csv"),
                           "--out", str(tmp / "s" / "split.csv"), "--tries", "300", *extra],
                          capture_output=True, text=True)


def read(tmp):
    return {r["video"]: r["split"] for r in csv.DictReader(open(tmp / "s" / "split.csv"))}


def test_split_rules(tmp_path):
    make(tmp_path)
    res = run(tmp_path)
    assert res.returncode == 0, res.stderr
    sp = read(tmp_path)
    assert len(sp) == 60 and set(sp.values()) == {"train", "valid", "test"}       # every conversation once
    assert sp["V03"] == sp["V17"]                                                # shared guest stays together
    assert sum(s == "test" for s in sp.values()) >= 10
    assert "hosts (people in many videos, seen everywhere): Alex" in res.stdout
    assert "Guests shared between sets: train-test 0, train-valid 0, valid-test 0" in res.stdout
    assert os.path.exists(tmp_path / "s" / "split.meta.json")


def test_same_seed_same_split_and_frozen(tmp_path):
    make(tmp_path)
    run(tmp_path)
    first = read(tmp_path)
    res = run(tmp_path)                                   # second run must refuse to overwrite
    assert res.returncode != 0 and "frozen" in res.stderr
    assert run(tmp_path, "--force").returncode == 0
    assert read(tmp_path) == first                        # same seed -> same split


def test_report_only_does_not_change_split(tmp_path):
    make(tmp_path)
    run(tmp_path)
    before = open(tmp_path / "s" / "split.csv").read()
    res = run(tmp_path, "--report_only")
    assert res.returncode == 0 and "Target share" in res.stdout
    assert open(tmp_path / "s" / "split.csv").read() == before
