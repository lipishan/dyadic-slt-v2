#!/usr/bin/env python3
"""Dyadic train/valid/test split by WHOLE conversation (plan section 0). Made once, then frozen.

What it does, in order:
  1. One conversation = one video. Every unit and turn of a conversation goes to the same set.
  2. People: names are mapped to person ids with signer_names.csv. A person in >= --host_min
     videos is a HOST (Alex, Renca, Callie...): hosts are seen in every set anyway. Everyone else
     is a GUEST. Conversations that share a guest are kept together in one block, so a guest
     never appears in two sets.
  3. Translation targets = units that are not 'suspicious', not likely back-channels and not
     signed by several people at once. Target counts drive the 70/10/20 ratio.
  4. Many random assignments of blocks (fixed seed) are tried; the one closest to the target
     ratios that also balances conversation length, long units and data source is kept.
  5. Writes splits/dyadic_split_v1.csv (video -> split) and a .meta.json (seed, settings,
     input checksums, counts). An existing split file is never overwritten without --force.

  python data/make_split.py --turns merged/turns_merged.csv --names signer_names.csv \
      --inventory merged/inventory.csv --out splits/dyadic_split_v1.csv

After editing signer_names.csv later, recompute only the report (the split does not change):
  python data/make_split.py --turns ... --names ... --inventory ... \
      --out splits/dyadic_split_v1.csv --report_only
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict

RATIOS = {"train": 0.7, "valid": 0.1, "test": 0.2}
SEVERAL = re.compile(r"\b(and|all|both)\b|[;/&+]", re.I)
VERSION = "make_split v1 (2026-11-25)"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def conversations(turns, names, inventory, host_min, long_words):
    """Per video: people, guests, counts used for balancing."""
    pid = {}
    for r in names:                              # (video-specific ids like 'VID:Woman' included)
        pid.setdefault(r["name"], {})
        for v in r["videos"].split():
            pid[r["name"]][v] = r["person_id"]
    kind = {r["video"]: r["kind"] for r in inventory}

    def person(video, name):
        m = pid.get(name, {})
        return m.get(video) or (next(iter(m.values())) if m else f"{video}:{name}")

    conv = defaultdict(lambda: dict(units=0, targets=0, turns=set(), long=0, hours=0.0, people=set()))
    for r in turns:
        c = conv[r["video"]]
        c["units"] += 1
        c["turns"].add(r["turn_id"])
        c["hours"] += max(0.0, float(r["duration"])) / 3600
        several = bool(SEVERAL.search(r["signer"] or ""))
        if not several and r["signer"]:
            c["people"].add(person(r["video"], r["signer"]))
        if (r["confidence"] != "suspicious" and r["likely_backchannel"] != "True" and not several):
            c["targets"] += 1
            c["long"] += int(r["words"]) >= long_words
    n_videos = Counter(p for c in conv.values() for p in c["people"])
    hosts = {p for p, n in n_videos.items() if n >= host_min}
    out = {}
    for v, c in conv.items():
        if not c["targets"]:
            continue
        out[v] = dict(video=v, units=c["units"], targets=c["targets"], turns=len(c["turns"]),
                      long=c["long"], hours=c["hours"], kind=kind.get(v, "unknown"),
                      people=sorted(c["people"]), guests=sorted(c["people"] - hosts))
    return out, hosts


def blocks(conv):
    """Union conversations that share a guest (so a guest stays in one set)."""
    parent = {v: v for v in conv}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    first = {}
    for v, c in sorted(conv.items()):
        for g in c["guests"]:
            if g in first:
                parent[find(v)] = find(first[g])
            else:
                first[g] = v
    out = defaultdict(list)
    for v in conv:
        out[find(v)].append(v)
    return sorted(sorted(b) for b in out.values())


def stats(conv, vids):
    cs = [conv[v] for v in vids]
    t = sum(c["targets"] for c in cs)
    return dict(conversations=len(cs), targets=t, units=sum(c["units"] for c in cs),
                turns=sum(c["turns"] for c in cs), hours=round(sum(c["hours"] for c in cs), 2),
                long_share=round(sum(c["long"] for c in cs) / t, 3) if t else 0,
                median_turns=st.median([c["turns"] for c in cs]) if cs else 0,
                kinds=dict(Counter(c["kind"] for c in cs)))


def score(conv, assign, total):
    """Lower is better: ratio error + balance of length, long units and data source."""
    by = defaultdict(list)
    for v, s in assign.items():
        by[s].append(v)
    if any(not by[s] for s in RATIOS):
        return float("inf")
    all_st = stats(conv, list(conv))
    err = 0.0
    for s, r in RATIOS.items():
        x = stats(conv, by[s])
        err += 4 * abs(x["targets"] / total - r)
        err += abs(x["long_share"] - all_st["long_share"])
        err += 0.02 * abs(x["median_turns"] - all_st["median_turns"]) / max(1, all_st["median_turns"])
        for k, n in all_st["kinds"].items():
            err += 0.5 * abs(x["kinds"].get(k, 0) / x["conversations"] - n / all_st["conversations"])
    return err


def search(conv, seed, tries, min_test):
    rng = random.Random(seed)
    bl = blocks(conv)
    total = sum(c["targets"] for c in conv.values())
    best, best_score = None, float("inf")
    for _ in range(tries):
        order = bl[:]
        rng.shuffle(order)
        assign, filled = {}, Counter()
        for b in order:
            t = sum(conv[v]["targets"] for v in b)
            s = ("test" if filled["test"] + t <= RATIOS["test"] * total * 1.05 or filled["test"] == 0
                 else "valid" if filled["valid"] + t <= RATIOS["valid"] * total * 1.05 or filled["valid"] == 0
                 else "train")
            filled[s] += t
            for v in b:
                assign[v] = s
        if sum(1 for s in assign.values() if s == "test") < min_test:
            continue
        sc = score(conv, assign, total)
        if sc < best_score:
            best, best_score = assign, sc
    if best is None:
        sys.exit(f"no split found with >= {min_test} test conversations; try more --tries")
    return best, best_score


def report(conv, hosts, assign):
    by = defaultdict(list)
    for v, s in assign.items():
        if v in conv:
            by[s].append(v)
    guests = {s: {g for v in by[s] for g in conv[v]["guests"]} for s in RATIOS}
    unseen = [v for v in by["test"] if not (set(conv[v]["guests"]) & (guests["train"] | guests["valid"]))
              and conv[v]["guests"]]
    L = [f"Conversations: {len(conv)}; hosts (people in many videos, seen everywhere): "
         f"{', '.join(sorted(hosts)) or 'none'}"]
    for s in RATIOS:
        x = stats(conv, by[s])
        L.append(f"{s:5}: {x['conversations']:3} conversations, {x['targets']:4} targets, {x['units']:4} units, "
                 f"{x['turns']:4} turns, {x['hours']:5} h, long units {x['long_share']:.0%}, "
                 f"median turns/conv {x['median_turns']}, sources {x['kinds']}")
    tot = sum(conv[v]["targets"] for v in conv)
    L.append("Target share: " + ", ".join(
        f"{s} {sum(conv[v]['targets'] for v in by[s]) / tot:.1%}" for s in RATIOS))
    L.append(f"Guests shared between sets: train-test {len(guests['train'] & guests['test'])}, "
             f"train-valid {len(guests['train'] & guests['valid'])}, "
             f"valid-test {len(guests['valid'] & guests['test'])}")
    L.append(f"Unseen-guest test conversations (guest never in train/valid): {len(unseen)} of {len(by['test'])}")
    missing = [v for v in conv if v not in assign]
    if missing:
        L.append(f"NOTE: {len(missing)} conversations are not in the split file (new data?): {missing[:10]}")
    return "\n".join(L), unseen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", default="splits/dyadic_split_v1.csv")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tries", type=int, default=5000)
    ap.add_argument("--min_test", type=int, default=10)
    ap.add_argument("--host_min", type=int, default=5, help="person in >= this many videos = host")
    ap.add_argument("--long_words", type=int, default=40, help="placeholder until the 'long' threshold is fixed")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()

    conv, hosts = conversations(read_csv(a.turns), read_csv(a.names), read_csv(a.inventory),
                                a.host_min, a.long_words)
    meta_path = os.path.splitext(a.out)[0] + ".meta.json"
    if a.report_only:
        assign = {r["video"]: r["split"] for r in read_csv(a.out)}
        text, unseen = report(conv, hosts, assign)
        print(text)
        return
    if os.path.exists(a.out) and not a.force:
        sys.exit(f"{a.out} exists and is frozen. Use --report_only to re-check it, or --force to replace it.")

    assign, sc = search(conv, a.seed, a.tries, a.min_test)
    text, unseen = report(conv, hosts, assign)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "split", "unseen_guest_test"])
        for v in sorted(assign):
            w.writerow([v, assign[v], v in unseen])
    meta = dict(version=VERSION, seed=a.seed, tries=a.tries, min_test=a.min_test, host_min=a.host_min,
                long_words=a.long_words, ratios=RATIOS, score=round(sc, 4),
                inputs={p: sha(p) for p in (a.turns, a.names, a.inventory)},
                hosts=sorted(hosts), report=text.splitlines())
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(text)
    print(f"\nWrote {a.out} and {meta_path}")


if __name__ == "__main__":
    main()
