#!/usr/bin/env python3
"""One review list per video for annotator 2: which lines to re-watch, and why.

Combines (all inputs optional, at least one needed):
  --tool_agreement DIR   compare_annotators.py output, tool vs annotator 2
  --old_agreement DIR    compare_annotators.py output, old baseline vs annotator 2
  --checked DIR          check_annotations.py output (annotator 2 vs transcript)

A line is listed if any source raises a concern about annotator 2's version, e.g.
  line 5 "Yes." | annotator 2 34.2-34.7 s | tool 31-35 s | reason: tool times differ a lot
The more sources flag a line, the higher it is in the "n_reasons" column.

  python data/review_list.py --tool_agreement agreement/ --old_agreement agreement_old/ \
      --checked checked/ --out_dir review/
Writes review/review_<video>.csv and prints a short count per video.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

FIELDS = ["video", "line", "text", "a2_start", "a2_end", "a2_signer",
          "tool_start", "tool_end", "tool_signer", "old_start", "old_end", "old_signer",
          "n_reasons", "reasons"]


def true(v):
    return str(v).strip() == "True"


def read_agreement(d, name, items, reasons):
    """name is 'tool' or 'old'; in those files 'a1' is that source and 'a2' is annotator 2."""
    with open(os.path.join(d, "per_turn.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["video"], int(r["line"]))
            it = items[key]
            it.setdefault("text", r.get("text", ""))
            if true(r["in_a2"]):
                it.update(a2_start=r["a2_start"], a2_end=r["a2_end"], a2_signer=r["a2_signer"])
            if true(r["in_a1"]):
                it.update({f"{name}_start": r["a1_start"], f"{name}_end": r["a1_end"],
                           f"{name}_signer": r["a1_signer"]})
            if true(r["in_a1"]) and true(r["in_a2"]):
                if true(r.get("big_diff")):
                    reasons[key].append(f"{name} times differ a lot (>2 s or overlap<0.5)")
                if not true(r.get("signer_agree")):
                    reasons[key].append(f"{name} names a different signer")
            elif true(r["in_a1"]) and not true(r.get("unsignable")):
                reasons[key].append(f"{name} marks this line as signed, annotator 2 does not")


def read_checked(d, items, reasons):
    with open(os.path.join(d, "turns_checked.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["video"], int(r["line"]))
            it = items[key]
            it.setdefault("text", r.get("text_clean", ""))
            it.setdefault("a2_start", r["start"]); it.setdefault("a2_end", r["end"])
            it.setdefault("a2_signer", r["signer"])
            for fl in filter(None, (r.get("flags") or "").split(";")):
                reasons[key].append(f"transcript check: {fl}")
    with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
        for s in json.load(f):
            for line, text in s["not_annotated_maybe_signed"]:
                key = (s["video"], int(line))
                items[key].setdefault("text", text)
                reasons[key].append("transcript line not annotated")


def build(tool_dir=None, old_dir=None, checked_dir=None):
    items, reasons = defaultdict(dict), defaultdict(list)
    if tool_dir:
        read_agreement(tool_dir, "tool", items, reasons)
    if old_dir:
        read_agreement(old_dir, "old", items, reasons)
    if checked_dir:
        read_checked(checked_dir, items, reasons)
    per_video = defaultdict(list)
    for key in sorted(reasons):
        vid, line = key
        rs = list(dict.fromkeys(reasons[key]))          # drop repeats, keep order
        per_video[vid].append({**{k: "" for k in FIELDS}, **items[key], "video": vid, "line": line,
                               "n_reasons": len(rs), "reasons": "; ".join(rs)})
    return per_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool_agreement"); ap.add_argument("--old_agreement"); ap.add_argument("--checked")
    ap.add_argument("--out_dir", default="review")
    a = ap.parse_args()
    if not (a.tool_agreement or a.old_agreement or a.checked):
        sys.exit("give at least one of --tool_agreement, --old_agreement, --checked")
    per_video = build(a.tool_agreement, a.old_agreement, a.checked)
    os.makedirs(a.out_dir, exist_ok=True)
    for vid, rows in per_video.items():
        with open(os.path.join(a.out_dir, f"review_{vid}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        top = sum(r["n_reasons"] >= 2 for r in rows)
        print(f"{vid}: {len(rows)} lines to re-watch ({top} flagged by 2+ reasons)")
    if not per_video:
        print("nothing to review")


if __name__ == "__main__":
    main()
