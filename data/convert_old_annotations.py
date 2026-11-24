#!/usr/bin/env python3
"""Turn the old (v1) per-video alignment JSONs into the CSV format of compare_annotators.py,
so the old timestamps can be compared with annotator 2.

Why a matching step is needed: the old files number their turns 0, 1, 2, ... over the lines
they managed to align, NOT over the transcript. Example: old turn 11 is transcript line 14.
So each old turn is matched to a transcript line by its text (exact match after normalising
quotes/spaces/case), searching forward only, so the order is kept.

Output goes in the "annotator 1" file format (turn_id, start_ms, end_ms, aligned_signer,
annotation_type, transcript), so this works unchanged:
  python data/convert_old_annotations.py --old_dir OLD_DIR --transcript_dir TR_DIR --out_dir old_csv/
  python data/compare_annotators.py --a1_dir old_csv/ --a2_dir ANN2_DIR --transcript_dir TR_DIR \
      --out_dir agreement_old/
In that report "annotator 1" means the OLD annotation.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_annotators import A1_PATTERN, TR_PATTERN, norm_text  # noqa: E402

HEADER = ["turn_id", "start_ms", "end_ms", "aligned_signer", "annotation_type", "transcript"]


def match_turns(old_turns, transcript_turns):
    """Return [(transcript line (1-based) or None, old turn)], keeping the order."""
    tr = [norm_text(t.get("text")) for t in transcript_turns]
    out, pos = [], 0
    for t in old_turns:
        key = norm_text(t.get("text"))
        line = next((i + 1 for i in range(pos, len(tr)) if tr[i] == key), None)
        if line is not None:
            pos = line
        out.append((line, t))
    return out


def convert_video(old_path, tr_path, out_path):
    with open(old_path, encoding="utf-8") as f:
        old = json.load(f)
    with open(tr_path, encoding="utf-8") as f:
        tr = json.load(f)["turns"]
    matched = match_turns(old["turns"], tr)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for line, t in matched:
            if line is None:
                continue
            w.writerow([line, round(1000 * t["start_time"]), round(1000 * t["end_time"]),
                        t.get("speaker") or "", "signed", t.get("text", "")])
    unmatched = [t["turn_index"] for line, t in matched if line is None]
    return dict(video=old.get("video_id"), old_turns=len(matched),
                matched=len(matched) - len(unmatched), unmatched_old_turns=unmatched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_dir", required=True)
    ap.add_argument("--transcript_dir", required=True)
    ap.add_argument("--out_dir", default="old_csv")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.old_dir, "*.json")))
    if not paths:
        sys.exit("no old JSON files found")
    for p in paths:
        vid = os.path.splitext(os.path.basename(p))[0]
        tr = os.path.join(a.transcript_dir, TR_PATTERN.format(vid=vid))
        if not os.path.exists(tr):
            print(f"{vid}: no transcript, skipped")
            continue
        s = convert_video(p, tr, os.path.join(a.out_dir, A1_PATTERN.format(vid=vid)))
        note = f"; not found in transcript: old turns {s['unmatched_old_turns']}" if s["unmatched_old_turns"] else ""
        print(f"{vid}: {s['matched']} of {s['old_turns']} old turns matched to transcript lines{note}")


if __name__ == "__main__":
    main()
