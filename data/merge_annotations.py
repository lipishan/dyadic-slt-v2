#!/usr/bin/env python3
"""Merge annotator 2 and annotator 1 (the tool) into ONE turn table for all videos.

Rules (agreed 2026-11-25):
  Videos both annotated:
    - use annotator 2's turn by default;
    - annotator 2 skipped a signed line but the tool has it        -> use the tool's turn;
    - annotator 2's turn fails a check but the tool's passes       -> use the tool's turn.
  Videos only one source annotated: use that source.
  Every turn gets the same automatic checks, a `source` and a `confidence`:
    confirmed  - chosen turn passes all checks AND both sources agree
                 (time overlap >= 0.5 and same signer)
    plausible  - passes all checks, but only one source has it (or the other disagrees)
    suspicious - fails a check -> we spot-check it; otherwise history text only
  (A pose check of hand movement can later raise single-source turns to confirmed.)

  python data/merge_annotations.py --ann2_dir ANN2_DIR --tool_dir TOOL_DIR \
      --transcript_dir TR_DIR --out_dir merged/

Outputs in out_dir:
  turns_merged.csv  one row per turn: chosen times, signer, source, confidence, reasons,
                    and both sources' times for comparison
  inventory.csv     one row per video: which files exist, kind (both / ann2_only / tool_only),
                    counts per confidence, signed lines that nobody annotated
  report.txt        plain-language summary (also printed)
"""
import argparse
import csv
import glob
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_annotations import (ANN_PATTERN, TOOL_PATTERN, TR_PATTERN, check_turns,  # noqa: E402
                               load_transcript, order_and_overlap, read_ann2, same_person, unsignable)
from compare_annotators import A1_COLS, iou, load_a1  # noqa: E402

# A turn "fails a check" if it has one of these flags. Name mismatches are NOT here on purpose:
# names are not yet unified across spellings (that is the signer-name table, a later step).
HARD_FLAGS = {"speed_implausible", "end_before_start", "no_signer",
              "line_order_differs_from_time_order", "overlaps_same_signer"}
AGREE_IOU = 0.5

FIELDS = ["video", "line", "start", "end", "duration", "signer", "source", "confidence", "reasons",
          "words", "wps", "overlaps_partner", "likely_backchannel", "text",
          "a2_start", "a2_end", "a2_signer", "tool_start", "tool_end", "tool_signer", "iou"]


def hard(row):
    return sorted(set(row["flags"]) & HARD_FLAGS)


def read_tool(tool_path):
    return [dict(line=i, start=t["start"], end=t["end"], signer=t["signer"] or "", text=t["text"],
                 sign_id=None) for i, t in sorted(load_a1(tool_path).items())]


def tool_kinds(tool_path):
    """Counts of the tool's annotation_type values (catches files where nothing is 'signed')."""
    with open(tool_path, newline="", encoding="utf-8-sig") as f:
        return Counter((r.get(A1_COLS["kind"]) or "").strip() for r in csv.DictReader(f))


def merge_video(vid, ann2_path, tool_path, tr_path):
    tr = load_transcript(tr_path)
    A = {r["line"]: r for r in check_turns(vid, read_ann2(ann2_path), tr)} if ann2_path else {}
    T = {r["line"]: r for r in check_turns(vid, read_tool(tool_path), tr)} if tool_path else {}
    kind = "both" if A and T else ("ann2_only" if A else "tool_only")

    out = []
    for line in sorted(set(A) | set(T)):
        a, t = A.get(line), T.get(line)
        agree = bool(a and t and iou(a, t) >= AGREE_IOU and
                     (not a["signer"] or not t["signer"] or same_person(a["signer"], t["signer"])))
        reasons = []
        if a and t:
            if not hard(a):
                chosen, source = a, "annotator2"
            elif not hard(t):
                chosen, source = t, "tool"
                reasons.append("annotator2 failed: " + ",".join(hard(a)))
            else:
                chosen, source = a, "annotator2"
            if not agree:
                reasons.append("sources disagree")
        elif a:
            chosen, source = a, "annotator2"
        else:
            chosen, source = t, "tool"
            if A:
                reasons.append("annotator2 skipped this line")
        out.append(dict(chosen, source=source, agree=agree, reasons=reasons,
                        a2=a, tool=t, iou=round(iou(a, t), 3) if a and t else ""))

    order_and_overlap(out)                      # mixing sources can create new order problems
    for x in out:
        h = hard(x)
        if h:
            x["confidence"] = "suspicious"
            x["reasons"].append("fails: " + ",".join(h))
        else:
            x["confidence"] = "confirmed" if x["agree"] else "plausible"

    done = {x["line"] for x in out}
    nobody = [i for i, tt in enumerate(tr or [], 1) if i not in done and not unsignable(tt.get("text"))]
    return out, kind, nobody


def to_row(x):
    a, t = x["a2"] or {}, x["tool"] or {}
    return dict(video=x["video"], line=x["line"], start=x["start"], end=x["end"],
                duration=x["duration"], signer=x["signer"], source=x["source"],
                confidence=x["confidence"], reasons="; ".join(x["reasons"]), words=x["words"],
                wps=x["wps"], overlaps_partner=x["overlaps_partner"],
                likely_backchannel=x["likely_backchannel"], text=x["text_clean"],
                a2_start=a.get("start", ""), a2_end=a.get("end", ""), a2_signer=a.get("signer", ""),
                tool_start=t.get("start", ""), tool_end=t.get("end", ""),
                tool_signer=t.get("signer", ""), iou=x["iou"])


def video_ids(folder, pattern):
    if not folder:
        return {}
    suffix = pattern.format(vid="")
    return {os.path.basename(p)[:-len(suffix)]: p
            for p in glob.glob(os.path.join(folder, pattern.format(vid="*")))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann2_dir", required=True)
    ap.add_argument("--tool_dir")
    ap.add_argument("--transcript_dir", required=True)
    ap.add_argument("--out_dir", default="merged")
    a = ap.parse_args()
    A, T = video_ids(a.ann2_dir, ANN_PATTERN), video_ids(a.tool_dir, TOOL_PATTERN)
    if not (A or T):
        sys.exit("no annotation files found")
    os.makedirs(a.out_dir, exist_ok=True)

    rows, inv, notes = [], [], []
    for vid in sorted(set(A) | set(T)):
        tr = os.path.join(a.transcript_dir, TR_PATTERN.format(vid=vid))
        has_tr = os.path.exists(tr)
        kinds = tool_kinds(T[vid]) if vid in T else Counter()
        tool_path = T.get(vid) if kinds.get("signed") else None
        if vid in T and not tool_path:
            notes.append(f"{vid}: tool file has no 'signed' rows (annotation_type values: {dict(kinds)})")
        if vid not in A and not tool_path:
            continue
        out, kind, nobody = merge_video(vid, A.get(vid), tool_path, tr if has_tr else None)
        rows += [to_row(x) for x in out]
        c = Counter(x["confidence"] for x in out)
        s = Counter(x["source"] for x in out)
        inv.append(dict(video=vid, kind=kind, has_ann2=vid in A, has_tool=vid in T,
                        has_transcript=has_tr, turns=len(out), confirmed=c["confirmed"],
                        plausible=c["plausible"], suspicious=c["suspicious"],
                        from_annotator2=s["annotator2"], from_tool=s["tool"],
                        signed_lines_nobody_annotated=" ".join(map(str, nobody))))
        if not has_tr:
            notes.append(f"{vid}: no transcript, so text/signer checks were skipped")

    with open(os.path.join(a.out_dir, "turns_merged.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out_dir, "inventory.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(inv[0].keys())); w.writeheader(); w.writerows(inv)

    k, c, s = Counter(v["kind"] for v in inv), Counter(r["confidence"] for r in rows), \
        Counter(r["source"] for r in rows)
    switched = sum("annotator2 failed" in r["reasons"] for r in rows)
    filled = sum("annotator2 skipped" in r["reasons"] for r in rows)
    nob = sum(len(v["signed_lines_nobody_annotated"].split()) for v in inv)
    L = [f"Videos: {len(inv)}  (both: {k['both']}, annotator 2 only: {k['ann2_only']}, "
         f"tool only: {k['tool_only']})",
         f"Turns: {len(rows)}  (from annotator 2: {s['annotator2']}, from tool: {s['tool']})",
         f"  tool used because annotator 2 skipped the line: {filled}",
         f"  tool used because annotator 2's turn failed a check: {switched}",
         f"Confidence: confirmed {c['confirmed']}, plausible {c['plausible']}, "
         f"suspicious {c['suspicious']}",
         f"Signed transcript lines that nobody annotated: {nob}"]
    if notes:
        L += ["", "Notes:"] + [f"  {n}" for n in notes]
    rep = "\n".join(L)
    with open(os.path.join(a.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
