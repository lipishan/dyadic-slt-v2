#!/usr/bin/env python3
"""Compare two annotators' turn alignments for the same video(s).

Both annotators matched stretches of video to transcript lines. Their CSVs use
different column names; both carry the transcript line number (1-based), which
is the join key.

Single video:
  python compare_annotators.py \
      --a1 VID_asl_turn_alignment.csv --a2 VID_annotations.csv \
      --transcript VID.transcript.json --out_dir agreement/

Many videos (every video found in both folders):
  python compare_annotators.py --a1_dir DIR1 --a2_dir DIR2 \
      --transcript_dir DIR3 --out_dir agreement/

Outputs in out_dir:
  per_turn.csv     one row per (video, transcript line) with both annotations
  per_video.csv    agreement numbers per video
  summary.json     pooled agreement numbers
  report.txt       plain-language report (also printed)
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys
import unicodedata

# ---- column maps (change here if a format differs) -------------------------
A1_COLS = dict(index="turn_id", start_ms="start_ms", end_ms="end_ms",
               signer="aligned_signer", kind="annotation_type", text="transcript")
A1_SIGNED_KINDS = {"signed"}
A2_COLS = dict(index="turn_index", start_s="start_time", end_s="end_time",
               signer="active_signer", text="text")
A1_PATTERN = "{vid}_asl_turn_alignment.csv"
A2_PATTERN = "{vid}_annotations.csv"
TR_PATTERN = "{vid}.transcript.json"

NULL_SIGNERS = {"", "none", "null", "nan", "unknown"}


def norm_text(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", s).strip().lower()


def norm_signer(s):
    s = (s or "").strip()
    return None if s.lower() in NULL_SIGNERS else s


def n_words(s):
    return len(re.findall(r"[A-Za-z0-9']+", s or ""))


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_annotations import same_person, unsignable as looks_unsignable  # noqa: E402  one shared rule set


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_a1(path):
    out = {}
    for r in read_csv(path):
        if r.get(A1_COLS["kind"], "").strip() not in A1_SIGNED_KINDS:
            continue
        if not r.get(A1_COLS["start_ms"]) or not r.get(A1_COLS["end_ms"]):
            continue
        i = int(float(r[A1_COLS["index"]]))
        out[i] = dict(start=float(r[A1_COLS["start_ms"]]) / 1000.0,
                      end=float(r[A1_COLS["end_ms"]]) / 1000.0,
                      signer=norm_signer(r.get(A1_COLS["signer"])),
                      text=r.get(A1_COLS["text"], ""))
    return out


def load_a2(path):
    out = {}
    for r in read_csv(path):
        i = int(float(r[A2_COLS["index"]]))
        out[i] = dict(start=float(r[A2_COLS["start_s"]]),
                      end=float(r[A2_COLS["end_s"]]),
                      signer=norm_signer(r.get(A2_COLS["signer"])),
                      text=r.get(A2_COLS["text"], ""))
    return out


def load_transcript(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return [dict(speaker=t.get("speaker"), text=t.get("text", "")) for t in d["turns"]]


def iou(a, b):
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def within_checks(ann):
    """Per-annotator habits: time resolution, gaps, overlaps between turns."""
    rows = sorted(ann.values(), key=lambda r: r["start"])
    whole = bool(rows) and all(abs(r["start"] - round(r["start"])) < 1e-6 and
                abs(r["end"] - round(r["end"])) < 1e-6 for r in rows)
    gaps = [b["start"] - a["end"] for a, b in zip(rows, rows[1:])]
    return dict(n=len(rows), whole_seconds_only=whole,
                n_zero_gaps=sum(1 for g in gaps if abs(g) < 1e-6),
                n_overlaps=sum(1 for g in gaps if g < -1e-6),
                n_gaps=len(gaps))


def compare_video(vid, a1_path, a2_path, tr_path):
    a1, a2, tr = load_a1(a1_path), load_a2(a2_path), load_transcript(tr_path)
    idxs = sorted(set(a1) | set(a2) | (set(range(1, len(tr) + 1)) if tr else set()))
    rows, text_mismatch = [], []
    for i in idxs:
        x, y = a1.get(i), a2.get(i)
        t = tr[i - 1] if tr and 1 <= i <= len(tr) else None
        ttext = t["text"] if t else (x or y or {}).get("text", "")
        for who, ann in (("a1", x), ("a2", y)):
            if ann and t and norm_text(ann["text"]) != norm_text(t["text"]):
                text_mismatch.append((i, who))
        r = dict(video=vid, line=i, transcript_speaker=(t or {}).get("speaker"),
                 words=n_words(ttext), unsignable=looks_unsignable(ttext),
                 in_a1=x is not None, in_a2=y is not None,
                 a1_start=x and x["start"], a1_end=x and x["end"], a1_signer=x and x["signer"],
                 a2_start=y and y["start"], a2_end=y and y["end"], a2_signer=y and y["signer"],
                 text=ttext[:80])
        if x and y:
            r.update(d_start=round(y["start"] - x["start"], 3),
                     d_end=round(y["end"] - x["end"], 3),
                     iou=round(iou(x, y), 3),
                     signer_agree=(x["signer"] == y["signer"] or same_person(x["signer"], y["signer"])))
            r["big_diff"] = max(abs(r["d_start"]), abs(r["d_end"])) > 2.0 or r["iou"] < 0.5
        for who, ann in (("a1", x), ("a2", y)):
            if ann:
                dur = ann["end"] - ann["start"]
                wps = r["words"] / dur if dur > 0 else float("inf")
                r[f"{who}_wps"] = round(wps, 2)
                r[f"{who}_wps_ok"] = 0.8 <= wps <= 6.0
        rows.append(r)

    both = [r for r in rows if r["in_a1"] and r["in_a2"]]
    ds = [abs(r["d_start"]) for r in both] + [abs(r["d_end"]) for r in both]
    summ = dict(
        video=vid,
        transcript_lines=len(tr) if tr else None,
        transcript_unsignable=sum(r["unsignable"] for r in rows) if tr else None,
        a1_signed=len(a1), a2_signed=len(a2), both=len(both),
        only_a1=[r["line"] for r in rows if r["in_a1"] and not r["in_a2"]],
        only_a2=[r["line"] for r in rows if r["in_a2"] and not r["in_a1"]],
        neither=[r["line"] for r in rows if not r["in_a1"] and not r["in_a2"]],
        signer_agree=sum(bool(r["signer_agree"]) for r in both),
        median_abs_boundary_diff_s=round(st.median(ds), 3) if ds else None,
        pct_boundaries_within_0_5s=round(100 * sum(d <= 0.5 for d in ds) / len(ds), 1) if ds else None,
        pct_boundaries_within_1s=round(100 * sum(d <= 1.0 for d in ds) / len(ds), 1) if ds else None,
        mean_iou=round(st.mean(r["iou"] for r in both), 3) if both else None,
        median_start_offset_a2_minus_a1_s=round(st.median(r["d_start"] for r in both), 3) if both else None,
        median_end_offset_a2_minus_a1_s=round(st.median(r["d_end"] for r in both), 3) if both else None,
        big_diff_lines=[r["line"] for r in both if r["big_diff"]],
        signer_disagree_lines=[r["line"] for r in both if not r["signer_agree"]],
        wps_fail_a1=[r["line"] for r in rows if r.get("a1_wps_ok") is False],
        wps_fail_a2=[r["line"] for r in rows if r.get("a2_wps_ok") is False],
        text_mismatch=text_mismatch,
        a1_habits=within_checks(a1), a2_habits=within_checks(a2),
    )
    return rows, summ


def plain_report(summs, pooled):
    L = []
    for s in summs:
        L.append(f"=== Video {s['video']} ===")
        if s["transcript_lines"] is not None:
            L.append(f"Transcript lines: {s['transcript_lines']} "
                     f"({s['transcript_unsignable']} are clearly not signing: links, images, clips, ads, speaker labels)")
        L.append(f"Lines marked as signing: annotator 1 = {s['a1_signed']}, annotator 2 = {s['a2_signed']}, "
                 f"both = {s['both']}")
        if s["only_a1"]:
            L.append(f"  Only annotator 1 marked lines: {s['only_a1']}")
        if s["only_a2"]:
            L.append(f"  Only annotator 2 marked lines: {s['only_a2']}")
        if s["both"]:
            L.append(f"Same signer named: {s['signer_agree']} of {s['both']} lines"
                     + (f" (disagree on lines {s['signer_disagree_lines']})" if s["signer_disagree_lines"] else ""))
            L.append(f"Start/end times: typical difference {s['median_abs_boundary_diff_s']} s; "
                     f"{s['pct_boundaries_within_0_5s']}% within 0.5 s, {s['pct_boundaries_within_1s']}% within 1 s; "
                     f"average overlap (IoU) {s['mean_iou']}")
            L.append(f"Habit: annotator 2 starts {s['median_start_offset_a2_minus_a1_s']:+} s and ends "
                     f"{s['median_end_offset_a2_minus_a1_s']:+} s relative to annotator 1 (median)")
            if s["big_diff_lines"]:
                L.append(f"  Lines that differ a lot (>2 s or IoU<0.5), worth a look: {s['big_diff_lines']}")
        for who in ("a1", "a2"):
            h = s[f"{who}_habits"]
            name = "Annotator 1" if who == "a1" else "Annotator 2"
            notes = []
            if h["whole_seconds_only"]:
                notes.append("all times are whole seconds")
            if h["n_gaps"] and h["n_zero_gaps"] == h["n_gaps"]:
                notes.append("every turn ends exactly where the next begins (no pauses)")
            elif h["n_zero_gaps"]:
                notes.append(f"{h['n_zero_gaps']} of {h['n_gaps']} turns end exactly where the next begins")
            if h["n_overlaps"]:
                notes.append(f"{h['n_overlaps']} overlapping turns")
            if notes:
                L.append(f"{name}: " + "; ".join(notes))
        if s["wps_fail_a1"] or s["wps_fail_a2"]:
            L.append(f"Signing speed outside 0.8-6 words/s: annotator 1 lines {s['wps_fail_a1']}, "
                     f"annotator 2 lines {s['wps_fail_a2']}")
        if s["text_mismatch"]:
            L.append(f"Text differs from transcript (line, annotator): {s['text_mismatch']}")
        L.append("")
    if len(summs) > 1:
        L.append("=== All videos together ===")
        for k, v in pooled.items():
            L.append(f"{k}: {v}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1"); ap.add_argument("--a2"); ap.add_argument("--transcript")
    ap.add_argument("--a1_dir"); ap.add_argument("--a2_dir"); ap.add_argument("--transcript_dir")
    ap.add_argument("--out_dir", default="agreement")
    a = ap.parse_args()

    jobs = []
    if a.a1 and a.a2:
        vid = os.path.basename(a.a2).split("_annotations")[0]
        jobs.append((vid, a.a1, a.a2, a.transcript))
    elif a.a1_dir and a.a2_dir:
        for p in sorted(glob.glob(os.path.join(a.a2_dir, A2_PATTERN.format(vid="*")))):
            vid = os.path.basename(p)[: -len(A2_PATTERN.format(vid=""))]
            p1 = os.path.join(a.a1_dir, A1_PATTERN.format(vid=vid))
            upd = p1.replace(".csv", "_updated.csv")
            if os.path.exists(upd):                     # a newer '_updated' file wins
                p1 = upd
            if os.path.exists(p1):
                tr = os.path.join(a.transcript_dir, TR_PATTERN.format(vid=vid)) if a.transcript_dir else None
                jobs.append((vid, p1, p, tr))
    else:
        sys.exit("give --a1/--a2 or --a1_dir/--a2_dir")
    if not jobs:
        sys.exit("no videos found in both annotators' files")

    os.makedirs(a.out_dir, exist_ok=True)
    all_rows, summs = [], []
    for job in jobs:
        rows, s = compare_video(*job)
        all_rows += rows; summs.append(s)

    both = [r for r in all_rows if r["in_a1"] and r["in_a2"]]
    ds = [abs(r["d_start"]) for r in both] + [abs(r["d_end"]) for r in both]
    pooled = dict(
        videos=len(summs),
        lines_both=len(both),
        turn_count_equal_videos=sum(s["a1_signed"] == s["a2_signed"] for s in summs),
        signer_agree_pct=round(100 * sum(bool(r["signer_agree"]) for r in both) / len(both), 1) if both else None,
        median_abs_boundary_diff_s=round(st.median(ds), 3) if ds else None,
        pct_within_0_5s=round(100 * sum(d <= 0.5 for d in ds) / len(ds), 1) if ds else None,
        pct_within_1s=round(100 * sum(d <= 1.0 for d in ds) / len(ds), 1) if ds else None,
        mean_iou=round(st.mean(r["iou"] for r in both), 3) if both else None,
    )

    keys = sorted({k for r in all_rows for k in r},
                  key=lambda k: list(all_rows[0].keys()).index(k) if k in all_rows[0] else 99)
    with open(os.path.join(a.out_dir, "per_turn.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_rows)
    flat = [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in s.items()} for s in summs]
    with open(os.path.join(a.out_dir, "per_video.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys())); w.writeheader(); w.writerows(flat)
    with open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(pooled=pooled, videos=summs), f, indent=2)
    rep = plain_report(summs, pooled)
    with open(os.path.join(a.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)
    print(rep)


if __name__ == "__main__":
    main()
