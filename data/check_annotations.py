#!/usr/bin/env python3
"""Check annotator 2's files (the reference annotation) against the transcripts.

For each video: are all signed transcript lines annotated, is every turn given a
signer, does the signer match the transcript speaker, does the text match, is the
signing speed plausible, which turns overlap the partner (likely back-channels).
Writes one cleaned table for all videos plus a plain-language report.

Single video:
  python check_annotations.py --ann VID_annotations.csv --transcript VID.transcript.json --out_dir checked/
All videos in folders:
  python check_annotations.py --ann_dir ANN_DIR --transcript_dir TR_DIR --out_dir checked/

Outputs in out_dir:
  turns_checked.csv   one row per annotated turn, with cleaned text and flags
  report.txt          plain-language report (also printed)
  summary.json        counts per video
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import unicodedata

COLS = dict(id="sign_id", video="video_id", start="start_time", end="end_time",
            signer="active_signer", line="turn_index", text="text")
ANN_PATTERN = "{vid}_annotations.csv"
TR_PATTERN = "{vid}.transcript.json"
NULL = {"", "none", "null", "nan", "unknown"}
WPS_MIN, WPS_MAX = 0.8, 6.0
SHORT_S, SHORT_WORDS = 1.5, 3          # "short reply" thresholds for back-channel guess
PREFIX_RE = re.compile(r"^\s*([A-Z][\w“”\"' .-]{0,40}?)\s*(\([^)]*\))?\s*:\s+")


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", s).strip().lower()


def words(s):
    return len(re.findall(r"[A-Za-z0-9']+", s or ""))


def unsignable(text):
    t = (text or "").strip()
    tl = t.lower()
    return (t.startswith("[") or tl.startswith("http") or tl.startswith("image of")
            or re.fullmatch(r"[—\-– ]+", t) is not None)


def nullable(s):
    s = (s or "").strip()
    return None if s.lower() in NULL else s


def split_prefix(text, known_speakers):
    """'Alex (Daily Moth): Together...' -> ('Alex', 'Together...') if Alex is a known speaker."""
    m = PREFIX_RE.match(text or "")
    if not m:
        return None, text
    name = m.group(1).strip()
    for sp in known_speakers:
        if sp and (name == sp or name in sp.split() or name in sp):
            return sp, text[m.end():].strip()
    return None, text


def check_video(vid, ann_path, tr_path):
    with open(ann_path, newline="", encoding="utf-8-sig") as f:
        ann = list(csv.DictReader(f))
    tr = None
    if tr_path and os.path.exists(tr_path):
        with open(tr_path, encoding="utf-8") as f:
            tr = json.load(f)["turns"]
    speakers = sorted({t.get("speaker") for t in tr or [] if t.get("speaker")} |
                      {nullable(r[COLS["signer"]]) for r in ann if nullable(r[COLS["signer"]])})

    rows = []
    for r in ann:
        line = int(float(r[COLS["line"]]))
        start, end = float(r[COLS["start"]]), float(r[COLS["end"]])
        t = tr[line - 1] if tr and 1 <= line <= len(tr) else None
        text = r[COLS["text"]]
        flags = []
        if t and norm(text) != norm(t.get("text")):
            flags.append("text_differs_from_transcript")
        pref_sp, clean = split_prefix(text, speakers)
        signer, source = nullable(r[COLS["signer"]]), "annotator"
        tr_sp = t.get("speaker") if t else None
        if signer is None:
            if tr_sp:
                signer, source = tr_sp, "transcript"
            elif pref_sp:
                signer, source = pref_sp, "text_prefix"
            else:
                source = "missing"
                flags.append("no_signer")
            if signer:
                flags.append(f"signer_filled_from_{source}")
        elif tr_sp and signer != tr_sp:
            flags.append("signer_differs_from_transcript")
        dur = end - start
        if dur <= 0:
            flags.append("end_before_start")
        w = words(clean)
        wps = w / dur if dur > 0 else float("inf")
        if not (WPS_MIN <= wps <= WPS_MAX):
            flags.append("speed_implausible")
        rows.append(dict(video=vid, sign_id=r.get(COLS["id"]), line=line, start=start, end=end,
                         duration=round(dur, 3), signer=signer, signer_source=source,
                         words=w, wps=round(wps, 2), text_clean=clean, flags=flags))

    # order, overlaps, back-channel guess
    rows.sort(key=lambda x: x["start"])
    for i, x in enumerate(rows):
        x["overlaps_partner"] = False
        for y in rows[max(0, i - 3):i] + rows[i + 1:i + 4]:
            if y["signer"] != x["signer"] and min(x["end"], y["end"]) > max(x["start"], y["start"]):
                x["overlaps_partner"] = True
        x["likely_backchannel"] = x["overlaps_partner"] and (
            x["duration"] <= SHORT_S or x["words"] <= SHORT_WORDS)
        if i and x["line"] < rows[i - 1]["line"]:
            x["flags"].append("line_order_differs_from_time_order")
        if i and x["signer"] == rows[i - 1]["signer"] and x["start"] < rows[i - 1]["end"]:
            x["flags"].append("overlaps_same_signer")
    rows.sort(key=lambda x: x["line"])

    annotated = {x["line"] for x in rows}
    missing = []
    if tr:
        for i, t in enumerate(tr, 1):
            if i not in annotated and not unsignable(t.get("text")):
                missing.append((i, (t.get("text") or "")[:70]))
    summ = dict(video=vid, transcript_lines=len(tr) if tr else None,
                transcript_unsignable=sum(unsignable(t.get("text")) for t in tr) if tr else None,
                annotated=len(rows), not_annotated_maybe_signed=missing,
                flagged=sum(1 for x in rows if x["flags"]),
                overlaps=sum(x["overlaps_partner"] for x in rows),
                likely_backchannel=[x["line"] for x in rows if x["likely_backchannel"]])
    return rows, summ


def report(rows, summs):
    L = []
    for s in summs:
        L.append(f"=== Video {s['video']} ===")
        if s["transcript_lines"] is not None:
            L.append(f"Transcript lines: {s['transcript_lines']} "
                     f"({s['transcript_unsignable']} clearly not signing: URLs, separators, images)")
        L.append(f"Annotated signing turns: {s['annotated']}")
        if s["not_annotated_maybe_signed"]:
            L.append("Transcript lines with no annotation (please confirm they are not signed):")
            for i, t in s["not_annotated_maybe_signed"]:
                L.append(f"  line {i}: {t}")
        L.append(f"Turns overlapping the partner: {s['overlaps']}; "
                 f"likely back-channels (short + overlapping): lines {s['likely_backchannel']}")
        vr = [x for x in rows if x["video"] == s["video"] and x["flags"]]
        if vr:
            L.append(f"Turns with something to check ({len(vr)}):")
            for x in vr:
                L.append(f"  line {x['line']} ({x['start']:.1f}-{x['end']:.1f} s, {x['signer']}, "
                         f"{x['words']} words, {x['wps']} w/s): {', '.join(x['flags'])}")
        else:
            L.append("No turns flagged.")
        L.append("")
    if len(summs) > 1:
        L.append(f"=== All videos: {len(summs)} videos, {len(rows)} turns, "
                 f"{sum(1 for x in rows if x['flags'])} flagged ===")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann"); ap.add_argument("--transcript")
    ap.add_argument("--ann_dir"); ap.add_argument("--transcript_dir")
    ap.add_argument("--out_dir", default="checked")
    a = ap.parse_args()
    jobs = []
    if a.ann:
        jobs.append((os.path.basename(a.ann).split("_annotations")[0], a.ann, a.transcript))
    elif a.ann_dir:
        for p in sorted(glob.glob(os.path.join(a.ann_dir, ANN_PATTERN.format(vid="*")))):
            vid = os.path.basename(p)[: -len(ANN_PATTERN.format(vid=""))]
            tr = os.path.join(a.transcript_dir, TR_PATTERN.format(vid=vid)) if a.transcript_dir else None
            jobs.append((vid, p, tr))
    else:
        sys.exit("give --ann or --ann_dir")
    if not jobs:
        sys.exit("no annotation files found")
    os.makedirs(a.out_dir, exist_ok=True)
    rows, summs = [], []
    for j in jobs:
        r, s = check_video(*j)
        rows += r; summs.append(s)
    if not rows:
        sys.exit("annotation files contain no turns")
    with open(os.path.join(a.out_dir, "turns_checked.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for x in rows:
            w.writerow({**x, "flags": ";".join(x["flags"])})
    with open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summs, f, indent=2, ensure_ascii=False)
    rep = report(rows, summs)
    with open(os.path.join(a.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)
    print(rep)


if __name__ == "__main__":
    main()
