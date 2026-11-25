#!/usr/bin/env python3
"""People index: one card per person with a video still, the names used, and the videos.

Use it to spot the same face under two names, or to name "Woman", "Interpreter",
"DAILY MOTH". Then correct person_id in signer_names.csv.

For each person it takes the longest clean unit (not suspicious, not a back-channel) from up
to --per_person different videos and grabs ONE frame from the middle of it with ffmpeg.
The picture is the whole frame (both people may be visible); the card says who is signing.
Cropped faces come later with the person-box step.

  python data/people_index.py --turns merged/turns_merged.csv --names signer_names.csv \
      --videos_dir /path/to/videos --out people_index.html

Needs ffmpeg. Videos are found as VIDEO_ID.<mp4|mkv|webm|mov> in --videos_dir (or one folder
below it). The page is a single file with the pictures inside, so it can be opened or sent as is.
"""
import argparse
import base64
import csv
import glob
import html
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

EXTS = ("mp4", "mkv", "webm", "mov")


def find_video(videos_dir, vid):
    for pat in ("{d}/{v}.{e}", "{d}/*/{v}.{e}"):
        for e in EXTS:
            hits = glob.glob(pat.format(d=glob.escape(videos_dir), v=glob.escape(vid), e=e))
            if hits:
                return hits[0]
    return None


def ffmpeg_grab(video, t, out_jpg):
    """One frame at time t (seconds), 360 px wide. Returns True on success."""
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", video,
           "-frames:v", "1", "-vf", "scale=360:-2", out_jpg]
    return subprocess.run(cmd, capture_output=True).returncode == 0 and os.path.exists(out_jpg)


def pick_units(turns, names, per_person):
    """person_id -> list of (video, mid_time, name), longest clean units, different videos."""
    pid = defaultdict(dict)
    for r in names:
        for v in r["videos"].split():
            pid[r["name"]][v] = r["person_id"]
    cand = defaultdict(list)
    for r in turns:
        p = pid.get(r["signer"], {}).get(r["video"])
        if not p or r["confidence"] == "suspicious" or r["likely_backchannel"] == "True":
            continue
        s, e = float(r["start"]), float(r["end"])
        cand[p].append((e - s, r["video"], (s + e) / 2, r["signer"]))
    out = {}
    for p, cs in cand.items():
        seen, chosen = set(), []
        for d, v, mid, nm in sorted(cs, reverse=True):
            if v not in seen:
                seen.add(v)
                chosen.append((v, mid, nm))
            if len(chosen) == per_person:
                break
        out[p] = chosen
    return out


def build_page(names, picks, images):
    """images: (person_id, video) -> jpeg bytes or None."""
    people = defaultdict(lambda: dict(names=set(), videos=set(), status=set(), notes=set()))
    for r in names:
        x = people[r["person_id"]]
        x["names"].add(r["name"]); x["videos"].update(r["videos"].split())
        x["status"].add(r["status"]); x["notes"].discard("")
        if r["note"]:
            x["notes"].add(r["note"])
    cards = []
    for p in sorted(people, key=lambda k: (k.split(":")[-1].lower(), k)):
        x = people[p]
        imgs = ""
        for v, mid, nm in picks.get(p, []):
            b = images.get((p, v))
            src = f"data:image/jpeg;base64,{base64.b64encode(b).decode()}" if b else ""
            cap = f"{html.escape(v)} @ {mid:.0f}s — signing: {html.escape(nm)}"
            imgs += (f'<figure><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'
                     if b else f'<figure class="none"><figcaption>no picture ({cap})</figcaption></figure>')
        flag = ' check' if "check" in x["status"] else ''
        cards.append(
            f'<section class="card{flag}"><h2>{html.escape(p)}</h2>'
            f'<p><b>Names:</b> {html.escape(", ".join(sorted(x["names"])))}</p>'
            f'<p><b>Videos ({len(x["videos"])}):</b> {html.escape(" ".join(sorted(x["videos"])[:8]))}'
            f'{" …" if len(x["videos"]) > 8 else ""}</p>'
            + (f'<p class="note">{html.escape("; ".join(sorted(x["notes"])))}</p>' if x["notes"] else "")
            + f'<div class="imgs">{imgs}</div></section>')
    css = ("body{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}"
           "h1{font-size:20px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:16px}"
           ".card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px}"
           ".card.check{border-color:#d9822b}.card h2{font-size:16px;margin:0 0 6px}"
           ".card p{margin:4px 0;font-size:13px}.note{color:#a35a00}"
           ".imgs{display:flex;gap:8px;flex-wrap:wrap}figure{margin:0;width:180px}"
           "img{width:180px;border-radius:4px}figcaption{font-size:11px;color:#555}"
           "input{padding:6px;width:300px;margin-bottom:12px}")
    js = ("document.getElementById('q').addEventListener('input',e=>{const q=e.target.value.toLowerCase();"
          "document.querySelectorAll('.card').forEach(c=>c.style.display=c.textContent.toLowerCase()"
          ".includes(q)?'':'none')})")
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>People index</title>"
            f"<style>{css}</style></head><body><h1>People index — {len(people)} people "
            f"(orange border = needs a check)</h1><input id='q' placeholder='Filter by name or video'>"
            f"<div class='grid'>{''.join(cards)}</div><script>{js}</script></body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--per_person", type=int, default=2)
    ap.add_argument("--out", default="people_index.html")
    a = ap.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found. Install it in your user environment (e.g. conda install ffmpeg) "
                 "or run this on a machine that has it.")
    with open(a.turns, newline="", encoding="utf-8") as f:
        turns = list(csv.DictReader(f))
    with open(a.names, newline="", encoding="utf-8") as f:
        names = list(csv.DictReader(f))
    picks = pick_units(turns, names, a.per_person)
    images, missing = {}, set()
    with tempfile.TemporaryDirectory() as tmp:
        for p, items in picks.items():
            for v, mid, _ in items:
                video = find_video(a.videos_dir, v)
                if not video:
                    missing.add(v)
                    continue
                jpg = os.path.join(tmp, "x.jpg")
                if ffmpeg_grab(video, mid, jpg):
                    with open(jpg, "rb") as f:
                        images[(p, v)] = f.read()
                    os.remove(jpg)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(build_page(names, picks, images))
    print(f"Wrote {a.out}: {len({p for p, _ in images})} people with a picture, "
          f"{len(images)} pictures." + (f" Videos not found: {len(missing)} (e.g. {sorted(missing)[:5]})"
                                         if missing else ""))


if __name__ == "__main__":
    main()
