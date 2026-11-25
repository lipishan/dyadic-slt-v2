"""Tests for data/merge_annotations.py: one small video per rule.

Video BOTH (transcript lines 1-6):
  1 "[Title]"                   not signed
  2 Alex  both agree            -> annotator2, confirmed
  3 Sam   annotator 2 too fast  -> tool (annotator 2 failed a check), plausible
  4 Alex  sources disagree      -> annotator2, plausible, reason "sources disagree"
  5 Sam   annotator 2 skipped   -> tool, plausible, reason "annotator2 skipped this line"
  6 Alex  nobody annotated      -> listed as signed line nobody annotated
Video TOOLONLY: tool only -> source tool. Video A2ONLY: annotator 2 only.
Video EMPTYTOOL: tool file whose rows are not 'signed' -> note, skipped.
"""
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "data", "merge_annotations.py")

TR = {"turns": [
    {"speaker": None, "text": "[Title]"},
    {"speaker": "Alex", "text": "Hello there my good friend"},
    {"speaker": "Sam", "text": "Nice to meet you here today"},
    {"speaker": "Alex", "text": "How are you doing these days"},
    {"speaker": "Sam", "text": "I am fine thank you"},
    {"speaker": "Alex", "text": "Great to hear that"},
]}
A2_HEAD = ["sign_id", "video_id", "start_time", "end_time", "active_signer", "turn_index", "text"]
TOOL_HEAD = ["turn_id", "start_ms", "end_ms", "aligned_signer", "annotation_type", "transcript"]


def w(path, head, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        c = csv.writer(f); c.writerow(head); c.writerows(rows)


def setup(tmp):
    d = {n: tmp / n for n in ("ann2", "tool", "tr", "out")}
    for n in ("ann2", "tool", "tr"):
        d[n].mkdir()
    for vid in ("BOTH", "TOOLONLY", "A2ONLY", "EMPTYTOOL"):
        (d["tr"] / f"{vid}.transcript.json").write_text(json.dumps(TR), encoding="utf-8")
    w(d["ann2"] / "BOTH_annotations.csv", A2_HEAD, [
        ["b1", "BOTH", "0.0", "3.0", "Alex", "2", "Hello there my good friend"],
        ["b2", "BOTH", "3.5", "3.9", "Sam", "3", "Nice to meet you here today"],     # 6 words in 0.4 s
        ["b3", "BOTH", "20.0", "23.0", "Alex", "4", "How are you doing these days"]])
    w(d["tool"] / "BOTH_asl_turn_alignment.csv", TOOL_HEAD, [
        ["2", "0", "3000", "Alex", "signed", "Hello there my good friend"],
        ["3", "3000", "6000", "Sam", "signed", "Nice to meet you here today"],
        ["4", "6000", "9000", "Alex", "signed", "How are you doing these days"],
        ["5", "24000", "27000", "Sam", "signed", "I am fine thank you"]])
    w(d["tool"] / "TOOLONLY_asl_turn_alignment.csv", TOOL_HEAD,
      [["2", "0", "3000", "Alex", "signed", "Hello there my good friend"]])
    w(d["ann2"] / "A2ONLY_annotations.csv", A2_HEAD,
      [["c1", "A2ONLY", "0.0", "3.0", "Alex", "2", "Hello there my good friend"]])
    w(d["tool"] / "EMPTYTOOL_asl_turn_alignment.csv", TOOL_HEAD,
      [["2", "0", "3000", "Alex", "Signed ", "Hello there my good friend"]])   # odd label
    res = subprocess.run([sys.executable, SCRIPT, "--ann2_dir", str(d["ann2"]), "--tool_dir", str(d["tool"]),
                          "--transcript_dir", str(d["tr"]), "--out_dir", str(d["out"])],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    turns = list(csv.DictReader(open(d["out"] / "turns_merged.csv", encoding="utf-8")))
    inv = {r["video"]: r for r in csv.DictReader(open(d["out"] / "inventory.csv", encoding="utf-8"))}
    return res.stdout, turns, inv


def test_rules_on_shared_video(tmp_path):
    _, turns, inv = setup(tmp_path)
    t = {int(r["line"]): r for r in turns if r["video"] == "BOTH"}
    assert (t[2]["source"], t[2]["confidence"]) == ("annotator2", "confirmed")
    assert (t[3]["source"], t[3]["confidence"]) == ("tool", "plausible")
    assert "annotator2 failed: speed_implausible" in t[3]["reasons"]
    assert t[3]["start"] == "3.0" and t[3]["a2_start"] == "3.5"          # both versions kept
    assert (t[4]["source"], t[4]["confidence"]) == ("annotator2", "plausible")
    assert "sources disagree" in t[4]["reasons"]
    assert (t[5]["source"], t[5]["confidence"]) == ("tool", "plausible")
    assert "annotator2 skipped this line" in t[5]["reasons"]
    assert inv["BOTH"]["kind"] == "both"
    assert inv["BOTH"]["signed_lines_nobody_annotated"] == "6"
    assert inv["BOTH"]["from_tool"] == "2"


def test_single_source_videos_and_odd_tool_file(tmp_path):
    out, turns, inv = setup(tmp_path)
    assert inv["TOOLONLY"]["kind"] == "tool_only"
    assert [r["source"] for r in turns if r["video"] == "TOOLONLY"] == ["tool"]
    assert inv["A2ONLY"]["kind"] == "ann2_only"
    assert "EMPTYTOOL" not in inv                                   # nothing usable
    assert "EMPTYTOOL: tool file has no 'signed' rows" in out
    assert "Videos: 3  (both: 1, annotator 2 only: 1, tool only: 1)" in out


def test_tool_overlapping_annotator2_turn_is_caught(tmp_path):
    """Mixing sources must re-run the order check: a tool turn that collides with an
    annotator-2 turn of the same signer becomes suspicious."""
    d = {n: tmp_path / n for n in ("ann2", "tool", "tr", "out")}
    for n in ("ann2", "tool", "tr"):
        d[n].mkdir()
    (d["tr"] / "V.transcript.json").write_text(json.dumps(TR), encoding="utf-8")
    w(d["ann2"] / "V_annotations.csv", A2_HEAD,
      [["v1", "V", "0.0", "3.0", "Alex", "2", "Hello there my good friend"],
       ["v2", "V", "3.5", "6.0", "Sam", "3", "Nice to meet you here today"]])
    w(d["tool"] / "V_asl_turn_alignment.csv", TOOL_HEAD,     # line 5 (Sam) placed inside Sam's line 3
      [["5", "4000", "6000", "Sam", "signed", "I am fine thank you"]])
    res = subprocess.run([sys.executable, SCRIPT, "--ann2_dir", str(d["ann2"]), "--tool_dir", str(d["tool"]),
                          "--transcript_dir", str(d["tr"]), "--out_dir", str(d["out"])],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    t = {int(r["line"]): r for r in csv.DictReader(open(d["out"] / "turns_merged.csv", encoding="utf-8"))}
    assert t[5]["confidence"] == "suspicious" and "overlaps_same_signer" in t[5]["reasons"]
