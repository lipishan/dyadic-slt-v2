"""Tests for data/convert_old_annotations.py.

Old files number turns over what they aligned, not over the transcript, so matching must go by
text. Here the transcript has a title, a skipped image line, and "Thank you." appearing both
inside a longer line and on its own: only the exact line may match.
"""
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))
import compare_annotators as cmp  # noqa: E402
import convert_old_annotations as conv  # noqa: E402

TRANSCRIPT = {"turns": [
    {"speaker": "Alex", "text": "Interview"},                       # 1
    {"speaker": "Alex", "text": "Here is an “interview”."},         # 2
    {"speaker": None, "text": "[Image of a photo]"},                # 3
    {"speaker": "Liz", "text": "Thanks. Thank you."},               # 4
    {"speaker": "Liz", "text": "Thank you."},                       # 5
]}
OLD = {"video_id": "VID", "turns": [
    {"turn_index": 0, "speaker": "Alex", "text": "Here is an \"interview\".", "start_time": 1.0, "end_time": 4.5},
    {"turn_index": 1, "speaker": "Liz", "text": "Thank you.", "start_time": 5.0, "end_time": 6.0},
    {"turn_index": 2, "speaker": "Liz", "text": "Something not in the transcript", "start_time": 7.0, "end_time": 8.0},
]}


def test_match_by_text_keeps_order():
    m = conv.match_turns(OLD["turns"], TRANSCRIPT["turns"])
    assert [line for line, _ in m] == [2, 5, None]


def test_convert_and_load_as_annotator_1(tmp_path):
    old_p, tr_p, out_p = tmp_path / "VID.json", tmp_path / "VID.transcript.json", tmp_path / "out.csv"
    old_p.write_text(json.dumps(OLD), encoding="utf-8")
    tr_p.write_text(json.dumps(TRANSCRIPT), encoding="utf-8")
    s = conv.convert_video(old_p, tr_p, out_p)
    assert s["matched"] == 2 and s["unmatched_old_turns"] == [2]
    a1 = cmp.load_a1(out_p)                      # readable by compare_annotators unchanged
    assert sorted(a1) == [2, 5]
    assert a1[2]["start"] == 1.0 and a1[2]["end"] == 4.5 and a1[5]["signer"] == "Liz"


def test_cli(tmp_path):
    old_d, tr_d, out_d = tmp_path / "old", tmp_path / "tr", tmp_path / "out"
    old_d.mkdir(); tr_d.mkdir()
    (old_d / "VID.json").write_text(json.dumps(OLD), encoding="utf-8")
    (tr_d / "VID.transcript.json").write_text(json.dumps(TRANSCRIPT), encoding="utf-8")
    res = subprocess.run([sys.executable, os.path.join(ROOT, "data", "convert_old_annotations.py"),
                          "--old_dir", str(old_d), "--transcript_dir", str(tr_d), "--out_dir", str(out_d)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "2 of 3 old turns matched" in res.stdout
    with open(out_d / "VID_asl_turn_alignment.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_empty_annotator_is_not_whole_seconds():
    assert cmp.within_checks({})["whole_seconds_only"] is False
