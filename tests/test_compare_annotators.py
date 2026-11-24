"""Tests for data/compare_annotators.py on a tiny made-up video.

Transcript: 1 "[Title]", 2 Alex "Hello there my friend", 3 Sam "Nice to meet you", 4 Alex "Yes".
Annotator 1 (ms, whole seconds, turns touching): lines 2, 3, 4 (line 4 with the wrong signer);
                                                 line 1 is a non-signed row and must be skipped.
Annotator 2 (s): lines 2 and 3 only.
"""
import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "data", "compare_annotators.py")
sys.path.insert(0, os.path.join(ROOT, "data"))
import compare_annotators as cmp  # noqa: E402

TRANSCRIPT = {"turns": [
    {"speaker": None, "text": "[Title]"},
    {"speaker": "Alex", "text": "Hello there my friend"},
    {"speaker": "Sam", "text": "Nice to meet you"},
    {"speaker": "Alex", "text": "Yes"},
]}
A1_HEADER = ["turn_id", "start_ms", "end_ms", "aligned_signer", "annotation_type", "transcript"]
A1 = [
    ["1", "", "", "", "title", "[Title]"],
    ["2", "0", "3000", "Alex", "signed", "Hello there my friend"],
    ["3", "3000", "6000", "Sam", "signed", "Nice to meet you"],
    ["4", "6000", "7000", "Sam", "signed", "Yes"],
]
A2_HEADER = ["sign_id", "video_id", "start_time", "end_time", "active_signer", "turn_index", "text"]
A2 = [
    ["s1", "VID", "0.5", "3.0", "Alex", "2", "Hello there my friend"],
    ["s2", "VID", "3.5", "5.5", "Sam", "3", "Nice to meet you"],
]


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_video(d1, d2, dt, a1=A1, a2=A2, vid="VID"):
    p1 = os.path.join(d1, f"{vid}_asl_turn_alignment.csv")
    p2 = os.path.join(d2, f"{vid}_annotations.csv")
    pt = os.path.join(dt, f"{vid}.transcript.json")
    write_csv(p1, A1_HEADER, a1)
    write_csv(p2, A2_HEADER, a2)
    with open(pt, "w", encoding="utf-8") as f:
        json.dump(TRANSCRIPT, f)
    return p1, p2, pt


@pytest.fixture
def compared(tmp_path):
    rows, summ = cmp.compare_video("VID", *write_video(tmp_path, tmp_path, tmp_path))
    return {r["line"]: r for r in rows}, summ


def test_iou():
    assert cmp.iou(dict(start=0, end=2), dict(start=1, end=3)) == pytest.approx(1 / 3)
    assert cmp.iou(dict(start=0, end=1), dict(start=2, end=3)) == 0.0


def test_loading_converts_units_and_skips_non_signed(tmp_path):
    p1, p2, _ = write_video(tmp_path, tmp_path, tmp_path)
    a1 = cmp.load_a1(p1)
    assert sorted(a1) == [2, 3, 4]                       # title row skipped
    assert a1[3]["start"] == 3.0 and a1[3]["end"] == 6.0  # ms -> s
    assert sorted(cmp.load_a2(p2)) == [2, 3]


def test_which_lines_each_annotator_has(compared):
    _, s = compared
    assert (s["a1_signed"], s["a2_signed"], s["both"]) == (3, 2, 2)
    assert s["only_a1"] == [4] and s["only_a2"] == [] and s["neither"] == [1]
    assert s["transcript_unsignable"] == 1


def test_boundary_numbers(compared):
    rows, s = compared
    assert rows[2]["d_start"] == 0.5 and rows[2]["d_end"] == 0.0
    assert rows[3]["iou"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["median_abs_boundary_diff_s"] == 0.5
    assert s["pct_boundaries_within_0_5s"] == 100.0
    assert s["mean_iou"] == pytest.approx(0.75, abs=1e-3)
    assert s["median_start_offset_a2_minus_a1_s"] == 0.5
    assert s["big_diff_lines"] == []
    assert s["signer_agree"] == 2


def test_habits(compared):
    _, s = compared
    assert s["a1_habits"]["whole_seconds_only"]
    assert s["a1_habits"]["n_zero_gaps"] == s["a1_habits"]["n_gaps"] == 2   # turns touch
    assert not s["a2_habits"]["whole_seconds_only"]


def test_big_diff_and_signer_disagreement(tmp_path):
    a2 = [["s1", "VID", "3.5", "6.0", "Sam", "2", "Hello there my friend"]]   # 3.5 s late, wrong signer
    rows, s = cmp.compare_video("VID", *write_video(tmp_path, tmp_path, tmp_path, a2=a2))
    assert s["big_diff_lines"] == [2]
    assert s["signer_disagree_lines"] == [2]


def test_cli_folder_mode(tmp_path):
    d1, d2, dt, out = (tmp_path / n for n in ("a1", "a2", "tr", "out"))
    for d in (d1, d2, dt):
        d.mkdir()
    write_video(d1, d2, dt)
    res = subprocess.run([sys.executable, SCRIPT, "--a1_dir", str(d1), "--a2_dir", str(d2),
                          "--transcript_dir", str(dt), "--out_dir", str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "Only annotator 1 marked lines: [4]" in res.stdout
    for name in ("per_turn.csv", "per_video.csv", "summary.json", "report.txt"):
        assert (out / name).exists()
    assert json.load(open(out / "summary.json"))["pooled"]["lines_both"] == 2


def test_cli_no_shared_videos(tmp_path):
    d1, d2 = tmp_path / "a1", tmp_path / "a2"
    d1.mkdir(); d2.mkdir()
    write_csv(d2 / "VID_annotations.csv", A2_HEADER, A2)   # no annotator-1 file
    res = subprocess.run([sys.executable, SCRIPT, "--a1_dir", str(d1), "--a2_dir", str(d2),
                          "--out_dir", str(tmp_path / "out")], capture_output=True, text=True)
    assert res.returncode != 0 and "no videos found" in res.stderr
