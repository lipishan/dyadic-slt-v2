"""Tests for data/review_list.py: run the real pipeline on a tiny video, then build the list."""
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))
import review_list as rl  # noqa: E402

TRANSCRIPT = {"turns": [
    {"speaker": "Alex", "text": "Headline here"},          # 1: tool says signed, annotator 2 skipped
    {"speaker": "Alex", "text": "Hello there my friend"},  # 2: tool agrees
    {"speaker": "Sam", "text": "Nice to meet you"},        # 3: tool times far off + other signer
]}
TOOL = [["turn_id", "start_ms", "end_ms", "aligned_signer", "annotation_type", "transcript"],
        ["1", "0", "2000", "Alex", "signed", "Headline here"],
        ["2", "2000", "5000", "Alex", "signed", "Hello there my friend"],
        ["3", "9000", "12000", "Alex", "signed", "Nice to meet you"]]
ANN2 = [["sign_id", "video_id", "start_time", "end_time", "active_signer", "turn_index", "text"],
        ["s1", "VID", "2.2", "5.0", "Alex", "2", "Hello there my friend"],
        ["s2", "VID", "5.5", "7.5", "Sam", "3", "Nice to meet you"]]


def run(*args):
    res = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return res.stdout


def setup(tmp_path):
    d = {n: tmp_path / n for n in ("tool", "ann2", "tr", "agree", "checked", "review")}
    for n in ("tool", "ann2", "tr"):
        d[n].mkdir()
    for path, rows in ((d["tool"] / "VID_asl_turn_alignment.csv", TOOL), (d["ann2"] / "VID_annotations.csv", ANN2)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
    (d["tr"] / "VID.transcript.json").write_text(json.dumps(TRANSCRIPT), encoding="utf-8")
    run(os.path.join(ROOT, "data", "compare_annotators.py"), "--a1_dir", str(d["tool"]),
        "--a2_dir", str(d["ann2"]), "--transcript_dir", str(d["tr"]), "--out_dir", str(d["agree"]))
    run(os.path.join(ROOT, "data", "check_annotations.py"), "--ann_dir", str(d["ann2"]),
        "--transcript_dir", str(d["tr"]), "--out_dir", str(d["checked"]))
    return d


def test_lines_and_reasons(tmp_path):
    d = setup(tmp_path)
    rows = {r["line"]: r for r in rl.build(tool_dir=d["agree"], checked_dir=d["checked"])["VID"]}
    assert sorted(rows) == [1, 3]                               # line 2 is fine, not listed
    assert rows[1]["n_reasons"] == 2                            # tool says signed + not annotated
    assert "transcript line not annotated" in rows[1]["reasons"]
    assert "tool times differ a lot" in rows[3]["reasons"]
    assert "tool names a different signer" in rows[3]["reasons"]
    assert rows[3]["a2_start"] == "5.5" and rows[3]["tool_start"] == "9.0"


def test_cli_writes_one_file_per_video(tmp_path):
    d = setup(tmp_path)
    out = run(os.path.join(ROOT, "data", "review_list.py"), "--tool_agreement", str(d["agree"]),
              "--checked", str(d["checked"]), "--out_dir", str(d["review"]))
    assert "VID: 2 lines to re-watch (1 flagged by 2+ reasons)" in out
    assert (d["review"] / "review_VID.csv").exists()
