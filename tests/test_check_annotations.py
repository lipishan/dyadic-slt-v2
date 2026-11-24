"""Tests for data/check_annotations.py on a tiny made-up video.

The fake video has 6 transcript lines:
  1  "[Title card]"                       not signing, not annotated  -> fine
  2  Alex: welcome line                   annotated, signer given
  3  Sam: thanks line                     annotated, signer left blank -> filled from transcript
  4  Alex: "Yes" while Sam still signs    short + overlapping         -> likely back-channel
  5  Sam: a real line                     NOT annotated               -> reported as missing
  6  "Alex (Daily Moth): Together ..."    no speaker anywhere         -> signer from text prefix
"""
import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "data", "check_annotations.py")
sys.path.insert(0, os.path.join(ROOT, "data"))
import check_annotations as ca  # noqa: E402

HEADER = ["sign_id", "video_id", "start_time", "end_time", "active_signer", "turn_index", "text"]
TRANSCRIPT = {"turns": [
    {"speaker": None, "text": "[Title card]"},
    {"speaker": "Alex", "text": "Welcome to the show today friends"},
    {"speaker": "Sam", "text": "Thanks for having me here"},
    {"speaker": "Alex", "text": "Yes"},
    {"speaker": "Sam", "text": "I grew up in Ohio and moved later"},
    {"speaker": None, "text": "Alex (Daily Moth): Together we can do this"},
]}
ANN = [
    ["s1", "VID", "0.0", "3.0", "Alex", "2", "Welcome to the show today friends"],
    ["s2", "VID", "3.5", "6.0", "", "3", "Thanks for having me here"],
    ["s3", "VID", "5.5", "6.0", "Alex", "4", "Yes"],
    ["s4", "VID", "7.0", "10.0", "", "6", "Alex (Daily Moth): Together we can do this"],
]


def write_video(folder, ann_rows=ANN, transcript=TRANSCRIPT, vid="VID"):
    ann_path = os.path.join(folder, f"{vid}_annotations.csv")
    tr_path = os.path.join(folder, f"{vid}.transcript.json")
    with open(ann_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(ann_rows)
    with open(tr_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f)
    return ann_path, tr_path


@pytest.fixture
def checked(tmp_path):
    ann, tr = write_video(tmp_path)
    rows, summ = ca.check_video("VID", ann, tr)
    return {r["line"]: r for r in rows}, summ


# --- small helpers -----------------------------------------------------------

def test_norm_ignores_case_quotes_and_spaces():
    assert ca.norm("  “Hello”   World ") == ca.norm('"hello" world')


def test_unsignable():
    assert ca.unsignable("[Music]")
    assert ca.unsignable("https://example.com")
    assert ca.unsignable("Image of a dog")
    assert ca.unsignable("— —")
    assert not ca.unsignable("Hello there")


def test_split_prefix_only_for_known_speakers():
    assert ca.split_prefix("Alex (Daily Moth): Hi there", ["Alex", "Sam"]) == ("Alex", "Hi there")
    assert ca.split_prefix("Bob: Hi there", ["Alex", "Sam"]) == (None, "Bob: Hi there")


# --- one whole video ------------------------------------------------------------

def test_signer_given_is_kept(checked):
    rows, _ = checked
    assert rows[2]["signer"] == "Alex" and rows[2]["signer_source"] == "annotator"
    assert rows[2]["flags"] == []


def test_blank_signer_filled_from_transcript(checked):
    rows, _ = checked
    assert rows[3]["signer"] == "Sam"
    assert "signer_filled_from_transcript" in rows[3]["flags"]


def test_signer_filled_from_text_prefix_and_prefix_stripped(checked):
    rows, _ = checked
    assert rows[6]["signer"] == "Alex"
    assert "signer_filled_from_text_prefix" in rows[6]["flags"]
    assert rows[6]["text_clean"] == "Together we can do this"
    assert rows[6]["words"] == 5          # the name is not counted as signed words


def test_overlap_and_backchannel(checked):
    rows, summ = checked
    assert rows[3]["overlaps_partner"] and rows[4]["overlaps_partner"]
    assert rows[4]["likely_backchannel"]          # short "Yes" over the partner
    assert not rows[3]["likely_backchannel"]      # long turn, only overlapped
    assert not rows[2]["overlaps_partner"]
    assert summ["likely_backchannel"] == [4]


def test_missing_lines_skip_titles(checked):
    _, summ = checked
    missing = [i for i, _ in summ["not_annotated_maybe_signed"]]
    assert missing == [5]                 # line 1 is a title card, so not reported
    assert summ["transcript_lines"] == 6 and summ["transcript_unsignable"] == 1
    assert summ["annotated"] == 4


def test_mismatches_and_bad_times(tmp_path):
    ann = [
        ["s1", "VID", "0.0", "3.0", "Sam", "2", "Welcome to the show today friends"],  # wrong signer
        ["s2", "VID", "4.0", "4.5", "Sam", "3", "Thanks for having me here and more words now"],
        ["s3", "VID", "9.0", "8.0", "Alex", "4", "Yes"],                               # end < start
    ]
    a, t = write_video(tmp_path, ann)
    rows = {r["line"]: r for r in ca.check_video("VID", a, t)[0]}
    assert "signer_differs_from_transcript" in rows[2]["flags"]
    assert "text_differs_from_transcript" in rows[3]["flags"]
    assert "speed_implausible" in rows[3]["flags"]        # 10 words in 0.5 s = 20 w/s
    assert "end_before_start" in rows[4]["flags"]


def test_works_without_transcript(tmp_path):
    a, _ = write_video(tmp_path)
    rows, summ = ca.check_video("VID", a, None)
    assert len(rows) == 4 and summ["transcript_lines"] is None


# --- command line ----------------------------------------------------------------

def test_cli_folder_mode_writes_outputs(tmp_path):
    ann_dir, tr_dir, out = tmp_path / "ann", tmp_path / "tr", tmp_path / "out"
    ann_dir.mkdir(); tr_dir.mkdir()
    a, t = write_video(ann_dir)
    os.replace(t, tr_dir / "VID.transcript.json")
    res = subprocess.run([sys.executable, SCRIPT, "--ann_dir", str(ann_dir),
                          "--transcript_dir", str(tr_dir), "--out_dir", str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "Video VID" in res.stdout
    with open(out / "turns_checked.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert json.load(open(out / "summary.json"))[0]["video"] == "VID"
    assert (out / "report.txt").exists()


def test_cli_empty_annotation_file_fails_cleanly(tmp_path):
    a, t = write_video(tmp_path, ann_rows=[])
    res = subprocess.run([sys.executable, SCRIPT, "--ann", a, "--transcript", t,
                          "--out_dir", str(tmp_path / "out")], capture_output=True, text=True)
    assert res.returncode != 0
    assert "no turns" in res.stderr
