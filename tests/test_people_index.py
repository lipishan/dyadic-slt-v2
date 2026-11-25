"""Tests for data/people_index.py without ffmpeg: unit picking, video lookup, page building."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))
import people_index as pi  # noqa: E402

NAMES = [dict(person_id="Alex", name="Alex", videos="V1 V2", status="auto", note=""),
         dict(person_id="Alex", name="Alex Abenchuchan", videos="V3", status="check", note="one-word name"),
         dict(person_id="V2:Woman", name="Woman", videos="V2", status="check", note="generic role")]


def U(v, who, s, e, conf="plausible", bc="False"):
    return dict(video=v, signer=who, start=str(s), end=str(e), confidence=conf, likely_backchannel=bc)


def test_pick_longest_clean_units_from_different_videos():
    turns = [U("V1", "Alex", 0, 5), U("V1", "Alex", 10, 30), U("V2", "Alex", 0, 50, conf="suspicious"),
             U("V2", "Alex", 60, 70), U("V3", "Alex Abenchuchan", 0, 8), U("V2", "Woman", 5, 9),
             U("V2", "Woman", 20, 21, bc="True")]
    picks = pi.pick_units(turns, NAMES, per_person=2)
    assert picks["Alex"] == [("V1", 20.0, "Alex"), ("V2", 65.0, "Alex")]    # suspicious skipped
    assert picks["V2:Woman"] == [("V2", 7.0, "Woman")]                       # back-channel skipped


def test_find_video(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "-HqBPJtdYfU.mp4").write_bytes(b"x")
    assert pi.find_video(str(tmp_path), "-HqBPJtdYfU").endswith("-HqBPJtdYfU.mp4")
    assert pi.find_video(str(tmp_path), "nope") is None


def test_page_has_cards_pictures_and_checks():
    picks = {"Alex": [("V1", 20.0, "Alex")], "V2:Woman": [("V2", 7.0, "Woman")]}
    page = pi.build_page(NAMES, picks, {("Alex", "V1"): b"\xff\xd8jpeg"})
    assert page.startswith("<!DOCTYPE html>")
    assert "Alex, Alex Abenchuchan" in page and "data:image/jpeg;base64," in page
    assert "no picture (V2 @ 7s" in page
    assert page.count('class="card check"') == 2                            # both have a 'check' row
