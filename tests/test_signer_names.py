"""Tests for data/signer_names.py (suggested name -> person grouping)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))
import signer_names as sn  # noqa: E402


def T(video, signer, n=1):
    return [dict(video=video, signer=signer)] * n


def groups(rows):
    g = {}
    for r in rows:
        g.setdefault(r["person_id"], set()).add(r["name"])
    return g


def test_groupings():
    turns = (T("v1", "Renca Dunn") + T("v2", "RENCA DUNN") + T("v3", "Renca", 3)        # same words
             + T("v4", "Callie") + T("v4", "Callie Frye")                              # same video
             + T("v5", "Troy") + T("v6", "Troy Kotsur")                                # one candidate
             + T("v7", "Tim") + T("v8", "Tim Cook") + T("v9", "Tim Burton")            # ambiguous
             + T("v10", "Alex; Maor Ben Zeev") + T("v10", "Maor Ben Zeev") + T("v10", "Alex")
             + T("v11", "Interpreter") + T("v12", "Interpreter") + T("v13", "DAILY MOTH"))
    rows = sn.build(turns)
    g = groups(rows)
    by = {r["name"]: r for r in rows if r["n_videos"] >= 1}
    assert {"Renca Dunn", "RENCA DUNN", "Renca"} in g.values()
    assert {"Callie", "Callie Frye"} in g.values()
    assert by["Troy"]["person_id"] == "Troy Kotsur" and by["Troy"]["status"] == "check"
    assert by["Tim"]["person_id"] == "Tim" and "several longer names" in by["Tim"]["note"]
    assert by["Maor Ben Zeev"]["person_id"] != by["Alex"]["person_id"]   # combined label does not merge
    assert by["Alex; Maor Ben Zeev"]["note"] == "several people in one turn"
    interp = [r["person_id"] for r in rows if r["name"] == "Interpreter"]
    assert sorted(interp) == ["v11:Interpreter", "v12:Interpreter"]       # kept apart per video
    assert by["DAILY MOTH"]["note"] == "show label without a person's name"
