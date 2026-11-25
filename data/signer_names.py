#!/usr/bin/env python3
"""Draft table of signer names -> person id, for a person to confirm.

The same person appears under many spellings ("Alex", "Alex Abenchuchan",
"THE DAILY MOTH (ALEX)"), and different people can share a first name ("Tim" in two
videos). The split needs real person ids, so this script SUGGESTS groupings and says how
sure it is; you correct the `person_id` column by hand where needed.

  python data/signer_names.py --turns merged/turns_merged.csv --out signer_names.csv

Suggestion rules:
  auto   same words, ignoring case/punctuation ("Renca Dunn" = "RENCA DUNN"), or one name's
         words are inside the other's AND both occur in the same video ("Callie" and
         "Callie Frye" in one video)
  check  one-word name inside exactly one longer name, never in the same video
         ("Alex" -> "Alex Abenchuchan"); please confirm
  check  label without a person's name ("DAILY MOTH"), generic role ("Interpreter",
         "Woman", "... (name unverified)") or several people ("Lorne and Emily"):
         kept separate per video until you decide
Columns: person_id (edit this), name, status, note, n_videos, n_units, videos
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_annotations import name_tokens  # noqa: E402

GENERIC = re.compile(r"\b(interpreter|woman|man|unverified|unknown|speaker|guest|host|interviewer|"
                     r"narrator|voice|audience|reporter only)\b", re.I)
SEVERAL = re.compile(r"\b(and|all|both)\b|[;/&+]", re.I)


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def build(turns):
    vids, units = defaultdict(set), Counter()
    for r in turns:
        n = (r.get("signer") or "").strip()
        if n:
            vids[n].add(r["video"]); units[n] += 1
    names = sorted(vids)
    tok = {n: frozenset(name_tokens(n)) for n in names}
    special = {}
    for n in names:
        if SEVERAL.search(n):
            special[n] = "several people in one turn"
        elif GENERIC.search(n):
            special[n] = "generic role, not a name"
        elif not tok[n]:
            special[n] = "show label without a person's name"
    parent = {n: n for n in names}
    status = {n: "auto" for n in names}
    note = {n: "" for n in names}
    normal = [n for n in names if n not in special]

    for a in normal:                                         # identical words
        for b in normal:
            if a < b and tok[a] == tok[b]:
                parent[find(parent, b)] = find(parent, a)
    for a in normal:                                         # words inside, same video
        for b in normal:
            if a != b and tok[a] < tok[b] and vids[a] & vids[b]:
                parent[find(parent, a)] = find(parent, b)
    for a in normal:                                         # one-word name, one candidate
        if len(tok[a]) != 1 or find(parent, a) != a:
            continue
        cands = {find(parent, b) for b in normal if tok[a] < tok[b]}
        if len(cands) == 1:
            parent[a] = cands.pop()
            status[a], note[a] = "check", "one-word name matched to the only longer name containing it"
        elif len(cands) > 1:
            status[a], note[a] = "check", "one-word name; several longer names contain it"

    groups = defaultdict(list)
    for n in normal:
        groups[find(parent, n)].append(n)
    rows = []
    for members in groups.values():
        pid = max(members, key=lambda m: (len(vids[m]) + units[m] / 1000, len(m)))
        for m in members:
            rows.append(dict(person_id=pid, name=m, status=status[m], note=note[m],
                             n_videos=len(vids[m]), n_units=units[m], videos=" ".join(sorted(vids[m])[:6])))
    for n, why in special.items():                           # kept separate per video
        for v in sorted(vids[n]):
            rows.append(dict(person_id=f"{v}:{n}", name=n, status="check", note=why, n_videos=1,
                             n_units=sum(1 for r in turns if r["video"] == v and r["signer"] == n),
                             videos=v))
    rows.sort(key=lambda r: (r["person_id"].lower(), -r["n_units"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--out", default="signer_names.csv")
    a = ap.parse_args()
    with open(a.turns, newline="", encoding="utf-8") as f:
        rows = build(list(csv.DictReader(f)))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    people = {r["person_id"] for r in rows}
    s = Counter(r["status"] for r in rows)
    print(f"{len({r['name'] for r in rows})} names -> {len(people)} suggested people; "
          f"{s['check']} rows to check, {s['auto']} automatic. Wrote {a.out}")


if __name__ == "__main__":
    main()
