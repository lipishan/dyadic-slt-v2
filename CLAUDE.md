# CLAUDE.md — Dyadic SLT (v2)

Fresh codebase for **dyadic ASL → English translation with interaction memory**.
The full plan is in `docs/PLAN.md`; this file is the short version plus the rules for working here.

## Working rules
1. **Explain in plain language**, with a small example where it helps.
2. **Small changes per turn.** One focused change at a time; no large rewrites.
3. **Test before pushing.** Run `pytest tests/` and make sure it passes before every push.
4. **List the changed files at the end of every turn** (added / modified / deleted).
5. **Never drop a step, tag or analysis without asking the user first.**
6. **No access to ASL experts or Deaf signers.** For any step that needs ASL knowledge, look for a
   way to do it without an expert (automatic proxy, visual comparison a non-signer can do,
   English-side evidence), and say in the paper how it was done.
7. Push to GitHub; the user runs `git pull` on the server. No zip files.

## Research questions
- **RQ1 (method):** Does interaction memory improve dyadic SLT over no memory and short windows,
  using *predicted* (never gold) history at test time?
- **RQ2 (linguistics):** Which content improves, and how does that relate to articulation?
  Compare monologue (OpenASL, self-history only) with dialogue (self vs partner history).

## Plan in brief (section numbers match `docs/PLAN.md`)
- **§0a Annotation checks** ← *current step* (details below).
- **§0 Dyadic split:** split by whole conversation (70/10/20 by targets, ≥ 10 test conversations),
  stratified by conversation length, long turns and source; plus a signer-unseen test subset if
  possible. Freeze it as `splits/dyadic_split_v1.csv` **before any training** and never change it.
- **§1 Data:** per-video annotation files; count check (annotated stretches vs transcript lines);
  back-channel labels; fixed padding added in code; six per-turn tags for RQ2; automatic person
  boxes; OpenASL pseudo-dialogue; SHuBERT features per stream + MediaPipe keypoints.
  **Causality rule:** a turn may enter memory only if it ended before the current turn starts.
- **§1b Long turns:** sentence segmentation by monotonic DP alignment (training data only), checked
  against an OpenASL answer key, with a go/no-go rule. At test, segmentation never uses reference
  text; scoring is at whole-turn level.
- **§2 Model:** frozen SHuBERT → small adapter → memory (one token per previous turn: visual vector
  + speaker + turn distance, last 16 turns). Two decoders behind one interface: T5-base (dev,
  ablations) and Qwen3-4B-Base + LoRA (main, soft prefix, Base checkpoint only).
- **§3 Training stages with gates:** (1) OpenASL, no memory — true video must beat shuffled video;
  (2) OpenASL pseudo-dialogue with self-history; (3) dyadic fine-tune with self + partner memory.
- **§4 Evaluation:** chrF, BLEU, a neural metric, per-word NLL; conversation-level paired
  bootstrap, 3 seeds, 95% CIs; model selection on valid only; shuffled-video check at test.
- **§5 RQ2 analyses:** mention-number analysis (visual-only memory), articulation link, tags,
  monologue vs dyadic, stream ablation.
- **§6 Pitfalls:** see the table in the plan; most are enforced by tests in `tests/`.

## Current step: §0a annotation checks
- **Annotator 2's files are the reference annotation.** Annotator 1 was a tool: it is reported
  only as an *automatic* annotation (tool quality), never as a second human annotator; its
  confidence column is ignored.
- Annotator column names differ and are mapped in code: annotator 1 `turn_id`, annotator 2
  `turn_index` (joined on transcript line number).
- `data/check_annotations.py` checks annotator 2 against the transcript: unannotated non-trivial
  lines, missing signer (filled from the transcript speaker or a `Name (...):` text prefix, which
  is also stripped), signer/text mismatches, signing speed (0.8–6 words/s), partner overlaps, likely
  back-channels (short + overlapping). Output `turns_checked.csv` is the input to the split.
- Old v1 timestamps are a third source, used only for comparison and ablation ABL-DATA1, never as
  training targets.
- **Next:** count check on all videos → dyadic split (§0) → person boxes (can run in parallel).

## Code layout
- `data/` manifests, split script, causal history builder, count check, long-turn DP, tests
- `features/` person tracking, SHuBERT per stream, MediaPipe
- `model/` shared adapter + memory; `decoders/t5.py`, `decoders/qwen.py`
- `train.py` + one config file per experiment (no ad-hoc flags)
- `eval/`, `analysis/`
- `tests/` causality, alignment, shapes, no-reference-at-test, no conversation overlap in split
