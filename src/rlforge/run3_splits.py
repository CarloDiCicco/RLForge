"""
run3_splits.py
──────────────
Pure logic for turning the run #3 informativeness sweep into train/val splits
(used by scripts/build_run3_splits.py and its tests). No file I/O here.

A sweep record is one training task answered `rounds` times by the untrained
model (scripts/sweep_run3.py). The steps, in order:

1. every task is re-checked against ALL quality rules (audit_task, the grading
   risks, gold check, not in the test set) — the pool was already filtered on
   2026-11-14, so the old rules should drop 0; re-running them is a guard, and the
   count proves it;
2. only tasks with MIXED results (0 < passes < rounds) give GRPO a signal;
3. each mixed task is labelled by WHY its failing answers failed ("cutoff",
   "code" or "both"), because cut-off-only tasks mostly teach shorter thinking,
   not better code (RESEARCH_LOG 2026-11-19) and the split should report it;
4. the validation set (PAPER_PLAN.md, "Validation (fixed)") is 25 code-side
   tasks (code-only and both in proportion), 25 cut-off-only, 30 always-solved
   and 20 never-solved; the always/never tasks show whether training breaks
   what the model already did, or unlocks what it never did;
5. the train list is the rest of the mixed tasks, ORDERED so every batch of 8
   holds 4 code-side and 4 cut-off-side tasks (option A, decided 2026-11-21):
   the mix is the same at every step, so a run stopped early is not a
   differently mixed run. The longer side's leftover tasks are spare.

"Code side" = code-only + both (at least one failure was wrong code);
"cut-off side" = cut-off-only (every failure ran out of tokens).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any


def outcome(n_passed: int, rounds: int) -> str:
    """'mixed', 'always' or 'never' from the number of passing answers."""
    if n_passed == 0:
        return "never"
    if n_passed == rounds:
        return "always"
    return "mixed"


def answer_cut_off(answer: dict[str, Any]) -> bool:
    """An answer failed because it ran out of tokens, not because of its code.

    `closed_thinking` False means no `</think>` → no code was graded at all.
    `hit_token_cap` with closed thinking means the code may have been cut too;
    both count as cut-off. Only called on FAILING answers (a passing answer that
    hit the cap still had complete, correct code)."""
    return bool(answer["hit_token_cap"]) or not answer["closed_thinking"]


def failure_kind(record: dict[str, Any]) -> str:
    """Why a mixed task's failing answers failed: 'cutoff' (all cut off),
    'code' (none cut off: wrong output, crash, timeout, no code block) or 'both'."""
    fails = [a for a in record["answers"] if not a["passed"]]
    if not fails:
        raise ValueError(f"{record['task_id']}: failure_kind called on a task with no failures")
    n_cut = sum(answer_cut_off(a) for a in fails)
    if n_cut == len(fails):
        return "cutoff"
    if n_cut == 0:
        return "code"
    return "both"


def validate_record(record: dict[str, Any], rounds: int) -> str | None:
    """Return a problem description, or None if the record is well formed.
    Catches a half-written or inconsistent line before it reaches a split."""
    answers = record.get("answers")
    if not isinstance(answers, list) or len(answers) != rounds:
        return f"has {len(answers) if isinstance(answers, list) else 'no'} answers, expected {rounds}"
    if sum(bool(a.get("passed")) for a in answers) != record.get("n_passed"):
        return "n_passed does not match the answers"
    for a in answers:
        for key in ("passed", "hit_token_cap", "closed_thinking", "text"):
            if key not in a:
                return f"answer missing field {key!r}"
    return None


def stratified_val(
    labels: dict[str, str], n_val: int, seed: int
) -> tuple[list[str], list[str]]:
    """Split task ids into (train, val) with `n_val` val tasks, keeping each
    label's share in val as close as possible to its share overall.

    Largest-remainder rounding gives exact `n_val` in total; sorting before the
    seeded shuffle makes the result independent of input order."""
    if n_val < 0 or n_val > len(labels):
        raise ValueError(f"n_val={n_val} but only {len(labels)} tasks")
    by_label: dict[str, list[str]] = defaultdict(list)
    for tid, lab in labels.items():
        by_label[lab].append(tid)
    total = len(labels)
    # ideal (fractional) val count per label, floored, then the leftover seats go
    # to the labels with the largest remainders (ties broken by label name)
    ideal = {lab: n_val * len(ids) / total for lab, ids in by_label.items()} if total else {}
    take = {lab: int(x) for lab, x in ideal.items()}
    leftover = n_val - sum(take.values())
    for lab in sorted(ideal, key=lambda k: (-(ideal[k] - take[k]), k))[:leftover]:
        take[lab] += 1
    rng = random.Random(seed)
    train, val = [], []
    for lab in sorted(by_label):
        ids = sorted(by_label[lab])
        rng.shuffle(ids)
        val += ids[: take[lab]]
        train += ids[take[lab]:]
    return sorted(train), sorted(val)


def label_counts(ids: list[str], labels: dict[str, str]) -> dict[str, int]:
    return dict(Counter(labels[t] for t in ids))


# ── option A: two sides, constant 4 + 4 per batch ──

CODE_SIDE_KINDS = ("code", "both")


def side(kind: str) -> str:
    """'code' for tasks with at least one wrong-code failure (code-only, both),
    'cutoff' for tasks whose every failure ran out of tokens."""
    if kind not in ("code", "both", "cutoff"):
        raise ValueError(f"unknown failure kind {kind!r}")
    return "code" if kind in CODE_SIDE_KINDS else "cutoff"


def random_draw(ids: list[str], n: int, seed: int) -> tuple[list[str], list[str]]:
    """(rest, drawn): `n` ids drawn at random. Same rule as stratified_val with a
    single label (sorted, then seeded shuffle), so it does not depend on input order."""
    return stratified_val({t: "_" for t in ids}, n, seed)


def interleave_batches(
    code_ids: list[str], cutoff_ids: list[str], per_side: int, seed: int
) -> tuple[list[str], list[str]]:
    """(order, spare): a training order where every consecutive block of
    2*per_side tasks has exactly `per_side` code-side and `per_side` cut-off
    tasks. Each side is shuffled once (seeded) to decide which tasks come early;
    the order inside a block is shuffled too, so position in the block carries
    no information (it does not change the update: one step sums over the block).

    The number of blocks is set by the SHORTER side; the other side's leftover
    tasks are returned as `spare` (not trained on, kept for a longer run)."""
    if per_side < 1:
        raise ValueError(f"per_side must be >= 1, got {per_side}")
    rng = random.Random(seed)
    code, cut = sorted(code_ids), sorted(cutoff_ids)
    rng.shuffle(code)
    rng.shuffle(cut)
    n_blocks = min(len(code), len(cut)) // per_side
    order: list[str] = []
    for b in range(n_blocks):
        block = code[b * per_side:(b + 1) * per_side] + cut[b * per_side:(b + 1) * per_side]
        rng.shuffle(block)
        order += block
    used = n_blocks * per_side
    spare = sorted(code[used:] + cut[used:])
    return order, spare
