"""Tests for the run #3 split builder: the grading-risk rules added 2026-11-21
(phrases taken from real tasks, both hits and look-alikes that must NOT hit)
and the split logic (failure kind, record checks, stratified validation)."""

from __future__ import annotations

import pytest

from rlforge.curation import audit_task, grading_risk_reasons
from rlforge.run3_splits import (
    failure_kind,
    interleave_batches,
    outcome,
    random_draw,
    side,
    stratified_val,
    validate_record,
)
from rlforge.schema.task import CodeTask
from rlforge.schema.testsuite import IOPair, TestSuite

BASE = "You are given N integers. Print their sum. " * 5


def _task(extra: str) -> CodeTask:
    pairs = [IOPair(stdin=f"{i}\n", stdout=f"{i}\n") for i in range(6)]
    return CodeTask(task_id="T/1", source="x", prompt=BASE + extra,
                    test_suites={"hidden": TestSuite(kind="stdin_stdout", io_pairs=pairs)})


# ── rules ──

def test_image_placeholder_now_caught():
    t = _task("Pig can use teleports. [Image] Determine if Pig can visit the friend.")
    assert "image" in audit_task(t, min_tests=5)


@pytest.mark.parametrize("phrase", [
    "The figure below shows the given graph.",
    "The picture shows how all 10 decimal digits are displayed:",
    "For example, 2 + 3 is denoted as in Figure 1.",
    "the following figure shows a frame which has a height of 6 cm",
    "The following figure visualizes the input and the answer for Sample 1.",
])
def test_figure_ref_hits(phrase):
    assert "figure_ref" in grading_risk_reasons(_task(phrase))


@pytest.mark.parametrize("phrase", [
    "You have to figure out how many squares we repainted.",
    "Paul would like to take a picture including at least k of them.",
    "It is a simple game that uses only picture cards.",
])
def test_figure_ref_lookalikes_not_hit(phrase):
    assert "figure_ref" not in grading_risk_reasons(_task(phrase))


@pytest.mark.parametrize("phrase", [
    "Your output will be considered correct if the absolute or relative error does not exceed 10^{-6}.",
    "Answers with an absolute or relative error of at most 10^{-9} will be accepted.",
    "Your answer for each city will be considered correct if its relative error does not exceed 10^{ - 6}.",
])
def test_float_tolerance_hits(phrase):
    assert "float_tolerance" in grading_risk_reasons(_task(phrase))


def test_clean_task_has_no_risk():
    assert grading_risk_reasons(_task("Print the count modulo 10^9 + 7.")) == []
    # real false hit of an earlier, looser rule (resistor colour-code task)
    assert grading_risk_reasons(_task("how tight a tolerance that value has")) == []


# ── split logic ──

def _ans(passed, cap=False, closed=True):
    return {"passed": passed, "hit_token_cap": cap, "closed_thinking": closed, "text": "x"}


def _rec(tid, answers):
    return {"task_id": tid, "n_passed": sum(a["passed"] for a in answers), "answers": answers}


def test_outcome():
    assert [outcome(k, 4) for k in (0, 2, 4)] == ["never", "mixed", "always"]


def test_failure_kind():
    assert failure_kind(_rec("a", [_ans(True), _ans(False, closed=False), _ans(False, cap=True), _ans(True)])) == "cutoff"
    assert failure_kind(_rec("b", [_ans(True), _ans(False), _ans(False), _ans(True)])) == "code"
    assert failure_kind(_rec("c", [_ans(True), _ans(False), _ans(False, closed=False), _ans(True)])) == "both"
    # a PASSING answer that hit the cap is not a failure and does not count
    assert failure_kind(_rec("d", [_ans(True, cap=True), _ans(False), _ans(True), _ans(True)])) == "code"
    with pytest.raises(ValueError):
        failure_kind(_rec("e", [_ans(True)] * 4))


def test_validate_record():
    good = _rec("a", [_ans(True), _ans(False), _ans(False), _ans(True)])
    assert validate_record(good, 4) is None
    assert "answers" in validate_record(_rec("b", [_ans(True)] * 3), 4)
    bad_count = dict(good, n_passed=3)
    assert "n_passed" in validate_record(bad_count, 4)
    no_field = _rec("c", [_ans(True), _ans(False), _ans(False), {"passed": True, "text": "x"}])
    assert "missing field" in validate_record(no_field, 4)


def test_stratified_val_proportions_and_disjoint():
    labels = {f"c{i}": "cutoff" for i in range(60)} | {f"k{i}": "code" for i in range(30)} \
        | {f"b{i}": "both" for i in range(10)}
    train, val = stratified_val(labels, 20, seed=0)
    assert len(val) == 20 and len(train) == 80
    assert not set(train) & set(val) and set(train) | set(val) == set(labels)
    kinds = [labels[t] for t in val]
    assert (kinds.count("cutoff"), kinds.count("code"), kinds.count("both")) == (12, 6, 2)


def test_stratified_val_deterministic_and_order_free():
    labels = {f"t{i}": ("cutoff" if i % 3 else "code") for i in range(50)}
    a = stratified_val(labels, 7, seed=1)
    b = stratified_val(dict(reversed(list(labels.items()))), 7, seed=1)
    assert a == b and len(a[1]) == 7
    assert stratified_val(labels, 7, seed=2) != a


def test_stratified_val_bounds():
    with pytest.raises(ValueError):
        stratified_val({"a": "code"}, 2, seed=0)
    assert stratified_val({"a": "code"}, 0, seed=0) == (["a"], [])


# ── option A: sides and the 4 + 4 order ──

def test_side():
    assert [side(k) for k in ("code", "both", "cutoff")] == ["code", "code", "cutoff"]
    with pytest.raises(ValueError):
        side("never")


def test_random_draw():
    ids = [f"t{i}" for i in range(30)]
    rest, drawn = random_draw(ids, 10, seed=0)
    assert len(drawn) == 10 and set(rest) | set(drawn) == set(ids) and not set(rest) & set(drawn)
    assert random_draw(list(reversed(ids)), 10, seed=0) == (rest, drawn)


def test_interleave_every_batch_is_4_plus_4():
    code = [f"k{i}" for i in range(18)]   # shorter side: 18 // 4 = 4 batches
    cut = [f"c{i}" for i in range(40)]
    order, spare = interleave_batches(code, cut, per_side=4, seed=0)
    assert len(order) == 32
    for b in range(4):
        block = order[b * 8:(b + 1) * 8]
        assert sum(t.startswith("k") for t in block) == 4
    # nothing lost, nothing repeated
    assert len(set(order)) == 32 and set(order) | set(spare) == set(code) | set(cut)
    assert not set(order) & set(spare)
    assert sum(t.startswith("k") for t in spare) == 2 and sum(t.startswith("c") for t in spare) == 24


def test_interleave_deterministic_and_order_free():
    code, cut = [f"k{i}" for i in range(12)], [f"c{i}" for i in range(12)]
    a = interleave_batches(code, cut, 4, seed=3)
    assert a == interleave_batches(code[::-1], cut[::-1], 4, seed=3)
    assert a != interleave_batches(code, cut, 4, seed=4)
    # not alphabetical: the old builder wrote a sorted list
    assert a[0] != sorted(a[0])
