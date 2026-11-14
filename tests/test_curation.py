"""Tests for run #3 data curation: audit rules, overlap scoring, the DeepCoder
wrapper stripping, and the committed-split leak guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlforge.adapters.deepcoder import strip_pi_wrapper, task_id_for, unfence
from rlforge.curation import audit_task, n_tests, overlap_pairs
from rlforge.schema.task import CodeTask
from rlforge.schema.testsuite import IOPair, TestSuite

STATEMENT = "You are given N integers. Print their sum. " * 5


def _stdio(prompt: str = STATEMENT, n: int = 6, tid: str = "T/1") -> CodeTask:
    pairs = [IOPair(stdin=f"{i}\n", stdout=f"{i}\n") for i in range(n)]
    return CodeTask(task_id=tid, source="x", prompt=prompt,
                    test_suites={"hidden": TestSuite(kind="stdin_stdout", io_pairs=pairs)})


# ── audit ──

def test_clean_task_is_kept():
    assert audit_task(_stdio(), min_tests=5) == []


@pytest.mark.parametrize("phrase", [
    "If there are multiple solutions, you may print any of them.",
    "If multiple solutions exist, any of them is accepted.",
    "If there are multiple such pairs, any of them will be accepted.",
])
def test_multi_answer_detected(phrase):
    assert "multi_answer" in audit_task(_stdio(STATEMENT + phrase), min_tests=5)


@pytest.mark.parametrize("phrase", [
    # benign uses of "any" that a loose pattern flagged on the real data
    "You can change at most k characters to any of the four directions.",
    "If there are multiple occurrences of the minimum value, select the first one.",
    "Return the minimum total cost possible from any valid division.",
])
def test_multi_answer_not_overmatched(phrase):
    assert "multi_answer" not in audit_task(_stdio(STATEMENT + phrase), min_tests=5)


def test_image_and_few_tests():
    assert "image" in audit_task(_stdio(STATEMENT + " ![fig](a.png)"), min_tests=5)
    assert "few_tests" in audit_task(_stdio(n=3), min_tests=5)
    assert "few_tests" not in audit_task(_stdio(n=3), min_tests=1)


def test_n_tests_skips_setup_line():
    suite = TestSuite(kind="assert", asserts=["sol = Solution()", "assert sol.f(1) == 1"])
    t = CodeTask(task_id="T/2", source="x", prompt=STATEMENT, entry_point="f",
                 test_suites={"hidden": suite})
    assert n_tests(t) == 1


# ── overlap ──

def test_copy_scores_high_boilerplate_low():
    test = _stdio("A frog jumps over stones numbered one to n and wants the cheapest path "
                  "to the last stone given heights h and a jump cost equal to the difference", tid="TEST/1")
    copy = _stdio(test.prompt + " Constraints follow.", tid="TRAIN/copy")
    other = _stdio("Count the vowels in a string and print the count for every query line "
                   "you read from standard input until the end of the file is reached now", tid="TRAIN/other")
    pairs = overlap_pairs([copy, other], [test])
    assert pairs[0][1] == "TRAIN/copy" and pairs[0][0] > 0.9
    assert all(tid != "TRAIN/other" for _, tid, _ in pairs)


# ── DeepCoder wrapper ──

HEADER = "Solve the following coding problem using the programming language python:\n\n"


def test_strip_stdio_wrapper():
    raw = (HEADER + "Body text.\n\nThe input will be given via stdin and the output should be "
           "printed to stdout by your code.\n\nNow solve the problem by providing the code.")
    assert strip_pi_wrapper(raw) == ("Body text.", "")


def test_strip_function_wrapper_keeps_starter():
    raw = (HEADER + "Body.\n\nWrite your solution by modifying this code:\n\n```python\n"
           "def f(a):\n\t\n```\n\nYour solution should implemented in the function \"f\". The inputs "
           "will be passed to it and it should return the correct solution.\n\n\nNow solve the "
           "problem and return the code.")
    statement, starter = strip_pi_wrapper(raw)
    assert statement == "Body." and starter.startswith("def f(a):")


def test_unknown_wrapper_is_rejected_not_guessed():
    assert strip_pi_wrapper("Some other prompt") is None


def test_ids_ignore_whitespace_and_case_and_unfence():
    assert task_id_for("Sum  Two\nNumbers") == task_id_for("sum two numbers")
    assert unfence("```python\nprint(1)\n```") == "print(1)"


# ── leak guard on committed splits ──

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits"


def test_run3_splits_disjoint():
    names = ["run3_train.json", "run3_val.json", "run3_test.json"]
    sets = {n: set(json.load(open(SPLITS / n))) for n in names if (SPLITS / n).exists()}
    if len(sets) < 2:
        pytest.skip("run #3 splits not built yet")
    items = list(sets.items())
    for i, (a, sa) in enumerate(items):
        for b, sb in items[i + 1:]:
            assert not sa & sb, f"{a} and {b} share task ids"
