"""Tests for adapters/humaneval.py — the HumanEval+ dataset adapter.

Skipped entirely if evalplus is not installed (optional [data] extra), same
pattern as test_sandbox_bwrap.py skipping when bwrap is unusable on the host.
"""

from __future__ import annotations

import pytest

pytest.importorskip("evalplus")

from rlforge.adapters.humaneval import _parse_check_asserts, load_humaneval_plus
from rlforge.sandbox.subprocess_sandbox import SubprocessSandbox
from rlforge.verify.compiler import suite_to_test_code

# The real check() body for HumanEval/0, captured from evalplus output this
# session (../ReasoningLab/data/humaneval_plus.jsonl, line 1).
_REAL_CHECK_BODY = (
    "\n\nMETADATA = {\n"
    "    'author': 'jt',\n"
    "    'dataset': 'test'\n"
    "}\n\n\n"
    "def check(candidate):\n"
    "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
    "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
    "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n"
    "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n"
    "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n"
    "    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n"
    "    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n\n"
    "check(has_close_elements)"
)


def test_parse_check_asserts_extracts_exact_count():
    asserts = _parse_check_asserts(_REAL_CHECK_BODY, "has_close_elements")
    assert len(asserts) == 7


def test_parse_check_asserts_substitutes_entry_point():
    asserts = _parse_check_asserts(_REAL_CHECK_BODY, "has_close_elements")
    assert all("candidate" not in a for a in asserts)
    assert all(a.startswith("assert has_close_elements(") for a in asserts)


def test_parse_check_asserts_drops_boilerplate():
    asserts = _parse_check_asserts(_REAL_CHECK_BODY, "has_close_elements")
    joined = "\n".join(asserts)
    assert "METADATA" not in joined
    assert "def check" not in joined
    assert not joined.rstrip().endswith("check(has_close_elements)")


def test_parse_check_asserts_raises_on_no_asserts():
    with pytest.raises(ValueError, match="no assert statements"):
        _parse_check_asserts("def check(candidate):\n    pass\n", "fn")


# The real check() body for HumanEval/1: two of its four asserts span multiple
# physical lines (a list literal broken across lines). A line-based
# `^assert...$` regex truncates these into invalid syntax — this is the actual
# bug caught by running the full pipeline against real HumanEval+ data.
_MULTILINE_CHECK_BODY = (
    "\n\nMETADATA = {\n"
    "    'author': 'jt',\n"
    "    'dataset': 'test'\n"
    "}\n\n\n"
    "def check(candidate):\n"
    "    assert candidate('(()()) ((())) () ((())()())') == [\n"
    "        '(()())', '((()))', '()', '((())()())'\n"
    "    ]\n"
    "    assert candidate('() (()) ((())) (((())))') == [\n"
    "        '()', '(())', '((()))', '(((())))'\n"
    "    ]\n"
    "    assert candidate('(()(())((())))') == [\n"
    "        '(()(())((())))'\n"
    "    ]\n"
    "    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n"
)


def test_parse_check_asserts_handles_multiline_assert_statements():
    asserts = _parse_check_asserts(_MULTILINE_CHECK_BODY, "separate_paren_groups")
    assert len(asserts) == 4
    for a in asserts:
        compile(a, "<test>", "exec")  # each must be syntactically complete
        assert "candidate" not in a
        assert a.startswith("assert separate_paren_groups(")


# The real check() body for HumanEval/32 ("find_zero"): its single assert is
# nested inside a `for` loop seeded with a fixed RNG, depending on loop-local
# variables (coeffs, solution) — not a standalone statement. 5/164 real
# HumanEval+ tasks have this shape; splitting would silently break them.
_NESTED_ASSERT_CHECK_BODY = (
    "\n\nMETADATA = {}\n\n\n"
    "def check(candidate):\n"
    "    import math\n"
    "    import random\n"
    "    rng = random.Random(42)\n"
    "    import copy\n"
    "    for _ in range(100):\n"
    "        ncoeff = 2 * rng.randint(1, 4)\n"
    "        coeffs = []\n"
    "        for _ in range(ncoeff):\n"
    "            coeff = rng.randint(-10, 10)\n"
    "            if coeff == 0:\n"
    "                coeff = 1\n"
    "            coeffs.append(coeff)\n"
    "        solution = candidate(copy.deepcopy(coeffs))\n"
    "        assert math.fabs(poly(coeffs, solution)) < 1e-4\n\n"
)


def test_parse_check_asserts_keeps_nested_assert_as_one_block():
    asserts = _parse_check_asserts(_NESTED_ASSERT_CHECK_BODY, "find_zero")
    # Not split: the assert depends on loop-local state, so the whole
    # procedural body is kept as a single runnable entry.
    assert len(asserts) == 1
    block = asserts[0]
    compile(block, "<test>", "exec")  # the whole block must be valid Python
    assert "candidate" not in block
    assert "find_zero(" in block
    assert "for _ in range(100)" in block  # loop context preserved


# The real check() body for HumanEval/151 ("double_the_difference"): every
# assert IS a direct top-level statement (no loop nesting), but the LAST one
# references `lst`/`odd_sum` bound by two ordinary top-level Assign statements
# a few lines above it. A "count top-level asserts vs. all asserts" heuristic
# would wrongly call this splittable (they're equal); it must be detected by
# the presence of the two Assign statements in the body instead.
_SIBLING_BINDING_CHECK_BODY = (
    "def check(candidate):\n\n"
    "    # Check some simple cases\n"
    "    assert candidate([]) == 0 , \"msg 1\"\n"
    "    assert candidate([5, 4]) == 25 , \"msg 2\"\n\n"
    "    # Check some edge cases that are easy to work out by hand.\n"
    "    assert candidate([-1, -2, 8]) == 0, \"msg 5\"\n"
    "    lst = list(range(-99, 100, 2))\n"
    "    odd_sum = sum([i**2 for i in lst if i%2!=0 and i > 0])\n"
    "    assert candidate(lst) == odd_sum , \"msg 7\"\n\n"
)


def test_parse_check_asserts_keeps_whole_body_when_asserts_depend_on_siblings():
    asserts = _parse_check_asserts(_SIBLING_BINDING_CHECK_BODY, "double_the_difference")
    # Not split: two of the "asserts" are top-level, but the body also
    # contains Assign statements a later assert depends on.
    assert len(asserts) == 1
    block = asserts[0]
    compile(block, "<test>", "exec")
    assert "candidate" not in block
    assert "lst = list(range(-99, 100, 2))" in block
    assert block.count("assert double_the_difference(") == 4


def test_load_humaneval_plus_limit_returns_valid_codetasks():
    tasks = load_humaneval_plus(limit=3)
    assert len(tasks) == 3
    for task in tasks:
        assert task.source == "humaneval_plus"
        assert "hidden" in task.test_suites
        assert task.test_suites["hidden"].asserts
        assert "visible" not in task.test_suites
        assert task.reference_solution


def test_prompt_is_not_rewrapped_in_instruction_template():
    tasks = load_humaneval_plus(limit=3)
    for task in tasks:
        # The ReasoningLab-style wrapper this adapter deliberately avoids.
        assert "Return only one Markdown fenced Python block" not in task.prompt
        assert "Complete the following Python function" not in task.prompt


def test_first_task_id_and_order_are_stable():
    tasks = load_humaneval_plus(limit=2)
    assert [t.task_id for t in tasks] == ["HumanEval/0", "HumanEval/1"]


def test_parsed_asserts_actually_run_against_the_reference_solution():
    task = load_humaneval_plus(limit=1)[0]
    suite = task.test_suites["hidden"]
    test_code = suite_to_test_code(suite, task.entry_point)
    result = SubprocessSandbox().run(task.reference_solution, test_code)
    assert result.passed
