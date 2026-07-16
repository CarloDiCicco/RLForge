"""Tests for verify/compiler.py — the TestSuite → test_code dispatch seam."""

from __future__ import annotations

import pytest

from rlforge.classify.taxonomy import FailureType
from rlforge.sandbox.base import SandboxConfig
from rlforge.sandbox.subprocess_sandbox import SubprocessSandbox
from rlforge.schema.testsuite import IOPair, TestSuite
from rlforge.verify.compiler import suite_to_test_code

ADD = "def add(a, b):\n    return a + b"


def test_assert_suite_compiles_to_joined_asserts():
    suite = TestSuite(kind="assert", asserts=["assert add(1, 1) == 2", "assert add(2, 2) == 4"])
    code = suite_to_test_code(suite)
    assert code == "assert add(1, 1) == 2\nassert add(2, 2) == 4"


def test_compiled_assert_code_runs_in_sandbox_pass():
    suite = TestSuite(kind="assert", asserts=["assert add(2, 3) == 5"])
    r = SubprocessSandbox().run(ADD, suite_to_test_code(suite))
    assert r.passed
    assert r.failure_type == FailureType.PASS


def test_compiled_assert_code_runs_in_sandbox_fail():
    suite = TestSuite(kind="assert", asserts=["assert add(2, 3) == 99"])
    r = SubprocessSandbox().run(ADD, suite_to_test_code(suite))
    assert not r.passed
    assert r.failure_type == FailureType.ASSERTION


def test_empty_assert_suite_compiles_to_empty_string():
    assert suite_to_test_code(TestSuite(kind="assert", asserts=[])) == ""


# ── stdin/stdout harness (kind="stdin_stdout") ──────────────────────────────
#
# A stdin_stdout candidate is a whole program that reads stdin and prints stdout,
# not a function. suite_to_test_code emits a fragment that execs the candidate
# (bound as an inert string by the sandbox in mode="stdin_stdout") once per IO
# pair, redirecting stdin/stdout and diffing the output. These tests exercise it
# end-to-end through the real subprocess sandbox.

# Reads two ints from one stdin line and prints their sum.
SUM_STDIN = "a, b = map(int, input().split())\nprint(a + b)"

SUM_SUITE = TestSuite(
    kind="stdin_stdout",
    io_pairs=[
        IOPair(stdin="2 3\n", stdout="5\n"),
        IOPair(stdin="10 20\n", stdout="30\n"),
    ],
)


def _run_stdin(candidate: str, suite: TestSuite):
    return SubprocessSandbox().run(
        candidate, suite_to_test_code(suite), mode="stdin_stdout"
    )


def test_stdin_stdout_correct_candidate_passes_all_pairs():
    r = _run_stdin(SUM_STDIN, SUM_SUITE)
    assert r.passed
    assert r.failure_type == FailureType.PASS


def test_stdin_stdout_wrong_output_fails_as_assertion():
    wrong = "a, b = map(int, input().split())\nprint(a - b)"  # wrong on every pair
    r = _run_stdin(wrong, SUM_SUITE)
    assert not r.passed
    assert r.failure_type == FailureType.ASSERTION
    # The diff detail is surfaced so a failing pair is debuggable.
    assert "IO pair" in (r.traceback or r.stderr)


def test_stdin_stdout_normalizes_trailing_whitespace():
    # Expected output has no trailing newline; the candidate's print() adds one.
    # Normalization (line rstrip + trailing-blank-line drop) must make them equal.
    suite = TestSuite(
        kind="stdin_stdout", io_pairs=[IOPair(stdin="4 5\n", stdout="9")]
    )
    r = _run_stdin(SUM_STDIN, suite)
    assert r.passed


def test_stdin_stdout_runtime_error_is_runtime_not_assertion():
    crash = "a, b = map(int, input().split())\nprint(a / 0)"
    r = _run_stdin(crash, SUM_SUITE)
    assert not r.passed
    assert r.failure_type == FailureType.RUNTIME


def test_stdin_stdout_infinite_loop_times_out():
    slow = "while True:\n    pass"
    r = SubprocessSandbox(
        SandboxConfig(backend="subprocess", timeout_s=2)
    ).run(slow, suite_to_test_code(SUM_SUITE), mode="stdin_stdout")
    assert not r.passed
    assert r.failure_type == FailureType.TIMEOUT


def test_checker_not_implemented_in_m1():
    suite = TestSuite(kind="checker", checker_code="def check(fn): return True")
    with pytest.raises(NotImplementedError):
        suite_to_test_code(suite)
