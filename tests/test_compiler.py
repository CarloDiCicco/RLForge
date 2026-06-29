"""Tests for verify/compiler.py — the TestSuite → test_code dispatch seam."""

from __future__ import annotations

import pytest

from rlforge.classify.taxonomy import FailureType
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


def test_stdin_stdout_not_implemented_in_m1():
    suite = TestSuite(kind="stdin_stdout", io_pairs=[IOPair(stdin="5", stdout="25")])
    with pytest.raises(NotImplementedError):
        suite_to_test_code(suite)


def test_checker_not_implemented_in_m1():
    suite = TestSuite(kind="checker", checker_code="def check(fn): return True")
    with pytest.raises(NotImplementedError):
        suite_to_test_code(suite)
