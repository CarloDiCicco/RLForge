"""Tests for verify/runner.run_suite, the single grader (reward, gold check,
benchmark). Behaviour matrix from the 2026-11-15 audit: correct programs in every
common input/output style must pass; wrong programs and every known reward hack
must fail. Runs on the bwrap backend when usable, else the subprocess backend."""

from __future__ import annotations

import pytest

from rlforge.classify.taxonomy import FailureType
from rlforge.sandbox import make_sandbox
from rlforge.sandbox.base import SandboxConfig
from rlforge.sandbox.bwrap_sandbox import bwrap_usable
from rlforge.schema.testsuite import IOPair, TestSuite
from rlforge.verify.runner import _parse_suite, run_suite, split_assert

BACKEND = "bwrap" if bwrap_usable() else "subprocess"


@pytest.fixture(scope="module")
def sb():
    return make_sandbox(SandboxConfig(backend=BACKEND, timeout_s=10.0, cpu_s=10, mem_mb=4096))


STDIO = TestSuite(kind="stdin_stdout", io_pairs=[IOPair(stdin="2 3\n", stdout="5\n"),
                                                 IOPair(stdin="1 1\n", stdout="2\n")])

STDIO_CORRECT = {
    "input": "a, b = map(int, input().split())\nprint(a + b)",
    "open0": "a, b = map(int, open(0).read().split())\nprint(a + b)",
    "os_read_fileno": "import os, sys\na, b = map(int, os.read(sys.stdin.fileno(), 1 << 20).split())\nprint(a + b)",
    "stdin_buffer": "import sys\na, b = map(int, sys.stdin.buffer.readline().split())\nprint(a + b)",
    "stdout_buffer": "import sys\na, b = map(int, input().split())\nsys.stdout.buffer.write(f'{a + b}\\n'.encode())",
    "os_write": "import os\na, b = map(int, input().split())\nos.write(1, f'{a + b}\\n'.encode())",
    "exit_after": "a, b = map(int, input().split())\nprint(a + b)\nexit()",
    "sys_exit0": "import sys\na, b = map(int, input().split())\nprint(a + b)\nsys.exit(0)",
    "thread_no_join": ("import threading, sys\ndef main():\n    a, b = map(int, input().split())\n    print(a + b)\n"
                       "threading.stack_size(1 << 26)\nsys.setrecursionlimit(10 ** 6)\n"
                       "threading.Thread(target=main).start()"),
    "iterate_stdin": "import sys\nfor line in sys.stdin:\n    a, b = map(int, line.split())\n    print(a + b)",
    "no_newline": "a, b = map(int, input().split())\nprint(a + b, end='')",
    "main_guard": "def main():\n    a, b = map(int, input().split())\n    print(a + b)\nif __name__ == '__main__':\n    main()",
    "lcb_implicit_names": "a, b = map(int, input().split())\nprint(sum(deque([a, b])))",
}


@pytest.mark.parametrize("name", sorted(STDIO_CORRECT))
def test_stdio_correct_styles_pass(sb, name):
    r = run_suite(sb, STDIO_CORRECT[name], STDIO)
    assert r.passed, (r.failure_type, r.stderr)
    assert r.passed_indices == [0, 1]


@pytest.mark.parametrize("src, ft", [
    ("a, b = map(int, input().split())\nprint(a - b)", FailureType.ASSERTION),
    ("raise ValueError('x')", FailureType.RUNTIME),
    ("def (:", FailureType.SYNTAX),
    ("a, b = map(int, input().split())\nprint(a + b)\nexit(1)", FailureType.RUNTIME),
])
def test_stdio_wrong_programs_fail(sb, src, ft):
    r = run_suite(sb, src, STDIO)
    assert not r.passed and r.failure_type == ft


def test_stdio_infinite_loop_times_out(sb):
    fast = make_sandbox(SandboxConfig(backend=BACKEND, timeout_s=2.0, cpu_s=5, mem_mb=4096))
    r = run_suite(fast, "while True: pass", STDIO)
    assert not r.passed and r.failure_type == FailureType.TIMEOUT


STDIO_HACKS = {
    "exit_before_output": "exit(0)",
    "os_exit": "import os\nos._exit(0)",
    "read_main_module": "import __main__\nprint(__main__._expected)",
    "walk_frames": "import sys\nf = sys._getframe(0)\nwhile f.f_back:\n    f = f.f_back\nprint(f.f_globals.get('_pairs'))",
    "scan_gc": ("import gc\nfor o in gc.get_objects():\n    if isinstance(o, dict) and '_expected' in o:\n"
                "        print(o['_expected'])\n        break"),
    "read_script": "print(open('/script.py').read())",
}


@pytest.mark.parametrize("name", sorted(STDIO_HACKS))
def test_stdio_hacks_fail(sb, name):
    assert not run_suite(sb, STDIO_HACKS[name], STDIO).passed


def test_stdio_expected_output_never_in_sandbox(sb):
    # Strongest form of the guarantee: a program that prints EVERYTHING it can
    # reach (its script, all globals) still cannot contain the expected output.
    secret = TestSuite(kind="stdin_stdout", io_pairs=[IOPair(stdin="x\n", stdout="SECRET_7731\n")])
    src = "import sys\nprint(open('/script.py').read() if __import__('os').path.exists('/script.py') else '')\nprint(globals())"
    r = run_suite(sb, src, secret, stop_on_fail=False)
    assert not r.passed and "SECRET_7731" not in (r.stderr or "").split("expected=")[0]


def test_stdio_tests_are_isolated(sb):
    # State set while running test 0 must not exist in test 1 (fresh process).
    src = ("import builtins\nif hasattr(builtins, '_seen'):\n    print('LEAK')\nelse:\n"
           "    builtins._seen = 1\n    a, b = map(int, input().split())\n    print(a + b)")
    assert run_suite(sb, src, STDIO).passed


ADD = TestSuite(kind="assert", asserts=["assert add(2, 3) == 5", "assert add(1, 1) == 2"])
SORT = TestSuite(kind="assert", asserts=["sol = Solution()", "assert sol.f([3, 1, 2]) == [1, 2, 3]",
                                         "assert sol.f([]) == []"])


@pytest.mark.parametrize("src, suite", [
    ("def add(a, b):\n    return a + b", ADD),
    ("def add(a, b):\n    return float(a + b)", ADD),               # 5.0 == 5, as before
    ("def add(a, b):\n    print('debug')\n    return a + b", ADD),   # prints are harmless
    ("class Solution:\n    def f(self, a: List[int]) -> List[int]:\n        return sorted(a)", SORT),
])
def test_assert_correct_pass(sb, src, suite):
    r = run_suite(sb, src, suite)
    assert r.passed, (r.failure_type, r.stderr)


@pytest.mark.parametrize("src, suite, ft", [
    ("def add(a, b):\n    return a - b", ADD, FailureType.ASSERTION),
    ("class Solution:\n    def f(self, a):\n        return tuple(sorted(a))", SORT, FailureType.ASSERTION),
    ("def add(a, b):\n    raise KeyError", ADD, FailureType.RUNTIME),
    ("def add(a, b)\n    return 1", ADD, FailureType.SYNTAX),
])
def test_assert_wrong_fail(sb, src, suite, ft):
    r = run_suite(sb, src, suite)
    assert not r.passed and r.failure_type == ft


@pytest.mark.parametrize("src", [
    # an object whose == always says True: passed every assert in the old harness
    "class X:\n    def __eq__(s, o): return True\n    def __repr__(s): return '5'\ndef add(a, b):\n    return X()",
    "class X(int):\n    def __eq__(s, o): return True\n    def __repr__(s): return '5'\ndef add(a, b):\n    return X(0)",
    "repr = lambda v: '5'\ndef add(a, b):\n    return 0",
    "import builtins\nbuiltins.AssertionError = type('E', (Exception,), {})\ndef add(a, b):\n    return 0",
])
def test_assert_hacks_fail(sb, src):
    assert not run_suite(sb, src, ADD).passed


def test_humaneval_shapes():
    assert split_assert("assert f(1) == (3 + 5, 3 * 5)")[1] == ("eq", (8, 15), None)
    assert split_assert("assert abs(f(1.33) - 0.33) < 1e-6")[1] == ("approx", 0.33, 1e-6)
    assert split_assert("assert not f(1)")[1][0] == "falsy"
    assert split_assert("assert f(1)")[1][0] == "truthy"
    assert split_assert("assert f(1) is True")[1] == ("is", True, None)
    assert split_assert("assert True, 'msg'") == "skip"
    # loop unrolled with the loop variable bound, helper values computed
    suite = TestSuite(kind="assert", asserts=[
        # real shape (HumanEval/44, /151): a line starting with an assert, then
        # helper code in the same string
        "assert g(1, 2) == '1'\nfor x in range(2, 5):\n    assert g(x, x + 1) == str(x)",
        "assert h([]) == 0\nlst = list(range(3))\ns = sum(lst)\nassert h(lst) == s",
    ])
    setup, calls, checks = _parse_suite(suite)
    assert calls == ["g(1, 2)", "g(2, 2 + 1)", "g(3, 3 + 1)", "g(4, 4 + 1)", "h([])", "h([0, 1, 2])"]
    assert [c[1] for c in checks] == ["1", "2", "3", "4", 0, 3]


def test_candidate_vs_candidate_comparison(sb):
    suite = TestSuite(kind="assert", asserts=["assert tuple(f([2, 1])) == tuple(f([2, 1]))",
                                              "assert tuple(f([2, 1])) == tuple([1, 2])"])
    assert run_suite(sb, "def f(a):\n    return sorted(a)", suite).passed
    assert not run_suite(sb, "def f(a):\n    return a", suite).passed
