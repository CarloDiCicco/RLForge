"""
verify/compiler.py
──────────────────
suite_to_test_code — turn a typed TestSuite into the `test_code` string the
sandbox already expects (PLAN §1.3 / §1.4).

This is the dispatch seam that keeps the sandbox mechanism-agnostic. The sandbox
`run(candidate_code, test_code)` takes test code as plain Python text and knows
nothing about test *kinds*; this compiler is the single place that translates a
suite's `kind` into that text. Adding a new verification mechanism later is a new
branch here, with no change to the sandbox, registry, or metrics.

`kind="assert"` and `kind="stdin_stdout"` are implemented. `stdin_stdout` uses an
in-process harness (`_stdin_stdout_test_code`): the candidate is a whole program,
so the sandbox binds its source as an inert string (run mode="stdin_stdout") and
the emitted fragment execs it once per IO pair with stdin/stdout redirected.
`checker` remains valid to *store* but raises NotImplementedError when compiled
until its harness lands — failing loudly is correct, so an unsupported suite can
never be silently scored as a pass.
"""

from __future__ import annotations

import json

from rlforge.schema.testsuite import TestSuite

# The variable the sandbox binds the candidate source to (as an inert string) in
# stdin_stdout mode; the fragment below compiles and execs it once per IO pair.
# Must match subprocess_sandbox.CANDIDATE_SRC_VAR.
_CANDIDATE_SRC_VAR = "__RLFORGE_CANDIDATE_SRC_7f3a__"


# Runs the compiled candidate `_code` once on one stdin text and returns what it
# printed. SHARED, verbatim, by the reward harness below and by the gold-check
# probe (verify/reference_verifier.py), so the check that selects training tasks
# and the reward that trains on them cannot drift apart.
#
# Why byte-backed streams instead of StringIO (bug found 2026-11-14 by the run #3
# gold check, where 10/100 known-correct TACO solutions failed EVERY test in 0.1s):
#   - competitive code reads fast input via `sys.stdin.buffer` and writes via
#     `sys.stdout.buffer`; StringIO has no `.buffer`, so a correct program died
#     with AttributeError. TextIOWrapper over BytesIO has a real `.buffer`.
#   - `exit()` / `sys.exit()` after printing is normal in contest code; it raised
#     SystemExit out of exec and aborted the whole harness. Here it simply ends
#     that run, like a real process exit (output so far is kept).
# Not covered: raw fd I/O (`os.read(0)`, `os.write(1)`), which bypasses sys.*.
#
# exit()/sys.exit(): the sandbox preamble sets them to None (anti reward hacking:
# in assert mode a program could exit 0 before the asserts run). That attack does
# not exist here: SystemExit is caught per test and the output is still compared,
# and the pass sentinel prints only after every pair. So for the duration of ONE
# run, exit/quit/sys.exit raise SystemExit like in a real process, then the
# nullified versions are restored. os._exit stays disabled (it would kill the
# harness itself).
STDIO_RUN_ONE = '''\
import builtins as _bi

def _raise_exit(*_a):
    raise SystemExit(*_a)

def _run_one(_stdin_text):
    _old_in, _old_out = _sys.stdin, _sys.stdout
    _old_exits = (_bi.exit, _bi.quit, _sys.exit)
    _bi.exit = _bi.quit = _sys.exit = _raise_exit
    _outb = _io.BytesIO()
    _out = _io.TextIOWrapper(_outb, encoding="utf-8", write_through=True)
    _sys.stdin = _io.TextIOWrapper(_io.BytesIO(_stdin_text.encode("utf-8")), encoding="utf-8")
    _sys.stdout = _out
    _got = ""
    try:
        exec(_code, {"__name__": "__main__"})
    except SystemExit:
        pass
    finally:
        # Read the bytes BEFORE the wrapper is released: a TextIOWrapper closes
        # its BytesIO when garbage-collected, after which getvalue() raises.
        try:
            _out.flush()
        except Exception:
            pass
        _got = _outb.getvalue().decode("utf-8", "replace")
        _sys.stdin, _sys.stdout = _old_in, _old_out
        _bi.exit, _bi.quit, _sys.exit = _old_exits
    return _got
'''


def _stdin_stdout_test_code(suite: TestSuite) -> str:
    """Compile a stdin_stdout suite into a self-checking test fragment.

    The fragment runs AFTER the sandbox has bound the candidate program source to
    `_CANDIDATE_SRC_VAR` (an inert string — see build_script's stdin_stdout mode).
    For each (stdin, expected stdout) pair it swaps in a StringIO stdin, captures
    stdout, execs the candidate once, and compares. Output is normalized (each
    line right-stripped, trailing blank lines dropped) because competitive-style
    expected outputs differ from real ones only by trailing whitespace far more
    often than by a real mismatch. A mismatch raises AssertionError, which the
    classifier maps to FailureType.ASSERTION exactly like a failed assert suite.
    """
    pairs = [[p.stdin, p.stdout] for p in suite.io_pairs]
    # json.dumps gives a safe Python-literal-compatible embedding of arbitrary
    # payloads (newlines, quotes, unicode) that eval-free `json.loads` reads back.
    pairs_literal = json.dumps(pairs)
    return f'''\
import io as _io, sys as _sys, json as _json
_pairs = _json.loads({pairs_literal!r})
_code = compile({_CANDIDATE_SRC_VAR}, "<candidate>", "exec")

def _norm(_s):
    _lines = [_ln.rstrip() for _ln in _s.splitlines()]
    while _lines and _lines[-1] == "":
        _lines.pop()
    # Leading blank lines are ignored too, like real judges: measured on the
    # run #3 pool, some expected outputs start with a stray "\\n" (data defect).
    while _lines and _lines[0] == "":
        _lines.pop(0)
    return "\\n".join(_lines)

{STDIO_RUN_ONE}
for _idx, (_stdin_text, _expected) in enumerate(_pairs):
    _raw = _run_one(_stdin_text)
    if _norm(_raw) != _norm(_expected):
        raise AssertionError(
            "IO pair %d failed: stdin=%r expected=%r got=%r"
            % (_idx, _stdin_text, _expected, _raw)
        )
'''


def suite_to_test_code(suite: TestSuite, entry_point: str | None = None) -> str:
    """Compile a TestSuite into runnable Python test code for the sandbox.

    Args:
        suite:       the suite to run (its `kind` selects the compilation).
        entry_point: the candidate's required function name, when known. Unused
                     for `assert` (the asserts name it themselves); reserved for
                     the stdin/stdout and checker harnesses in M2.

    Returns:
        A Python snippet to append after the candidate code. For `assert` suites
        this is just the asserts joined by newlines — they raise AssertionError
        on failure, which the classifier maps to FailureType.ASSERTION.
    """
    if suite.kind == "assert":
        # The asserts are already runnable statements; one per line. An empty
        # assert suite compiles to "" — the caller (pipeline) should treat an
        # empty grading suite as 'cannot grade', not as a trivial pass.
        return "\n".join(suite.asserts)

    if suite.kind == "stdin_stdout":
        # The candidate is a whole program (reads stdin, prints stdout), so the
        # sandbox must assemble it in stdin_stdout mode: run(..., mode="stdin_stdout")
        # binds the source to _CANDIDATE_SRC_VAR instead of executing it inline.
        # This fragment then drives it once per IO pair. See build_script.
        return _stdin_stdout_test_code(suite)

    if suite.kind == "checker":
        raise NotImplementedError(
            "checker test suites are not runnable in M1; the checker harness "
            "lands in M2 (synthetic tasks). The suite can be stored now."
        )

    # Unreachable while TestKind is the only source of `kind`, but explicit so a
    # future kind added to the Literal without a branch here fails loudly.
    raise ValueError(f"unknown test suite kind: {suite.kind!r}")
