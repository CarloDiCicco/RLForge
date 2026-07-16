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
import io as _io, sys as _sys, contextlib as _ctx, json as _json
_pairs = _json.loads({pairs_literal!r})
_code = compile({_CANDIDATE_SRC_VAR}, "<candidate>", "exec")

def _norm(_s):
    _lines = [_ln.rstrip() for _ln in _s.splitlines()]
    while _lines and _lines[-1] == "":
        _lines.pop()
    return "\\n".join(_lines)

for _idx, (_stdin_text, _expected) in enumerate(_pairs):
    _old_stdin = _sys.stdin
    _sys.stdin = _io.StringIO(_stdin_text)
    _buf = _io.StringIO()
    try:
        with _ctx.redirect_stdout(_buf):
            exec(_code, {{"__name__": "__main__"}})
    finally:
        _sys.stdin = _old_stdin
    _got = _norm(_buf.getvalue())
    if _got != _norm(_expected):
        raise AssertionError(
            "IO pair %d failed: stdin=%r expected=%r got=%r"
            % (_idx, _stdin_text, _expected, _buf.getvalue())
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
