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

M1 implements only `kind="assert"`. `stdin_stdout` and `checker` are valid to
*store* (so datasets that need them can be ingested now) but raise
NotImplementedError when compiled, until M2 wires their harnesses — failing loudly
is correct, so an unsupported suite can never be silently scored as a pass.
"""

from __future__ import annotations

from rlforge.schema.testsuite import TestSuite


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
        raise NotImplementedError(
            "stdin_stdout test suites are not runnable in M1; a stdin/stdout "
            "harness lands in M2 (LiveCodeBench). The suite can be stored now."
        )

    if suite.kind == "checker":
        raise NotImplementedError(
            "checker test suites are not runnable in M1; the checker harness "
            "lands in M2 (synthetic tasks). The suite can be stored now."
        )

    # Unreachable while TestKind is the only source of `kind`, but explicit so a
    # future kind added to the Literal without a branch here fails loudly.
    raise ValueError(f"unknown test suite kind: {suite.kind!r}")
