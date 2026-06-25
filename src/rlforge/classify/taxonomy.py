"""
classify/taxonomy.py
────────────────────
FailureType — the normalized failure category for one execution (PLAN §1.4),
plus `classify_failure`, which maps raw execution signals to a FailureType.

Why the classifier takes primitive signals (passed / timed_out / stderr) rather
than an ExecutionResult object: `schema.execution.ExecutionResult` carries a
`failure_type` field, so it must import this module. Keeping the classifier
dependency-free of the schema breaks that import cycle and keeps the taxonomy a
leaf module the whole codebase can use freely.

Classification priority (first match wins) — the order is deliberate:
  1. PASS       – passed is True (exit 0 AND sentinel; see sandbox executor)
  2. EXTRACTION – no runnable code was extracted from the model output
  3. TIMEOUT    – process killed for exceeding the wall-clock limit
  4. SYNTAX     – SyntaxError / IndentationError / TabError (code never ran)
  5. ASSERTION  – AssertionError (tests ran; logic is wrong)
  6. RUNTIME    – any other non-empty stderr (NameError, TypeError, …)
  7. UNKNOWN    – failed with empty stderr (OOM / hard crash / C-level exit)

Separating SYNTAX from ASSERTION matters for repair analysis: unparseable code
needs a different fix strategy than parseable code with wrong logic, and a model
that reduces one but not the other is learning something different (PLAN §1.4).
"""

from __future__ import annotations

from enum import Enum


class FailureType(str, Enum):
    """Normalized failure category for a single execution attempt.

    Inherits from str so each member IS its string value (FailureType.PASS ==
    "pass"). Consequences: json.dumps serializes members as plain strings
    without a custom encoder, and equality with raw strings works directly, so
    JSONL logs can be filtered without importing this module.
    """

    PASS       = "pass"        # exit 0 AND sentinel found in stdout
    EXTRACTION = "extraction"  # no runnable code extracted from model output
    TIMEOUT    = "timeout"     # process killed for exceeding the wall-clock limit
    SYNTAX     = "syntax"      # SyntaxError / IndentationError / TabError
    ASSERTION  = "assertion"   # AssertionError (tests ran; logic is wrong)
    RUNTIME    = "runtime"     # any other exception (NameError, TypeError, …)
    UNKNOWN    = "unknown"     # failure with no stderr (OOM / hard crash)


# Exception class names that indicate a parse/compile failure. IndentationError
# and TabError are SyntaxError subclasses but appear under their own names in
# tracebacks, so all three must be matched.
_SYNTAX_MARKERS: tuple[str, ...] = ("SyntaxError", "IndentationError", "TabError")


def classify_failure(
    *,
    passed: bool,
    extracted: bool,
    timed_out: bool,
    stderr: str,
) -> FailureType:
    """Map raw execution signals to a FailureType (priority order; first wins).

    Args:
        passed:    True iff the executor confirmed a pass (exit 0 + sentinel).
        extracted: True iff runnable code was extracted from the model output.
                   When False, the candidate never reached the sandbox.
        timed_out: True iff the process was killed by the wall-clock timeout.
        stderr:    Captured standard error (raw tracebacks).
    """
    # 1. Pass is authoritative regardless of anything else in the result.
    if passed:
        return FailureType.PASS

    # 2. Extraction failure: nothing runnable was produced, so no execution
    #    signal (stderr/timeout) is meaningful. Checked early for that reason.
    if not extracted:
        return FailureType.EXTRACTION

    # 3. Timeout: the OS killed the process, so stderr is empty — rely on the
    #    explicit flag, not stderr parsing.
    if timed_out:
        return FailureType.TIMEOUT

    # 4. Syntax: interpreter rejected the code before any test ran. Checked
    #    before ASSERTION so malformed code is never read as a logic error.
    if any(marker in stderr for marker in _SYNTAX_MARKERS):
        return FailureType.SYNTAX

    # 5. Assertion: code ran but a test assertion failed (logic error).
    if "AssertionError" in stderr:
        return FailureType.ASSERTION

    # 6. Runtime: any other non-empty stderr (the long exception tail).
    if stderr.strip():
        return FailureType.RUNTIME

    # 7. Unknown: failed with empty stderr — OOM kill, segfault, or a C-level
    #    exit that bypassed Python. Cannot say more; flagged for inspection.
    return FailureType.UNKNOWN
