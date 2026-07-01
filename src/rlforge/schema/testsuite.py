"""
schema/testsuite.py
───────────────────
TestSuite — a typed, polymorphic container for one named test suite (PLAN §1.1).

Why this is not just `list[str]`. Different datasets verify a solution in
genuinely different ways, and a research harness must store each dataset's
*canonical* grading semantics faithfully (see the Data & Evaluation Methodology
note in PLAN.md — benchmark numbers are only comparable to the literature if we
grade the way the dataset intends):

  - "assert"        — Python assert snippets (HumanEval / MBPP / EvalPlus).
  - "stdin_stdout"  — feed stdin, diff stdout against expected (LiveCodeBench and
                      most competitive-programming sets).
  - "checker"       — a custom predicate / property-based checker (synthetic tasks).

A bare `list[str]` of asserts cannot represent stdin/stdout pairs or a checker, so
the value type is this tagged object instead. The sandbox never sees a TestSuite:
`verify/compiler.py` turns a suite into the `test_code` string the sandbox already
expects, dispatching on `kind`. Adding a new verification style later is a new
`kind` + one compiler branch — no change to the sandbox, registry, or metrics.

This module is a leaf (only stdlib + pydantic) so both the schema and the
compiler can import it without cycles.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# The verification mechanisms we know we will need. Only "assert" has a compiler
# in M1; the others are reserved so datasets that need them (M2) do not force a
# schema migration — they are valid to *store* now, just not yet runnable.
TestKind = Literal["assert", "stdin_stdout", "checker"]


class IOPair(BaseModel):
    """
    One stdin→expected-stdout example for a `stdin_stdout` suite, it's one
    pair of the test suite list.
    """

    stdin: str
    stdout: str


class TestSuite(BaseModel):
    """One named suite of tests, tagged by the mechanism used to run them.

    Exactly one payload field is populated, matching `kind`:
      kind="assert"       → `asserts`      (the others empty)
      kind="stdin_stdout" → `io_pairs`     (the others empty)
      kind="checker"      → `checker_code` (the others empty)
    The `_payload_matches_kind` validator enforces this so a malformed suite is
    rejected at ingestion, not discovered at run time.
    """

    # Tell pytest this is not a test class (its name starts with "Test", which
    # pytest would otherwise try to collect). It is a domain model.
    __test__ = False

    kind: TestKind = "assert"

    # Payload — one of these is used depending on `kind`.
    asserts: list[str] = Field(default_factory=list)
    io_pairs: list[IOPair] = Field(default_factory=list)
    checker_code: str | None = None

    @model_validator(mode="after")
    def _payload_matches_kind(self) -> "TestSuite":
        """Reject a suite whose populated payload does not match its kind.

        Note an *empty* assert suite is still valid: a task with no held-out
        grading tests is a meaningful, analysable state (PLAN §4.4 wants weak /
        missing suites surfaced, not silently dropped). We only forbid payload of
        the *wrong* kind, not an empty payload of the right kind.
        """
        # Trim blank assert snippets, preserving order (was in CodeTask before).
        self.asserts = [s for s in self.asserts if s and s.strip()]
        if self.checker_code is not None and not self.checker_code.strip():
            self.checker_code = None

        if self.kind == "assert":
            wrong = self.io_pairs or self.checker_code is not None
        elif self.kind == "stdin_stdout":
            wrong = bool(self.asserts) or self.checker_code is not None
        else:  # checker
            wrong = bool(self.asserts) or bool(self.io_pairs)
        if wrong:
            raise ValueError(
                f"TestSuite kind={self.kind!r} has payload of a different kind"
            )
        return self

    def is_empty(self) -> bool:
        """True iff the suite has no runnable tests (cannot grade with it)."""
        return not (self.asserts or self.io_pairs or self.checker_code)

    def size(self) -> int:
        """Verifier strictness: how many individual tests this suite runs.

        Asserts and IO pairs each count as one test each; a checker is a single
        pass/fail predicate, so it counts as 1 (0 if absent — an empty suite).
        Feeds ExecutionRecord.test_suite_size so pass-rate can be sliced by
        verifier strictness without joining back to the task.
        """
        if self.asserts:
            return len(self.asserts)
        if self.io_pairs:
            return len(self.io_pairs)
        return 1 if self.checker_code else 0

    def as_prompt_examples(self) -> str:
        """Render the suite as human-style spec-examples for a prompt.

        Used ONLY for the visible suite (the only one allowed in a prompt). The
        rendering is mechanism-appropriate but deliberately uniform in style, so
        the prompt cues the task contract (function vs. program) without leaking
        the grading mechanism's exact phrasing. Returns "" for an empty suite.

        (For benchmark data the adapter may instead use the dataset's canonical
        prompt; this renderer is the default / synthetic-data path.)
        """
        if self.kind == "assert":
            return "\n".join(self.asserts)
        if self.kind == "stdin_stdout":
            return "\n".join(
                f"input:\n{p.stdin}\noutput:\n{p.stdout}" for p in self.io_pairs
            )
        # checker: there is nothing illustrative to show the model.
        return ""
