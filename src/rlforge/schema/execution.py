"""
schema/execution.py
────────────────────
ExecutionResult — the in-memory result of running one candidate in the sandbox.
ExecutionRecord  — the persisted, provenance-carrying form written to the run
                   registry (PLAN §1.4 / §1.5).

Two types on purpose: the sandbox produces a lightweight `ExecutionResult`
(what happened when this code ran), and the registry wraps it into an
`ExecutionRecord` that adds the linkage ids needed to join executions back to
the generation that produced them and the task/suite they were run against.
Keeping them separate means the sandbox stays ignorant of run bookkeeping.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rlforge import FORMAT_VERSION
from rlforge.classify.taxonomy import FailureType


class ExecutionResult(BaseModel):
    """Structured outcome of executing one candidate solution (PLAN §1.4)."""

    # Did the candidate pass its test suite? (sentinel + exit-code protocol; see
    # the sandbox executor for why exit_code == 0 alone is insufficient.)
    passed: bool
    # Categorical failure label; PASS when passed is True. Set by the classifier.
    failure_type: FailureType

    # Captured streams. stderr holds raw tracebacks; traceback is the extracted
    # final traceback text when one is identifiable (None otherwise).
    stdout: str = ""
    stderr: str = ""
    traceback: str | None = None
    # The specific failing test/assertion, when identifiable — invaluable for
    # repair analysis (which test did the model's code actually fail?).
    failed_test: str | None = None

    # Resource accounting.
    execution_time_s: float = Field(ge=0.0)
    memory_peak_mb: float | None = None
    # Process exit code; None when the process was killed (e.g. timeout).
    exit_code: int | None = None


class ExecutionRecord(BaseModel):
    """Persisted execution row with provenance/linkage (PLAN §1.5)."""

    format_version: str = Field(default=FORMAT_VERSION)

    # Identity + linkage. execution_result_id is this record's own unique id;
    # sample_id is the FOREIGN KEY to the GenerationRecord whose code was run
    # (one generation → many executions, one per test suite, all sharing this
    # sample_id). task_id is denormalized in for convenient task-level joins.
    execution_result_id: str
    sample_id: str
    task_id: str
    # Which sandbox configuration (limits/backend) produced this result, so a
    # result is always interpretable against the cage it ran in.
    sandbox_config_id: str

    # The execution outcome, embedded verbatim.
    result: ExecutionResult

    # Which suite was graded, and its role. test_suite_id names the concrete
    # suite; visible_or_hidden_tests records the role ("visible"/"hidden"/...)
    # so reward-hacking analysis (PLAN §4.3) can compare across roles.
    timeout_s: float = Field(gt=0.0)
    test_suite_id: str
    visible_or_hidden_tests: str

    # Wall-clock timestamp (ISO-8601, UTC) when the execution was recorded.
    timestamp: str
