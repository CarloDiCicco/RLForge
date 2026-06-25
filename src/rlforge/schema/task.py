"""
schema/task.py
──────────────
CodeTask — the canonical *task* schema (PLAN §1.1).

This is the single internal representation of a coding task. Dataset adapters
(HumanEval, MBPP, LiveCodeBench, EvalPlus, future repair sets) and synthetic
generators all convert their raw rows into this one shape at the ingestion
boundary. Nothing downstream of an adapter is allowed to see a source-specific
shape — origin survives only as the `source` field, for analysis.

Design choice: **named test suites** (PLAN §4.3 reward-hacking analysis).
Instead of two fixed `visible_tests` / `hidden_tests` lists, tests live in a
`test_suites: dict[str, list[str]]` keyed by role. Why:

  - The *grading / reward* suite must never be placed in the prompt. During RL
    training that would apply optimization pressure toward hard-coding test
    answers (reward hacking); at inference it would only yield a misleading
    score. The model may see small spec-examples (the `visible` suite) that
    define the task, human-style — never the set it is graded on.
  - The analysis in PLAN §4.3 needs ≥3 distinct roles kept auditable: the
    visible spec-examples, the hidden grading set, and the EvalPlus-extended
    set. A dict keeps these distinct and lets new suites be added without a
    schema migration (migrations are the expensive thing per PLAN).

Conventional suite keys (not enforced, but used across the codebase):
    "visible"            — small spec-examples; the only suite allowed in prompts
    "hidden"             — held-out grading suite; the honest pass/fail authority
    "evalplus_extended"  — EvalPlus's expanded tests (third, tougher check)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from rlforge import FORMAT_VERSION

# Suite key whose tests are the only ones permitted in a model prompt.
# Centralized here so prompt-building code references the constant, not a
# string literal, and a future rename is a one-line change.
PROMPTABLE_SUITE = "visible"


class CodeTask(BaseModel):
    """A single code-generation task in RLForge's canonical format."""

    # Stamps the format that produced this record so old data stays
    # interpretable as the schema evolves (PLAN: versioned + validated).
    format_version: str = Field(default=FORMAT_VERSION)

    # Unique identifier used in logs, records, and error reporting.
    task_id: str
    # Origin of the task (e.g. "humaneval", "mbpp", "synthetic:self-gen").
    # Analysis-only: it must never become a code path or branch.
    source: str
    # Natural-language problem statement sent to the model.
    prompt: str

    # Optional signature / entry point the solution must provide. Kept separate
    # from the prompt so adapters can populate them structurally and the
    # extractor/executor can reference them without re-parsing the prompt.
    function_signature: str | None = None
    entry_point: str | None = None

    # Named test suites, keyed by role (see module docstring). Values are
    # Python test snippets the executor runs against a candidate solution.
    test_suites: dict[str, list[str]] = Field(default_factory=dict)

    # A known-correct solution, when the source provides one. Used in M1 as the
    # generation stub (echoed back to prove the pipeline wiring) and later for
    # diagnostics; never used as a verifier.
    reference_solution: str | None = None

    # Per-task wall-clock timeout the sandbox applies; overridable per task.
    timeout_s: float = Field(default=8.0, gt=0.0)

    # Free-form bookkeeping (difficulty, original ids, tags, etc.).
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id", "source", "prompt")
    @classmethod
    def _non_empty_required_text(cls, value: str) -> str:
        """Trim required text fields and reject empty content."""
        value = value.strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("function_signature", "entry_point", "reference_solution")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        """Collapse blank optional strings to None (one 'empty' representation)."""
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("test_suites")
    @classmethod
    def _normalize_test_suites(
        cls, value: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Trim suite names and drop empty snippets, preserving order.

        We keep a suite key even if its list ends up empty: an explicitly empty
        "hidden" suite (a task with no held-out grading tests) is a meaningful,
        analysable state — distinct from the key being absent — and PLAN §4.4
        wants weak/missing test suites surfaced, not silently dropped.
        """
        normalized: dict[str, list[str]] = {}
        for raw_name, snippets in value.items():
            name = raw_name.strip()
            if not name:
                raise ValueError("test-suite name must be non-empty")
            normalized[name] = [s for s in snippets if s and s.strip()]
        return normalized

    def prompt_tests(self) -> list[str]:
        """The only tests allowed into a model prompt — the spec-examples.

        Returns the PROMPTABLE_SUITE tests (empty list if absent). Prompt
        builders must go through this, never read test_suites directly, so the
        grading suites cannot leak into a prompt by accident.
        """
        return self.test_suites.get(PROMPTABLE_SUITE, [])

    def grading_suite(self, suite: str = "hidden") -> list[str]:
        """The tests used as the pass/fail authority for the given role.

        Defaults to the "hidden" held-out suite. Returns an empty list if the
        task has no such suite (callers should treat that as 'cannot grade').
        """
        return self.test_suites.get(suite, [])
