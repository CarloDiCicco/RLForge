"""
metrics/metrics.py
──────────────────
Aggregate metrics computed from a completed run's ExecutionRecords (PLAN §1.6).

Two things a run must report:
  - pass@1 / pass@k: an UNBIASED estimate of "if I only kept k of my n samples,
    what's the chance at least one is correct?" (Codex/HumanEval estimator, not
    ReasoningLab's biased "did any of n attempts pass" — see pass_at_k below).
  - failure_distribution: for the FAILED executions only, what fraction fall
    into each FailureType. Where pass@k asks "did it work", this asks "when it
    didn't, why not" — the diagnostic PLAN §1.4 needs to tell "model writes
    malformed code" (SYNTAX) apart from "model's logic is wrong" (ASSERTION).

Adapted from ReasoningLab `eval/metrics.py`: keep its shape (a RunMetrics result
type, Counter-based failure distribution, empty-input guard) but replace the pass
logic entirely — theirs counts a task as passed if ANY attempt passed, which is
biased upward the more samples you draw and not comparable across runs sampled
at different n. Ours is a function of k alone.

Consumed by:
  - pipeline.py: calls compute_metrics() once per run, after all executions are
    recorded, and writes the result via registry.jsonl_logger.log_summary().
"""

from __future__ import annotations

from collections import Counter
from math import comb
from typing import Iterable, Sequence

from pydantic import BaseModel, Field

from rlforge import FORMAT_VERSION
from rlforge.classify.taxonomy import FailureType
from rlforge.schema.execution import ExecutionRecord


class RunMetrics(BaseModel):
    """Aggregate statistics for one pipeline run (PLAN §1.6)."""

    format_version: str = Field(default=FORMAT_VERSION)

    # Mean per-sample correctness (c/n per task, averaged over tasks). The
    # number that appears in every paper's headline table.
    pass_at_1: float
    # pass@k for each requested k, e.g. {1: 0.42, 5: 0.71, 10: 0.85}.
    pass_at_k: dict[int, float]
    # Fraction of FAILED executions per FailureType value. PASS is excluded —
    # it is not a failure mode. Empty dict if every execution passed.
    failure_dist: dict[str, float]

    total_tasks: int
    total_samples: int
    total_executions: int


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased probability that a random draw of k of n samples has ≥1 pass.

    The Codex/HumanEval estimator: P(all k drawn are failures) = C(n-c, k) /
    C(n, k), so pass@k = 1 minus that. Unlike "did any of the n samples pass",
    this depends only on k, not on how large n happened to be — so pass@1
    computed from n=200 samples is comparable to pass@1 computed from n=10.

    Args:
        n: total samples generated for this task. Must be >= k.
        c: how many of those n samples passed (0 <= c <= n).
        k: the budget being evaluated (e.g. 1, 5, 10).
    """
    if c == 0:
        # No passing sample exists at all; no k-sized draw can contain one.
        return 0.0
    if n - c < k:
        # Fewer failures than k: any draw of k must include at least one pass.
        return 1.0
    # Numerically stable product form of 1 - C(n-c,k)/C(n,k). Avoids forming
    # C(n,k) directly, which overflows/loses precision for realistic n (e.g.
    # n=200). Mathematically identical to the binomial-ratio definition.
    prob_all_fail = 1.0
    for i in range(n - c + 1, n + 1):
        prob_all_fail *= 1.0 - k / i
    return 1.0 - prob_all_fail


def failure_distribution(failure_types: Iterable[FailureType]) -> dict[str, float]:
    """Fraction of FAILED executions per FailureType, PASS excluded.

    Args:
        failure_types: the failure_type of every execution in the run (passes
            included — they are filtered out here, not by the caller).

    Returns:
        {} if there are no failures (every execution passed); otherwise a dict
        of FailureType value -> fraction, summing to 1.0.
    """
    failed = [ft for ft in failure_types if ft != FailureType.PASS]
    if not failed:
        return {}
    counts = Counter(ft.value for ft in failed)
    total = len(failed)
    return {value: count / total for value, count in counts.items()}


def compute_metrics(
    grading_executions: Sequence[ExecutionRecord], ks: Sequence[int] = (1,)
) -> RunMetrics:
    """Aggregate RunMetrics from one run's grading-suite ExecutionRecords.

    Only pass executions from the GRADING suite (e.g. "hidden"), never the
    "visible" suite — the visible suite is what the model may have seen in its
    prompt, so it must never be the pass/fail authority (the reward-hacking
    guard from schema/task.py). The pipeline is responsible for filtering to
    the grading role before calling this.

    Args:
        grading_executions: every ExecutionRecord from the run's grading suite
            (all tasks, all samples). Must be non-empty.
        ks: the k values to report pass@k for (e.g. (1,) in M1's n=1 smoke run;
            (1, 5, 10) once M2 samples more per task).

    Returns:
        RunMetrics with pass@1, pass@k for each requested k, and the failure
        distribution.

    Raises:
        ValueError: if grading_executions is empty (nothing to aggregate).
    """
    if not grading_executions:
        raise ValueError(
            "compute_metrics received no grading executions — nothing to aggregate."
        )

    # Group by task_id to get (n, c) per task: n = samples seen, c = passes.
    # Every task counts equally in the average regardless of how many samples
    # it happened to have, or how easy/hard it is — matching "67% of problems
    # solved", not "67% of attempts".
    per_task: dict[str, list[bool]] = {}
    for rec in grading_executions:
        per_task.setdefault(rec.task_id, []).append(rec.result.passed)

    pass_at_k_by_k: dict[int, float] = {}
    for k in ks:
        scores = [
            pass_at_k(n=len(passes), c=sum(passes), k=k)
            for passes in per_task.values()
        ]
        pass_at_k_by_k[k] = sum(scores) / len(scores)

    # pass@1 is pass_at_k(n, c, 1), which algebraically is just c/n — but we
    # reuse pass_at_k directly so there is exactly one implementation of the
    # estimator, not a separately-maintained special case.
    pass_1_scores = [
        pass_at_k(n=len(passes), c=sum(passes), k=1) for passes in per_task.values()
    ]
    pass_1 = sum(pass_1_scores) / len(pass_1_scores)

    return RunMetrics(
        pass_at_1=pass_1,
        pass_at_k=pass_at_k_by_k,
        failure_dist=failure_distribution(
            rec.result.failure_type for rec in grading_executions
        ),
        total_tasks=len(per_task),
        total_samples=sum(len(passes) for passes in per_task.values()),
        total_executions=len(grading_executions),
    )
