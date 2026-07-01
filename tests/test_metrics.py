"""Tests for metrics/metrics.py — pass@k estimator and aggregate RunMetrics."""

from __future__ import annotations

import pytest

from rlforge.classify.taxonomy import FailureType
from rlforge.metrics.metrics import compute_metrics, failure_distribution, pass_at_k
from rlforge.schema.execution import ExecutionRecord, ExecutionResult


def _exec(task_id: str, passed: bool, ft: FailureType, eid: str) -> ExecutionRecord:
    return ExecutionRecord(
        execution_result_id=eid,
        sample_id=f"{task_id}-{eid}",
        task_id=task_id,
        sandbox_config_id="cfg",
        result=ExecutionResult(passed=passed, failure_type=ft, execution_time_s=0.1),
        timeout_s=8.0,
        test_suite_id="hidden",
        visible_or_hidden_tests="hidden",
        test_suite_size=1,
        timestamp="2026-06-25T00:00:00Z",
    )


# ── pass_at_k ──────────────────────────────────────────────────────────────


def test_pass_at_k_zero_when_nothing_passed():
    assert pass_at_k(n=5, c=0, k=1) == 0.0


def test_pass_at_k_one_when_everything_passed():
    assert pass_at_k(n=5, c=5, k=3) == 1.0


def test_pass_at_k_known_value():
    # n=5, c=2, k=1: probability a single random draw is a pass = c/n = 0.4.
    assert pass_at_k(n=5, c=2, k=1) == pytest.approx(0.4)


def test_pass_at_k_one_when_fewer_failures_than_k():
    # n=5, c=4 -> only 1 failure; any draw of k=2 must include a pass.
    assert pass_at_k(n=5, c=4, k=2) == 1.0


def test_pass_at_1_equals_c_over_n():
    for n, c in [(10, 3), (1, 1), (1, 0), (7, 7)]:
        assert pass_at_k(n=n, c=c, k=1) == pytest.approx(c / n)


# ── failure_distribution ─────────────────────────────────────────────────


def test_failure_distribution_empty_when_all_pass():
    assert failure_distribution([FailureType.PASS, FailureType.PASS]) == {}


def test_failure_distribution_excludes_pass_and_sums_to_one():
    types = [
        FailureType.PASS,
        FailureType.ASSERTION,
        FailureType.ASSERTION,
        FailureType.SYNTAX,
    ]
    dist = failure_distribution(types)
    assert dist["assertion"] == pytest.approx(2 / 3)
    assert dist["syntax"] == pytest.approx(1 / 3)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert "pass" not in dist


# ── compute_metrics ───────────────────────────────────────────────────────


def test_compute_metrics_aggregates_per_task_and_averages_evenly():
    # Task A: 2/2 pass. Task B: 0/2 pass. Each task counts equally regardless
    # of sample count, so pass@1 == mean(1.0, 0.0) == 0.5, not 2/4.
    execs = [
        _exec("A", True, FailureType.PASS, "e1"),
        _exec("A", True, FailureType.PASS, "e2"),
        _exec("B", False, FailureType.ASSERTION, "e3"),
        _exec("B", False, FailureType.SYNTAX, "e4"),
    ]
    metrics = compute_metrics(execs, ks=(1,))
    assert metrics.pass_at_1 == pytest.approx(0.5)
    assert metrics.pass_at_k[1] == pytest.approx(0.5)
    assert metrics.total_tasks == 2
    assert metrics.total_samples == 4
    assert metrics.total_executions == 4
    assert metrics.failure_dist["assertion"] == pytest.approx(0.5)
    assert metrics.failure_dist["syntax"] == pytest.approx(0.5)


def test_compute_metrics_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_metrics([])
