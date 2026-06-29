"""Tests for the run registry (PLAN §1.5): provenance ids + JSONL round-trips."""

from __future__ import annotations

from rlforge import FORMAT_VERSION
from rlforge.classify.taxonomy import FailureType
from rlforge.registry.jsonl_logger import (
    log_execution,
    log_generation,
    read_records,
)
from rlforge.registry.runinfo import (
    git_commit,
    hardware_info,
    new_execution_id,
    new_run_id,
    new_sample_id,
    utc_now_iso,
)
from rlforge.schema.execution import ExecutionRecord, ExecutionResult
from rlforge.schema.records import GenerationRecord


# ── fixtures ───────────────────────────────────────────────────────────────

def _generation(run_id: str, sample_id: str) -> GenerationRecord:
    return GenerationRecord(
        run_id=run_id,
        sample_id=sample_id,
        task_id="t1",
        dataset_source="fixture",
        split="test",
        model_name="stub-echo",
        prompt_template_id="bare",
        decoding_temperature=0.0,
        top_p=1.0,
        max_new_tokens=256,
        seed=0,
        input_tokens=10,
        output_tokens=20,
        raw_output="def add(a, b):\n    return a + b",
        extracted_code="def add(a, b):\n    return a + b",
        extraction_status="raw_fallback",
        timestamp=utc_now_iso(),
        git_commit=git_commit(),
        hardware_info=hardware_info(),
    )


def _execution(sample_id: str, execution_id: str) -> ExecutionRecord:
    return ExecutionRecord(
        execution_result_id=execution_id,
        sample_id=sample_id,
        task_id="t1",
        sandbox_config_id="subprocess-deadbeef",
        result=ExecutionResult(
            passed=True,
            failure_type=FailureType.PASS,
            execution_time_s=0.01,
        ),
        timeout_s=8.0,
        test_suite_id="hidden",
        visible_or_hidden_tests="hidden",
        timestamp=utc_now_iso(),
    )


# ── runinfo ──────────────────────────────────────────────────────────────────

def test_sample_id_deterministic_and_unique():
    run = "run-x"
    a = new_sample_id(run, "t1", 0)
    assert a == new_sample_id(run, "t1", 0)          # stable
    assert a != new_sample_id(run, "t1", 1)          # differs by sample_idx
    assert a != new_sample_id(run, "t2", 0)          # differs by task
    assert a != new_sample_id("run-y", "t1", 0)      # differs by run


def test_execution_id_deterministic_per_suite():
    s = new_sample_id("run-x", "t1", 0)
    e = new_execution_id(s, "hidden")
    assert e == new_execution_id(s, "hidden")        # stable
    assert e != new_execution_id(s, "visible")       # differs by suite


def test_run_id_is_fresh_each_call():
    assert new_run_id() != new_run_id()


def test_git_commit_and_hardware_never_raise():
    assert isinstance(git_commit(), str)
    hw = hardware_info()
    assert isinstance(hw, dict)
    assert "python_version" in hw


# ── jsonl round-trips ─────────────────────────────────────────────────────────

def test_generation_round_trip_revalidates(tmp_path):
    run = new_run_id()
    sid = new_sample_id(run, "t1", 0)
    path = tmp_path / "generations.jsonl"
    log_generation(path, _generation(run, sid))

    back = read_records(path, GenerationRecord)
    assert len(back) == 1
    assert back[0].sample_id == sid
    assert back[0].format_version == FORMAT_VERSION


def test_execution_round_trip_revalidates(tmp_path):
    sid = new_sample_id("run-x", "t1", 0)
    eid = new_execution_id(sid, "hidden")
    path = tmp_path / "executions.jsonl"
    log_execution(path, _execution(sid, eid))

    back = read_records(path, ExecutionRecord)
    assert len(back) == 1
    assert back[0].sample_id == sid                  # child-side FK preserved
    assert back[0].result.failure_type == FailureType.PASS


def test_append_only_does_not_rewrite(tmp_path):
    run = new_run_id()
    path = tmp_path / "generations.jsonl"
    log_generation(path, _generation(run, new_sample_id(run, "t1", 0)))
    first_bytes = path.read_bytes()
    log_generation(path, _generation(run, new_sample_id(run, "t1", 1)))

    # Two lines, and the first line's bytes are unchanged (prefix-preserved).
    assert len(read_records(path, GenerationRecord)) == 2
    assert path.read_bytes().startswith(first_bytes)


def test_read_missing_file_returns_empty(tmp_path):
    assert read_records(tmp_path / "nope.jsonl", GenerationRecord) == []
