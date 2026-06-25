"""Tests for the canonical schemas (PLAN §1.1, §1.4, §1.5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rlforge import FORMAT_VERSION
from rlforge.classify.taxonomy import FailureType
from rlforge.schema.execution import ExecutionRecord, ExecutionResult
from rlforge.schema.records import GenerationRecord, TrainingRun
from rlforge.schema.task import PROMPTABLE_SUITE, CodeTask


def _minimal_task(**overrides) -> CodeTask:
    base = dict(
        task_id="t1",
        source="humaneval",
        prompt="Write add(a, b).",
        test_suites={
            "visible": ["assert add(2, 3) == 5"],
            "hidden": ["assert add(-1, 1) == 0", "assert add(0, 0) == 0"],
        },
        reference_solution="def add(a, b):\n    return a + b",
    )
    base.update(overrides)
    return CodeTask(**base)


# ── CodeTask ─────────────────────────────────────────────────────────────────


def test_codetask_stamps_format_version():
    assert _minimal_task().format_version == FORMAT_VERSION


def test_codetask_roundtrips_through_json():
    task = _minimal_task()
    restored = CodeTask.model_validate_json(task.model_dump_json())
    assert restored == task


def test_codetask_rejects_empty_required_text():
    with pytest.raises(ValidationError):
        _minimal_task(task_id="   ")


def test_codetask_rejects_nonpositive_timeout():
    with pytest.raises(ValidationError):
        _minimal_task(timeout_s=0.0)


def test_blank_optionals_collapse_to_none():
    task = _minimal_task(entry_point="  ", function_signature="")
    assert task.entry_point is None
    assert task.function_signature is None


def test_prompt_tests_returns_only_visible_suite():
    task = _minimal_task()
    # The promptable accessor exposes the visible spec-examples only — never
    # the hidden grading suite (the reward-hacking guard).
    assert task.prompt_tests() == ["assert add(2, 3) == 5"]
    assert PROMPTABLE_SUITE == "visible"


def test_grading_suite_defaults_to_hidden_and_is_disjoint_from_prompt():
    task = _minimal_task()
    grading = task.grading_suite()
    assert grading == ["assert add(-1, 1) == 0", "assert add(0, 0) == 0"]
    # The whole point: nothing in the prompt overlaps the grading authority.
    assert set(task.prompt_tests()).isdisjoint(grading)


def test_missing_suite_returns_empty_not_error():
    task = _minimal_task(test_suites={"visible": ["assert add(1, 1) == 2"]})
    assert task.grading_suite("hidden") == []


def test_empty_snippets_dropped_but_key_kept():
    task = _minimal_task(test_suites={"hidden": ["", "  ", "assert add(1,1)==2"]})
    assert task.test_suites["hidden"] == ["assert add(1,1)==2"]


def test_blank_suite_name_rejected():
    with pytest.raises(ValidationError):
        _minimal_task(test_suites={"  ": ["assert True"]})


# ── ExecutionResult / ExecutionRecord ──────────────────────────────────────────


def test_execution_result_carries_failure_type_enum():
    r = ExecutionResult(passed=True, failure_type=FailureType.PASS, execution_time_s=0.1)
    # str-Enum: serializes as a plain string.
    assert r.model_dump()["failure_type"] == "pass"


def test_execution_record_embeds_result_and_roundtrips():
    rec = ExecutionRecord(
        execution_result_id="e1",
        sample_id="s1",
        task_id="t1",
        sandbox_config_id="cfg-default",
        result=ExecutionResult(
            passed=False, failure_type=FailureType.ASSERTION, execution_time_s=0.2
        ),
        timeout_s=8.0,
        test_suite_id="hidden",
        visible_or_hidden_tests="hidden",
        timestamp="2026-06-25T00:00:00Z",
    )
    restored = ExecutionRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec
    assert restored.result.failure_type == FailureType.ASSERTION


# ── Records ────────────────────────────────────────────────────────────────────


def test_generation_record_roundtrips():
    rec = GenerationRecord(
        run_id="r1", sample_id="s1", task_id="t1", dataset_source="humaneval",
        split="test", model_name="qwen", prompt_template_id="v1",
        decoding_temperature=0.7, top_p=0.95, max_new_tokens=512, seed=0,
        input_tokens=10, output_tokens=20, raw_output="...", extracted_code="...",
        extraction_status="ok", execution_result_id="e1",
        timestamp="2026-06-25T00:00:00Z", git_commit="abc123",
    )
    assert GenerationRecord.model_validate_json(rec.model_dump_json()) == rec


def test_training_run_stamps_format_version():
    run = TrainingRun(
        train_run_id="tr1", base_model="qwen", method="SFT", data_mixture_id="M0",
        num_train_samples=100, num_eval_samples=10, optimizer="adamw",
        learning_rate=1e-4, batch_size=8, gradient_accumulation_steps=4,
        max_steps=1000, seed=0, metrics_path="/x", git_commit="abc123",
    )
    assert run.format_version == FORMAT_VERSION
