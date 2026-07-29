"""End-to-end smoke test for pipeline.py (PLAN §1.6 verification).

Uses backend="subprocess" (not bwrap) so this test is host-independent — CI/dev
machines without a working bwrap cage (see RESEARCH_LOG.md) still exercise the
full wiring. bwrap itself is already covered by tests/test_sandbox_bwrap.py.
"""

from __future__ import annotations

from pathlib import Path

from rlforge import FORMAT_VERSION
from rlforge.adapters.task_loader import load_tasks
from rlforge.registry.jsonl_logger import read_records
from rlforge.sandbox.base import SandboxConfig
from rlforge.schema.execution import ExecutionRecord
from rlforge.schema.records import GenerationRecord


def _run(tmp_path):
    from rlforge.pipeline import RunConfig, run_pipeline

    tasks = load_tasks(
        Path(__file__).parent / "fixtures" / "tasks_smoke.jsonl"
    )
    config = RunConfig(
        out_dir=str(tmp_path),
        sandbox_config=SandboxConfig(backend="subprocess"),
        n_samples=1,
        ks=(1,),
        tasks_source="tests/fixtures/tasks_smoke.jsonl",
    )
    metrics = run_pipeline(tasks, config=config)
    # run_id is the only entry under out_dir.
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    return metrics, run_dirs[0]


def test_pipeline_produces_all_three_logs(tmp_path):
    _, run_dir = _run(tmp_path)
    assert (run_dir / "generations.jsonl").exists()
    assert (run_dir / "executions.jsonl").exists()
    assert (run_dir / "summary.jsonl").exists()


def test_generations_reload_and_carry_provenance(tmp_path):
    _, run_dir = _run(tmp_path)
    gens = read_records(run_dir / "generations.jsonl", GenerationRecord)
    # Two tasks have a reference_solution (add, is_even); the third is skipped.
    assert len(gens) == 2
    for g in gens:
        assert g.format_version == FORMAT_VERSION
        assert g.dataset_source == "fixture"
        assert g.git_commit  # never empty (git_commit() degrades to "unknown")
        assert g.hardware_info


def test_executions_reload_and_carry_linkage(tmp_path):
    _, run_dir = _run(tmp_path)
    execs = read_records(run_dir / "executions.jsonl", ExecutionRecord)
    # 2 generated samples x 2 suites each (visible + hidden) = 4.
    assert len(execs) == 4
    gens = read_records(run_dir / "generations.jsonl", GenerationRecord)
    gen_sample_ids = {g.sample_id for g in gens}
    for e in execs:
        assert e.sample_id in gen_sample_ids  # child-side FK resolves
        assert e.test_suite_size > 0


def test_ref_less_task_is_skipped_entirely(tmp_path):
    _, run_dir = _run(tmp_path)
    gens = read_records(run_dir / "generations.jsonl", GenerationRecord)
    execs = read_records(run_dir / "executions.jsonl", ExecutionRecord)
    assert all(g.task_id != "no_ref" for g in gens)
    assert all(e.task_id != "no_ref" for e in execs)


def test_pass_at_1_is_perfect_for_echoed_reference_solutions(tmp_path):
    metrics, _ = _run(tmp_path)
    assert metrics.pass_at_1 == 1.0
    assert metrics.pass_at_k[1] == 1.0
    assert metrics.failure_dist == {}
    assert metrics.total_tasks == 2


def test_summary_reloads_with_run_provenance(tmp_path):
    import json

    _, run_dir = _run(tmp_path)
    lines = (run_dir / "summary.jsonl").read_text().splitlines()
    assert len(lines) == 1
    summary = json.loads(lines[0])
    assert summary["pass_at_1"] == 1.0
    assert summary["model_name"] == "stub:echo-reference"
    assert "git_commit" in summary and "hardware_info" in summary
    # The run-level summary must name what the tasks were loaded from: with
    # several benchmarks coexisting (LeetCode vs stdin/stdout), a summary line
    # without the source is ambiguous exactly when it matters.
    assert summary["tasks_source"] == "tests/fixtures/tasks_smoke.jsonl"
