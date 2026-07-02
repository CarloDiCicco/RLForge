"""
pipeline.py
───────────
run_pipeline — the end-to-end wiring: load tasks → generate → extract → sandbox
→ classify → record → metrics (PLAN §1.6). This is the final M1 infrastructure
step: every earlier piece (schemas, sandbox, extractor, registry, metrics) is
an isolated part, and this module is what makes them cooperate on a real task
set.

Generation is INJECTED, not hardcoded: `generate_fn(task, sample_idx) ->
GenOutput` is a plain callable. M1 passes `echo_reference_stub` (returns the
task's `reference_solution`, unchanged — a correct wiring should then pass
~100% of the time, which is the smoke test's whole point). M2 will pass the
real model adapter through the exact same signature; nothing in this module
needs to change when that happens.

Adapted from ReasoningLab `eval/runner.py`: keep its shape (log every record the
moment it's produced, so a crash mid-run loses nothing already on disk;
aggregate metrics only after the full loop) but drop everything about
model/policy dispatch and hidden-state capture — M1 has no model yet.

Consumed by:
  - the `python -m rlforge.pipeline <tasks.jsonl>` CLI shim (bottom of file)
  - tests/test_pipeline.py (end-to-end smoke test)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, NamedTuple

from pydantic import BaseModel, Field

from rlforge.classify.taxonomy import FailureType
from rlforge.extract.extractor import PRE_EXECUTION_FAILURES, extract_code
from rlforge.metrics.metrics import RunMetrics, compute_metrics
from rlforge.registry.jsonl_logger import log_execution, log_generation, log_summary
from rlforge.registry.runinfo import (
    git_commit,
    hardware_info,
    new_execution_id,
    new_run_id,
    new_sample_id,
    utc_now_iso,
)
from rlforge.sandbox import make_sandbox
from rlforge.sandbox.base import SandboxConfig
from rlforge.sandbox.bwrap_sandbox import bwrap_usable
from rlforge.schema.execution import ExecutionRecord, ExecutionResult
from rlforge.schema.records import GenerationRecord
from rlforge.schema.task import CodeTask
from rlforge.verify.compiler import suite_to_test_code


class GenOutput(NamedTuple):
    """What a generate_fn returns for one (task, sample_idx) call.

    `raw_output` is `None` when there is nothing to generate from (the M1 stub's
    case: the task has no reference_solution) — the pipeline skips that sample
    entirely rather than fabricating an empty generation record for it.
    """

    raw_output: str | None
    model_name: str
    decoding_temperature: float
    top_p: float
    max_new_tokens: int
    seed: int
    input_tokens: int
    output_tokens: int
    # The exact, post-template string sent to the model (or, for a stub that
    # never calls a model, an honest note of what the stub actually used).
    # Never reconstructed later from task_id + prompt_template_id — see
    # GenerationRecord.final_prompt for why that would be unreliable.
    final_prompt: str
    prompt_template_id: str = "bare"


GenerateFn = Callable[[CodeTask, int], GenOutput]


def echo_reference_stub(task: CodeTask, sample_idx: int) -> GenOutput:
    """M1's generation stand-in: echo the task's reference_solution verbatim.

    Proves the extract -> sandbox -> classify -> record -> metrics wiring is
    correct (a correct pipeline should score ~100% pass on echoed reference
    solutions) without depending on a real model, which lands in M2.

    Fields are honest sentinels, not fabricated "real-looking" values, so stub
    data can never be mistaken for a genuine generation run in the logs:
    model_name is tagged "stub:echo-reference"; temperature/top_p/seed are the
    deterministic-decoding values that make sense for a non-sampling stub;
    token counts are a transparent whitespace-split proxy (not a real
    tokenizer, since M1 has none).
    """
    raw = task.reference_solution  # None if the task has nothing to echo
    token_proxy = len(raw.split()) if raw else 0
    return GenOutput(
        raw_output=raw,
        model_name="stub:echo-reference",
        decoding_temperature=0.0,
        top_p=1.0,
        max_new_tokens=0,
        seed=0,
        input_tokens=len(task.prompt.split()),
        output_tokens=token_proxy,
        # The stub never calls a model, so it never builds a real prompt via
        # prompt_template.build_prompt — recording that plainly (task.prompt,
        # unwrapped) is more honest than fabricating a template call that
        # didn't happen. A real generate_fn sets this to build_prompt(task).
        final_prompt=task.prompt,
    )


class RunConfig(BaseModel):
    """Knobs for one pipeline run (PLAN §1.6)."""

    out_dir: str
    sandbox_config: SandboxConfig = Field(default_factory=lambda: SandboxConfig(backend="bwrap"))
    n_samples: int = 1
    ks: tuple[int, ...] = (1,)
    # Which named suite is the pass/fail authority for metrics (never "visible":
    # the model may have seen those tests, so they cannot be the grading truth).
    grading_role: str = "hidden"


def _default_sandbox_config() -> SandboxConfig:
    """Prefer the real bwrap cage; fall back to subprocess-only if unusable.

    Only the pipeline's default prefers bwrap — SandboxConfig's own class
    default (sandbox/base.py) stays "subprocess", so unit tests that construct
    a bare SandboxConfig remain host-independent. A real run should get the
    strongest cage available, with a visible warning if it can't.
    """
    if bwrap_usable():
        return SandboxConfig(backend="bwrap")
    print(
        "[pipeline] bwrap is not usable on this host; falling back to "
        "backend='subprocess' (layers A+B only, no network/filesystem cage). "
        "See RESEARCH_LOG.md for the bwrap fix.",
        file=sys.stderr,
    )
    return SandboxConfig(backend="subprocess")


def run_pipeline(
    tasks: list[CodeTask],
    generate_fn: GenerateFn = echo_reference_stub,
    config: RunConfig | None = None,
) -> RunMetrics:
    """Run the full pipeline over `tasks` and return the aggregate RunMetrics.

    For each task, generates `config.n_samples` candidates. Each candidate is
    extracted, then run against EVERY suite the task defines (not just the
    grading suite) — one ExecutionRecord per suite, exercising the one-
    generation-fans-to-many-executions design (PLAN §1.5). Only the
    `config.grading_role` suite's executions feed compute_metrics(); a task
    with no reference_solution (nothing to generate from) is skipped entirely
    with a logged warning, producing no records.

    Every record is written the moment it's produced (log_generation /
    log_execution), so a crash mid-run loses nothing already on disk.
    """
    if config is None:
        config = RunConfig(out_dir=".", sandbox_config=_default_sandbox_config())

    run_id = new_run_id()
    out_dir = Path(config.out_dir) / run_id
    generations_path = out_dir / "generations.jsonl"
    executions_path = out_dir / "executions.jsonl"
    summary_path = out_dir / "summary.jsonl"

    sandbox = make_sandbox(config.sandbox_config)
    grading_execs: list[ExecutionRecord] = []
    last_model_name = "unknown"

    for task in tasks:
        for sample_idx in range(config.n_samples):
            gen = generate_fn(task, sample_idx)
            if gen.raw_output is None:
                # Nothing to generate from (M1 stub: no reference_solution).
                # Skip the whole sample: no GenerationRecord, no executions.
                print(
                    f"[pipeline] skip {task.task_id} sample {sample_idx}: "
                    "no reference_solution",
                    file=sys.stderr,
                )
                continue

            last_model_name = gen.model_name
            sample_id = new_sample_id(run_id, task.task_id, sample_idx)
            res = extract_code(gen.raw_output)

            gen_record = GenerationRecord(
                run_id=run_id,
                sample_id=sample_id,
                task_id=task.task_id,
                dataset_source=task.source,
                split="smoke",
                model_name=gen.model_name,
                final_prompt=gen.final_prompt,
                prompt_template_id=gen.prompt_template_id,
                decoding_temperature=gen.decoding_temperature,
                top_p=gen.top_p,
                max_new_tokens=gen.max_new_tokens,
                seed=gen.seed,
                input_tokens=gen.input_tokens,
                output_tokens=gen.output_tokens,
                raw_output=gen.raw_output,
                extracted_code=res.code,
                extraction_status=res.status,
                timestamp=utc_now_iso(),
                git_commit=git_commit(),
                hardware_info=hardware_info(),
            )
            log_generation(generations_path, gen_record)

            for suite_name, suite in task.test_suites.items():
                if res.status in PRE_EXECUTION_FAILURES:
                    # Unparseable/empty code never reaches the sandbox — an
                    # EXTRACTION failure is not a claim about runtime behavior.
                    result = ExecutionResult(
                        passed=False,
                        failure_type=FailureType.EXTRACTION,
                        execution_time_s=0.0,
                    )
                else:
                    # suite_to_test_code raises NotImplementedError for kinds
                    # M1 cannot run (stdin_stdout/checker) — intentionally not
                    # caught here; see verify/compiler.py and Step 7.5. An
                    # unsupported suite must stop the run, never be silently
                    # scored as a pass.
                    test_code = suite_to_test_code(suite, task.entry_point)
                    result = sandbox.run(res.code, test_code)

                exec_record = ExecutionRecord(
                    execution_result_id=new_execution_id(sample_id, suite_name),
                    sample_id=sample_id,
                    task_id=task.task_id,
                    sandbox_config_id=config.sandbox_config.sandbox_config_id,
                    result=result,
                    timeout_s=config.sandbox_config.timeout_s,
                    test_suite_id=suite_name,
                    visible_or_hidden_tests=suite_name,
                    test_suite_size=suite.size(),
                    timestamp=utc_now_iso(),
                )
                log_execution(executions_path, exec_record)
                if suite_name == config.grading_role:
                    grading_execs.append(exec_record)

    metrics = compute_metrics(grading_execs, config.ks)
    log_summary(
        summary_path,
        run_id,
        metrics,
        extra={
            "model_name": last_model_name,
            "git_commit": git_commit(),
            "hardware_info": hardware_info(),
            "sandbox_config_id": config.sandbox_config.sandbox_config_id,
            "n_samples": config.n_samples,
        },
    )
    print(
        f"[pipeline] run {run_id}: pass@1={metrics.pass_at_1:.3f} "
        f"pass@k={metrics.pass_at_k} failures={metrics.failure_dist}"
    )
    return metrics


if __name__ == "__main__":
    from rlforge.adapters.task_loader import load_tasks

    if len(sys.argv) < 2:
        print("usage: python -m rlforge.pipeline <tasks.jsonl>", file=sys.stderr)
        raise SystemExit(1)

    loaded_tasks = load_tasks(sys.argv[1])
    run_pipeline(
        loaded_tasks,
        config=RunConfig(
            # In-repo, matching ReasoningLab's convention: runs/ is a tracked
            # (but gitignored-contents) folder in the repo, not on ~/data.
            out_dir="runs",
            sandbox_config=_default_sandbox_config(),
        ),
    )
