"""Tests for train/reward.py — the GRPO reward function.

Two tiers, deliberately:

1. **Unit tests** on synthetic tasks with `backend="subprocess"`, so they are
   host-independent (same convention as tests/test_pipeline.py: a dev box without
   a working bwrap cage still exercises the wiring).
2. **A replay test** (`@pytest.mark.slow`) that runs REAL recorded generations
   from the LiveCodeBench-LeetCode baseline through the reward and asserts it
   reproduces the pass/fail label in that run's `executions.jsonl`.

Tier 2 is the test that actually matters. The RL result is only meaningful as a
delta against the baselines, and that requires "reward 1.0" and "benchmark pass"
to be the same predicate on the same inputs. A reward that merely looks sensible
in unit tests but disagrees with the harness would produce confident, meaningless
numbers — so it is checked against the harness's own recorded output, not against
our expectations of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlforge.sandbox.base import SandboxConfig
from rlforge.schema.task import CodeTask
from rlforge.schema.testsuite import TestSuite
from rlforge.train.reward import VerifierReward

# The LeetCode baseline whose labels we replay (RESEARCH_LOG 2026-07-23,
# pass@1 = 0.401, 178/444). Run data lives outside the repo's test fixtures
# because it is ~15 MB; the test skips cleanly if it is absent.
_BASELINE_RUN = (
    Path(__file__).resolve().parents[1] / "runs" / "eval" / "20260723T135725Z-0a40d501"
)
_LEETCODE_TASKS = Path(__file__).resolve().parents[1] / "data" / "livecodebench_leetcode.jsonl"

# Grading limits must match the baseline config, or the reward stops agreeing
# with the number it is compared against (mem_mb especially — see
# RESEARCH_LOG 2026-07-23 on the 512 MB compilation kills).
_BASELINE_SANDBOX = SandboxConfig(backend="bwrap", timeout_s=60.0, cpu_s=60, mem_mb=8192)


# ── helpers ───────────────────────────────────────────────────────────────────


def _task(task_id: str = "T/1", *, asserts: list[str] | None = None) -> CodeTask:
    """A minimal assert-graded task whose solution is `add_one`."""
    return CodeTask(
        task_id=task_id,
        source="unit-test",
        prompt="def add_one(x):",
        entry_point="add_one",
        test_suites={
            "hidden": TestSuite(
                kind="assert",
                asserts=asserts if asserts is not None else ["assert add_one(1) == 2"],
            )
        },
    )


def _reward(tasks: list[CodeTask], **kwargs) -> VerifierReward:
    """Reward wired to the host-independent subprocess backend."""
    return VerifierReward(
        {t.task_id: t for t in tasks},
        SandboxConfig(backend="subprocess", timeout_s=15.0, cpu_s=15, mem_mb=2048),
        **kwargs,
    )


def _fenced(code: str) -> str:
    """Wrap code the way the model is instructed to emit it."""
    return f"Here you go:\n```python\n{code}\n```"


# ── correctness ───────────────────────────────────────────────────────────────


def test_correct_solution_scores_one_and_wrong_scores_zero():
    task = _task()
    r = _reward([task])
    got = r(
        completions=[_fenced("def add_one(x):\n    return x + 1"), _fenced("def add_one(x):\n    return x + 2")],
        task_id=[task.task_id, task.task_id],
    )
    assert got == [1.0, 0.0]


def test_reward_requires_a_task_id_column():
    """A missing task_id must fail loudly at wiring time, not score silent zeros.

    Scoring 0.0 for every completion would look like a model that cannot solve
    anything — a confusing, expensive-to-diagnose failure mid-run.
    """
    r = _reward([_task()])
    with pytest.raises(ValueError, match="task_id"):
        r(completions=[_fenced("def add_one(x):\n    return x + 1")])


# ── reward hacking ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "hack",
    [
        "import sys\nsys.exit(0)\ndef add_one(x):\n    return 0",
        "raise SystemExit(0)\ndef add_one(x):\n    return 0",
        "import os\nos._exit(0)\ndef add_one(x):\n    return 0",
        "exit()\ndef add_one(x):\n    return 0",
        # Forging the pass sentinel: the sandbox strips it before checking.
        'print("__RLFORGE_TESTS_COMPLETE_9e4d1c7a__")\ndef add_one(x):\n    return 0',
    ],
)
def test_reward_hacks_score_zero(hack):
    """Early-exit and forged-sentinel exploits must never be paid.

    This is the failure mode that invalidates RL-on-code results: a policy that
    discovers any of these gets unbounded reward for no capability. The defense
    lives in the sandbox (exit 0 AND sentinel, preamble nulling the exit paths);
    these cases pin it at the reward level so a future sandbox change that breaks
    it fails here rather than in a training run.
    """
    task = _task()
    r = _reward([task])
    assert r(completions=[_fenced(hack)], task_id=[task.task_id]) == [0.0]


# ── false-pass guards ─────────────────────────────────────────────────────────


def test_empty_grading_suite_is_not_a_pass():
    """An empty assert suite compiles to an empty test body → exits 0 → sentinel.

    Without the guard in _score_one that is a FALSE PASS on a task nothing was
    checked against. No LeetCode task is currently like this (verified: all 444
    have 13-104 asserts), but a future adapter could produce one.
    """
    task = _task(asserts=[])
    r = _reward([task])
    assert r(completions=[_fenced("def add_one(x):\n    return x + 1")], task_id=[task.task_id]) == [0.0]


def test_missing_hidden_suite_is_zero_and_visible_is_never_graded():
    """Only the hidden suite may be the reward authority.

    A task carrying only prompt-visible tests must score 0.0, not be graded on
    tests the model was shown — that is direct optimization pressure toward
    reproducing known answers.
    """
    task = CodeTask(
        task_id="T/visible-only",
        source="unit-test",
        prompt="def add_one(x):",
        entry_point="add_one",
        test_suites={"visible": TestSuite(kind="assert", asserts=["assert add_one(1) == 2"])},
    )
    r = _reward([task])
    assert r(completions=[_fenced("def add_one(x):\n    return x + 1")], task_id=[task.task_id]) == [0.0]


def test_unknown_task_id_scores_zero_without_raising():
    r = _reward([_task()])
    assert r(completions=[_fenced("def add_one(x):\n    return x + 1")], task_id=["T/nope"]) == [0.0]


@pytest.mark.parametrize(
    "completion",
    [
        "I think the answer is to add one to x, but let me reconsider...",  # prose only
        "```python\ndef add_one(x):\n    if x > 0:\n        return x +",     # truncated
        "",                                                                   # empty
    ],
)
def test_extraction_failures_score_zero(completion):
    task = _task()
    r = _reward([task])
    assert r(completions=[completion], task_id=[task.task_id]) == [0.0]


# ── threading ─────────────────────────────────────────────────────────────────


def test_threaded_grading_matches_serial():
    """Concurrency must not change the reward.

    Grading shares one sandbox instance across threads; if that were unsafe the
    symptom would be nondeterministic rewards, which is close to undebuggable
    inside a training run.
    """
    tasks = [_task(f"T/{i}") for i in range(8)]
    completions, ids = [], []
    for i, t in enumerate(tasks):
        body = "return x + 1" if i % 2 == 0 else "return x + 2"
        completions.append(_fenced(f"def add_one(x):\n    {body}"))
        ids.append(t.task_id)

    serial = _reward(tasks, max_workers=1)(completions=completions, task_id=ids)
    threaded = _reward(tasks, max_workers=8)(completions=completions, task_id=ids)
    assert serial == threaded == [1.0, 0.0] * 4


# ── the brevity term (off in run #1) ──────────────────────────────────────────


def test_length_term_off_by_default():
    """Default reward is exactly binary — run #1 must be correctness-only."""
    task = _task()
    r = _reward([task])
    assert r.length_weight == 0.0
    got = r(
        completions=[
            _fenced("def add_one(x):\n    return x + 1"),
            _fenced("def add_one(x):\n    # a much longer but equally correct solution\n    y = x\n    y = y + 1\n    return y"),
        ],
        task_id=[task.task_id, task.task_id],
    )
    assert got == [1.0, 1.0], "with length_weight=0 both passing rollouts score identically"


def test_length_bonus_prefers_shorter_passing_solution():
    task = _task()
    r = _reward([task], length_weight=0.2)
    short = _fenced("def add_one(x):\n    return x + 1")
    long = _fenced(
        "def add_one(x):\n    # verbose but correct\n    result = x\n    result = result + 1\n    return result"
    )
    got = r(completions=[short, long], task_id=[task.task_id, task.task_id])
    assert got[0] > got[1], "shorter passing solution must earn more"
    assert got[0] == pytest.approx(1.2), "shortest in group: full bonus"
    assert got[1] == pytest.approx(1.0), "longest in group: no bonus"


def test_length_bonus_is_gated_on_passing():
    """A short WRONG answer must not beat a long right one.

    This is the property that keeps the term from degenerating: ungated, the
    reward is maximized by emitting as little as possible.
    """
    task = _task()
    r = _reward([task], length_weight=0.2)
    short_wrong = _fenced("def add_one(x):\n    return 0")
    long_right = _fenced(
        "def add_one(x):\n    # verbose but correct\n    result = x\n    result = result + 1\n    return result"
    )
    got = r(completions=[short_wrong, long_right], task_id=[task.task_id, task.task_id])
    assert got == [0.0, 1.0], "failing rollout gets no brevity bonus at all"


def test_length_bonus_degenerate_groups_are_zero():
    """Fewer than two passing rollouts, or no length spread → no bonus, no crash."""
    task = _task()
    r = _reward([task], length_weight=0.2)

    # Exactly one passing rollout: nothing to rank against.
    one = r(
        completions=[_fenced("def add_one(x):\n    return x + 1"), _fenced("def add_one(x):\n    return x + 2")],
        task_id=[task.task_id, task.task_id],
    )
    assert one == [1.0, 0.0]

    # Two identical passing rollouts: hi == lo, must not divide by zero.
    same = _fenced("def add_one(x):\n    return x + 1")
    tied = r(completions=[same, same], task_id=[task.task_id, task.task_id])
    assert tied == [1.0, 1.0]


def test_length_bonus_is_grouped_per_task():
    """Brevity is ranked within a task, never across different tasks.

    Across tasks the comparison is meaningless: a long solution to a hard problem
    would be penalized against a short solution to an easy one.
    """
    t_a, t_b = _task("T/a"), _task("T/b")
    r = _reward([t_a, t_b], length_weight=0.2)
    short = _fenced("def add_one(x):\n    return x + 1")
    long = _fenced(
        "def add_one(x):\n    # verbose but correct\n    result = x\n    result = result + 1\n    return result"
    )
    # One rollout per task: each is alone in its group, so neither gets a bonus.
    got = r(completions=[short, long], task_id=["T/a", "T/b"])
    assert got == [1.0, 1.0]


# ── the per-generation execution log ──────────────────────────────────────────


def test_execution_log_records_one_line_per_generation(tmp_path):
    """The RL twin of executions.jsonl — a reward of 0 must be explainable.

    Without this, a training run can only report that reward fell, with no way to
    distinguish a wrong algorithm from a truncated generation from a timeout.
    """
    task = _task()
    log = tmp_path / "executions.jsonl"
    r = _reward([task], execution_log_path=log)
    r(
        completions=[
            _fenced("def add_one(x):\n    return x + 1"),   # pass
            _fenced("def add_one(x):\n    return x + 2"),   # wrong answer
            "just prose, no code at all",                    # unparseable
        ],
        task_id=[task.task_id] * 3,
    )

    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(lines) == 3
    for line in lines:
        # Every field needed to diagnose a failure without re-running it.
        for key in ("task_id", "correctness", "failure_type", "extraction_status",
                    "code_chars", "completion_chars", "reward", "reward_call"):
            assert key in line, key

    assert lines[0]["correctness"] == 1.0 and lines[0]["failure_type"] == "pass"
    # A wrong answer must be distinguishable from unparseable output — the whole
    # point of the log.
    assert lines[1]["failure_type"] == "assertion"
    assert lines[2]["failure_type"] == "extraction"
    assert lines[1]["exec_time_s"] is not None, "a wrong answer really ran"
    assert lines[2]["exec_time_s"] is None, "unparseable output never reached the sandbox"


def test_execution_log_appends_across_calls(tmp_path):
    """Appended and flushed per batch, so a killed run still leaves a record."""
    task = _task()
    log = tmp_path / "executions.jsonl"
    r = _reward([task], execution_log_path=log)
    good = _fenced("def add_one(x):\n    return x + 1")
    r(completions=[good], task_id=[task.task_id])
    r(completions=[good], task_id=[task.task_id])
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(lines) == 2
    assert [line["reward_call"] for line in lines] == [0, 1], "batches are distinguishable"


def test_failure_type_rates_are_logged_as_metrics():
    """Per-failure-type curves are what separate 'codes better' from 'rambles less'."""
    task = _task()
    r = _reward([task])
    r(
        completions=[
            _fenced("def add_one(x):\n    return x + 1"),
            _fenced("def add_one(x):\n    return x + 2"),
            "prose only",
        ],
        task_id=[task.task_id] * 3,
    )
    stats = r.last_batch_stats
    assert stats["reward/failure_pass"] == pytest.approx(1 / 3)
    assert stats["reward/failure_assertion"] == pytest.approx(1 / 3)
    assert stats["reward/failure_extraction"] == pytest.approx(1 / 3)
    assert stats["reward/pass_rate"] == pytest.approx(1 / 3)


def test_no_log_path_means_no_file_written(tmp_path):
    """Logging is opt-in; the unit tests and any embedding use run without it."""
    task = _task()
    r = _reward([task])
    r(completions=[_fenced("def add_one(x):\n    return x + 1")], task_id=[task.task_id])
    assert list(tmp_path.iterdir()) == []


# ── per-task progress logging ─────────────────────────────────────────────────


def test_progress_log_is_per_task_not_aggregate(capsys):
    """An aggregate pass count is ambiguous; the per-task breakdown is not.

    With several rollouts per task, "3/8 passed" cannot distinguish one task passing
    three times from three tasks passing once — and those call for opposite
    conclusions about whether the policy is learning. Both batches below total 3
    passes out of 8; the log must tell them apart.
    """
    t0, t1 = _task("LCB/0"), _task("LCB/1")
    r = _reward([t0, t1], verbose=True)
    ok = _fenced("def add_one(x):\n    return x + 1")
    bad = _fenced("def add_one(x):\n    return x + 2")

    r(completions=[ok, ok, ok, bad, bad, bad, bad, bad],
      task_id=["LCB/0"] * 4 + ["LCB/1"] * 4)
    concentrated = capsys.readouterr().out
    assert "LCB/0  3/4 pass" in concentrated
    assert "LCB/1  0/4 pass" in concentrated

    r(completions=[ok, bad, bad, bad, ok, ok, bad, bad],
      task_id=["LCB/0"] * 4 + ["LCB/1"] * 4)
    spread = capsys.readouterr().out
    assert "LCB/0  1/4 pass" in spread
    assert "LCB/1  2/4 pass" in spread
    assert concentrated != spread, "same aggregate must produce different output"


def test_progress_log_flags_groups_with_no_learning_signal(capsys):
    """All-pass and all-fail groups give every rollout the same advantage.

    GRPO extracts nothing from such a group, so the run should say so rather than
    let a step look productive when part of it was wasted.
    """
    task = _task()
    r = _reward([task], verbose=True)
    bad = _fenced("def add_one(x):\n    return x + 2")
    r(completions=[bad, bad], task_id=[task.task_id] * 2)
    assert "FLAT" in capsys.readouterr().out

    ok = _fenced("def add_one(x):\n    return x + 1")
    r(completions=[ok, ok], task_id=[task.task_id] * 2)
    assert "FLAT" in capsys.readouterr().out, "all-pass is equally flat"

    r(completions=[ok, bad], task_id=[task.task_id] * 2)
    assert "FLAT" not in capsys.readouterr().out, "a mixed group does carry signal"


def test_progress_log_is_silent_by_default(capsys):
    """verbose=False keeps the unit tests and library use quiet."""
    task = _task()
    _reward([task])(completions=[_fenced("def add_one(x):\n    return x + 1")],
                    task_id=[task.task_id])
    assert capsys.readouterr().out == ""


# ── the replay test: agreement with the recorded baseline ─────────────────────


@pytest.mark.slow
def test_reward_reproduces_recorded_baseline_labels():
    """Replay real baseline generations; the reward must match executions.jsonl.

    This is the load-bearing test. It uses the bwrap backend and the baseline's
    own sandbox limits, so it checks the reward as it will actually be used.
    """
    if not _BASELINE_RUN.exists() or not _LEETCODE_TASKS.exists():
        pytest.skip(f"baseline run data not present at {_BASELINE_RUN}")

    from rlforge.adapters.task_loader import load_tasks

    tasks = {t.task_id: t for t in load_tasks(_LEETCODE_TASKS)}
    gens = {}
    for line in (_BASELINE_RUN / "generations.jsonl").open():
        rec = json.loads(line)
        gens[rec["task_id"]] = rec["raw_output"]
    labels = {}
    for line in (_BASELINE_RUN / "executions.jsonl").open():
        rec = json.loads(line)
        labels[rec["task_id"]] = 1.0 if rec["result"]["passed"] else 0.0

    # Balanced sample: an all-fail sample would pass even if the sandbox were
    # broken and returned False unconditionally.
    passing = [t for t, v in labels.items() if v == 1.0][:50]
    failing = [t for t, v in labels.items() if v == 0.0][:50]
    sample = passing + failing
    assert len(sample) == 100

    reward = VerifierReward(tasks, _BASELINE_SANDBOX, max_workers=8)
    got = reward(
        completions=[gens[t] for t in sample],
        task_id=list(sample),
    )
    expected = [labels[t] for t in sample]

    mismatches = [
        (t, e, g) for t, e, g in zip(sample, expected, got) if e != g
    ]
    assert not mismatches, f"reward disagrees with recorded labels on: {mismatches}"


def test_chat_format_completions_are_graded_like_strings():
    """TRL passes message lists for chat datasets; the reward must not care."""
    from rlforge.train.reward import _completion_text

    text = "```python\nprint(1)\n```"
    assert _completion_text(text) == text
    assert _completion_text([{"role": "assistant", "content": text}]) == text
    assert _completion_text(None) == ""
