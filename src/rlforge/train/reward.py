"""
train/reward.py
───────────────
`VerifierReward` — the GRPO reward function: a TRL-compatible callable that
scores a batch of model completions by actually running them against their
hidden test suite in the bwrap sandbox.

Why this is a thin wrapper and not new code
───────────────────────────────────────────
The reward replicates `pipeline.py`'s per-sample sequence (pipeline.py:248-265)
step for step:

    extract_code → grading_suite("hidden") → suite_to_test_code → sandbox.run

That identity is the point. The RL result is only meaningful as a delta against
the three baselines in RESEARCH_LOG, and that delta only holds if "reward 1.0"
and "benchmark pass" are the same predicate. Re-implementing grading here — even
slightly differently — would silently break the comparison while still producing
plausible-looking numbers. So: no new verification logic, only orchestration.

Verified against real data (this is the check that matters): replaying recorded
generations from the LeetCode baseline run through this reward reproduces the
pass/fail label in `executions.jsonl` exactly. See `tests/test_reward.py`.

Reward-hacking posture
──────────────────────
The sandbox — not this module — is the defense, and it is a real one: a pass
requires exit code 0 AND a sentinel token in stdout (`subprocess_sandbox.py`
:180-181), with `sys.exit`/`os._exit`/`exit()` nulled by the safety preamble and
a pre-printed sentinel stripped before the check. Empirically `sys.exit(0)`,
`raise SystemExit(0)`, `os._exit(0)`, `exit()` and a forged sentinel all score
0.0. This module adds the two guards the sandbox cannot make for itself: an
absent or empty grading suite must score 0.0, never a pass, because an empty
assert suite compiles to an empty test body — a script that trivially exits 0
and prints the sentinel.

Cost (measured, drives the threading)
─────────────────────────────────────
~1.7 s per completion on real LiveCodeBench-LeetCode suites (13-104 asserts each,
run sequentially in one process). At `num_generations=8` that is ~13.6 s per GRPO
step if graded serially, against ~31 s of generation — worth overlapping. Grading
is I/O-bound (each call waits on a subprocess), so threads work despite the GIL;
a single sandbox instance is safe to share and was verified to give identical
rewards at 1 and 8 workers.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rlforge.extract.extractor import extract_code
from rlforge.sandbox import make_sandbox
from rlforge.sandbox.base import SandboxConfig
from rlforge.schema.task import CodeTask
from rlforge.verify.compiler import suite_to_test_code


class VerifierReward:
    """Score completions by executing them against their hidden test suite.

    Call signature follows TRL's custom-reward contract: the trainer passes
    `prompts` and `completions` plus every other column of the training dataset
    as keyword arguments. We rely on a `task_id` column to recover which task a
    completion belongs to — matching on the prompt text would be fragile, and
    the group structure (`num_generations` completions per prompt) means the same
    prompt legitimately appears many times in one batch.
    """

    # TRL names the logged reward column after the function; a class instance has
    # no __name__ of its own, so set one or the metric shows up unnamed.
    __name__ = "verifier_reward"

    def __init__(
        self,
        tasks: dict[str, CodeTask],
        sandbox_config: SandboxConfig | None = None,
        *,
        correctness_weight: float = 1.0,
        length_weight: float = 0.0,
        max_workers: int = 8,
        execution_log_path: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            tasks: task_id → CodeTask, normally `{t.task_id: t for t in
                load_tasks(...)}`. Held by reference; the reward never mutates it.
            sandbox_config: limits for grading. Use the SAME values as the
                benchmark config (bwrap, timeout_s 60, cpu_s 60, mem_mb 8192) or
                the reward stops agreeing with the baseline it is compared to —
                notably mem_mb, where the 512 MB default silently failed tasks
                during test COMPILATION (RESEARCH_LOG 2026-07-23).
            correctness_weight: weight on the pass/fail term.
            length_weight: weight on the brevity bonus. **0.0 for run #1.**
                Kept as a config knob rather than absent code so enabling it is a
                one-number change and the two runs differ by exactly one variable.
            max_workers: threads used to grade one batch.
        """
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")

        self._tasks = tasks
        # Running tallies for the metrics the trainer logs each step. These exist
        # because of what the baseline data showed: LeetCode scores 0.401 overall
        # but 0.511 on non-truncated generations, so ~11 points of the gap is the
        # model rambling past the token budget rather than getting the algorithm
        # wrong. If GRPO improves pass@1 mainly by shortening outputs, that is a
        # legitimate result — but only if it is MEASURED. Without these counters,
        # "the policy learned to code better" and "the policy learned to stop
        # rambling" are indistinguishable in the final numbers.
        self.last_batch_stats: dict[str, float] = {}
        # Build the sandbox ONCE. BwrapSandbox's constructor probes the kernel for
        # usable user namespaces; doing that per completion would add a process
        # spawn to every one of the thousands of reward calls in a run.
        self._sandbox = make_sandbox(sandbox_config or SandboxConfig(backend="bwrap"))
        self.correctness_weight = correctness_weight
        self.length_weight = length_weight
        self.max_workers = max_workers
        # Print a per-task breakdown after each graded batch. On for real runs: TRL's
        # progress bar only advances when a whole optimizer step completes, which here
        # means ~23 minutes of total silence per step. Off by default so the unit
        # tests stay quiet.
        self.verbose = verbose
        # Per-generation execution log (the RL twin of the benchmark's
        # executions.jsonl). None disables it, which is what the unit tests use.
        self._log_path = Path(execution_log_path) if execution_log_path else None
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # Counts reward invocations so a log line can be tied back to the batch it
        # came from. Not the optimizer step: with gradient accumulation the trainer
        # calls the reward once per generation batch, several times per step.
        self._call_count = 0

    # ── the TRL entry point ────────────────────────────────────────────────────

    def __call__(
        self,
        prompts: list[str] | None = None,
        completions: list[str] | None = None,
        **kwargs: object,
    ) -> list[float]:
        """Return one reward per completion.

        Never raises: an exception here would kill a multi-hour training run, so
        a completion that cannot be graded scores 0.0 (the same outcome the
        benchmark harness gives an unusable generation) rather than propagating.
        """
        completions = completions or []
        task_ids = kwargs.get("task_id")
        if not isinstance(task_ids, list) or len(task_ids) != len(completions):
            raise ValueError(
                "VerifierReward requires a `task_id` column on the training "
                f"dataset, one per completion (got {type(task_ids).__name__} "
                f"for {len(completions)} completions). Add task_id to the dataset."
            )

        # Grade concurrently — each call blocks on a sandbox subprocess.
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            graded = list(pool.map(self._score_one, task_ids, completions))

        correctness = [g["correctness"] for g in graded]
        lengths = [g["code_chars"] for g in graded]
        rewards = [self.correctness_weight * c for c in correctness]

        if self.length_weight:
            bonuses = self._length_bonus(correctness, lengths, task_ids)
            rewards = [r + self.length_weight * b for r, b in zip(rewards, bonuses)]

        self._record_batch_stats(correctness, lengths, completions, graded)
        self._append_execution_log(graded, rewards)
        if self.verbose:
            self._print_progress(graded, task_ids)
        self._call_count += 1
        return rewards

    def _print_progress(self, graded: list[dict], task_ids: list[str]) -> None:
        """Print a PER-TASK breakdown of the batch just graded.

        Deliberately per-task rather than per-batch. An aggregate like "5/16 passed"
        is ambiguous in exactly the way that matters here: with several rollouts per
        task it cannot distinguish one easy task passing five times from five
        different tasks passing once. Those two cases call for opposite conclusions
        about whether the policy is learning, so the aggregate is not worth printing.

        One line per task, showing how many of ITS OWN rollouts passed and what the
        rest failed on:

            [reward] call 3  LCB/2755  3/8 pass  (assertion 3, extraction 2)

        A task at 0/8 or 8/8 is also flagged, because those are the groups GRPO
        learns nothing from — every rollout gets the same advantage, so the step is
        wasted on them.
        """
        # Preserve first-appearance order so the printout follows the batch.
        order: list[str] = []
        per_task: dict[str, list[dict]] = {}
        for task_id, rec in zip(task_ids, graded):
            if task_id not in per_task:
                per_task[task_id] = []
                order.append(task_id)
            per_task[task_id].append(rec)

        for task_id in order:
            recs = per_task[task_id]
            n_pass = sum(1 for r in recs if r["correctness"] > 0.0)
            fails = Counter(
                r["failure_type"] for r in recs if r["correctness"] == 0.0
            )
            detail = ", ".join(f"{k} {v}" for k, v in fails.most_common()) or "-"
            # No spread within a group => zero advantage => nothing learned from it.
            flat = " [FLAT: no learning signal]" if n_pass in (0, len(recs)) else ""
            print(
                f"[reward] call {self._call_count}  {task_id}  "
                f"{n_pass}/{len(recs)} pass  ({detail}){flat}",
                flush=True,
            )

    def _append_execution_log(self, graded: list[dict], rewards: list[float]) -> None:
        """Append one JSONL line per generation, if a log path was configured.

        The RL analogue of the benchmark's executions.jsonl. Written append-only and
        flushed per batch so a run killed at hour six still leaves a complete record
        up to that point — the wiring-check runs in this project were repeatedly
        killed mid-flight, and a log that only materializes at the end would have
        been empty every time.
        """
        if self._log_path is None:
            return
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                for rec, reward in zip(graded, rewards):
                    fh.write(json.dumps({**rec, "reward": reward,
                                         "reward_call": self._call_count}) + "\n")
        except OSError:
            # Logging must never take down a training run.
            pass

    def _record_batch_stats(
        self,
        correctness: list[float],
        lengths: list[int],
        completions: list[str],
        graded: list[dict],
    ) -> None:
        """Stash per-batch diagnostics for the trainer callback to log.

        `extraction_failure_rate` is the truncation proxy: a generation cut off at
        the token cap does not parse, so it lands in `extract_code`'s
        `syntax_error`/`empty` bucket and scores 0 with `n_code_chars == 0`. That
        is the same mechanism that turned 96/444 LeetCode generations into
        `extraction` failures in the baseline, so the rate is directly comparable
        to it.
        """
        n = len(correctness) or 1
        unparseable = sum(1 for c, ln in zip(correctness, lengths) if ln == 0)
        passing_lengths = [ln for c, ln in zip(correctness, lengths) if c > 0.0]
        self.last_batch_stats = {
            "reward/pass_rate": sum(1 for c in correctness if c > 0.0) / n,
            "reward/extraction_failure_rate": unparseable / n,
            "reward/mean_completion_chars": sum(len(c) for c in completions) / n,
            "reward/mean_passing_code_chars": (
                sum(passing_lengths) / len(passing_lengths) if passing_lengths else 0.0
            ),
        }
        # Per-failure-type rates, logged as separate scalars so tensorboard shows
        # each as its own curve. This is what answers the question the pass@1 number
        # cannot: if reward rises while `assertion` falls, the policy is solving
        # more tasks; if reward rises only because `extraction` falls, it merely
        # learned to stop running past the token budget. Both are real results, but
        # they are different claims and must be distinguishable.
        for failure_type in {g["failure_type"] for g in graded}:
            count = sum(1 for g in graded if g["failure_type"] == failure_type)
            self.last_batch_stats[f"reward/failure_{failure_type}"] = count / n

    # ── per-completion grading ────────────────────────────────────────────────

    def _score_one(self, task_id: str, completion: str) -> dict:
        """Grade one completion and return a full diagnostic record.

        Returns a dict rather than a bare float because the reward pathway already
        computes everything needed to explain a failure (extraction status, failure
        type, stderr, execution time) and throwing it away is what makes an RL run
        undebuggable: the trainer would log "reward went down" with no way to tell a
        wrong algorithm from a truncated generation from a timeout. The benchmark
        harness records exactly this in executions.jsonl; the RL loop should not be
        blinder than the evaluation it is trying to improve.

        `code_chars` counts characters of EXTRACTED code, not of the whole
        completion: the brevity term is meant to prefer concise *solutions*, and
        measuring the full generation would instead reward suppressing commentary.
        """
        rec = {
            "task_id": task_id,
            "correctness": 0.0,
            "code_chars": 0,
            # Mirrors the benchmark's FailureType vocabulary so RL-side and
            # benchmark-side failure distributions are directly comparable.
            "failure_type": "extraction",
            "extraction_status": None,
            "exec_time_s": None,
            "stderr": "",
            "completion_chars": len(completion),
        }
        try:
            res = extract_code(completion)
            rec["extraction_status"] = res.status
            # Unparseable or empty output. Truncated generations land here too,
            # which is why the trainer also sets mask_truncated_completions (an
            # unfinished answer is not evidence of a wrong answer).
            if not res.ok:
                return rec

            task = self._tasks.get(task_id)
            if task is None:
                rec["failure_type"] = "unknown_task"
                return rec

            # Grading authority is the HIDDEN suite, never PROMPTABLE_SUITE
            # ("visible"): tests the model may have seen in its prompt cannot be
            # the reward signal, or the policy is optimized toward reproducing
            # known answers instead of solving the task.
            suite = task.grading_suite("hidden")
            if suite is None or suite.is_empty():
                rec["failure_type"] = "no_grading_suite"
                return rec

            test_code = suite_to_test_code(suite, task.entry_point)
            # An empty test body would make the assembled script exit 0 and print
            # the sentinel with nothing checked — a FALSE PASS. `is_empty()` above
            # catches the normal case; this guards the compiled result directly.
            if not test_code.strip():
                rec["failure_type"] = "no_grading_suite"
                return rec

            # mode MUST be the suite's own kind: a stdin_stdout candidate pasted
            # in assert mode executes at import on empty stdin and dies with
            # EOFError before any test runs (RESEARCH_LOG 2026-07-16).
            result = self._sandbox.run(res.code, test_code, mode=suite.kind)
            rec["correctness"] = 1.0 if result.passed else 0.0
            rec["code_chars"] = len(res.code)
            rec["exec_time_s"] = round(result.execution_time_s, 4)
            rec["failure_type"] = getattr(
                result.failure_type, "value", str(result.failure_type)
            )
            # Truncated to keep the log small; the head carries the exception type,
            # which is what distinguishes the failure modes.
            rec["stderr"] = (result.stderr or "")[:400]
            return rec
        except Exception as exc:
            # Defensive: keep a long run alive. An ungradable completion is a zero,
            # exactly as the harness would score it — but say so in the record
            # rather than silently looking like an extraction failure.
            rec["failure_type"] = "harness_error"
            rec["stderr"] = f"{type(exc).__name__}: {exc}"[:400]
            return rec

    # ── the brevity term (implemented, disabled by default) ───────────────────

    @staticmethod
    def _length_bonus(
        correctness: list[float],
        lengths: list[int],
        task_ids: list[str],
    ) -> list[float]:
        """Brevity bonus in [0, 1], computed WITHIN each task's rollout group.

        Two properties make this safe, and both matter:

        1. **Gated on correctness.** A completion that fails scores 0.0 bonus, so
           brevity can only ever break ties among solutions that already pass.
           An ungated length penalty is optimized by emitting nothing at all.
        2. **Normalized within the group**, not against a global token target.
           "Short" for a two-line easy task and a hard dynamic-programming task
           are different quantities; comparing each rollout only against its
           siblings on the SAME task removes the need for a hand-tuned constant
           and keeps the term scale-free across a mixed-difficulty benchmark.

        Degenerate groups yield 0.0 for everyone: with fewer than two passing
        rollouts there is nothing to rank, and with all passing lengths equal
        there is no spread to normalize (and dividing by it would be a zero
        division).
        """
        # Group indices by task. GRPO emits `num_generations` completions per
        # prompt, but we group by task_id rather than assuming a fixed stride, so
        # this stays correct regardless of how the trainer orders a batch.
        groups: dict[str, list[int]] = {}
        for i, tid in enumerate(task_ids):
            groups.setdefault(tid, []).append(i)

        bonuses = [0.0] * len(correctness)
        for idxs in groups.values():
            passing = [i for i in idxs if correctness[i] > 0.0 and lengths[i] > 0]
            if len(passing) < 2:
                continue
            lo = min(lengths[i] for i in passing)
            hi = max(lengths[i] for i in passing)
            if hi == lo:
                continue
            for i in passing:
                # Shortest passing rollout in the group → 1.0, longest → 0.0.
                bonuses[i] = (hi - lengths[i]) / (hi - lo)
        return bonuses
