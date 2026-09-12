"""
verify/reference_verifier.py
─────────────────────────────
Stage 3 of the TACO training-pool pipeline: run each problem's OWN official
solution against its OWN tests, and keep only the tests that solution satisfies.

WHY THIS EXISTS. A training pool's reward is only as trustworthy as its tests.
DeepCoder documented models learning to print memorised answers when a verifier
is weak, and a test that a *correct* solution fails is worse than useless in RL:
it actively teaches the policy that correct code is wrong. Because TACO ships
reference solutions, that failure mode is measurable rather than assumed — this
module is what measures it.

It is also the correctness proof for adapters/taco.py. Converting TACO's
function-call rows into assert strings needs guesses about an under-documented
format (argument packing, expected-value wrapping). Those guesses do not have to
be right, because a wrong one makes the official solution fail here and the
affected tests are dropped. A wrong heuristic costs recall, never correctness.

PRUNING, NOT PASS/FAIL — the design the data forced. The plan originally scored
each problem all-or-nothing. Measured on 25 real stdio tasks, that discards far
too much: only 2/25 were wholly broken, while 11/25 satisfied 50-90% of their
pairs, the rest failing on a minority of genuinely corrupt cases. One verified
example is TACO test row 7 (hackerrank newyear-game), where the upstream data
pairs a 3-test-case stdin against a 2-line expected stdout — the row is
mis-shaped in TACO itself, not by our conversion. So the verifier removes the
individual failing tests and keeps the problem, provided enough tests survive
(the >=5 threshold is applied downstream in Stage 4).

HOW PRUNING RUNS IN ONE SANDBOX CALL. Naively, pruning means one sandbox
execution per test (26k problems x tens of tests = far too slow). Instead this
module compiles a *probe* fragment that runs every test in a single process,
catches per-test exceptions, and prints one machine-readable verdict line. The
sandbox contract is unchanged: the probe still ends with the pass sentinel, so
a crash or timeout is still an ordinary failure, not a silent pass.

CONCURRENCY: threads, not processes. sandbox.run() spawns a subprocess and
blocks on it, so nearly all wall time is I/O wait — a ThreadPoolExecutor gets
full parallelism with no GIL contention and no pickling of CodeTask objects.
SubprocessSandbox is safe to share: run() allocates everything per call and the
only instance state is the immutable config. Two caveats:
  - preexec_fn after fork() in a threaded parent is safe ONLY because
    _set_limits does nothing but resource.setrlimit (no allocation, no locks).
    Do not add anything to it.
  - memory_peak_mb comes from getrusage(RUSAGE_CHILDREN), which is process-wide
    and cumulative, so it is meaningless under concurrency. It is deliberately
    NOT recorded here.

RESUME: the ledger is append-only JSONL, flushed per line under a lock, so a
kill loses at most a partial trailing line (tolerated on read). Re-running skips
task_ids already present. This matters because the full pass is hours long.

Duplication note: this re-implements a small slice of pipeline.py's inner loop
(compile suite -> run sandbox -> read result). That is deliberate — run_pipeline
is coupled to generation, the registry and metrics, none of which apply to
verifying a known solution — but the compile-and-run contract now lives in two
places and must be changed in both.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rlforge.sandbox.base import SandboxConfig
from rlforge.sandbox.subprocess_sandbox import CANDIDATE_SRC_VAR
from rlforge.schema.task import CodeTask
from rlforge.schema.testsuite import TestSuite

# Printed by the probe immediately before the verdict payload, so the verdict can
# be located in stdout even if the candidate program printed arbitrary text.
_VERDICT_MARKER = "__RLFORGE_VERDICT_5c1a__"

# Hard ceiling on tests probed per task. Suites in TACO run to hundreds of pairs
# (one row has 304); an RL rollout re-executes the suite thousands of times, so a
# large suite is a permanent throughput tax for no extra signal. Capping here
# also bounds Stage 3's own cost.
DEFAULT_MAX_TESTS = 30


@dataclass
class VerificationOutcome:
    """One task's verdict: which of its tests the official solution satisfies."""

    task_id: str
    kind: str
    status: str  # "verified" | "no_solution" | "no_tests" | "harness_error"
    n_tests_probed: int = 0
    n_tests_passed: int = 0
    passed_indices: list[int] = field(default_factory=list)
    execution_time_s: float = 0.0
    exit_code: int | None = None
    detail: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class VerificationStats:
    tasks_seen: int = 0
    tasks_skipped_resume: int = 0
    verified: int = 0
    no_solution: int = 0
    no_tests: int = 0
    harness_error: int = 0
    tests_probed: int = 0
    tests_passed: int = 0
    fully_passing: int = 0
    partially_passing: int = 0
    zero_passing: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    by_platform_passed: dict[str, int] = field(default_factory=dict)
    by_platform_total: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        rate = (self.tests_passed / self.tests_probed) if self.tests_probed else 0.0
        return (
            f"verify: {self.tasks_seen} tasks "
            f"(skipped {self.tasks_skipped_resume} already done); "
            f"verified={self.verified} no_solution={self.no_solution} "
            f"no_tests={self.no_tests} harness_error={self.harness_error}; "
            f"tests {self.tests_passed}/{self.tests_probed} kept ({rate:.3f}); "
            f"tasks fully={self.fully_passing} partial={self.partially_passing} "
            f"zero={self.zero_passing}"
        )


# ── Probe construction ───────────────────────────────────────────────────────


def _assert_probe(suite: TestSuite, max_tests: int) -> tuple[str, int]:
    """Probe fragment for an assert suite; returns (code, n_tests_probed).

    Non-assert setup lines (e.g. `sol = Solution()`) are preamble that every
    assert depends on, so they run once up front rather than being probed.
    """
    setup = [line for line in suite.asserts if not line.lstrip().startswith("assert ")]
    tests = [line for line in suite.asserts if line.lstrip().startswith("assert ")]
    tests = tests[:max_tests]
    code = f"""\
{chr(10).join(setup)}
import json as _json
_tests = _json.loads({json.dumps(tests)!r})
_ok = []
for _i, _t in enumerate(_tests):
    try:
        exec(_t, globals())
        _ok.append(_i)
    except BaseException:
        pass
print("{_VERDICT_MARKER}" + _json.dumps(_ok))
"""
    return code, len(tests)


def _stdio_probe(suite: TestSuite, max_tests: int) -> tuple[str, int]:
    """Probe fragment for a stdin_stdout suite; returns (code, n_tests_probed).

    Mirrors verify/compiler.py's normalization exactly (right-strip each line,
    drop trailing blanks) so a test kept here behaves identically when the real
    harness grades a model's attempt later. Any divergence would mean training
    against a subtly different verifier than the one that selected the tests.
    """
    pairs = [[p.stdin, p.stdout] for p in suite.io_pairs[:max_tests]]
    code = f"""\
import io as _io, sys as _sys, contextlib as _ctx, json as _json
_pairs = _json.loads({json.dumps(pairs)!r})
_code = compile({CANDIDATE_SRC_VAR}, "<candidate>", "exec")

def _norm(_s):
    _lines = [_ln.rstrip() for _ln in _s.splitlines()]
    while _lines and _lines[-1] == "":
        _lines.pop()
    return "\\n".join(_lines)

_ok = []
for _idx, (_stdin_text, _expected) in enumerate(_pairs):
    _old_stdin = _sys.stdin
    _sys.stdin = _io.StringIO(_stdin_text)
    _buf = _io.StringIO()
    try:
        with _ctx.redirect_stdout(_buf):
            exec(_code, {{"__name__": "__main__"}})
        _got = _norm(_buf.getvalue())
        if _got == _norm(_expected):
            _ok.append(_idx)
    except BaseException:
        pass
    finally:
        _sys.stdin = _old_stdin
print("{_VERDICT_MARKER}" + _json.dumps(_ok))
"""
    return code, len(pairs)


def build_probe(suite: TestSuite, max_tests: int = DEFAULT_MAX_TESTS) -> tuple[str, int]:
    """Compile a suite into a probe that reports WHICH tests pass, not just if."""
    if suite.kind == "assert":
        return _assert_probe(suite, max_tests)
    if suite.kind == "stdin_stdout":
        return _stdio_probe(suite, max_tests)
    raise NotImplementedError(f"no probe for suite kind {suite.kind!r}")


def parse_verdict(stdout: str) -> list[int] | None:
    """Extract the passing-test indices the probe printed, or None if absent.

    Absent means the probe never reached its final print — a crash, a timeout,
    or a candidate that killed the interpreter. Treated as "nothing verified"
    rather than silently as zero passes, so the two are distinguishable.
    """
    idx = stdout.rfind(_VERDICT_MARKER)
    if idx < 0:
        return None
    payload = stdout[idx + len(_VERDICT_MARKER):].splitlines()[0].strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


# ── Verifying one task ───────────────────────────────────────────────────────


def verify_one(
    task: CodeTask, sandbox: Any, max_tests: int = DEFAULT_MAX_TESTS
) -> VerificationOutcome:
    """Probe one task's official solution against its own tests.

    Every failure path returns an outcome rather than raising: one malformed row
    must not kill a multi-hour job over 26k problems.
    """
    suite = task.test_suites.get("hidden")
    if suite is None or suite.is_empty():
        return VerificationOutcome(task.task_id, "unknown", "no_tests")
    if not task.reference_solution:
        return VerificationOutcome(task.task_id, suite.kind, "no_solution")

    try:
        probe, n_probed = build_probe(suite, max_tests)
        if n_probed == 0:
            return VerificationOutcome(task.task_id, suite.kind, "no_tests")
        result = sandbox.run(task.reference_solution, probe, mode=suite.kind)
    except Exception as exc:  # noqa: BLE001 — deliberately broad, see docstring
        return VerificationOutcome(
            task.task_id, suite.kind, "harness_error", detail=f"{type(exc).__name__}: {exc}"
        )

    passed = parse_verdict(result.stdout or "")
    if passed is None:
        # No verdict line: the probe died (timeout / OOM / hard exit). Recorded
        # as verified-with-zero-passes plus a detail, so Stage 4 drops the task
        # and the log can still distinguish this from genuine test failures.
        return VerificationOutcome(
            task.task_id,
            suite.kind,
            "verified",
            n_tests_probed=n_probed,
            n_tests_passed=0,
            execution_time_s=getattr(result, "execution_time_s", 0.0) or 0.0,
            exit_code=getattr(result, "exit_code", None),
            detail="no verdict emitted (crash/timeout)",
        )

    valid = [i for i in passed if isinstance(i, int) and 0 <= i < n_probed]
    return VerificationOutcome(
        task.task_id,
        suite.kind,
        "verified",
        n_tests_probed=n_probed,
        n_tests_passed=len(valid),
        passed_indices=valid,
        execution_time_s=getattr(result, "execution_time_s", 0.0) or 0.0,
        exit_code=getattr(result, "exit_code", None),
    )


# ── Ledger I/O ───────────────────────────────────────────────────────────────


def read_done_ids(path: Path) -> set[str]:
    """task_ids already in the ledger, tolerating a truncated final line.

    A partial trailing line is the expected shape of a SIGKILL mid-write; that
    task simply gets re-verified. Any OTHER malformed line would be corruption
    and is not silently ignored — it raises.
    """
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["task_id"])
        except (json.JSONDecodeError, KeyError):
            if i == len(lines) - 1:
                break  # truncated tail from an interrupted run
            raise
    return done


def verify_tasks(
    tasks: Iterable[CodeTask],
    *,
    out_path: Path,
    sandbox_config: SandboxConfig,
    workers: int = 16,
    max_tests: int = DEFAULT_MAX_TESTS,
    resume: bool = True,
    progress_every: int = 200,
    stats: VerificationStats | None = None,
) -> VerificationStats:
    """Verify many tasks concurrently, appending verdicts as they complete."""
    from rlforge.sandbox import make_sandbox

    stats = stats if stats is not None else VerificationStats()
    done = read_done_ids(out_path) if resume else set()
    sandbox = make_sandbox(sandbox_config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("a", encoding="utf-8")
    write_lock = threading.Lock()

    def pending() -> Iterator[CodeTask]:
        """Feed the pool lazily so tasks are not all held in memory at once."""
        for task in tasks:
            stats.tasks_seen += 1
            if task.task_id in done:
                stats.tasks_skipped_resume += 1
                continue
            yield task

    def record(task: CodeTask, outcome: VerificationOutcome) -> None:
        with write_lock:
            handle.write(outcome.to_json() + "\n")
            handle.flush()  # per-line durability is what makes resume reliable
            stats.by_kind[outcome.kind] = stats.by_kind.get(outcome.kind, 0) + 1
            if outcome.status == "verified":
                stats.verified += 1
                stats.tests_probed += outcome.n_tests_probed
                stats.tests_passed += outcome.n_tests_passed
                if outcome.n_tests_passed == outcome.n_tests_probed:
                    stats.fully_passing += 1
                elif outcome.n_tests_passed > 0:
                    stats.partially_passing += 1
                else:
                    stats.zero_passing += 1
                platform = str(task.metadata.get("platform") or "unknown")
                stats.by_platform_total[platform] = (
                    stats.by_platform_total.get(platform, 0) + outcome.n_tests_probed
                )
                stats.by_platform_passed[platform] = (
                    stats.by_platform_passed.get(platform, 0) + outcome.n_tests_passed
                )
            else:
                setattr(stats, outcome.status, getattr(stats, outcome.status, 0) + 1)
            n = stats.verified + stats.no_solution + stats.no_tests + stats.harness_error
            if progress_every and n % progress_every == 0:
                print(f"  [verify] {n} done | {stats.summary()}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # pool.map consumes the generator lazily with a bounded look-ahead,
            # which is what keeps memory flat across ~26k tasks.
            queued = list(pending())
            for task, outcome in zip(
                queued, pool.map(lambda t: verify_one(t, sandbox, max_tests), queued)
            ):
                record(task, outcome)
    finally:
        handle.close()

    return stats
