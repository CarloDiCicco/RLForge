"""
sandbox/subprocess_sandbox.py
─────────────────────────────
SubprocessSandbox — layers A + B of the cage (PLAN §1.3).

  A. Subprocess isolation + sentinel pass-detection.
       The candidate runs in a separate `python` process in a throwaway cwd.
       passed=True requires exit code 0 AND a sentinel token in stdout, so a
       candidate cannot fake a pass by exiting early (e.g. raise SystemExit(0))
       before the tests run.
  B. OS-enforced resource limits via resource.setrlimit in a preexec_fn.
       Applied in the child after fork, before exec, so the limits are in force
       for the whole interpreter lifetime: address space (RLIMIT_AS), CPU time
       (RLIMIT_CPU), file size (RLIMIT_FSIZE), and process count (RLIMIT_NPROC).
       A memory bomb is killed by the kernel, not left to OOM the host.

This is the portable base backend. BwrapSandbox (layer C) reuses this script
builder + classification and adds kernel-level network/filesystem isolation.

Not a defense against a determined adversary (ctypes can issue raw syscalls);
it is a robust cage against the realistic threat — buggy/hallucinated code from
a code model run thousands of times. Layer C tightens it further.
"""

from __future__ import annotations

import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rlforge.classify.taxonomy import classify_failure
from rlforge.sandbox.base import SandboxConfig
from rlforge.sandbox.preamble import LIVECODEBENCH_IMPORTS, SAFETY_PREAMBLE
from rlforge.schema.execution import ExecutionResult

# Unique token printed only after the test code completes. Required (with exit
# code 0) before we report passed=True — proves execution reached the end.
_PASS_SENTINEL = "__RLFORGE_TESTS_COMPLETE_9e4d1c7a__"

_MB = 1024 * 1024


def build_script(candidate_code: str, test_code: str) -> str:
    """Assemble preamble + candidate + test + sentinel into one runnable script.

    The sentinel print is the last statement, reached only if candidate and test
    code both ran without raising or exiting early.
    """
    return (
        LIVECODEBENCH_IMPORTS
        + "\n"
        + SAFETY_PREAMBLE
        + "\n\n"
        + candidate_code.rstrip()
        + "\n\n"
        + test_code.lstrip()
        + f'\nprint("{_PASS_SENTINEL}")\n'
    )


def _make_limit_preexec(config: SandboxConfig, *, apply_nproc: bool = True):
    """Build a preexec_fn that applies setrlimit in the forked child.

    Runs after fork() and before exec(), so the limits bind the launched process
    and (since rlimits inherit across fork/exec) any descendants. Hard == soft,
    so they cannot be raised. Kept tiny and import-free: preexec_fn runs in a
    fragile post-fork context where almost nothing is safe to call.

    apply_nproc: when False, RLIMIT_NPROC is NOT set. The bwrap backend needs
    this off — bwrap is the process we launch and it must fork once to start the
    interpreter, so capping its process count at 0 would make bwrap itself fail
    ("Creating new namespace failed: Resource temporarily unavailable"). The
    other limits (AS/CPU/FSIZE) still inherit to the caged interpreter, and
    bwrap's PID namespace already constrains process spawning inside the cage.
    """
    mem_bytes = config.mem_mb * _MB
    cpu_s = config.cpu_s
    fsize_bytes = config.fsize_mb * _MB
    nproc = config.nproc

    def _set_limits() -> None:
        # Address space: the hard memory cap. Allocation past this fails with
        # MemoryError (or the kernel kills the process), never the host OOM.
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        # CPU seconds: backstops the wall-clock timeout against busy loops.
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        # Max single-file size: stops a runaway write filling the disk.
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
        # Max processes: 0 blocks fork/spawn. Skipped for bwrap (see docstring).
        if apply_nproc:
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))

    return _set_limits


class SubprocessSandbox:
    """Runs candidate code in a resource-limited subprocess (layers A+B)."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig(backend="subprocess")

    def _command(self, script_path: Path) -> list[str]:
        """The argv to execute the script. Overridden by BwrapSandbox to wrap
        this command inside `bwrap`."""
        return [sys.executable, str(script_path)]

    def _preexec(self):
        """The preexec_fn applying resource limits. Overridden by BwrapSandbox
        to skip the NPROC cap (bwrap must fork to launch the interpreter)."""
        return _make_limit_preexec(self.config, apply_nproc=True)

    def run(self, candidate_code: str, test_code: str) -> ExecutionResult:
        script = build_script(candidate_code, test_code)
        tmp_path: Path | None = None
        try:
            # Write the script to a temp file (not -c): no argv length limit, and
            # tracebacks carry real line numbers for classification.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(script)
                tmp_path = Path(tmp.name)

            with tempfile.TemporaryDirectory() as work_dir:
                start = time.perf_counter()
                try:
                    proc = subprocess.run(
                        self._command(tmp_path),
                        capture_output=True,
                        text=True,
                        timeout=self.config.timeout_s,
                        cwd=work_dir,              # isolate file I/O to a temp dir
                        preexec_fn=self._preexec(),  # layer B (resource limits)
                    )
                    elapsed = time.perf_counter() - start
                    return self._result_from_proc(proc, elapsed)
                except subprocess.TimeoutExpired as exc:
                    elapsed = time.perf_counter() - start
                    return self._timeout_result(exc, elapsed)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()

    # ── result construction ────────────────────────────────────────────────

    def _result_from_proc(
        self, proc: subprocess.CompletedProcess[str], elapsed: float
    ) -> ExecutionResult:
        sentinel_found = _PASS_SENTINEL in proc.stdout
        passed = proc.returncode == 0 and sentinel_found
        # Strip the sentinel so downstream sees only real model output.
        clean_stdout = proc.stdout.replace(_PASS_SENTINEL, "").rstrip("\n")

        failure_type = classify_failure(
            passed=passed,
            extracted=True,        # code reached the sandbox; extraction succeeded
            timed_out=False,
            stderr=proc.stderr,
        )
        return ExecutionResult(
            passed=passed,
            failure_type=failure_type,
            stdout=clean_stdout,
            stderr=proc.stderr,
            traceback=_last_traceback(proc.stderr),
            execution_time_s=elapsed,
            memory_peak_mb=_child_peak_mem_mb(),
            exit_code=proc.returncode,
        )

    def _timeout_result(
        self, exc: subprocess.TimeoutExpired, elapsed: float
    ) -> ExecutionResult:
        # stdout/stderr may be None if nothing flushed before the kill.
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return ExecutionResult(
            passed=False,
            failure_type=classify_failure(
                passed=False, extracted=True, timed_out=True, stderr=stderr
            ),
            stdout=stdout.replace(_PASS_SENTINEL, "").rstrip("\n"),
            stderr=stderr,
            traceback=None,
            execution_time_s=elapsed,
            memory_peak_mb=_child_peak_mem_mb(),
            exit_code=None,            # killed before it produced an exit code
        )


def _child_peak_mem_mb() -> float | None:
    """Physical RAM peak used by a terminated children since process start, in MB.

    Uses getrusage(RUSAGE_CHILDREN). ru_maxrss is kilobytes on Linux. This is
    cumulative across all children of this process, so it is an upper bound when
    many executions share one parent — adequate for analysis, not exact
    per-execution accounting. Returns None if unavailable.
    """
    try:
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        return ru.ru_maxrss / 1024.0
    except (ValueError, OSError):
        return None


def _last_traceback(stderr: str) -> str | None:
    """Extract the final traceback block from stderr, if present."""
    marker = "Traceback (most recent call last):"
    idx = stderr.rfind(marker)
    if idx == -1:
        return None
    return stderr[idx:].strip() or None
