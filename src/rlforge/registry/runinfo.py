"""
registry/runinfo.py
───────────────────
Provenance + identity for the run registry (PLAN §1.5).

The schemas say *what* a record holds; this module mints the bookkeeping every
record needs so a result is reproducible and joinable: the run id, the per-sample
and per-execution ids, a timestamp, the git commit, and the hardware it ran on.

Id scheme — deterministic sha256 (same approach as `SandboxConfig.
sandbox_config_id`): an id is a hash of the tuple that uniquely identifies the
thing. Determinism gives reproducible joins (re-deriving an id finds its record)
and turns a duplicated input into a visible collision rather than a silently
duplicated record. Uniqueness rests on the *inputs* being distinct, which the
pipeline guarantees by visiting each (task, sample) and each (sample, suite) once
(and the loader rejecting duplicate task_ids); sha256 itself never collides on
distinct inputs.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Current time as an ISO-8601 string in UTC (the records' `timestamp`)."""
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    """A fresh, filesystem-safe run id: UTC timestamp + short random suffix.

    Used as the run folder name (runs/<run_id>/). The timestamp makes runs sort
    chronologically; the uuid suffix disambiguates two runs started in the same
    second. This one is intentionally random — a run is a fresh event, not a
    content-addressed derivation.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _digest(*parts: str, prefix: str) -> str:
    """sha256 of the parts joined by '|', truncated, with a readable prefix.

    Mirrors SandboxConfig.sandbox_config_id so all content-addressed ids in the
    codebase look and behave the same. 16 hex chars (64 bits) is ample to avoid
    accidental collisions across a research run's record count.
    """
    payload = "|".join(parts)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{prefix}-{digest}"


def new_sample_id(run_id: str, task_id: str, sample_idx: int) -> str:
    """Deterministic id for one generated candidate (the FK executions point to).

    Unique by construction: within a run_id the pipeline emits each
    (task_id, sample_idx) pair exactly once.
    """
    return _digest(run_id, task_id, str(sample_idx), prefix="smp")


def new_execution_id(sample_id: str, test_suite_id: str) -> str:
    """Deterministic id for running one sample against one test suite.

    Unique by construction: a sample is run against each suite at most once.
    """
    return _digest(sample_id, test_suite_id, prefix="exe")


def git_commit() -> str:
    """Current git HEAD short of `git rev-parse HEAD`, or "unknown".

    Never raises: a research log must still write outside a git checkout, with no
    git installed, or in CI without history. Any failure degrades to "unknown".
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(__file__),
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def hardware_info() -> dict[str, Any]:
    """Best-effort snapshot of the machine that produced the records.

    Kept dependency-light: platform/python/cpu always; the CUDA device name only
    if torch is importable (RLForge has no torch dep in M1, so this stays
    optional and must never fail the run if torch is absent).
    """
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "executable": sys.executable,
    }
    try:
        import torch  # type: ignore

        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        # torch absent or CUDA query failed — provenance is still valid without it.
        pass
    return info
