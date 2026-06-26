"""
sandbox/base.py
───────────────
The Sandbox abstraction (PLAN §1.3).

Everything upstream (extraction, registry, metrics, pipeline) talks to the
`Sandbox` Protocol and consumes `ExecutionResult` — it never knows which backend
ran the code. That is the seam that keeps the project scalable: the M1 backends
are subprocess+rlimit (A+B) and bubblewrap (C); a future `ContainerSandbox`
(podman/docker) is a new class implementing this Protocol plus a config switch,
with zero changes to the pipeline. `SandboxConfig.sandbox_config_id` is what the
run registry stamps onto each ExecutionRecord so a result is always
interpretable against the cage that produced it.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from rlforge.schema.execution import ExecutionResult

# Backends selectable from config. "subprocess" = A+B (rlimit, no namespaces);
# "bwrap" = C (kernel-enforced cage). "container" is reserved, not implemented.
SandboxBackend = Literal["subprocess", "bwrap", "container"]


class SandboxConfig(BaseModel):
    """Resource + isolation limits for one sandbox, and its identity.

    The config is content-addressed: `sandbox_config_id` is derived from the
    limit fields, so two configs with identical limits share an id and results
    are comparable, while any change to a limit yields a new id (and old records
    stay correctly attributed to the cage they actually ran in).
    """

    backend: SandboxBackend = "subprocess"

    # Wall-clock kill (covers infinite loops and sleep hangs).
    timeout_s: float = Field(default=8.0, gt=0.0)
    # Hard address-space cap (RLIMIT_AS), OS-enforced — a memory bomb is killed
    # by the kernel, not left to take down the host via OOM.
    mem_mb: int = Field(default=512, gt=0)
    # CPU-time cap (RLIMIT_CPU), seconds. Backstops the wall-clock timeout
    # against busy loops; rounded up to a whole second by the kernel.
    cpu_s: int = Field(default=10, gt=0)
    # Max bytes any single file the code writes may reach (RLIMIT_FSIZE) — stops
    # a runaway write from filling the disk.
    fsize_mb: int = Field(default=64, gt=0)
    # Max child processes (RLIMIT_NPROC) — 0 means the code may not fork at all,
    # blocking fork bombs and subprocess spawning at the OS level.
    nproc: int = Field(default=0, ge=0)

    @property
    def sandbox_config_id(self) -> str:
        """Stable, content-derived id for this configuration.

        Hashes the limit fields (not transient state) so the id is reproducible
        across runs and processes. Prefixed with the backend for readability in
        logs, e.g. "subprocess-1a2b3c4d".
        """
        payload = (
            f"{self.backend}|t={self.timeout_s}|m={self.mem_mb}|c={self.cpu_s}"
            f"|f={self.fsize_mb}|n={self.nproc}"
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
        return f"{self.backend}-{digest}"


@runtime_checkable
class Sandbox(Protocol):
    """A cage that runs candidate code against a test suite, safely.

    Implementations execute `candidate_code` followed by `test_code` in an
    isolated environment and return a structured `ExecutionResult`. They must
    never let candidate code touch the host beyond the cage's declared limits.
    """

    config: SandboxConfig

    def run(self, candidate_code: str, test_code: str) -> ExecutionResult:
        """Execute candidate + test code; return the structured outcome.

        Must not raise on candidate misbehavior (crashes, timeouts, bad syntax):
        those are reported inside the ExecutionResult. May raise only on
        environment/infrastructure faults (e.g. the backend binary is missing).
        """
        ...
