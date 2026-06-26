"""Sandbox backends and the factory that selects one from config."""

from __future__ import annotations

from rlforge.sandbox.base import Sandbox, SandboxConfig
from rlforge.sandbox.bwrap_sandbox import BwrapSandbox, bwrap_available
from rlforge.sandbox.subprocess_sandbox import SubprocessSandbox


def make_sandbox(config: SandboxConfig) -> Sandbox:
    """Construct the sandbox backend named by `config.backend`.

    This is the only place the pipeline chooses a backend; adding a future
    `ContainerSandbox` is a new branch here plus the class, nothing else.
    """
    if config.backend == "subprocess":
        return SubprocessSandbox(config)
    if config.backend == "bwrap":
        return BwrapSandbox(config)
    raise ValueError(f"unknown or unimplemented sandbox backend: {config.backend!r}")


__all__ = [
    "Sandbox",
    "SandboxConfig",
    "SubprocessSandbox",
    "BwrapSandbox",
    "bwrap_available",
    "make_sandbox",
]
