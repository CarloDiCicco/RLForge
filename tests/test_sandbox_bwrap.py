"""Tests for the bubblewrap sandbox, layer C (PLAN §1.3 isolation probes).

Skipped automatically if bubblewrap is not installed, so the suite stays green
on hosts without it (the real target, the DGX, has bwrap).
"""

from __future__ import annotations

import pytest

from rlforge.classify.taxonomy import FailureType
from rlforge.sandbox import make_sandbox
from rlforge.sandbox.base import Sandbox, SandboxConfig
from rlforge.sandbox.bwrap_sandbox import BwrapSandbox, bwrap_usable

pytestmark = pytest.mark.skipif(
    not bwrap_usable(),
    reason="bwrap cannot create a namespace here (e.g. AppArmor userns restriction)",
)

ADD = "def add(a, b):\n    return a + b"


def _bwrap(**cfg) -> BwrapSandbox:
    return BwrapSandbox(SandboxConfig(backend="bwrap", **cfg))


def test_satisfies_sandbox_protocol():
    assert isinstance(_bwrap(), Sandbox)


def test_factory_builds_bwrap_backend():
    sb = make_sandbox(SandboxConfig(backend="bwrap"))
    assert isinstance(sb, BwrapSandbox)


def test_normal_solution_still_passes_inside_cage():
    # The cage must not break legitimate code: stdlib import + arithmetic.
    r = _bwrap().run(ADD, "import math\nassert add(2, 3) == 5")
    assert r.passed
    assert r.failure_type == FailureType.PASS


def test_network_is_off():
    # --unshare-net: no interface exists, so a connect attempt fails. The code
    # raises (caught as a runtime failure), proving no outbound connection.
    code = (
        "def add(a, b):\n"
        "    import socket\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "    return a + b\n"
    )
    r = _bwrap().run(code, "assert add(1, 1) == 2")
    assert not r.passed
    assert r.failure_type in (FailureType.RUNTIME, FailureType.TIMEOUT)


def test_home_and_ssh_keys_not_visible():
    # The home dir is not bind-mounted, so it does not exist inside the cage.
    code = (
        "import os\n"
        "def add(a, b):\n"
        "    assert not os.path.exists(os.path.expanduser('~/.ssh')), 'home leaked!'\n"
        "    return a + b\n"
    )
    r = _bwrap().run(code, "assert add(1, 1) == 2")
    assert r.passed  # the in-code assertion held → home was absent


def test_repo_files_not_visible():
    # The repo is not bind-mounted either.
    repo = "/home/carlo/repos/RLForge/pyproject.toml"
    code = (
        "import os\n"
        "def add(a, b):\n"
        f"    assert not os.path.exists({repo!r}), 'repo leaked!'\n"
        "    return a + b\n"
    )
    r = _bwrap().run(code, "assert add(1, 1) == 2")
    assert r.passed
