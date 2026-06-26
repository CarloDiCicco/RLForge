"""
sandbox/bwrap_sandbox.py
────────────────────────
BwrapSandbox — layer C of the cage (PLAN §1.3).

Wraps the SubprocessSandbox command (layers A+B) inside `bubblewrap` (`bwrap`),
so the candidate runs in a kernel-enforced room:

  - No network: `--unshare-net` gives the process a fresh network namespace with
    no interfaces, so outbound connections cannot be made at all (stronger than
    the preamble nulling socket.socket).
  - No host filesystem: only a minimal read-only view of the interpreter and the
    one script file is mounted; the home dir, SSH keys, tokens, and the repo are
    simply not present in the namespace. `open("~/.ssh/id_rsa")` fails because
    the path does not exist inside the cage.
  - Writable scratch only: a private tmpfs at /tmp and the cwd; everything the
    candidate writes is discarded when the namespace tears down.
  - Reduced privilege: `--unshare-all`, `--die-with-parent`, and a new PID/IPC/
    UTS namespace, so the candidate cannot see or signal host processes.

Reuse, not rewrite: only `_command()` is overridden. build_script (preamble +
sentinel), setrlimit (preexec_fn still applies — limits compose with bwrap),
failure classification, and result construction are all inherited from
SubprocessSandbox unchanged.
"""

from __future__ import annotations

import shutil
import sys
import sysconfig
from pathlib import Path

from rlforge.sandbox.base import SandboxConfig
from rlforge.sandbox.subprocess_sandbox import (
    SubprocessSandbox,
    _make_limit_preexec,
)

# Resolved once: the bwrap binary. None means bubblewrap is not installed.
_BWRAP = shutil.which("bwrap")


def bwrap_available() -> bool:
    """True iff the bubblewrap binary is present on this host."""
    return _BWRAP is not None


def _real_python() -> str:
    """The resolved interpreter binary (following venv symlinks).

    sys.executable in a venv is a symlink chain (.venv/bin/python3 → … →
    /usr/bin/python3.12). Inside the bwrap namespace that relative chain does not
    resolve, so we run the real target directly — it lives under the already
    bound base prefix, and site-packages are bound separately by
    _read_only_binds(), so imports still work.
    """
    return str(Path(sys.executable).resolve())


# Cached result of the runtime usability probe (None = not yet probed).
_USABLE: bool | None = None


def bwrap_usable() -> bool:
    """True iff bwrap can actually create a sandbox namespace on this host.

    Having the binary is not enough: on hardened Ubuntu,
    `kernel.apparmor_restrict_unprivileged_userns=1` blocks a non-setuid bwrap
    from creating a user namespace ("setting up uid map: Permission denied"),
    so the cage fails at runtime. This runs a trivial `bwrap … true` once and
    caches whether it succeeds, so callers (and tests) can detect a usable cage
    rather than discovering it only when running real code.
    """
    global _USABLE
    if _USABLE is not None:
        return _USABLE
    if _BWRAP is None:
        _USABLE = False
        return False
    import subprocess

    # Probe with the SAME read-only binds the real sandbox uses, so the test
    # exercises namespace creation (the thing AppArmor can block) rather than
    # failing spuriously because the interpreter's lib dirs weren't mounted.
    cmd = (
        [_BWRAP, "--unshare-all", "--unshare-net", "--die-with-parent", "--tmpfs", "/tmp"]
        + _read_only_binds()
        + [_real_python(), "-c", "pass"]
    )
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=10)
        _USABLE = proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        _USABLE = False
    return _USABLE


def _read_only_binds() -> list[str]:
    """Bind-mount flags exposing only what the interpreter needs to start.

    We give the candidate a read-only view of the Python installation (stdlib,
    the executable's prefix) and nothing else from the host. site-packages under
    the same prefixes come along, which is intended: the candidate may import
    stdlib and installed libs, but cannot read arbitrary host files.
    """
    # Directories required for the interpreter to import its standard library.
    # Using the real prefixes keeps this correct under uv/venv layouts.
    candidates = {
        sys.base_prefix,                       # base interpreter install
        sys.prefix,                            # active venv (may equal base_prefix)
        sysconfig.get_path("stdlib"),          # standard library
        sysconfig.get_path("platstdlib"),      # platform-specific stdlib
        sysconfig.get_path("purelib"),         # site-packages (pure)
        sysconfig.get_path("platlib"),         # site-packages (platform)
        str(Path(sys.executable).resolve().parent),  # the python binary's dir
        # Top-level lib/bin dirs that are NOT symlinks into /usr on this distro
        # (here /lib and /lib64 are real dirs holding the dynamic linker
        # ld-linux-*.so — without them execvp of the interpreter fails with a
        # misleading "No such file or directory").
        "/lib", "/lib64", "/bin", "/sbin",
    }
    flags: list[str] = []
    for path in sorted(p for p in candidates if p and Path(p).exists()):
        # --ro-bind SRC DEST: mount host SRC read-only at the same path inside.
        flags += ["--ro-bind", path, path]
    return flags


class BwrapSandbox(SubprocessSandbox):
    """SubprocessSandbox (A+B) wrapped in a bubblewrap cage (C)."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        super().__init__(config or SandboxConfig(backend="bwrap"))
        if not bwrap_available():
            # Fail loudly at construction, not silently at run time: a missing
            # cage must never degrade to running untrusted code unsandboxed.
            raise RuntimeError(
                "bubblewrap (bwrap) not found on PATH; cannot build the bwrap "
                "sandbox. Install it or use backend='subprocess'."
            )

    def _preexec(self):
        """Apply resource limits but NOT the NPROC cap: the launched process is
        bwrap, which must fork to start the interpreter. AS/CPU/FSIZE still
        inherit into the caged interpreter; bwrap's PID namespace constrains
        spawning inside the cage."""
        return _make_limit_preexec(self.config, apply_nproc=False)

    def _command(self, script_path: Path) -> list[str]:
        """Wrap `python script.py` in a bwrap cage (overrides layer A+B argv).

        The script is mounted read-only at a fixed in-cage path (/script.py)
        rather than at its host path: the host path is under /tmp, which we
        shadow with a private tmpfs, so re-binding it there would be fragile.
        """
        in_cage_script = "/script.py"
        bwrap_flags = [
            _BWRAP,
            "--unshare-all",        # new user/net/pid/ipc/uts/cgroup namespaces
            "--unshare-net",        # explicit: no network interface at all
            "--die-with-parent",    # killed if the parent (our process) dies
            "--new-session",        # detach controlling terminal (no TIOCSTI)
            "--proc", "/proc",      # a private /proc the interpreter expects
            "--dev", "/dev",        # minimal device nodes (null, zero, urandom…)
            "--tmpfs", "/tmp",      # private, in-memory scratch; discarded on exit
            "--chdir", "/tmp",      # run with cwd inside the writable tmpfs
            "--ro-bind", str(script_path), in_cage_script,  # the script, read-only
        ]
        bwrap_flags += _read_only_binds()
        # Then the layer-A command, using the resolved real interpreter and the
        # in-cage script path.
        return bwrap_flags + [_real_python(), in_cage_script]
