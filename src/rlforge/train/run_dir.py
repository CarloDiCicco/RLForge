"""
train/run_dir.py
────────────────
Run folder + provenance for TRAINING runs, shared by every algorithm script
(train_grpo.py today; a future train_dpo.py etc. calls the same functions).

Why this exists. Before it, a train run wrote into a FIXED folder named after the
config (`runs/train/grpo-lcb-leetcode-run12`), so:
  - relaunching the same config mixed files into the previous run's folder, and
    old runs had to be renamed by hand (`-KILLED-partial-128`, `-OLD-1task`);
  - nothing in the folder recorded WHAT produced it — no config copy, no git
    commit, no start/end time — so if the YAML changed later, the run's settings
    were unrecoverable. The eval pipeline already recorded this; training did not.

Layout produced (one folder per launch, never reused):

    <out_dir>/<UTC timestamp>-<experiment_id>/
        run_config.yaml       the config file EXACTLY as launched (byte copy)
        config_resolved.json  the parsed config incl. defaults and derived values
        run_meta.json         git commit/dirty, command, versions, hardware,
                              start/end time, final status
        ... everything the trainer writes (checkpoints, logs, adapter)
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rlforge.registry.runinfo import git_commit, git_dirty, hardware_info, utc_now_iso

# Packages whose version changes what training computes. Recorded so a result can
# be tied to the exact library behaviour (e.g. TRL changing a loss default).
_TRACKED_PACKAGES = ("torch", "transformers", "trl", "peft", "accelerate", "datasets")

META_FILE = "run_meta.json"


def create_run_dir(out_dir: str | Path, experiment_id: str) -> Path:
    """Create and return a fresh folder `<out_dir>/<UTC timestamp>-<experiment_id>`.

    Timestamp to the second, in UTC so folders sort chronologically regardless of
    the machine's timezone. `exist_ok=False` is deliberate: two launches in the
    same second would otherwise share (and corrupt) one folder, so that case fails
    loudly instead of silently mixing runs.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out_dir) / f"{stamp}-{experiment_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _package_versions() -> dict[str, str]:
    """Installed versions of the tracked packages; "not installed" if absent."""
    versions = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def write_run_provenance(
    run_dir: Path,
    config_path: str | Path,
    config: BaseModel,
    *,
    algorithm: str,
) -> None:
    """Record what produced this run, BEFORE training starts.

    Written first so that even a run that crashes in step 1 is fully described.
    Two config files on purpose: the raw YAML keeps the author's comments and exact
    text; the resolved JSON shows what the code actually used after defaults and
    derived values (e.g. gradient_accumulation_steps) were applied.
    """
    shutil.copyfile(config_path, run_dir / "run_config.yaml")
    (run_dir / "config_resolved.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2) + "\n"
    )
    meta: dict[str, Any] = {
        "experiment_id": getattr(config, "experiment_id", None),
        "algorithm": algorithm,
        "status": "running",
        "started_at": utc_now_iso(),
        "ended_at": None,
        "config_path": str(config_path),
        "command": sys.argv,
        "git_commit": git_commit(),
        # A dirty tree means the commit alone does NOT reproduce the code that ran.
        "git_dirty": git_dirty(),
        "model_id": getattr(config, "model_id", None),
        "model_revision": getattr(config, "model_revision", None),
        "packages": _package_versions(),
        "hardware": hardware_info(),
    }
    _write_meta(run_dir, meta)


def finish_run(run_dir: Path, status: str, error: str | None = None, **extra: Any) -> None:
    """Stamp the end time and final status into run_meta.json.

    status: "completed", "failed" (exception) or "interrupted" (Ctrl-C). A run
    killed by a signal Python cannot catch (SIGKILL, host crash) never gets here
    and stays "running" with ended_at null. That is still informative, because a
    folder that says "running" while no process exists means it died hard.
    """
    path = run_dir / META_FILE
    meta = json.loads(path.read_text())
    meta.update(status=status, ended_at=utc_now_iso(), **extra)
    if error is not None:
        meta["error"] = error
    _write_meta(run_dir, meta)


def _write_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    (run_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
