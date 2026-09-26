"""Tests for train/run_dir.py — per-launch run folders and their provenance.

These guard reproducibility, not behaviour: a train run whose settings can't be
recovered from its own folder is not a citable result.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import BaseModel

from rlforge.train.run_dir import create_run_dir, finish_run, write_run_provenance


class _Cfg(BaseModel):
    experiment_id: str = "exp-x"
    model_id: str = "Qwen/Qwen3.5-4B"
    model_revision: str = "abc123"
    derived_value: int = 32  # stands in for e.g. gradient_accumulation_steps


def test_run_dir_is_timestamped_and_named(tmp_path):
    run_dir = create_run_dir(tmp_path, "exp-x")
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path
    # UTC timestamp to the second, then the experiment id.
    assert re.fullmatch(r"\d{8}T\d{6}Z-exp-x", run_dir.name), run_dir.name


def test_existing_run_dir_is_never_reused(tmp_path, monkeypatch):
    """A same-second relaunch must fail loudly, not mix files into the old run."""
    import rlforge.train.run_dir as mod

    class _FrozenClock:
        @staticmethod
        def now(tz=None):
            from datetime import datetime, timezone
            return datetime(2026, 9, 19, 16, 48, 7, tzinfo=timezone.utc)

    monkeypatch.setattr(mod, "datetime", _FrozenClock)
    create_run_dir(tmp_path, "exp-x")
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "exp-x")


def test_provenance_records_config_and_meta(tmp_path):
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("# a comment the author wrote\nexperiment_id: exp-x\n")
    run_dir = create_run_dir(tmp_path / "runs", "exp-x")

    write_run_provenance(run_dir, config_file, _Cfg(), algorithm="grpo")

    # Raw YAML is a byte copy, so comments survive.
    assert (run_dir / "run_config.yaml").read_text() == config_file.read_text()
    # Resolved config includes values the YAML never stated.
    resolved = json.loads((run_dir / "config_resolved.json").read_text())
    assert resolved["derived_value"] == 32

    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["status"] == "running"
    assert meta["ended_at"] is None
    assert meta["algorithm"] == "grpo"
    assert meta["experiment_id"] == "exp-x"
    assert meta["model_revision"] == "abc123"
    for key in ("started_at", "git_commit", "git_dirty", "command", "packages", "hardware"):
        assert key in meta, key
    assert "trl" in meta["packages"] and "torch" in meta["packages"]


@pytest.mark.parametrize("status,error", [
    ("completed", None),
    ("failed", "OutOfMemoryError: CUDA out of memory"),
    ("interrupted", None),
])
def test_finish_run_stamps_status(tmp_path, status, error):
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("experiment_id: exp-x\n")
    run_dir = create_run_dir(tmp_path / "runs", "exp-x")
    write_run_provenance(run_dir, config_file, _Cfg(), algorithm="grpo")

    finish_run(run_dir, status, error=error)

    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["status"] == status
    assert meta["ended_at"] is not None
    assert meta.get("error") == error
    # Start-time fields must survive the update.
    assert meta["algorithm"] == "grpo"


def test_shipped_train_configs_use_parent_out_dir():
    """out_dir is the PARENT; a per-run name there would nest runs inside it."""
    from rlforge.train.grpo_config import load_grpo_config

    configs = Path(__file__).resolve().parents[1] / "configs"
    for path in sorted(configs.glob("grpo-*.yaml")):
        assert load_grpo_config(path).out_dir == "runs/train", path.name
