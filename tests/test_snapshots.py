"""Snapshot interval rule and the callback's save schedule (train/snapshots.py)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rlforge.train.snapshots import LoraSnapshotCallback, snapshot_interval


def test_interval_rule():
    assert snapshot_interval(2, 10) == 2
    # never sparser than the checkpoints
    assert snapshot_interval(25, 10) == 10
    with pytest.raises(ValueError):
        snapshot_interval(0, 10)


class _FakeModel:
    def save_pretrained(self, path):
        from pathlib import Path
        Path(path).mkdir(parents=True)
        (Path(path) / "adapter_model.safetensors").write_text("x")


def test_callback_saves_every_interval_and_last_step(tmp_path):
    cb = LoraSnapshotCallback(tmp_path, interval=2)
    args = SimpleNamespace(max_steps=7)
    for step in range(1, 8):
        cb.on_step_end(args, SimpleNamespace(global_step=step), None, model=_FakeModel())
    saved = sorted(p.name for p in (tmp_path / "snapshots").iterdir())
    # 2, 4, 6 by interval, and 7 because it is the last step
    assert saved == ["step-0002", "step-0004", "step-0006", "step-0007"]
    meta = json.loads((tmp_path / "snapshots" / "step-0004" / "snapshot_meta.json").read_text())
    assert meta == {"step": 4, "max_steps": 7}
