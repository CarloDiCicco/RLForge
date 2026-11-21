"""
snapshots.py
────────────
Save the LoRA adapter every few optimizer steps, beside the run's checkpoints.

Why: run #3 has no validation probe during training (a probe costs ~2 h each at
32k). Instead the adapter is saved often and the snapshots are scored on the
validation split AFTER the run, which gives the learning curve and decides the
run length for the later runs. Snapshots are also the input for weight analysis.

A snapshot is only the adapter (tens of MB), not a checkpoint: no optimizer
state, so a run cannot be resumed from it. Resuming stays the job of the regular
`save_steps` checkpoints.

Layout: <run_dir>/snapshots/step-0002/{adapter files, snapshot_meta.json}
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import TrainerCallback


def snapshot_interval(every_steps: int, save_steps: int) -> int:
    """Steps between snapshots: the requested interval, but never sparser than
    the checkpoints (a snapshot interval above save_steps would add nothing the
    checkpoints do not already have, and would leave long gaps in the curve)."""
    if every_steps < 1 or save_steps < 1:
        raise ValueError(f"intervals must be >= 1, got every_steps={every_steps}, save_steps={save_steps}")
    return min(every_steps, save_steps)


class LoraSnapshotCallback(TrainerCallback):
    """Writes the adapter to <run_dir>/snapshots/step-XXXX every `interval` steps
    and at the last step (so the final model is always part of the curve)."""

    def __init__(self, run_dir: Path, interval: int) -> None:
        self.root = Path(run_dir) / "snapshots"
        self.interval = interval

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step % self.interval != 0 and step != args.max_steps:
            return control
        out = self.root / f"step-{step:04d}"
        # On a PEFT model save_pretrained writes the adapter only, which is
        # exactly what the eval script loads with --adapter.
        model.save_pretrained(str(out))
        # Only the step: HF calls on_step_end BEFORE it logs that step's metrics,
        # so the step's reward is not in state.log_history yet. The metrics live
        # in TensorBoard, keyed by the same step.
        meta = {"step": step, "max_steps": args.max_steps}
        (out / "snapshot_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"[snapshot] step {step}: adapter saved to {out}")
        return control
