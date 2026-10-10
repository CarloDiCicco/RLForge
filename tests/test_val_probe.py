"""Tests for the validation probe, the three-way split, and the serious-run config.

The probe produces the learning curve a paper would show, and the split decides
what "held-out" means. Both fail silently if wrong, so their invariants are pinned.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from rlforge.train.grpo_config import ValidationBlock, load_grpo_config
from rlforge.train.val_probe import ValidationProbe, summarize

_REPO = Path(__file__).resolve().parents[1]
_CONFIGS = _REPO / "configs"
_SPLITS = _REPO / "data" / "splits"

_PASS = "```python\nclass Solution:\n    def f(self):\n        return 1\n```"


def _load_train_script():
    spec = importlib.util.spec_from_file_location("train_grpo", _REPO / "scripts" / "train_grpo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeGrader:
    """Grades 'GOOD' as a pass. Stands in for VerifierReward (no sandbox needed)."""

    max_workers = 2

    def grade(self, task_id: str, completion: str) -> dict:
        ok = completion == "GOOD"
        return {"task_id": task_id, "correctness": 1.0 if ok else 0.0,
                "failure_type": "pass" if ok else "assertion",
                "extraction_status": "fenced_python", "exec_time_s": 0.1,
                "code_chars": 10, "stderr": "", "completion_chars": len(completion)}


class _FakeModel:
    def __init__(self) -> None:
        self.training = True
        self.modes: list[bool] = []

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self


def _probe(tmp_path, generate_fn, every=5, k=2) -> ValidationProbe:
    cfg = ValidationBlock(split="unused.json", every_steps=every, samples_per_task=k)
    return ValidationProbe(
        prompts={"T/a": "prompt a", "T/b": "prompt b"}, grader=_FakeGrader(),
        tokenizer=None, cfg=cfg, enable_thinking=False, max_new_tokens=16,
        out_dir=tmp_path, generate_fn=generate_fn,
    )


# ── the probe ─────────────────────────────────────────────────────────────────


def test_probe_scores_every_task_k_times_and_writes_records(tmp_path):
    seen: list[str] = []

    def gen(model, prompts):
        seen.extend(prompts)
        # Task a always passes, task b never does.
        return [("GOOD" if p == "prompt a" else "BAD", 7, False) for p in prompts]

    metrics = _probe(tmp_path, gen, k=2).run(0, _FakeModel())

    assert sorted(seen) == ["prompt a", "prompt a", "prompt b", "prompt b"]
    assert metrics["pass_rate"] == 0.5
    assert metrics["tasks_solved_any"] == 0.5 and metrics["tasks_solved_all"] == 0.5
    records = [json.loads(l) for l in (tmp_path / "val_executions.jsonl").open()]
    assert len(records) == 4
    assert {(r["task_id"], r["sample"]) for r in records} == {
        ("T/a", 0), ("T/a", 1), ("T/b", 0), ("T/b", 1)}
    assert all(r["step"] == 0 for r in records)
    summary = [json.loads(l) for l in (tmp_path / "val_summary.jsonl").open()]
    assert summary[0]["step"] == 0 and summary[0]["pass_rate"] == 0.5


def test_probe_restores_train_mode_and_rng(tmp_path):
    """The probe must not change what the training loop does next."""

    def gen(model, prompts):
        assert model.training is False, "generation must run in eval mode"
        torch.rand(100)  # consume randomness, as real sampling does
        return [("BAD", 3, True) for _ in prompts]

    model = _FakeModel()
    torch.manual_seed(7)
    expected = torch.rand(3)

    torch.manual_seed(7)
    _probe(tmp_path, gen).run(0, model)
    assert torch.equal(torch.rand(3), expected), "probe leaked into the training RNG"
    assert model.training is True


def test_probe_schedule_and_no_double_run(tmp_path):
    steps: list[int] = []
    probe = _probe(tmp_path, lambda m, p: [("BAD", 1, False) for _ in p], every=5)
    original = probe.run
    probe.run = lambda step, model: (steps.append(step), original(step, model))[1]

    model = _FakeModel()
    probe.on_train_begin(None, SimpleNamespace(global_step=0, max_steps=12), None, model=model)
    for step in range(1, 13):
        probe.on_step_end(None, SimpleNamespace(global_step=step, max_steps=12), None, model=model)

    assert steps == [0, 5, 10, 12], "step 0, every 5, and the final step"
    # A final step that is also a multiple must be probed once, not twice.
    probe.on_step_end(None, SimpleNamespace(global_step=12, max_steps=12), None, model=model)
    summary = [json.loads(l) for l in (tmp_path / "val_summary.jsonl").open()]
    assert [s["step"] for s in summary] == [0, 5, 10, 12]


def test_summarize_counts_truncation_and_extraction():
    records = [
        {"task_id": "a", "correctness": 1.0, "hit_token_cap": False,
         "failure_type": "pass", "completion_tokens": 100},
        {"task_id": "a", "correctness": 0.0, "hit_token_cap": True,
         "failure_type": "extraction", "completion_tokens": 2048},
    ]
    m = summarize(records, samples_per_task=2)
    assert m["pass_rate"] == 0.5 and m["truncated_rate"] == 0.5
    assert m["extraction_fail_rate"] == 0.5 and m["mean_completion_tokens"] == 1074
    assert m["tasks_solved_any"] == 1.0 and m["tasks_solved_all"] == 0.0


# ── the three-way split ───────────────────────────────────────────────────────


def test_v2_split_is_disjoint_complete_and_keeps_the_test_set():
    train = json.loads((_SPLITS / "lcb_leetcode_v2_train.json").read_text())
    val = json.loads((_SPLITS / "lcb_leetcode_v2_val.json").read_text())
    test = json.loads((_SPLITS / "lcb_leetcode_v2_test.json").read_text())
    assert (len(train), len(val), len(test)) == (300, 44, 100)
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
    assert len(set(train) | set(val) | set(test)) == 444
    # The test set is the original 100 held-out tasks: results stay comparable, and
    # nothing that was ever trained on (old train = v2 train + v2 val) is in it.
    assert test == json.loads((_SPLITS / "lcb_leetcode_eval.json").read_text())
    old_train = json.loads((_SPLITS / "lcb_leetcode_train.json").read_text())
    assert sorted(train + val) == old_train


def test_v2_val_difficulty_tracks_train():
    stats = json.loads((_SPLITS / "lcb_leetcode_v2_split_stats.json").read_text())
    n_train, n_val = stats["n_train"], stats["n_val"]
    for difficulty, count in stats["difficulty_train"].items():
        share_train = count / n_train
        share_val = stats["difficulty_val"].get(difficulty, 0) / n_val
        assert abs(share_train - share_val) < 0.03, difficulty


# ── the serious-run config ────────────────────────────────────────────────────


def test_serious_config_matches_validated_run_except_scale():
    """Only scale, the warmup and the probe may differ from the validated 12-step run."""
    small = load_grpo_config(_CONFIGS / "grpo-lcb-leetcode-12step-lr2e-5.yaml")
    big = load_grpo_config(_CONFIGS / "grpo-leetcode-3ep.yaml")
    # Warmup was added for the long run (decided 2026-10-10); the peak rate is the same.
    scale = {"max_steps", "save_steps", "lr_scheduler_type", "warmup_steps"}
    assert (big.grpo.lr_scheduler_type, big.grpo.warmup_steps) == ("constant_with_warmup", 5)
    assert big.grpo.learning_rate == small.grpo.learning_rate == 2e-5
    assert small.grpo.model_dump(exclude=scale) == big.grpo.model_dump(exclude=scale)
    for block in ("reward", "sandbox_config"):
        assert getattr(small, block) == getattr(big, block), block
    # Same adapter shape; the long run additionally adapts the linear-attention
    # projections, so that every block's attention has an adapter (2026-10-10).
    assert small.lora.model_dump(exclude={"target_modules"}) == \
        big.lora.model_dump(exclude={"target_modules"})
    assert set(big.lora.target_modules) - set(small.lora.target_modules) == {
        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}
    assert set(small.lora.target_modules) <= set(big.lora.target_modules)
    assert (small.model_revision, small.enable_thinking, small.seed) == \
        (big.model_revision, big.enable_thinking, big.seed)
    assert big.max_train_tasks is None and big.train_split.endswith("v2_train.json")


def test_serious_config_schedule_is_three_passes_with_aligned_probes():
    cfg = load_grpo_config(_CONFIGS / "grpo-leetcode-3ep.yaml")
    n_train = len(json.loads((_REPO / cfg.train_split).read_text()))
    micro = n_train * cfg.grpo.num_generations // cfg.grpo.per_device_train_batch_size
    acc = cfg.grpo.gradient_accumulation_steps
    steps_per_pass = micro // acc + (micro % acc > 0)  # HF Trainer's rule
    assert steps_per_pass == 38
    assert cfg.grpo.max_steps == 3 * steps_per_pass
    # A checkpoint exists for every probed step, and probes land on pass ends.
    assert cfg.grpo.save_steps == cfg.validation.every_steps
    assert steps_per_pass % cfg.validation.every_steps == 0
    # Probe sampling is the frozen baseline's, not the rollout's.
    v = cfg.validation
    assert (v.temperature, v.top_p, v.top_k, v.presence_penalty) == (0.7, 0.8, 20, 1.5)


def test_wiring_config_differs_from_serious_only_in_scale():
    big = load_grpo_config(_CONFIGS / "grpo-leetcode-3ep.yaml")
    wiring = load_grpo_config(_CONFIGS / "grpo-leetcode-3ep-wiring.yaml")
    scale = {"max_steps", "save_steps"}
    assert big.grpo.model_dump(exclude=scale) == wiring.grpo.model_dump(exclude=scale)
    probe_scale = {"every_steps", "samples_per_task"}
    assert big.validation.model_dump(exclude=probe_scale) == \
        wiring.validation.model_dump(exclude=probe_scale)
    assert (big.train_split, big.lora, big.reward, big.sandbox_config) == \
        (wiring.train_split, wiring.lora, wiring.reward, wiring.sandbox_config)


def test_probe_refuses_validation_tasks_that_are_trained_on(tmp_path):
    module = _load_train_script()
    cfg = load_grpo_config(_CONFIGS / "grpo-leetcode-3ep.yaml")
    leaky = cfg.model_copy(update={
        "validation": cfg.validation.model_copy(update={"split": cfg.train_split})})
    try:
        module.build_validation_probe(leaky, {}, None, tmp_path)
    except ValueError as exc:
        assert "also in the train split" in str(exc)
    else:
        raise AssertionError("a probe on trained-on tasks must be refused")


def test_tensorboard_dir_and_short_symlink(tmp_path, monkeypatch):
    module = _load_train_script()
    monkeypatch.delenv("TENSORBOARD_LOGGING_DIR", raising=False)
    run_dir = tmp_path / "runs" / "train" / "20261010T000000Z-exp"
    run_dir.mkdir(parents=True)

    tb_dir = module.setup_tensorboard_dir(run_dir, "exp")

    assert tb_dir == run_dir / "tb" and tb_dir.is_dir()
    assert os.environ["TENSORBOARD_LOGGING_DIR"] == str(tb_dir)
    links = list((tmp_path / "runs" / "tb").iterdir())
    assert len(links) == 1 and links[0].name.endswith("-exp")
    assert links[0].is_symlink() and links[0].resolve() == tb_dir.resolve()
