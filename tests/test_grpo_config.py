"""Tests for train/grpo_config.py and the RL split, plus the config fields the
RL evaluation depends on.

These guard the settings that silently change what an experiment MEASURES — the
class of error that produces a confident, wrong paper number rather than a crash.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlforge.config.schema import load_experiment_config
from rlforge.train.grpo_config import GRPOExperimentConfig, load_grpo_config

_REPO = Path(__file__).resolve().parents[1]
_CONFIGS = _REPO / "configs"
_SPLITS = _REPO / "data" / "splits"


def _minimal(**overrides) -> dict:
    base = {
        "experiment_id": "t",
        "model_id": "Qwen/Qwen3.5-4B",
        "tasks_file": "data/x.jsonl",
        "train_split": "data/splits/train.json",
        "out_dir": "/tmp/out",
    }
    base.update(overrides)
    return base


# ── defaults encode the run-#1 decisions ──────────────────────────────────────


def test_defaults_match_run1_decisions():
    cfg = GRPOExperimentConfig.model_validate(_minimal())
    # Rollout sampling must be the diversity-preserving values, not the
    # benchmark's 0.7/0.8/20 — at low temperature advantages collapse to zero.
    assert (cfg.grpo.temperature, cfg.grpo.top_p, cfg.grpo.top_k) == (1.0, 1.0, 0)
    assert cfg.grpo.num_generations == 8
    assert cfg.grpo.beta == 0.0, "no KL term in run #1"
    assert cfg.grpo.mask_truncated_completions is True
    assert cfg.grpo.max_completion_length == 2048, "must match the baselines"
    assert cfg.reward.length_weight == 0.0, "run #1 is correctness-only"
    assert cfg.reward.correctness_weight == 1.0


def test_memory_safety_defaults():
    """Pins the settings that exist because of two real failures.

    Both were found the expensive way: a 64-rollout update OOMed in `backward`
    with checkpointing off, and the resulting ~105 GB claim on the GB10's UNIFIED
    CPU/GPU memory pool locked the entire host (SSH included) rather than just
    failing the job. Regressing either of these silently reintroduces a
    machine-level outage, so they are asserted rather than left to a comment.
    """
    cfg = GRPOExperimentConfig.model_validate(_minimal())
    assert cfg.grpo.gradient_checkpointing is True, (
        "off + 64 rollouts/update OOMed in backward at ~105 GB"
    )
    assert cfg.grpo.max_memory_fraction < 1.0, (
        "an uncapped run can starve the OS on unified memory and lock the host"
    )
    assert cfg.grpo.max_memory_fraction <= 0.8, "leave headroom for OS + bwrap sandboxes"


@pytest.mark.parametrize("name", ["wiring", "smoke", "full"])
def test_shipped_configs_carry_the_memory_guards(name):
    """The guards must be in the files that actually run, not only in defaults."""
    cfg = load_grpo_config(_CONFIGS / f"grpo-lcb-leetcode-{name}.yaml")
    assert cfg.grpo.gradient_checkpointing is True, name
    assert cfg.grpo.max_memory_fraction == 0.75, name
    assert cfg.grpo.generation_batch_size == 8, name


def test_default_sandbox_matches_the_baseline_cage():
    """The reward must grade under the same limits as the benchmark.

    If these drift, reward == 1.0 stops meaning "the benchmark would score this a
    pass", and the RL delta is no longer measured against the baseline. mem_mb in
    particular: 512 MB silently failed tasks during test compilation
    (RESEARCH_LOG 2026-07-23).
    """
    cfg = GRPOExperimentConfig.model_validate(_minimal())
    sc = cfg.sandbox_config
    assert (sc.backend, sc.timeout_s, sc.cpu_s, sc.mem_mb) == ("bwrap", 60.0, 60, 8192)
    # Same content-addressed id as the three baseline runs.
    assert sc.sandbox_config_id == "bwrap-c02cbdf4"


# ── validation catches misconfiguration before the model loads ─────────────────


def test_micro_batch_must_divide_the_rollouts_per_update():
    """Failing here beats failing minutes into a run, after the model has loaded."""
    with pytest.raises(ValueError, match="divisible by per_device_train_batch_size"):
        GRPOExperimentConfig.model_validate(
            _minimal(grpo={"num_generations": 8, "per_device_train_batch_size": 6})
        )


def test_accumulation_is_derived_from_tasks_per_update():
    """The batch identity must hold automatically, not by hand.

    tasks_per_update * num_generations == per_device_train_batch_size * accum.
    Getting this wrong silently changes how many tasks inform each update — a
    training-quality change that produces no error, which is exactly how the first
    configuration ended up training on ONE task per update.
    """
    cfg = GRPOExperimentConfig.model_validate(
        _minimal(grpo={"tasks_per_update": 8, "num_generations": 8,
                       "per_device_train_batch_size": 2})
    )
    assert cfg.grpo.gradient_accumulation_steps == 32  # 8*8/2
    assert (cfg.grpo.per_device_train_batch_size
            * cfg.grpo.gradient_accumulation_steps
            == cfg.grpo.tasks_per_update * cfg.grpo.num_generations)


def test_contradictory_accumulation_is_rejected():
    """Setting both must fail loudly rather than one silently winning."""
    with pytest.raises(ValueError, match="contradicts tasks_per_update"):
        GRPOExperimentConfig.model_validate(
            _minimal(grpo={"tasks_per_update": 8, "num_generations": 8,
                           "per_device_train_batch_size": 2,
                           "gradient_accumulation_steps": 4})
        )


def test_multiple_tasks_inform_each_update_by_default():
    """Guards the fix for the original 1-task-per-update mistake."""
    cfg = GRPOExperimentConfig.model_validate(_minimal())
    assert cfg.grpo.tasks_per_update > 1, (
        "one task per update makes every gradient step hostage to a single task"
    )


def test_unknown_loss_type_is_rejected():
    with pytest.raises(ValueError, match="loss_type"):
        GRPOExperimentConfig.model_validate(_minimal(grpo={"loss_type": "ppo"}))


def test_train_split_is_required():
    with pytest.raises(ValueError):
        GRPOExperimentConfig.model_validate(
            {k: v for k, v in _minimal().items() if k != "train_split"}
        )


# ── the shipped configs ───────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["grpo-lcb-leetcode-smoke", "grpo-lcb-leetcode-full"])
def test_shipped_grpo_configs_load(name):
    cfg = load_grpo_config(_CONFIGS / f"{name}.yaml")
    assert cfg.model_revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert cfg.sandbox_config.sandbox_config_id == "bwrap-c02cbdf4"
    assert cfg.reward.length_weight == 0.0


def test_smoke_and_full_differ_only_in_scale():
    """Every knob that affects what is measured must be identical.

    The smoke run is a rehearsal; if it differs in sampling or reward, clearing
    its gates tells you nothing about the real run.
    """
    smoke = load_grpo_config(_CONFIGS / "grpo-lcb-leetcode-smoke.yaml")
    full = load_grpo_config(_CONFIGS / "grpo-lcb-leetcode-full.yaml")
    assert smoke.grpo.model_dump(exclude={"max_steps", "save_steps"}) == \
        full.grpo.model_dump(exclude={"max_steps", "save_steps"})
    assert smoke.reward == full.reward
    assert smoke.lora == full.lora
    assert smoke.sandbox_config == full.sandbox_config
    # Scale is the only intended difference.
    assert smoke.grpo.max_steps < full.grpo.max_steps
    assert smoke.max_train_tasks is not None and full.max_train_tasks is None


def test_eval_configs_use_frozen_baseline_sampling():
    """Evaluation must NOT inherit the training temperature.

    Rollouts run at temperature 1.0 for advantage diversity; evaluating there
    would make the number incomparable to the three baselines.
    """
    for name in ["qwen3.5-4b-lcb-leetcode-eval100-rl", "qwen3.5-4b-lcb-leetcode-eval100-base"]:
        cfg = load_experiment_config(_CONFIGS / f"{name}.yaml")
        g = cfg.generation
        assert (g.temperature, g.top_p, g.top_k) == (0.7, 0.8, 20), name
        assert g.max_new_tokens == 2048, name
        assert g.presence_penalty == 1.5, name
        assert g.enable_thinking is False, name
        assert cfg.run.grading_role == "hidden", name
        assert cfg.task_ids_file is not None, f"{name} must restrict to held-out ids"


def test_eval_pair_differs_only_by_the_adapter():
    """The RL and base eval configs must be otherwise identical.

    Any other difference would be an alternative explanation for the delta.
    """
    rl = load_experiment_config(_CONFIGS / "qwen3.5-4b-lcb-leetcode-eval100-rl.yaml")
    base = load_experiment_config(_CONFIGS / "qwen3.5-4b-lcb-leetcode-eval100-base.yaml")
    assert rl.generation == base.generation
    # out_dir differs by design (runs/eval/<name>), so compare everything that
    # affects the MEASUREMENT and exclude the destination it is written to.
    assert rl.run.model_dump(exclude={"out_dir"}) == base.run.model_dump(exclude={"out_dir"})
    assert rl.run.sandbox_config == base.run.sandbox_config
    assert rl.task_ids_file == base.task_ids_file
    assert rl.model_revision == base.model_revision
    assert rl.seed == base.seed
    assert rl.adapter_path is not None and base.adapter_path is None


def test_all_baseline_configs_pin_the_model_revision():
    """A baseline compared against an RL run must name its exact snapshot."""
    for name in [
        "qwen3.5-4b-humaneval-full",
        "qwen3.5-4b-livecodebench-full",
        "qwen3.5-4b-livecodebench-stdio-full",
    ]:
        cfg = load_experiment_config(_CONFIGS / f"{name}.yaml")
        assert cfg.model_revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", name


# ── the split itself ──────────────────────────────────────────────────────────


def test_split_is_disjoint_and_complete():
    """The committed split defines the experiment; these are its invariants."""
    train = json.loads((_SPLITS / "lcb_leetcode_train.json").read_text())
    ev = json.loads((_SPLITS / "lcb_leetcode_eval.json").read_text())
    assert len(ev) == 100
    assert len(train) == 344
    assert not set(train) & set(ev), "a trained-on task in the eval set invalidates the result"
    assert len(set(train)) == len(train) and len(set(ev)) == len(ev)
    assert len(set(train) | set(ev)) == 444


def test_split_difficulty_is_stratified():
    """Eval difficulty mix must track the population within rounding.

    Difficulty dominates pass@1 on this benchmark, so an unrepresentative eval set
    would shift the number for reasons unrelated to the policy.
    """
    stats = json.loads((_SPLITS / "lcb_leetcode_split_stats.json").read_text())
    pop, ev = stats["difficulty_population"], stats["difficulty_eval"]
    for difficulty, count in pop.items():
        expected = 100 * count / stats["n_tasks"]
        assert abs(ev.get(difficulty, 0) - expected) <= 1.0, difficulty
