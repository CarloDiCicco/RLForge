"""
train/grpo_config.py
────────────────────
`GRPOExperimentConfig` — one YAML file describing a full RL run.

Same rationale as `config/schema.py` for the evaluation side: every knob that
changes what an experiment measures must live in a versioned file, not in
arguments typed into a script. For RL this matters more than for evaluation,
because the run is long and the hyperparameters are the finding — "GRPO improved
pass@1" is uninterpretable without the exact sampling temperature, group size and
reward weights that produced it.

Deliberately mirrors the existing config module's shape (nested pydantic blocks,
the same JSON/YAML dispatch) and REUSES `SandboxConfig` rather than redefining
grading limits, so the reward provably runs under the same cage as the benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from rlforge.sandbox.base import SandboxConfig


class LoraConfigBlock(BaseModel):
    """LoRA adapter geometry.

    LoRA rather than full fine-tuning: on a single GB10 it keeps the trainable
    parameter count at ~1% (42.5M of 4.21B, measured), which leaves the memory
    budget for rollout generation — the throughput bottleneck — instead of
    optimizer state. It is also the standard choice for single-GPU RL studies, so
    it needs no special defense in a paper.
    """

    r: int = Field(default=32, ge=1)
    lora_alpha: int = Field(default=64, ge=1)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    # All seven projections (attention + MLP). Restricting to attention-only is
    # common but measurably weaker for code tasks; the extra cost here is
    # negligible against the generation time.
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


class RewardConfigBlock(BaseModel):
    """Weights for the reward terms. See train/reward.py for the semantics."""

    correctness_weight: float = Field(default=1.0, ge=0.0)
    # 0.0 for run #1 — correctness only. Implemented but disabled so that
    # enabling it in run #2 changes exactly one number, making the two runs a
    # clean one-variable ablation rather than a code change mid-study.
    length_weight: float = Field(default=0.0, ge=0.0)
    # Grading is subprocess-bound (~1.7s per completion on real LeetCode suites),
    # so threads overlap it with nothing lost; 8 matches num_generations.
    max_workers: int = Field(default=8, ge=1)


class GRPOBlock(BaseModel):
    """GRPO / TRL training knobs.

    Defaults are the run-#1 decisions, each chosen for a reason recorded here so a
    later reader does not have to reverse-engineer intent from the numbers.
    """

    # ── sampling during rollouts ──────────────────────────────────────────────
    # TRL's defaults, NOT the benchmark's (0.7/0.8/20). GRPO's advantage is the
    # spread of rewards *within* a group: if all rollouts for a prompt are the
    # same, every advantage is zero and the step teaches nothing. Training and
    # evaluation sampling are different jobs — evaluation stays at the frozen
    # baseline values so the resulting number remains comparable.
    temperature: float = Field(default=1.0, gt=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    # Matches the baselines, so a rollout is the same length budget as a
    # benchmark generation and truncation rates are comparable across both.
    max_completion_length: int = Field(default=2048, ge=1)

    # ── group structure ───────────────────────────────────────────────────────
    # 8, not 4: at 4 rollouts with a ~0.40 pass rate, ~13% of prompts produce an
    # all-fail (or all-pass) group → zero advantage → a wasted step. At 8 that is
    # ~1.7%, and measured throughput is best at batch 8-16 anyway, so the extra
    # rollouts are nearly free.
    num_generations: int = Field(default=8, ge=2)

    # ── objective ─────────────────────────────────────────────────────────────
    # 0.0 = no KL penalty and no reference model loaded. TRL's own default and
    # current practice (DAPO); it also removes a hyperparameter from the first
    # experiment. Risk accepted: no anchor against policy drift, which the
    # HumanEval+ transfer eval is there to detect. If drift shows up, 0.001 is
    # DeepSeek-R1's value.
    beta: float = Field(default=0.0, ge=0.0)
    loss_type: str = Field(default="dapo")
    scale_rewards: str = Field(default="group")
    # A truncated rollout is an UNFINISHED answer, not a wrong one. Training on it
    # as a negative teaches the model that its (possibly correct) approach was
    # bad. Critical here: 21.6% of LeetCode baseline generations truncate.
    mask_truncated_completions: bool = Field(default=True)

    # ── optimization ──────────────────────────────────────────────────────────
    learning_rate: float = Field(default=1e-6, gt=0.0)
    # MEASURED CONSTRAINT (wiring check, 2026-09-12): this is the knob that OOMs,
    # and it must stay small. Qwen3.5-4B has a 248,320-token vocabulary, so the
    # logits tensor the GRPO loss materializes is ~1 GB per sequence at 2048
    # tokens in bf16 — and the backward pass needs several copies of it. At
    # per_device_train_batch_size=8 the run died having claimed 117.5 GB of the
    # GB10's 121.7 GB. 2 keeps the loss-side footprint to a few GB.
    #
    # This does NOT reduce the number of rollouts per prompt: num_generations is
    # unchanged and generation_batch_size below keeps rollout generation batched
    # for throughput. Only the backward pass is split into micro-batches.
    #
    # It is also the ONLY knob that affects the memory spike, which is why
    # tasks_per_update below can be raised freely without risking another OOM.
    per_device_train_batch_size: int = Field(default=2, ge=1)
    # How many DISTINCT TASKS contribute to one weight update.
    #
    # This is the knob that matters for training quality, and the first
    # configuration got it wrong: at 1 task per update, every gradient step is
    # dominated by whichever single task was drawn, so the update direction is
    # extremely noisy — the RL analogue of pretraining on one document at a time.
    # Advantages are still computed WITHIN each task's own group of
    # `num_generations` rollouts (that is GRPO's mechanism and must stay per-task);
    # this only controls how many such groups are averaged before stepping.
    #
    # 8 is standard practice and reduces gradient variance ~8x. It costs TIME, not
    # memory: total runtime tracks total generations, so 8 tasks x 25 steps and
    # 1 task x 200 steps are the same compute — but the former makes 25 informed
    # updates instead of 200 noisy ones.
    tasks_per_update: int = Field(default=8, ge=1)
    # Derived, not configured: gradient_accumulation_steps is whatever it takes to
    # push tasks_per_update * num_generations rollouts through a
    # per_device_train_batch_size-sized micro-batch. Set in the validator below so
    # the three cannot silently disagree.
    gradient_accumulation_steps: int = Field(default=0, ge=0)
    # Rollout generation batch, set INDEPENDENTLY of the training micro-batch.
    # Generation runs under no_grad and materializes one token's logits at a time,
    # so it is far cheaper than the loss: measured throughput rises from 16 tok/s
    # at batch 1 to 145 tok/s at batch 16. Left at None it would inherit the small
    # training batch and throw that speedup away.
    # 8, not 16: lowered after the 64-rollout OOM. Generation is not where that
    # OOM happened (the traceback was in `backward`), but 16 concurrent 2048-token
    # sequences hold a KV cache that is still resident when the loss runs, so it
    # raises the floor the backward pass has to fit above. 8 keeps most of the
    # batching speedup (108 vs 145 tok/s measured) for noticeably less peak.
    generation_batch_size: int | None = Field(default=8, ge=1)
    # Hard ceiling on the fraction of VRAM this process may allocate.
    #
    # This exists because of a HOST-LEVEL failure, not a training one: the GB10 has
    # unified CPU/GPU memory, so a run that claims ~105-117 of 121.7 GB starves the
    # operating system and locks the whole machine — SSH and the desktop included,
    # which is what happened on 2026-09-13. A cap turns that into a plain
    # OutOfMemoryError that kills the job and leaves the box usable.
    #
    # 0.75 leaves ~30 GB for the OS, the bwrap sandboxes grading rollouts (up to
    # 8192 MB each by sandbox_config) and everything else on the host.
    max_memory_fraction: float = Field(default=0.75, gt=0.0, le=1.0)
    # 25, not 200: with 8 tasks per update, 25 steps already sees 200 tasks and
    # 1600 rollouts — the same total compute as the original 1-task x 200-step
    # plan, because runtime tracks total generations rather than step count.
    max_steps: int = Field(default=25, ge=1)
    # ON. Turned off once, on the strength of a 22.9 GB peak — but that peak was
    # measured when an update held only 8 rollouts. At 64 rollouts (8 tasks x 8)
    # the backward pass over 32 layers of stored activations OOMed at ~105 GB, with
    # the traceback landing squarely in `backward`. Recomputing activations costs
    # ~30% step time and changes no result; an OOM ten hours into the full run
    # costs the run. Keep it on.
    gradient_checkpointing: bool = Field(default=True)

    # ── logging / checkpoints ─────────────────────────────────────────────────
    logging_steps: int = Field(default=1, ge=1)
    save_steps: int = Field(default=50, ge=1)
    # tensorboard, not wandb: runs happen on the DGX over SSH, and tensorboard
    # writes plain event files into the artifact dir — no account, no network
    # egress, and it survives a dropped connection.
    report_to: str = Field(default="tensorboard")
    log_completions: bool = Field(default=True)
    num_completions_to_print: int = Field(default=2, ge=0)

    @field_validator("loss_type")
    @classmethod
    def _known_loss(cls, value: str) -> str:
        allowed = {"dapo", "grpo", "bnpo", "dr_grpo"}
        if value not in allowed:
            raise ValueError(f"loss_type must be one of {sorted(allowed)}, got {value!r}")
        return value


class GRPOExperimentConfig(BaseModel):
    """Top-level contract for one RL run."""

    experiment_id: str
    seed: int = Field(default=0, ge=0)
    model_id: str
    # Pinned for the same reason as the eval configs: the baselines recorded a
    # null revision, so "same base model" was an assumption. An RL delta measured
    # against a baseline on a silently different snapshot would be meaningless.
    model_revision: str | None = Field(default=None)

    tasks_file: str
    # Path to the committed split (data/splits/*.json). Required, not optional:
    # training on all 444 and evaluating on a subset of them would make the
    # headline number train-set performance.
    train_split: str
    eval_split: str | None = Field(default=None)
    # Cap for smoke runs. None = use the whole train split.
    max_train_tasks: int | None = Field(default=None, ge=1)

    lora: LoraConfigBlock = Field(default_factory=LoraConfigBlock)
    grpo: GRPOBlock = Field(default_factory=GRPOBlock)
    reward: RewardConfigBlock = Field(default_factory=RewardConfigBlock)
    # Same cage AND same limits as the benchmark runs (bwrap, 60s/60s, 8192 MB).
    # Diverging here would silently decouple the reward from the pass@1 it is
    # meant to improve — mem_mb especially, where the 512 MB default failed tasks
    # during test compilation (RESEARCH_LOG 2026-07-23).
    sandbox_config: SandboxConfig = Field(
        default_factory=lambda: SandboxConfig(
            backend="bwrap", timeout_s=60.0, cpu_s=60, mem_mb=8192
        )
    )
    # Run outputs go under runs/train/<experiment_id> in the repo (gitignored).
    # Relative paths resolve against the CWD, so launch from the repo root.
    out_dir: str

    @field_validator("experiment_id", "model_id", "tasks_file", "train_split", "out_dir")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value

    @model_validator(mode="after")
    def _derive_accumulation(self) -> "GRPOExperimentConfig":
        """Derive gradient_accumulation_steps from tasks_per_update.

        The three batch knobs are related by one identity:

            tasks_per_update * num_generations
                = per_device_train_batch_size * gradient_accumulation_steps

        Rather than ask a config author to keep that consistent by hand (where a
        mistake silently changes how many tasks inform each update — a training
        quality change that no error would reveal), the accumulation steps are
        COMPUTED here and any hand-set value is checked against the identity.
        """
        g = self.grpo
        rollouts_per_update = g.tasks_per_update * g.num_generations

        if rollouts_per_update % g.per_device_train_batch_size != 0:
            raise ValueError(
                f"tasks_per_update * num_generations ({rollouts_per_update}) must be "
                f"divisible by per_device_train_batch_size "
                f"({g.per_device_train_batch_size})"
            )
        required = rollouts_per_update // g.per_device_train_batch_size

        if g.gradient_accumulation_steps == 0:
            # Sentinel default: derive it.
            g.gradient_accumulation_steps = required
        elif g.gradient_accumulation_steps != required:
            raise ValueError(
                f"gradient_accumulation_steps={g.gradient_accumulation_steps} "
                f"contradicts tasks_per_update={g.tasks_per_update} "
                f"(needs {required} = {g.tasks_per_update} tasks * "
                f"{g.num_generations} generations / "
                f"{g.per_device_train_batch_size} micro-batch). Set one or the "
                "other, not both."
            )

        # Generation batch must hold whole rollout groups, or a group would be
        # split across batches and its advantages computed over a partial group.
        gbs = g.generation_batch_size
        if gbs is not None and gbs % g.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({gbs}) must be divisible by "
                f"num_generations ({g.num_generations})"
            )
        return self


def _load_raw_config(config_path: Path) -> dict:
    """Read JSON/YAML as a raw dict — same dispatch as config/schema.py."""
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML configs.") from exc
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raise ValueError(f"Unsupported config extension: {suffix}. Use .json, .yaml or .yml.")


def load_grpo_config(config_path: str | Path) -> GRPOExperimentConfig:
    """Load and validate a GRPO run config from disk."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        return GRPOExperimentConfig.model_validate(_load_raw_config(path))
    except ValidationError as exc:
        raise ValueError(f"Invalid GRPO config at {path}: {exc}") from exc
