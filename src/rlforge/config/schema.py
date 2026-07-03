"""
config/schema.py
─────────────────
ExperimentConfig — one YAML/JSON file describing a full experiment end-to-end:
which model, which data, which decoding params, plus the existing pipeline-
level knobs (PLAN §2.4).

Before this, every knob (model_id, max_new_tokens, which task file, sandbox
backend, n_samples) only existed as Python arguments typed into a one-off
script — no way to change a setting without editing code, and no single
artifact recording "what was this experiment's overall configuration"
(individual records already carry per-record provenance — git_commit,
sandbox_config_id — but not the experiment's intent as a whole).

Adapted from ReasoningLab `config/schema.py`'s ExperimentConfig / GenerationConfig
/ load_experiment_config — same nested-block Pydantic structure and JSON/YAML
dispatch (a pattern already reused once in adapters/task_loader.py). Genuinely
new here: ExperimentConfig wraps RLForge's EXISTING pipeline.RunConfig as a
nested field rather than redefining sandbox/n_samples/ks/grading_role — those
knobs are already correctly modeled, so this config only adds what's missing
(model + generation + data selection), matching the pipeline's real shape
instead of ReasoningLab's separate RuntimeConfig/PathsConfig.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from rlforge.pipeline import RunConfig

# The dataset adapters with a built-in loader (adapters.humaneval, currently
# the only one). Extend this Literal as more built-in adapters land.
BuiltinDataset = Literal["humaneval_plus"]


class GenerationConfig(BaseModel):
    """Decoding options passed to QwenModel.generate().

    Defaults are Qwen3.5's own model-card-recommended settings for non-
    thinking-mode generation — not a generic placeholder — so a config that
    omits this block entirely still gets a sensible, intentional setup.
    """

    max_new_tokens: int = Field(default=1024, ge=1)
    do_sample: bool = Field(default=True)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    top_k: int = Field(default=20, ge=0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=1.0)
    presence_penalty: float = Field(default=1.5, ge=0.0, le=2.0)
    # Explicit, not inferred: thinking vs non-thinking are different model
    # behaviors with their own model-card sampling rows (this config's other
    # defaults are the non-thinking, general-tasks row). A config author must
    # opt into thinking mode deliberately, not have it silently mismatch the
    # rest of the sampling params above.
    enable_thinking: bool = Field(default=False)


class ExperimentConfig(BaseModel):
    """Top-level experiment contract loaded from a config file."""

    experiment_id: str
    seed: int = Field(default=0, ge=0)
    model_id: str

    # Exactly one of these selects the task source — enforced below.
    tasks_file: str | None = Field(default=None)
    dataset: BuiltinDataset | None = Field(default=None)
    # Optional cap forwarded to whichever loader is selected (both
    # adapters.task_loader.load_tasks and adapters.humaneval.load_humaneval_plus
    # already accept `limit` with this exact meaning).
    max_tasks: int | None = Field(default=None, ge=1)

    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    # The EXISTING pipeline.RunConfig, reused as-is (out_dir, sandbox_config,
    # n_samples, ks, grading_role) — not redefined here.
    run: RunConfig

    @field_validator("experiment_id", "model_id")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value

    @model_validator(mode="after")
    def _exactly_one_task_source(self) -> "ExperimentConfig":
        if (self.tasks_file is None) == (self.dataset is None):
            raise ValueError(
                "exactly one of `tasks_file` or `dataset` must be set "
                f"(got tasks_file={self.tasks_file!r}, dataset={self.dataset!r})"
            )
        return self


def _load_raw_config(config_path: Path) -> dict:
    """Read JSON/YAML config as a raw dictionary before validation."""
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to read YAML configs. Install `pyyaml`."
            ) from exc
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return raw or {}

    raise ValueError(
        f"Unsupported config extension: {suffix}. Use .json, .yaml or .yml."
    )


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Load config from disk and return a validated ExperimentConfig."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _load_raw_config(path)
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid experiment config at {path}: {exc}") from exc
