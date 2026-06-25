"""
schema/records.py
──────────────────
GenerationRecord — one (task, sample) generation event (PLAN §1.5).
TrainingRun      — one training run's configuration + outputs (PLAN §1.5).

These are the provenance backbone of the run registry: every generated sample
and every training run is recoverable, so later analysis can answer questions
like "did this improvement come from temperature?" or "did teacher data help?".

TrainingRun is defined now for schema stability but is not exercised until
Milestone 3; its fields mirror PLAN §1.5 so M3 does not require a schema change.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rlforge import FORMAT_VERSION


class GenerationRecord(BaseModel):
    """One candidate generation, with full decoding + provenance metadata."""

    format_version: str = Field(default=FORMAT_VERSION)

    # Identity + linkage. run_id groups all samples of one generation run;
    # sample_id is unique per candidate and links to its ExecutionRecord.
    run_id: str
    sample_id: str
    task_id: str
    dataset_source: str
    split: str

    # Model provenance — enough to reproduce which model produced the sample.
    model_name: str
    model_size: str | None = None
    model_revision: str | None = None
    adapter_name: str | None = None

    # Prompt + decoding parameters (the knobs analysis attributes effects to).
    prompt_template_id: str
    decoding_temperature: float
    top_p: float
    top_k: int | None = None
    max_new_tokens: int
    seed: int

    # Token accounting.
    input_tokens: int
    output_tokens: int

    # The generation itself and what extraction made of it.
    raw_output: str
    extracted_code: str
    extraction_status: str
    # Links to the ExecutionRecord produced from extracted_code.
    execution_result_id: str

    # Environment provenance.
    timestamp: str
    git_commit: str
    hardware_info: dict[str, Any] = Field(default_factory=dict)


class TrainingRun(BaseModel):
    """One training run's configuration and outputs (PLAN §1.5; used in M3)."""

    format_version: str = Field(default=FORMAT_VERSION)

    train_run_id: str
    base_model: str
    method: str  # SFT, DPO, GRPO/RLVR
    data_mixture_id: str
    dataset_sources: list[str] = Field(default_factory=list)
    num_train_samples: int
    num_eval_samples: int

    lora_config: dict[str, Any] = Field(default_factory=dict)
    quantization_config: dict[str, Any] = Field(default_factory=dict)
    optimizer: str
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    seed: int

    checkpoints: list[str] = Field(default_factory=list)
    metrics_path: str
    git_commit: str
