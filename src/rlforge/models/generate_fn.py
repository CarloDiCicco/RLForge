"""
models/generate_fn.py
──────────────────────
make_qwen_generate_fn — turn a loaded QwenModel into the GenerateFn
pipeline.run_pipeline expects (PLAN §2.3 generation).

This is the first `generate_fn` that fills every GenOutput field with a real
value: earlier the only implementation was pipeline.echo_reference_stub, which
never calls a model and stamps placeholder values (final_prompt=task.prompt
unwrapped, prompt_template_id="bare"). Here, final_prompt is the actual
build_prompt() output that was sent to the model, and prompt_template_id names
the real template — this is exactly what that schema work (see
schema/records.py::GenerationRecord.final_prompt) was built to record.

A factory (not a bare function): the model must be loaded once, outside the
per-(task, sample) loop run_pipeline drives — loading happens when
make_qwen_generate_fn is called, not on every generated sample.
"""

from __future__ import annotations

from typing import Protocol

from rlforge.pipeline import GenerateFn, GenOutput
from rlforge.prompt_template.default_prompt_template import PROMPT_TEMPLATE_ID, build_prompt
from rlforge.schema.task import CodeTask


class SupportsGenerate(Protocol):
    """The type of object make_qwen_generate_fn() expects to receive as `model`.

    A Protocol, not a direct QwenModel type hint, so tests can pass a small
    fake/mock implementing just this method — no GPU or real model needed to
    verify the glue logic in this file (see tests/test_generate_fn.py).
    """

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        seed: int = 0,
        enable_thinking: bool = False,
    ):
        ...


def make_qwen_generate_fn(
    model: SupportsGenerate,
    *,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 1.5,
    seed: int = 0,
    enable_thinking: bool = False,
) -> GenerateFn:
    """Build a GenerateFn that calls `model` for real generation.

    Defaults mirror QwenModel.generate()'s own defaults (Qwen3.5's model-card
    -recommended non-thinking-mode sampling settings), so a caller that only
    cares about generate_fn glue doesn't need to restate them.

    max_new_tokens defaults to 1024, not ReasoningLab's 256 — that budget was
    sized for a 6GB-VRAM card's speed constraints, not for correctness. A
    Qwen3.5 model (thinking-mode-capable, even with enable_thinking=False
    suppressing the shown trace) plus a full function completion needs real
    headroom; a truncated generation before reaching code would surface as a
    false EXTRACTION failure, not a genuine model-capability failure.

    seed is offset by sample_idx in the returned function, so multiple samples
    of the same task (n_samples>1, a later step) are each reproducible AND
    distinct from one another, rather than all identical.
    """

    def generate_fn(task: CodeTask, sample_idx: int) -> GenOutput:
        final_prompt = build_prompt(task)
        sample_seed = seed + sample_idx
        out = model.generate(
            final_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=sample_seed,
            enable_thinking=enable_thinking,
        )
        return GenOutput(
            raw_output=out.text,
            model_name=getattr(model, "model_id", model.__class__.__name__),
            # Recorded as 0.0/1.0 (no-op values) when do_sample=False, since
            # temperature/top_p have no effect on greedy decoding — the record
            # should reflect what actually influenced the output, not the
            # unused argument values.
            decoding_temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            max_new_tokens=max_new_tokens,
            seed=sample_seed,
            input_tokens=out.prompt_tokens,
            output_tokens=out.completion_tokens,
            hit_token_cap=out.hit_token_cap,
            generation_time_s=out.elapsed_s,
            model_revision=getattr(model, "model_revision", None),
            enable_thinking=enable_thinking,
            final_prompt=final_prompt,
            prompt_template_id=PROMPT_TEMPLATE_ID,
        )

    return generate_fn
