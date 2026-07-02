"""Tests for models/generate_fn.py — the QwenModel-to-GenerateFn glue.

Uses a fake model (implements only .generate()) so this exercises the glue
logic without needing transformers, a GPU, or real weights — always runs.
The real QwenModel is covered separately in tests/test_qwen_model.py, gated
on transformers + CUDA being available.
"""

from __future__ import annotations

from typing import NamedTuple

from rlforge.models.generate_fn import make_qwen_generate_fn
from rlforge.prompt_template.default_prompt_template import PROMPT_TEMPLATE_ID, build_prompt
from rlforge.schema.task import CodeTask


class _FakeGenOutput(NamedTuple):
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float = 0.01


class _FakeModel:
    """Records every call it receives, returns canned text."""

    model_id = "fake:model-v1"

    def __init__(self, text: str = "```python\ndef add(a, b):\n    return a + b\n```"):
        self._text = text
        self.calls: list[dict] = []

    def generate(self, prompt, *, max_new_tokens, do_sample=True, temperature=0.7,
                 top_p=0.8, top_k=20, min_p=0.0, repetition_penalty=1.0,
                 presence_penalty=1.3, seed=0):
        self.calls.append(dict(
            prompt=prompt, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty, seed=seed,
        ))
        return _FakeGenOutput(text=self._text, prompt_tokens=42, completion_tokens=13)


def _task(**overrides) -> CodeTask:
    base = dict(task_id="t1", source="fixture", prompt="def add(a, b):\n")
    base.update(overrides)
    return CodeTask(**base)


def test_generate_fn_builds_prompt_via_build_prompt():
    model = _FakeModel()
    fn = make_qwen_generate_fn(model)
    task = _task()
    fn(task, 0)
    assert model.calls[0]["prompt"] == build_prompt(task)


def test_generate_fn_sets_real_prompt_template_id_not_bare():
    model = _FakeModel()
    fn = make_qwen_generate_fn(model)
    out = fn(_task(), 0)
    assert out.prompt_template_id == PROMPT_TEMPLATE_ID
    assert out.prompt_template_id != "bare"


def test_generate_fn_final_prompt_matches_what_was_sent():
    model = _FakeModel()
    fn = make_qwen_generate_fn(model)
    task = _task()
    out = fn(task, 0)
    assert out.final_prompt == build_prompt(task)
    assert out.final_prompt == model.calls[0]["prompt"]


def test_generate_fn_offsets_seed_by_sample_idx():
    model = _FakeModel()
    fn = make_qwen_generate_fn(model, seed=100)
    fn(_task(), 0)
    fn(_task(), 3)
    assert model.calls[0]["seed"] == 100
    assert model.calls[1]["seed"] == 103


def test_generate_fn_maps_output_fields_from_model_result():
    model = _FakeModel(text="the raw model text")
    fn = make_qwen_generate_fn(model, max_new_tokens=777)
    out = fn(_task(), 0)
    assert out.raw_output == "the raw model text"
    assert out.model_name == "fake:model-v1"
    assert out.input_tokens == 42
    assert out.output_tokens == 13
    assert out.max_new_tokens == 777


def test_generate_fn_passes_max_new_tokens_to_model():
    model = _FakeModel()
    fn = make_qwen_generate_fn(model, max_new_tokens=2048)
    fn(_task(), 0)
    assert model.calls[0]["max_new_tokens"] == 2048


def test_generate_fn_forwards_all_generation_kwargs_to_model():
    model = _FakeModel()
    fn = make_qwen_generate_fn(
        model, do_sample=True, temperature=0.9, top_p=0.5, top_k=40,
        min_p=0.1, repetition_penalty=1.2, presence_penalty=1.5,
    )
    fn(_task(), 0)
    call = model.calls[0]
    assert call["do_sample"] is True
    assert call["temperature"] == 0.9
    assert call["top_p"] == 0.5
    assert call["top_k"] == 40
    assert call["min_p"] == 0.1
    assert call["repetition_penalty"] == 1.2
    assert call["presence_penalty"] == 1.5


def test_generate_fn_records_actual_sampling_params_when_sampling():
    model = _FakeModel()
    fn = make_qwen_generate_fn(model, do_sample=True, temperature=0.7, top_p=0.8)
    out = fn(_task(), 0)
    assert out.decoding_temperature == 0.7
    assert out.top_p == 0.8


def test_generate_fn_records_noop_values_when_greedy():
    model = _FakeModel()
    fn = make_qwen_generate_fn(model, do_sample=False, temperature=0.7, top_p=0.8)
    out = fn(_task(), 0)
    # temperature/top_p had no effect on greedy decoding; the record should
    # reflect that, not the (unused) argument values.
    assert out.decoding_temperature == 0.0
    assert out.top_p == 1.0
