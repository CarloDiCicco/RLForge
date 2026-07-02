"""Tests for models/qwen.py — the real QwenModel, against real weights.

Gated on transformers being installed AND a CUDA device being available (same
skip-if-unavailable pattern as tests/test_adapter_humaneval.py for evalplus and
tests/test_sandbox_bwrap.py for bwrap_usable()). Loads Qwen3.5-4B for real —
slow (weight download + load) and hardware-dependent, so this is intentionally
NOT part of the fast default test loop; test_generate_fn.py covers the glue
logic with a fake model on every run instead.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA device available on this host",
)

from rlforge.models.qwen import QwenModel  # noqa: E402

_MODEL_ID = "Qwen/Qwen3.5-4B"
_PROMPT = (
    "Complete the following Python function. "
    "Return only one Markdown fenced Python block (```python ... ```), "
    "with no extra explanation.\n\n"
    "```python\ndef add(a: int, b: int) -> int:\n    \"\"\"Return a + b.\"\"\"\n```"
)


@pytest.fixture(scope="module")
def model() -> QwenModel:
    # Loaded once per test module (real weight load is slow — ~tens of
    # seconds), reused across the tests below.
    return QwenModel(_MODEL_ID)


def test_generate_returns_nonempty_text(model):
    out = model.generate(_PROMPT, max_new_tokens=256)
    assert out.text.strip()


def test_thinking_block_suppressed_at_source(model):
    # enable_thinking=False must prevent a literal <think> tag from appearing
    # at all, not merely rely on extract_code()'s post-hoc stripping.
    out = model.generate(_PROMPT, max_new_tokens=256)
    assert "<think>" not in out.text


def test_token_counts_are_positive(model):
    out = model.generate(_PROMPT, max_new_tokens=256)
    assert out.prompt_tokens > 0
    assert out.completion_tokens > 0


def test_greedy_decoding_is_deterministic(model):
    out1 = model.generate(_PROMPT, max_new_tokens=64, do_sample=False, seed=0)
    out2 = model.generate(_PROMPT, max_new_tokens=64, do_sample=False, seed=0)
    assert out1.text == out2.text


def test_seeded_sampling_is_deterministic(model):
    # generate()'s defaults are now Qwen3.5's card-recommended SAMPLING
    # settings (do_sample=True), not greedy — same seed must still reproduce
    # the same output, since torch.manual_seed(seed) is set before every call.
    out1 = model.generate(_PROMPT, max_new_tokens=64, seed=0)
    out2 = model.generate(_PROMPT, max_new_tokens=64, seed=0)
    assert out1.text == out2.text


def test_presence_penalty_runs_against_real_model(model):
    # Confirms _PresencePenaltyLogitsProcessor actually executes without
    # erroring against the real model (not just unit-tested in isolation),
    # and that <think> is still suppressed under sampling + the penalty.
    out = model.generate(_PROMPT, max_new_tokens=256, do_sample=True, presence_penalty=1.3)
    assert out.text.strip()
    assert "<think>" not in out.text
