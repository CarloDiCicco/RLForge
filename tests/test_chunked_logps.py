"""Sliced log-probs (train/chunked_logps.py) must equal TRL's original exactly.

The slicing exists only to save memory on long thinking answers; any numerical
difference would silently change the GRPO update. Checked on a tiny random
model on CPU, against TRL's own implementation, with and without gradients.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from trl import GRPOTrainer  # noqa: E402

from rlforge.train.chunked_logps import ChunkedLogpsGRPOTrainer  # noqa: E402


class _Acc:
    is_main_process = True
    @staticmethod
    def unwrap_model(m):
        return m


def _trainer(model, chunk: int) -> ChunkedLogpsGRPOTrainer:
    # Bypass __init__ (it needs a dataset, reward, ...): set only what the
    # log-prob code reads.
    t = object.__new__(ChunkedLogpsGRPOTrainer)
    t.logits_chunk_size = chunk
    t.temperature = 0.6
    t.accelerator = _Acc()
    t.model_kwarg_keys = set()
    t._is_vlm = False
    t._entropy_bonus_enabled = False
    return t


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=257, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = Qwen3ForCausalLM(cfg).eval()
    ids = torch.randint(0, 257, (3, 40))
    mask = torch.ones_like(ids)
    return model, ids, mask


@pytest.mark.parametrize("chunk", [1, 7, 30, 64])
def test_sliced_logps_equal_trl(setup, chunk):
    model, ids, mask = setup
    t = _trainer(model, chunk)
    with torch.no_grad():
        ref, ref_ent, _ = GRPOTrainer._get_per_token_logps_and_entropies(
            t, model, ids, mask, 30, batch_size=2, compute_entropy=True)
        got, got_ent, _ = t._get_per_token_logps_and_entropies(
            model, ids, mask, 30, batch_size=2, compute_entropy=True)
    assert got.shape == ref.shape == (3, 30)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_ent, ref_ent, rtol=1e-5, atol=1e-5)


def test_sliced_gradients_equal_trl(setup):
    model, ids, mask = setup
    t = _trainer(model, 7)
    grads = []
    for fn in (lambda: GRPOTrainer._get_per_token_logps_and_entropies(t, model, ids, mask, 30)[0],
               lambda: t._get_per_token_logps_and_entropies(model, ids, mask, 30)[0]):
        model.zero_grad()
        fn().sum().backward()
        grads.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
    for a, b in zip(*grads, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
