"""Sliced log-probs (train/chunked_logps.py) must equal TRL's original exactly.

The slicing exists only to save memory on long thinking answers; any numerical
difference would silently change the GRPO update. Checked on a tiny random
model on CPU, against TRL's own implementation, with and without gradients.
"""

from __future__ import annotations

import collections

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


def _trainer(model, chunk: int, trim: bool = False) -> ChunkedLogpsGRPOTrainer:
    # Bypass __init__ (it needs a dataset, reward, ...): set only what the
    # log-prob code reads.
    t = object.__new__(ChunkedLogpsGRPOTrainer)
    t.logits_chunk_size = chunk
    t.trim_trailing_padding = trim
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


def test_no_grad_pass_logs_entropy(setup):
    # With the Liger loss TRL logs no entropy; the sliced no-grad pass logs it.
    import collections
    model, ids, mask = setup
    t = _trainer(model, 7)
    t._metrics = {"train": collections.defaultdict(list)}
    with torch.no_grad():
        _, ref_ent, _ = GRPOTrainer._get_per_token_logps_and_entropies(
            t, model, ids, mask, 30, compute_entropy=True)
        _, ent, _ = t._get_per_token_logps_and_entropies(model, ids, mask, 30)
    assert ent is None  # not returned when the caller did not ask
    logged = t._metrics["train"]["entropy"]
    assert len(logged) == 1
    assert abs(logged[0] - ref_ent.mean().item()) < 1e-4


# ── trailing-padding trim (trim_trailing_padding) ─────────────────────────────
# The trim must not change anything the loss or the logs read: log-probs and
# gradients on real completion tokens, and the Liger training loss, are compared
# against TRL's own untrimmed computation on TRL's batch layout.

from trl.losses import FusedLinearGRPOLoss  # noqa: E402

from rlforge.train.chunked_logps import trailing_padding  # noqa: E402

_P, _C = 6, 30  # prompt width, completion width (= logits_to_keep)


def _padded_batch(prompt_lens, completion_lens, seed=1):
    """TRL's layout: prompts left-padded, completions right-padded, pad id 0."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(len(prompt_lens), _P + _C, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, (p, c) in enumerate(zip(prompt_lens, completion_lens, strict=True)):
        ids[i, _P - p:_P] = torch.randint(1, 257, (p,), generator=g)
        ids[i, _P:_P + c] = torch.randint(1, 257, (c,), generator=g)
        mask[i, _P - p:_P + c] = 1
    return ids, mask


# 3 answers of 12, 20 and 7 tokens in a 30-wide completion block: 10 trailing
# positions are padding in every row; per row, 18 / 10 / 23.
_BATCH = ([6, 4, 5], [12, 20, 7])


def test_trailing_padding_count():
    ids, mask = _padded_batch(*_BATCH)
    assert trailing_padding(mask, _C) == 10
    assert [trailing_padding(mask[i:i + 1], _C) for i in range(3)] == [18, 10, 23]
    assert trailing_padding(torch.ones(2, 36, dtype=torch.long), _C) == 0  # no padding
    # an empty completion: everything after the prompt is padding, but at least
    # one completion position is kept (0 would mean "all" to transformers)
    _, empty = _padded_batch([6], [0])
    assert trailing_padding(empty, _C) == _C - 1
    assert trailing_padding(empty, 1) == 0


def test_trim_hidden_state_same_shape_real_positions_equal(setup):
    model = setup[0]
    ids, mask = _padded_batch([6], [12])
    with torch.no_grad():
        ref = _trainer(model, 7)._get_last_hidden_state(model, ids, mask, _C)
        got = _trainer(model, 7, trim=True)._get_last_hidden_state(model, ids, mask, _C)
    assert got.shape == ref.shape == (1, _C, 32)
    # the 12 hidden states that predict the 12 completion tokens are unchanged
    torch.testing.assert_close(got[:, :12], ref[:, :12], rtol=1e-5, atol=1e-6)
    assert (got[:, 12:] == 0).all()  # trimmed positions come back as zeros


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_trim_logps_equal_trl_on_real_tokens(setup, batch_size):
    model = setup[0]
    ids, mask = _padded_batch(*_BATCH)
    real = mask[:, -_C:].bool()
    t = _trainer(model, 7, trim=True)
    with torch.no_grad():
        ref, _, _ = GRPOTrainer._get_per_token_logps_and_entropies(
            t, model, ids, mask, _C, batch_size=batch_size)  # TRL, untrimmed
        got, _, _ = t._get_per_token_logps_and_entropies(model, ids, mask, _C, batch_size=batch_size)
    assert got.shape == ref.shape == (3, _C)
    torch.testing.assert_close(got[real], ref[real], rtol=1e-5, atol=1e-5)
    assert torch.isfinite(got).all()  # padded positions finite: masked products stay 0


def test_trim_gradients_equal_trl(setup):
    model = setup[0]
    ids, mask = _padded_batch(*_BATCH)
    real = mask[:, -_C:].float()
    t = _trainer(model, 7, trim=True)
    grads = []
    for fn in (lambda: GRPOTrainer._get_per_token_logps_and_entropies(t, model, ids, mask, _C)[0],
               lambda: t._get_per_token_logps_and_entropies(model, ids, mask, _C, batch_size=1)[0]):
        model.zero_grad()
        (fn() * real).sum().backward()  # a masked loss, as in GRPO
        grads.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
    assert len(grads[0]) == len(grads[1]) > 0
    for a, b in zip(*grads, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def _liger_trainer(model, trim: bool):
    """Enough of a trainer for TRL's compute_liger_loss (the training pass)."""
    t = _trainer(model, 7, trim=trim)
    t.liger_loss = FusedLinearGRPOLoss(
        beta=0.0, compiled=False, use_ref_model=False, epsilon_low=0.2, epsilon_high=0.2,
        loss_type="grpo", max_completion_length=_C, importance_sampling_level="token",
        temperature=t.temperature)
    t.beta, t.loss_type = 0.0, "grpo"
    t.model = model
    t.current_gradient_accumulation_steps = 1
    t.args = type("A", (), {"steps_per_generation": 1})()
    t._metrics = {"train": collections.defaultdict(list), "eval": collections.defaultdict(list)}
    t.accelerator.gather = lambda x: x  # single process
    return t


@pytest.mark.parametrize("rows", [[0], [1], [2], [0, 1, 2]])
def test_trim_liger_training_loss_and_gradients_equal(setup, rows):
    # The real training path: old log-probs from the no-grad pass, then TRL's
    # compute_liger_loss (vanilla GRPO loss, token-level IS ratio clipped at 3).
    # Each mode uses its own old log-probs, as in a real run.
    model = setup[0].train()
    ids, mask = _padded_batch(*_BATCH)
    ids, mask = ids[rows], mask[rows]
    real = mask[:, -_C:]
    g = torch.Generator().manual_seed(3)
    sampling_noise = 0.05 * torch.randn(len(rows), _C, generator=g)
    advantages = torch.tensor([0.9, -0.4, -0.5])[rows]
    results = []
    for trim in (False, True):
        t = _liger_trainer(model, trim)
        with torch.no_grad():
            old, _, _ = t._get_per_token_logps_and_entropies(model, ids, mask, _C, batch_size=1)
        sampling = old + sampling_noise
        is_ratio = torch.exp((old - sampling) * real).clamp(max=3.0)  # as TRL computes it
        inputs = {"prompt_ids": ids[:, :_P], "prompt_mask": mask[:, :_P],
                  "completion_ids": ids[:, _P:], "completion_mask": real,
                  "advantages": advantages, "old_per_token_logps": old,
                  "importance_sampling_ratio": is_ratio}
        model.zero_grad()
        loss = GRPOTrainer.compute_liger_loss(t, model, inputs)
        loss.backward()
        results.append((loss.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None],
                        t._metrics["train"]["clip_ratio"][-1]))
    model.eval()
    (l0, g0, c0), (l1, g1, c1) = results
    torch.testing.assert_close(l1, l0, rtol=1e-5, atol=1e-7)
    assert c0 == c1
    assert len(g0) == len(g1) > 0
    for a, b in zip(g0, g1, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_trim_logs_trimmed_fraction_in_training_pass_only(setup):
    model = setup[0]
    ids, mask = _padded_batch([6], [12])
    t = _trainer(model, 7, trim=True)
    t._metrics = {"train": collections.defaultdict(list)}
    with torch.no_grad():
        t._get_last_hidden_state(model, ids, mask, _C)
    assert "padding/trimmed_frac" not in t._metrics["train"]  # no-grad pass: not logged
    t._get_last_hidden_state(model, ids, mask, _C)
    assert t._metrics["train"]["padding/trimmed_frac"] == [18 / 36]


def test_trim_off_or_images_leave_trl_untouched(setup, monkeypatch):
    model = setup[0]
    ids, mask = _padded_batch([6], [12])
    widths = []
    orig = GRPOTrainer._get_last_hidden_state

    def spy(self, m, input_ids, attention_mask, logits_to_keep, *a, **k):
        # record the width TRL's method receives; no model call needed
        widths.append(input_ids.size(1))
        return torch.zeros(input_ids.size(0), logits_to_keep, 32)

    monkeypatch.setattr(GRPOTrainer, "_get_last_hidden_state", spy)
    with torch.no_grad():
        _trainer(model, 7)._get_last_hidden_state(model, ids, mask, _C)                 # flag off
        _trainer(model, 7, trim=True)._get_last_hidden_state(model, ids, mask, _C)      # trimmed
        out = _trainer(model, 7, trim=True)._get_last_hidden_state(                    # images
            model, ids, mask, _C, torch.zeros(1))
    assert widths == [36, 18, 36]
    assert out.shape == (1, _C, 32)


# ── exactness on the REAL architecture (Qwen3.5), bit for bit ─────────────────
# The tests above use a plain-attention model and float32 tolerances. Here:
#   - the model is a tiny Qwen3.5, the run's architecture: Gated DeltaNet linear
#     attention with its causal convolution (3 layers) + full attention (1 layer);
#   - LoRA on the run's target modules (non-zero B) and gradient checkpointing,
#     as in training;
#   - float64 on CPU, which is deterministic, so "same math" must mean EQUAL
#     (torch.equal), not "close". Any real mistake (a dropped or shifted token,
#     a padded position leaking into the loss) changes the numbers at once.
# On CPU the linear attention runs transformers' PyTorch reference kernel (the
# installed flash-linear-attention kernels are GPU-only; same math). The GPU
# kernels are checked on the real model by scripts/verify_trim_padding.py.

import inspect  # noqa: E402
import types  # noqa: E402

from rlforge.train import chunked_logps as _chunked  # noqa: E402

_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
                 "in_proj_a", "in_proj_b", "out_proj", "gate_proj", "up_proj", "down_proj"]


@pytest.fixture(scope="module")
def qwen35():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35_module

    mp = pytest.MonkeyPatch()
    # use the PyTorch reference of the linear-attention kernel (CPU)
    mp.setattr(qwen35_module, "torch_chunk_gated_delta_rule",
               inspect.unwrap(qwen35_module.torch_chunk_gated_delta_rule))
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=257, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"])  # same 3:1 pattern as the real model
    model = Qwen3_5ForCausalLM(cfg).double()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                             target_modules=_LORA_TARGETS))
    with torch.no_grad():  # B = 0 at init would make LoRA a no-op
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0.0, 0.1)
    yield model
    mp.undo()


def _qwen35_trainer(model, trim: bool):
    t = _liger_trainer(model, trim)
    t.model_kwarg_keys = inspect.signature(model.get_base_model().forward).parameters.keys()
    return t


def _training_pass(model, t, ids, mask, advantages, sampling_noise):
    """One real training micro-step: no-grad pass (old log-probs + logged
    entropy), TRL's IS ratio, TRL's Liger loss, backward. Returns everything a
    step reads or logs. The completion width is whatever `ids` has after the
    prompt (_P)."""
    width = ids.size(1) - _P
    real = mask[:, _P:]
    with torch.no_grad():
        old, _, _ = t._get_per_token_logps_and_entropies(model, ids, mask, width, batch_size=1)
    sampling = old + sampling_noise[:, :width]                     # stand-in for vLLM's log-probs
    is_ratio = torch.exp((old - sampling) * real).clamp(max=3.0)   # as TRL computes it
    inputs = {"prompt_ids": ids[:, :_P], "prompt_mask": mask[:, :_P],
              "completion_ids": ids[:, _P:], "completion_mask": real,
              "advantages": advantages, "old_per_token_logps": old,
              "importance_sampling_ratio": is_ratio}
    model.zero_grad(set_to_none=True)
    loss = GRPOTrainer.compute_liger_loss(t, model, inputs)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    m = t._metrics["train"]
    return {"old_logps_real": old[real.bool()], "loss": loss.detach(), "grads": grads,
            "clip_ratio": m["clip_ratio"][-1], "entropy": m["entropy"][-1]}


def _assert_equal(a, b, grads=True):
    for k in ("old_logps_real", "loss"):
        assert torch.equal(a[k], b[k]), k
    assert a["clip_ratio"] == b["clip_ratio"] and a["entropy"] == b["entropy"]
    if grads:
        assert a["grads"].keys() == b["grads"].keys() and len(a["grads"]) > 0
        for n in a["grads"]:
            assert torch.equal(a["grads"][n], b["grads"][n]), n


_ADV = torch.tensor([0.9, -0.4, -0.5], dtype=torch.float64)


def _noise(rows):
    g = torch.Generator().manual_seed(3)
    return 0.05 * torch.randn(3, _C, generator=g, dtype=torch.float64)[rows]


@pytest.mark.parametrize("rows", [[0], [1], [2], [0, 1, 2]])
def test_trim_bit_identical_to_trl_on_qwen35(qwen35, rows):
    # Three computations of the same micro-batch:
    #   pad  = TRL on TRL's layout (padded to the step's longest answer, 30)
    #   cut  = TRL on the same micro-batch with its common trailing padding
    #          removed: what TRL itself computes when the longest answer of the
    #          step is the longest of this micro-batch
    #   trim = ours, on TRL's layout
    # trim == cut in EVERY number (log-probs, IS inputs, loss, clip ratio,
    # logged entropy, all LoRA and model gradients), bit for bit.
    # trim == pad in every number except some weight gradients: those sum over
    # sequence positions, and TRL's own sum over a longer sequence (with exact
    # zeros for the padding) rounds differently, as `cut` vs `pad` does without
    # any trim (checked below).
    ids, mask = _padded_batch(*_BATCH)
    ids, mask = ids[rows], mask[rows]
    w = _P + max(_BATCH[1][r] for r in rows)
    qwen35.train()
    run = lambda trim, x, m: _training_pass(  # noqa: E731
        qwen35, _qwen35_trainer(qwen35, trim), x, m, _ADV[rows], _noise(rows))
    pad, cut, trim = run(False, ids, mask), run(False, ids[:, :w], mask[:, :w]), run(True, ids, mask)
    _assert_equal(trim, cut)
    _assert_equal(trim, pad, grads=False)
    # the gradient differences from `pad` are TRL's own, not the trim's
    differs = lambda a: {n for n in a["grads"] if not torch.equal(a["grads"][n], pad["grads"][n])}  # noqa: E731
    assert differs(trim) == differs(cut)
    lora_grads = [g for n, g in trim["grads"].items() if "lora_" in n]
    assert lora_grads and all(g.abs().sum() > 0 for g in lora_grads)  # the update is not trivially 0


def test_values_put_back_at_padded_positions_are_never_read(qwen35, monkeypatch):
    # The trim puts zeros back where the padding was. If anything downstream
    # (log-probs, IS ratio, loss, gradients, logged metrics) used those
    # positions, replacing the zeros with large random numbers would change it.
    # It changes nothing, bit for bit.
    rows = [0, 1, 2]
    ids, mask = _padded_batch(*_BATCH)
    qwen35.train()
    zeros = _training_pass(qwen35, _qwen35_trainer(qwen35, True), ids, mask, _ADV, _noise(rows))

    real_pad = _chunked.F.pad

    def garbage_pad(x, pad, *a, **k):
        out = real_pad(x, pad, *a, **k)
        n = pad[-1]  # positions added at the end of the sequence dimension
        g = torch.Generator().manual_seed(7)  # same garbage in both passes
        junk = 50.0 * torch.randn(out[:, -n:].shape, generator=g, dtype=out.dtype)
        return torch.cat([out[:, :-n], junk], dim=1)

    # replace F only inside chunked_logps (the trim's one F.pad call), not torch's
    # F.pad itself, which the model's own layers use
    monkeypatch.setattr(_chunked, "F", types.SimpleNamespace(pad=garbage_pad))
    garbage = _training_pass(qwen35, _qwen35_trainer(qwen35, True), ids, mask, _ADV, _noise(rows))
    _assert_equal(garbage, zeros)
