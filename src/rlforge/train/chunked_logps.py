"""
train/chunked_logps.py
──────────────────────
GRPOTrainer whose per-token log-probabilities are computed in slices along the
answer, so the full logits tensor is never built.

Why: TRL computes log-probs by running the model to logits for the WHOLE
sequence, (batch, length, vocabulary), then taking a log-softmax. Qwen3.5 has a
248,320-token vocabulary, so one 32k-token thinking answer gives 32,768 x 248,320
logits: ~15 GB in bf16 and ~30 GB in float32 after the division by temperature
and the softmax. The thinking smoke test (2026-10-15) ran out of memory exactly
there ("Tried to allocate 30.31 GiB"), in the no-grad pass TRL makes before each
update to correct for vLLM's sampling probabilities. The Liger fused loss
(`use_liger_kernel`) only covers the loss itself, not this pass.

Here the model runs once to its last hidden state (length x 2560, small), and the
output layer + log-softmax are applied `chunk` positions at a time. The result is
identical (same operations, in pieces); peak memory for the logits drops from
length x vocabulary to chunk x vocabulary. Under autograd each slice is
recomputed in the backward pass (torch checkpoint), so the saving also holds
when gradients are needed.

Only the plain text path is replaced. Images, MoE auxiliary losses and the
differentiable entropy bonus fall back to TRL's original implementation.

Trailing-padding trim (optional, `trim_trailing_padding`)
─────────────────────────────────────────────────────────
TRL pads all answers of a step to the longest one (right padding; 32,768 tokens
as soon as one answer is cut off) and runs the model on the padded width, both
in the no-grad log-prob pass and in the training pass. With one answer per
micro-batch, a 15k-token answer then costs a 34k-token forward and backward.
Measured on Qwen3.5-4B (2026-11-21): 39.6 s per answer at 34k vs 21.7 s at 20k.

The trim removes the trailing positions that are padding in EVERY row of the
micro-batch before the forward pass, then pads the hidden states back to the
original width with zeros, so every tensor downstream keeps its shape.
Why the result is the same:
- the model is causal: the output at a position depends only on that position
  and earlier ones (attention mask, causal convolution, left-to-right linear
  attention), so removing later positions cannot change it;
- the removed positions are padding; their values are masked out of the loss,
  of the importance-sampling ratio and of every logged metric. The zeros put
  back give finite values there (logits 0), so a masked product stays 0.
Same math, not the same bits: GPU kernels round differently for a different
sequence length. Checked on Qwen3.5-4B with real answers (2026-11-21,
scripts/verify_trim_padding.py): the trim moves log-probs and per-answer
gradients exactly as much as TRL itself moves them when the same answer sits in
a step padded to another width (bf16, 899-token answer: mean log-prob change
6.6e-3 trimmed vs 7.0e-3 for TRL at width 8192 instead of 32768; float32: 8.7e-5
vs 8.4e-5 for +128 padding). In a run the padded width changes every step (it
follows the longest answer), so this variation is already there.
Left padding of the prompts is not touched.
Both passes go through `_get_last_hidden_state` (the no-grad pass via the
slicing below, the training pass via TRL's Liger loss), so one override
covers both. This is the per-answer form of "remove padding" used by RL
frameworks such as verl (`use_remove_padding`), without packing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from trl import GRPOTrainer
from trl.trainer.utils import entropy_from_logits, selective_log_softmax


def trailing_padding(attention_mask: torch.Tensor, logits_to_keep: int) -> int:
    """Number of trailing positions that are padding in every row.

    attention_mask: (B, L), 1 = real token, 0 = padding (prompt + completion).
    Capped at logits_to_keep - 1 so at least one completion position is kept:
    TRL asks the model for `logits_to_keep + 1` positions, and 0 would mean
    "keep all" to transformers.
    """
    real = attention_mask.bool().any(dim=0).nonzero()  # positions real in some row
    last = int(real[-1]) if real.numel() else -1
    n = attention_mask.size(1) - (last + 1)
    return max(0, min(n, logits_to_keep - 1))


class ChunkedLogpsGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with sliced log-prob computation (see module docstring)."""

    def __init__(self, *args, logits_chunk_size: int = 2048,
                 trim_trailing_padding: bool = False, **kwargs) -> None:
        if logits_chunk_size < 1:
            raise ValueError(f"logits_chunk_size must be >= 1, got {logits_chunk_size}")
        self.logits_chunk_size = logits_chunk_size
        self.trim_trailing_padding = trim_trailing_padding
        super().__init__(*args, **kwargs)

    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask,
                               logits_to_keep, *multimodal, **multimodal_kw):
        """TRL's method, run on the micro-batch without its trailing padding.

        Returns the same shape as TRL's, (B, logits_to_keep, H): the trimmed
        positions come back as zeros (see the module docstring for why that is
        safe). Image inputs are left to TRL unchanged.
        """
        has_images = any(v is not None for v in (*multimodal, *multimodal_kw.values()))
        if not getattr(self, "trim_trailing_padding", False) or has_images:
            return super()._get_last_hidden_state(
                unwrapped_model, input_ids, attention_mask, logits_to_keep,
                *multimodal, **multimodal_kw)

        n_trim = trailing_padding(attention_mask, logits_to_keep)
        if torch.is_grad_enabled():
            # One value per training micro-batch: share of positions skipped.
            metrics = getattr(self, "_metrics", None)
            if metrics is not None:
                metrics["train"]["padding/trimmed_frac"].append(n_trim / input_ids.size(1))
        if n_trim == 0:
            return super()._get_last_hidden_state(
                unwrapped_model, input_ids, attention_mask, logits_to_keep)

        width = input_ids.size(1) - n_trim
        hidden = super()._get_last_hidden_state(
            unwrapped_model, input_ids[:, :width], attention_mask[:, :width],
            logits_to_keep - n_trim)
        # (B, logits_to_keep - n_trim, H) -> (B, logits_to_keep, H), zeros at the end.
        return F.pad(hidden, (0, 0, 0, n_trim))

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        compute_aux_loss=False,
        **multimodal,
    ):
        needs_original = (
            any(v is not None for v in multimodal.values())
            or compute_aux_loss
            or (compute_entropy and getattr(self, "_entropy_bonus_enabled", False))
        )
        if needs_original:
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, batch_size=batch_size,
                compute_entropy=compute_entropy, compute_aux_loss=compute_aux_loss, **multimodal,
            )

        # Entropy for LOGGING. With the Liger loss TRL's loss path returns before it
        # computes entropy, so no "entropy" curve is logged (run #2). The no-grad
        # pass TRL makes before each update (old log-probs) already builds each
        # slice's logits, so entropy is computed there at no extra memory and
        # logged under the same name TRL uses.
        log_entropy = not torch.is_grad_enabled() and not compute_entropy
        if log_entropy:
            compute_entropy = True

        lm_head = self.accelerator.unwrap_model(model).get_output_embeddings()
        temperature = self.temperature
        batch_size = batch_size or input_ids.size(0)
        all_logps, all_entropies = [], []

        def slice_logps(h: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
            # Same three operations as TRL, on `chunk` positions only.
            logits = lm_head(h) / temperature
            return selective_log_softmax(logits, ids)

        for start in range(0, input_ids.size(0), batch_size):
            end = min(start + batch_size, input_ids.size(0))
            ids_b = input_ids[start:end]
            # (B, logits_to_keep, H): hidden states that predict the completion tokens
            hidden = self._get_last_hidden_state(
                model, ids_b, attention_mask[start:end], logits_to_keep
            )
            completion_ids = ids_b[:, -logits_to_keep:]
            logps_parts, ent_parts = [], []
            for c in range(0, logits_to_keep, self.logits_chunk_size):
                h = hidden[:, c:c + self.logits_chunk_size]
                ids = completion_ids[:, c:c + self.logits_chunk_size]
                if torch.is_grad_enabled():
                    # Recompute this slice's logits in backward instead of keeping them.
                    logps_parts.append(checkpoint(slice_logps, h, ids, use_reentrant=False))
                else:
                    logps_parts.append(slice_logps(h, ids))
                if compute_entropy:
                    with torch.no_grad():  # logging / masking only (bonus case handled above)
                        ent_parts.append(entropy_from_logits(lm_head(h) / temperature))
            all_logps.append(torch.cat(logps_parts, dim=1))
            if compute_entropy:
                all_entropies.append(torch.cat(ent_parts, dim=1))

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        if log_entropy:
            # Mean over completion tokens only (padding masked out), like TRL.
            mask = attention_mask[:, -logits_to_keep:].bool()
            metrics = getattr(self, "_metrics", None)
            if metrics is not None and mask.any():
                metrics["train"]["entropy"].append(entropies[mask].float().mean().item())
            entropies = None  # the caller did not ask for them
        return logps, entropies, None
