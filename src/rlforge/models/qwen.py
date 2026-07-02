"""
models/qwen.py
───────────────
QwenModel — local Qwen inference (PLAN §2.3 generation).

Adapted from ReasoningLab `models/qwen_local.py`: same overall shape (load
tokenizer + model once in __init__, reuse across many generate() calls — avoids
paying the ~20s model-load cost per sample) and the same fix for Qwen3.5's
thinking-mode default (`enable_thinking=False` in `apply_chat_template`, which
suppresses the `<think>...</think>` block at generation time, before it would
otherwise consume most of `max_new_tokens` and truncate the actual code).

NOT reused: ReasoningLab's 4-bit NF4 quantization + bitsandbytes + CPU/disk
offload machinery. That was built for a 6GB-VRAM card (RTX 4050); this host
(the GB10 DGX Spark) has 128GB unified memory, so a Qwen3.5-4B/9B model loads
natively in bfloat16 with no quantization — simpler code and no quantization
capability cost, for hardware that doesn't need the tradeoff.

extract.extractor.extract_code already strips any literal <think> block that
slips through (built in Step 6, before any model existed) — a safety net, not
the primary defense; enable_thinking=False at the source is the real fix.

REUSED (ported near-verbatim): `_PresencePenaltyLogitsProcessor`. HF's native
generate() has no presence_penalty support (only vLLM/SGLang do), so
ReasoningLab implements it as a custom LogitsProcessor — a flat additive
penalty on any token already generated in this sequence, discouraging
repetition regardless of how many times a token recurred (unlike the
multiplicative repetition_penalty, which scales with frequency). Prompt tokens
are explicitly excluded from the penalty: penalizing them would suppress
unavoidable Python keywords (def, return, for, if, self, ...) on the very
first generated token.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList


class _PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Flat additive penalty on tokens already seen in the generated portion.

    penalty=0.0 is a no-op (never constructed in that case — see generate()).
    Qwen3.5's model card recommends 1.3-1.5 for non-thinking-mode generation.
    """

    def __init__(self, penalty: float, prompt_length: int) -> None:
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for i, seq in enumerate(input_ids):
            generated_ids = seq[self.prompt_length:]
            unique_tokens = generated_ids.unique()
            scores[i, unique_tokens] -= self.penalty
        return scores


class QwenGenerationOutput(NamedTuple):
    """Result of one QwenModel.generate() call."""

    text: str  # generated completion only, prompt not included
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float


class QwenModel:
    """Local Qwen inference wrapper: loads once, generates many times."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        # eval() disables dropout — required for deterministic greedy decoding.
        self.model.eval()

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
        presence_penalty: float = 1.3,
        seed: int = 0,
    ) -> QwenGenerationOutput:
        """Generate a completion for one prompt.

        Defaults are Qwen3.5's own model-card-recommended settings for
        non-thinking-mode generation (do_sample=True, temperature=0.7,
        top_p=0.8, top_k=20, repetition_penalty=1.0, presence_penalty=1.3) —
        not a generic/greedy default. Sampling params are only applied to
        gen_kwargs when do_sample=True; passing do_sample=False still works
        (e.g. tests wanting deterministic output) and ignores them.
        """
        # enable_thinking=False: the source-level fix for Qwen3.5's reasoning-
        # mode default. Without it the model emits a <think>...</think> block
        # first, which would eat most of max_new_tokens before any code appears.
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False, # the tokenizer will tokenize below
            add_generation_prompt=True, # adds the "Assistant:" prompt to the end of the message list
            enable_thinking=False,
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        # Seeded here (not globally at import time) so each generate() call is
        # independently reproducible given the same seed, regardless of how
        # many prior calls happened on this model instance.
        torch.manual_seed(seed)

        # HF generate() has no native presence_penalty (only vLLM/SGLang do);
        # implemented via a custom LogitsProcessor. penalty=0.0 stays a no-op —
        # constructing it unconditionally would still evaluate correctly, but
        # skipping it entirely when unused avoids the per-token Python loop
        # overhead in _PresencePenaltyLogitsProcessor.__call__ for the common
        # case where no penalty is wanted.
        logits_processors = LogitsProcessorList()
        if do_sample and presence_penalty > 0.0:
            logits_processors.append(
                _PresencePenaltyLogitsProcessor(presence_penalty, prompt_tokens)
            )

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            # Qwen models may not have pad_token_id set; eos is the standard
            # fallback to suppress a padding-related warning.
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
            "logits_processor": logits_processors,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k
            gen_kwargs["min_p"] = min_p

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - start

        # output_ids contains prompt tokens followed by completion tokens;
        # strip the prompt portion so we return only the newly generated text.
        new_ids = output_ids[0][prompt_tokens:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        return QwenGenerationOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=new_ids.shape[0],
            elapsed_s=elapsed,
        )
