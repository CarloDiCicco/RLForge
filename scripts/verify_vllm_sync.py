"""
scripts/verify_vllm_sync.py
───────────────────────────
Does vLLM generate with the CURRENT training weights, LoRA included?

In a run, TRL copies the training model into the colocated vLLM engine before
each step's generation (LoRA merged into the base weights, copied, unmerged).
With sleep mode vLLM throws its weights away after every generation, so the
copy runs every step. If it silently copied the wrong thing (e.g. the base
weights without LoRA), rollouts would come from the untrained model for the
whole run. The smoke runs could not show this: the only real update before
step 3 was tiny (warmup).

The test: build the trainer exactly as scripts/train_grpo.py does (same config:
vLLM colocated at the run's memory share, sleep mode, LoRA, chunked trainer),
then for several LoRA states set the weights, run TRL's own copy + generation
(the same two calls the trainer makes each step) and compare vLLM's per-token
log-probs of what it generated with the trainer's log-probs of the same tokens
under every state:
  - same state:      gap ~0.02, the vLLM/trainer difference seen in the runs;
  - any other state: much larger.
Negative control: the same with LoRA merging disabled, i.e. vLLM receives base
weights; the test must then flag it (gap to the LoRA state large, to base small).

LoRA states use large random B so that every state is clearly different from
the others; the copy does not care about the values.

Usage (repo root, GPU free, ~10 min):
    .venv/bin/python scripts/verify_vllm_sync.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl import GRPOConfig  # noqa: E402

from rlforge.train.chunked_logps import ChunkedLogpsGRPOTrainer  # noqa: E402
from rlforge.train.grpo_config import load_grpo_config  # noqa: E402


def build_trainer(cfg, prompts, max_tokens, out_dir):
    """The trainer as scripts/train_grpo.py builds it (vLLM and model settings
    from the run config); only the dataset, reward and answer length differ."""
    model_source = snapshot_download(cfg.model_id, revision=cfg.model_revision)
    tok = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision,
                                        trust_remote_code=True)
    args = GRPOConfig(
        output_dir=out_dir,
        model_init_kwargs={"dtype": "bfloat16"},
        temperature=cfg.grpo.temperature, top_p=cfg.grpo.top_p, top_k=cfg.grpo.top_k,
        min_p=cfg.grpo.min_p,
        max_completion_length=max_tokens,  # short answers: the copy does not depend on length
        chat_template_kwargs={"enable_thinking": cfg.enable_thinking},
        num_generations=2, per_device_train_batch_size=1, gradient_accumulation_steps=2,
        generation_batch_size=2, max_steps=1,
        beta=cfg.grpo.beta, loss_type=cfg.grpo.loss_type,
        gradient_checkpointing=cfg.grpo.gradient_checkpointing, bf16=True, seed=cfg.seed,
        use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg.grpo.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=cfg.grpo.vllm_enable_sleep_mode,
        vllm_max_model_length=cfg.grpo.max_completion_length + 2048,
        vllm_importance_sampling_correction=cfg.grpo.vllm_importance_sampling_correction,
        vllm_importance_sampling_mode=cfg.grpo.vllm_importance_sampling_mode,
        use_liger_kernel=cfg.grpo.use_liger_kernel,
        report_to="none",
    )
    dataset = Dataset.from_dict({"prompt": prompts})
    trainer = ChunkedLogpsGRPOTrainer(
        logits_chunk_size=cfg.grpo.logits_chunk_size,
        trim_trailing_padding=cfg.grpo.trim_trailing_padding,
        model=model_source,
        reward_funcs=lambda completions, **kw: [0.0] * len(completions),
        args=args, train_dataset=dataset, processing_class=tok,
        peft_config=LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.lora_alpha,
                               lora_dropout=cfg.lora.lora_dropout,
                               target_modules=cfg.lora.target_modules, task_type="CAUSAL_LM"),
    )
    return trainer, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo-vanilla-run3.yaml")
    ap.add_argument("--completions", default="runs/train/20261121T232950Z-grpo-vanilla-run3-smoke/"
                                             "completions/completions_00001.parquet",
                    help="any logged step; its first two task prompts are used")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--sigma", type=float, default=0.005, help="std of the random LoRA B")
    ap.add_argument("--out-dir", default=None, help="trainer output dir (nothing is saved); default: a temp dir")
    args = ap.parse_args()
    cfg = load_grpo_config(args.config)

    d = pd.read_parquet(args.completions)
    prompt_texts = [d.prompt[0], d.prompt[8]]  # two different tasks (rows come in groups of 8)
    out_dir = args.out_dir or tempfile.mkdtemp(prefix="vllm_sync_")
    trainer, tok = build_trainer(cfg, [[{"role": "user", "content": "x"}]] * 2, args.max_tokens, out_dir)
    model, vg = trainer.model, trainer.vllm_generation
    prompt_ids = [tok(p, add_special_tokens=False).input_ids for p in prompt_texts]
    print(f"vLLM: colocate, memory share {vg.gpu_memory_utilization}, sleep mode {vg.enable_sleep_mode}, "
          f"max length {vg.max_model_length}; LoRA B ~ N(0, {args.sigma})")
    lora_b = [(n, p) for n, p in model.named_parameters() if "lora_B" in n]
    print(f"LoRA B tensors: {len(lora_b)}")

    def set_lora(state):
        """state None = B zero (base model), else an integer seed."""
        with torch.no_grad():
            for k, (_, p) in enumerate(lora_b):
                if state is None:
                    p.zero_()
                else:
                    g = torch.Generator().manual_seed(1000 * state + k)
                    p.copy_((args.sigma * torch.randn(p.shape, generator=g)).to(p.dtype))

    def generate():
        # the trainer's two calls at each step (grpo_trainer._generate_single_turn)
        vg.sync_weights()
        _, completion_ids, logprobs, _ = vg.generate(prompts=prompt_ids, images=None, num_generations=1)
        return completion_ids, [[lp[0] for lp in row] for row in logprobs]

    def trainer_logps(completion_ids):
        """Trainer's log-probs of the generated tokens (TRL's no-grad pass)."""
        out = []
        for p, c in zip(prompt_ids, completion_ids, strict=True):
            ids = torch.tensor([p + list(c)], device=model.device)
            mask = torch.ones_like(ids)
            with torch.no_grad():
                lp = trainer._get_per_token_logps_and_entropies(model, ids, mask, len(c), batch_size=1)[0]
            out.append(lp[0].float().cpu())
        return out

    def gap(vllm_lps, train_lps):
        diffs = [(torch.tensor(v) - t).abs() for v, t in zip(vllm_lps, train_lps, strict=True)]
        return torch.cat(diffs).mean().item()

    states = {"base (B=0)": None, "LoRA A": 1, "LoRA B": 2}

    def round_(name, state, merge_ok=True):
        set_lora(state)
        merge = model.merge_adapter
        if not merge_ok:
            model.merge_adapter = lambda *a, **k: None  # vLLM then receives base weights
        try:
            completion_ids, vllm_lps = generate()
        finally:
            model.merge_adapter = merge
        n_tok = sum(len(c) for c in completion_ids)
        cols = []
        for other, ostate in states.items():
            set_lora(ostate)
            cols.append(gap(vllm_lps, trainer_logps(completion_ids)))
        set_lora(state)
        print(f"{name:<34} {n_tok:>6} | " + " | ".join(f"{c:>12.4f}" for c in cols))
        return cols

    model.eval()
    print("\nmean |vLLM log-prob - trainer log-prob| per generated token")
    print(f"{'vLLM generated with':<34} {'tokens':>6} | " + " | ".join(f"{s:>12}" for s in states))
    results = {name: round_(name, state) for name, state in states.items()}
    ctrl = round_("LoRA A, merge disabled (control)", 1, merge_ok=False)

    names = list(states)
    ok = all(results[n][i] == min(results[n]) for i, n in enumerate(names))
    print("\ncopy works (each row smallest on its own state):", "YES" if ok else "NO")
    print("control flagged (vLLM matches base, not LoRA A):",
          "YES" if ctrl[0] < ctrl[1] else "NO")


if __name__ == "__main__":
    main()
