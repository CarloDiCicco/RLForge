"""
scripts/verify_trim_padding.py
──────────────────────────────
Checks the trailing-padding trim (train/chunked_logps.py, `trim_trailing_padding`)
on the REAL model and REAL answers, before it is used in a training run.

The CPU tests (tests/test_chunked_logps.py) prove the logic bit for bit on a tiny
Qwen3.5 in float64. This script answers the remaining question: on Qwen3.5-4B in
bf16 on the GPU, with the real kernels, LoRA and gradient checkpointing as in
training, what does the trim change?

Data: one logged step of a training run (`completions/completions_0000N.parquet`,
prompt and answer text). The answers are re-tokenized and laid out exactly as
TRL does for a step: prompts left-padded to the longest prompt, answers
right-padded to the longest answer (~32k when one was cut off).

Each answer is computed one per micro-batch, as in training, in up to five ways:
  TRL          TRL's own code on TRL's layout (padded to the step's longest answer)
  trim         ours, on TRL's layout
  cut          TRL's own code on the answer with no trailing padding: what TRL
               itself computes when this answer is the longest of its step
  other width  TRL's own code with the answer padded to another width
  again        the same computation repeated (GPU run-to-run noise)

  1. no-grad log-prob pass (old log-probs): trim vs TRL, other width, again; speed
  2. training pass (TRL's Liger GRPO loss, LoRA gradients): same comparisons; speed
  3. exactness: trim vs cut (expected: identical), and the values the trim puts
     back at the padded positions replaced by garbage (expected: nothing changes)
  4. whole step: per-token log-prob changes over every answer (tails included),
     and the summed step gradient of whole task groups

LoRA B is set to small random values (about what ~40 steps at LR 2e-5 give),
so the adapter path is active, not a no-op.

Usage (repo root, GPU free; parts 1-2 ~10 min, part 3 ~10 min):
    .venv/bin/python scripts/verify_trim_padding.py \
        runs/train/<run>/completions/completions_00001.parquet [--parts 3]
"""

from __future__ import annotations

import argparse
import collections
import inspect
import time
import types
from types import SimpleNamespace

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer
from trl.losses import FusedLinearGRPOLoss

from rlforge.train import chunked_logps
from rlforge.train.chunked_logps import ChunkedLogpsGRPOTrainer, trailing_padding
from rlforge.train.grpo_config import load_grpo_config


class _Acc:
    """Single-process stand-in for accelerate (what the trainer code calls)."""
    is_main_process = True
    unwrap_model = staticmethod(lambda m: m)
    gather = staticmethod(lambda x: x)


def _trainer(model, cfg) -> ChunkedLogpsGRPOTrainer:
    # Bypass __init__ (dataset, rewards, vLLM...): set only what the two
    # passes read, with the run's own values.
    t = object.__new__(ChunkedLogpsGRPOTrainer)
    t.logits_chunk_size = cfg.grpo.logits_chunk_size
    t.trim_trailing_padding = False
    t.temperature = cfg.grpo.temperature
    t.accelerator = _Acc()
    t.model_kwarg_keys = inspect.signature(model.get_base_model().forward).parameters.keys()
    t._is_vlm = False
    t._entropy_bonus_enabled = False
    t.model = model
    t.beta, t.loss_type = 0.0, cfg.grpo.loss_type
    t.current_gradient_accumulation_steps = 1
    t.args = SimpleNamespace(steps_per_generation=1)
    t._metrics = {"train": collections.defaultdict(list), "eval": collections.defaultdict(list)}
    # compiled=True as in training (TRL's default)
    t.liger_loss = FusedLinearGRPOLoss(
        beta=0.0, use_ref_model=False, epsilon_low=0.2, epsilon_high=0.2,
        loss_type=cfg.grpo.loss_type, max_completion_length=cfg.grpo.max_completion_length,
        importance_sampling_level="token", temperature=cfg.grpo.temperature)
    return t


def _timed(fn):
    torch.cuda.synchronize()
    s = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - s


def _rel(a, b):
    return ((a - b).float().norm() / b.float().norm()).item()


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0).item()


class Step:
    """The real model + one logged step laid out as TRL lays it out."""

    def __init__(self, completions: str, config: str):
        cfg = load_grpo_config(config)
        torch.manual_seed(0)
        tok = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, revision=cfg.model_revision, dtype=torch.bfloat16, device_map="cuda")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=cfg.lora.r, lora_alpha=cfg.lora.lora_alpha, lora_dropout=cfg.lora.lora_dropout,
            target_modules=cfg.lora.target_modules, task_type="CAUSAL_LM"))
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    p.normal_(0.0, 1e-3)
        model.train()  # as in TRL's passes (dropout is 0)
        self.model, self.t = model, _trainer(model, cfg)
        self.lora = [p for n, p in model.named_parameters() if "lora_" in n]

        d = pd.read_parquet(completions)
        prompts = [tok(p, add_special_tokens=False).input_ids for p in d.prompt]
        answers = [tok(c, add_special_tokens=False).input_ids[:cfg.grpo.max_completion_length]
                   for c in d.completion]
        self.pw, self.cw = max(map(len, prompts)), max(map(len, answers))
        ids = torch.full((len(d), self.pw + self.cw), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, (p, c) in enumerate(zip(prompts, answers, strict=True)):
            ids[i, self.pw - len(p):self.pw] = torch.tensor(p)
            ids[i, self.pw:self.pw + len(c)] = torch.tensor(c)
            mask[i, self.pw - len(p):self.pw + len(c)] = 1
        self.ids, self.mask = ids.cuda(), mask.cuda()
        self.pad_id = tok.pad_token_id
        self.lens = [len(c) for c in answers]
        self.advantages = d.advantage.tolist()  # TRL's logged advantages, row order kept
        self.order = sorted(range(len(d)), key=self.lens.__getitem__)
        print(f"step layout: {len(d)} answers, prompt width {self.pw}, answer width {self.cw}; "
              f"answer length mean {sum(self.lens) / len(self.lens):.0f}, "
              f"min {min(self.lens)}, max {max(self.lens)}")

    def spread(self, n, below_cap=False):
        """n answers spread over the length range (sorted, evenly spaced)."""
        order = [i for i in self.order if self.lens[i] < self.cw] if below_cap else self.order
        return [order[round(k * (len(order) - 1) / max(n - 1, 1))] for k in range(n)]

    def at_width(self, i, width):
        """Answer i laid out as in a step whose answers are padded to `width`."""
        if width == self.cw:
            return self.ids[i:i + 1], self.mask[i:i + 1]
        n = self.pw + min(self.lens[i], width)
        x = torch.full((1, self.pw + width), self.pad_id, dtype=torch.long, device="cuda")
        m = torch.zeros_like(x)
        x[:, :n], m[:, :n] = self.ids[i:i + 1, :n], self.mask[i:i + 1, :n]
        return x, m

    def other_width(self, i):
        w = -(-self.lens[i] // 1024) * 1024  # length rounded up to 1024
        return w if w < self.cw else self.cw + 1024

    def old_logps(self, i, trim, width=None):
        """TRL's no-grad log-prob pass (ours when trim=True)."""
        self.t.trim_trailing_padding = trim
        width = width or self.cw
        x, m = self.at_width(i, width)
        with torch.no_grad():
            return self.t._get_per_token_logps_and_entropies(self.model, x, m, width, batch_size=1)[0]

    def train_pass(self, i, trim, adv, noise, width=None):
        """Old log-probs of the same mode and width, then TRL's Liger loss and
        backward. Returns (loss, LoRA gradient, old log-probs)."""
        width = width or self.cw
        old = self.old_logps(i, trim, width)
        x, m = self.at_width(i, width)
        real = m[:, self.pw:]
        is_ratio = torch.exp(-noise[:, :width] * real).clamp(max=3.0)  # as TRL builds it
        self.t.trim_trailing_padding = trim
        inputs = {"prompt_ids": x[:, :self.pw], "prompt_mask": m[:, :self.pw],
                  "completion_ids": x[:, self.pw:], "completion_mask": real,
                  "advantages": adv, "old_per_token_logps": old,
                  "importance_sampling_ratio": is_ratio}
        self.model.zero_grad(set_to_none=True)
        loss = GRPOTrainer.compute_liger_loss(self.t, self.model, inputs)
        loss.backward()
        return loss.detach(), torch.cat([p.grad.flatten() for p in self.lora]).float(), old


def part1(s: Step, n_answers: int) -> None:
    print("\n1. no-grad log-prob pass (log-prob differences on real answer tokens; max / mean)")
    print(f"{'answer':>6} {'length':>6} {'trimmed':>7} | {'trim vs TRL':>19} | "
          f"{'TRL other width':>19} | {'TRL again':>19} | {'time TRL':>8} {'trim':>6}")
    tot, worst = collections.defaultdict(float), collections.defaultdict(float)
    n_tok, t_ref, t_trim = 0, 0.0, 0.0
    for i in s.spread(n_answers):
        n = s.lens[i]
        ref, s_ref = _timed(lambda: s.old_logps(i, False))
        got, s_trim = _timed(lambda: s.old_logps(i, True))
        assert torch.isfinite(got).all(), "non-finite log-prob after trimming"
        diffs = {"trim": got, "width": s.old_logps(i, False, s.other_width(i)),
                 "again": s.old_logps(i, False)}
        row = []
        for k, x in diffs.items():
            dd = (x[:, :n] - ref[:, :n]).abs().float()
            tot[k] += dd.sum().item()
            worst[k] = max(worst[k], dd.max().item())
            row.append(f"{dd.max().item():>9.2e} {dd.mean().item():>9.2e}")
        n_tok, t_ref, t_trim = n_tok + n, t_ref + s_ref, t_trim + s_trim
        print(f"{i:>6} {n:>6} {trailing_padding(s.mask[i:i + 1], s.cw):>7} | " + " | ".join(row)
              + f" | {s_ref:>7.1f}s {s_trim:>5.1f}s")
    print("all: " + " | ".join(f"{k} max {worst[k]:.2e} mean {tot[k] / n_tok:.2e}" for k in tot)
          + f" | time {t_ref:.0f}s -> {t_trim:.0f}s")


def _noise_adv(s: Step, k: int):
    noise = 0.05 * torch.randn(1, s.cw + 1024, device="cuda",
                               generator=torch.Generator("cuda").manual_seed(k))
    adv = torch.tensor([0.7 if k % 2 == 0 else -0.6], device="cuda")
    return noise, adv


def part2(s: Step, n_answers: int) -> None:
    print("\n2. training pass (Liger GRPO loss and LoRA gradients, one answer per micro-batch)")
    print("gradient: relative difference / cosine, against TRL")
    print(f"{'answer':>6} {'length':>6} | {'loss TRL':>10} {'loss trim':>10} | {'trim':>17} | "
          f"{'TRL other width':>17} | {'TRL again':>17} | {'time TRL':>8} {'trim':>6}")
    for k, i in enumerate(s.spread(n_answers)):
        noise, adv = _noise_adv(s, k)
        (l_ref, g_ref, _), s_ref = _timed(lambda: s.train_pass(i, False, adv, noise))
        (l_trim, g_trim, _), s_trim = _timed(lambda: s.train_pass(i, True, adv, noise))
        g_w = s.train_pass(i, False, adv, noise, s.other_width(i))[1]
        g_again = s.train_pass(i, False, adv, noise)[1]
        cols = " | ".join(f"{_rel(g, g_ref):>8.2e} {_cos(g, g_ref):>8.5f}" for g in (g_trim, g_w, g_again))
        print(f"{i:>6} {s.lens[i]:>6} | {l_ref.item():>10.6f} {l_trim.item():>10.6f} | {cols} | "
              f"{s_ref:>7.1f}s {s_trim:>5.1f}s")


def part3(s: Step, n_answers: int) -> None:
    """Exactness. `cut` = TRL's own code on the answer with no trailing padding.
    The trim runs TRL's model call on exactly that input, so everything that is
    deterministic must be identical; gradients are compared with the GPU's own
    run-to-run noise (cut computed twice), since the backward kernels are not
    deterministic."""
    print("\n3a. trim vs cut (TRL on the same answer without trailing padding)")
    print(f"{'answer':>6} {'length':>6} | {'old log-probs':>22} | {'loss':>22} | "
          f"{'gradient trim vs cut':>20} | {'cut vs cut again':>20}")
    for k, i in enumerate(s.spread(n_answers, below_cap=True)):
        n = s.lens[i]
        noise, adv = _noise_adv(s, k)
        l_trim, g_trim, o_trim = s.train_pass(i, True, adv, noise)
        l_cut, g_cut, o_cut = s.train_pass(i, False, adv, noise, n)
        l_cut2, g_cut2, _ = s.train_pass(i, False, adv, noise, n)
        o_eq = torch.equal(o_trim[:, :n], o_cut)
        o_d = (o_trim[:, :n] - o_cut).abs().max().item()
        l_eq = torch.equal(l_trim, l_cut)
        print(f"{i:>6} {n:>6} | {'IDENTICAL' if o_eq else 'differ':>10} max {o_d:.1e} | "
              f"{'IDENTICAL' if l_eq else 'differ':>10} {(l_trim - l_cut).abs().item():.1e} | "
              f"{_rel(g_trim, g_cut):>9.2e} {_cos(g_trim, g_cut):.5f} | "
              f"{_rel(g_cut2, g_cut):>9.2e} {_cos(g_cut2, g_cut):.5f}"
              + ("" if l_eq == torch.equal(l_cut2, l_cut) else "  (loss not repeatable)"))

    print("\n3b. values put back at the padded positions: zeros vs large random numbers")
    print("    (if anything read those positions, the results would change)")
    print(f"{'answer':>6} {'length':>6} | {'old log-probs':>13} | {'loss':>9} | "
          f"{'gradient zeros vs garbage':>25} | {'zeros vs zeros again':>20}")
    real_pad = chunked_logps.F.pad

    def garbage_pad(x, pad, *a, **k):
        out = real_pad(x, pad, *a, **k)
        n_pad = pad[-1]
        g = torch.Generator(device=out.device).manual_seed(7)  # same garbage in both passes
        junk = 50.0 * torch.randn(out[:, -n_pad:].shape, generator=g, device=out.device, dtype=out.dtype)
        return torch.cat([out[:, :-n_pad], junk], dim=1)

    for k, i in enumerate(s.spread(n_answers, below_cap=True)):
        n = s.lens[i]
        noise, adv = _noise_adv(s, k)
        l0, g0, o0 = s.train_pass(i, True, adv, noise)
        _, g0b, _ = s.train_pass(i, True, adv, noise)
        chunked_logps.F = types.SimpleNamespace(pad=garbage_pad)  # only the trim's call
        try:
            l1, g1, o1 = s.train_pass(i, True, adv, noise)
        finally:
            chunked_logps.F = torch.nn.functional
        o_eq = torch.equal(o0[:, :n], o1[:, :n])
        print(f"{i:>6} {n:>6} | {'IDENTICAL' if o_eq else 'DIFFER':>13} | "
              f"{'IDENTICAL' if torch.equal(l0, l1) else 'DIFFER':>9} | "
              f"{_rel(g1, g0):>14.2e} {_cos(g1, g0):.5f} | {_rel(g0b, g0):>9.2e} {_cos(g0b, g0):.5f}")


def part4(s: Step, n_groups: int, sub: set, modes: str = "all") -> None:
    """Whole step, as the run sees it. In a run the trim and TRL differ by
    rounding; how big is that at the level of a full step, compared with what
    TRL itself does when the same step is padded 1024 positions wider, or simply
    computed again?
      a. old log-probs of EVERY answer: distribution of the per-token change,
         tails included (the logged max metrics are single extreme tokens);
      b. the step gradient (sum over answers, real advantages) of `n_groups`
         task groups: norm, relative difference, cosine.
    IS ratio = 1 here (vLLM's log-probs are not saved); the comparison between
    the modes is what matters."""
    wide = s.cw + 1024
    if "4a" in sub:
        _part4a(s, wide)
    if "4b" in sub:
        _part4b(s, n_groups, wide, modes)


def _part4a(s: Step, wide: int) -> None:
    print("\n4a. old log-probs, every answer of the step: |change| per token vs TRL")
    d_trim, d_wide = [], []
    for k, i in enumerate(s.order, 1):
        n = s.lens[i]
        ref = s.old_logps(i, False)[:, :n]
        d_trim.append((s.old_logps(i, True)[:, :n] - ref).abs().flatten().float().cpu())
        d_wide.append((s.old_logps(i, False, wide)[:, :n] - ref).abs().flatten().float().cpu())
        if k % 8 == 0:
            print(f"  {k}/{len(s.order)} answers done; max so far: trim {torch.cat(d_trim).max().item():.3f}, "
                  f"TRL +1024 {torch.cat(d_wide).max().item():.3f}", flush=True)
    q = [0.5, 0.9, 0.99, 0.999, 0.9999]
    print(f"{'':<22} {'tokens':>8} {'median':>9} {'90%':>9} {'99%':>9} {'99.9%':>9} {'99.99%':>9} "
          f"{'max':>7} {'>0.5':>5} {'>1':>4} {'>2':>4}")
    for name, d in (("trim", d_trim), ("TRL +1024 padding", d_wide)):
        d = torch.cat(d)
        # torch.quantile has an input-size limit; sort once instead
        srt = d.sort().values
        qs = [srt[min(int(x * len(srt)), len(srt) - 1)].item() for x in q]
        print(f"{name:<22} {len(d):>8} " + " ".join(f"{v:>9.2e}" for v in qs)
              + f" {d.max().item():>7.3f} {(d > 0.5).sum().item():>5} {(d > 1).sum().item():>4} "
              f"{(d > 2).sum().item():>4}")


def _part4b(s: Step, n_groups: int, wide: int, modes_sel: str = "all") -> None:
    print(f"\n4b. step gradient over {n_groups} task groups (real advantages)")
    adv_all = torch.tensor(s.advantages, device="cuda", dtype=torch.float32)  # as TRL
    groups = [g for g in range(len(s.lens) // 8) if adv_all[8 * g:8 * g + 8].abs().sum() > 0][:n_groups]
    # width None = the step's width; "narrow" = this answer's length rounded up to
    # 1024 (TRL in a step whose longest answer is shorter: the yardstick that
    # moves the GPU's matrix-multiply rounding the way the trim does; padding
    # 1024 WIDER does not, it only shows the backward pass's run-to-run noise)
    modes = {"TRL": (False, None), "TRL again": (False, None), "trim": (True, None),
             "TRL +1024 padding": (False, wide), "TRL narrower width": (False, "narrow")}
    if modes_sel != "all":
        modes = {m: v for m, v in modes.items() if m in modes_sel.split(",")}
    assert "TRL" in modes, "TRL is the reference"
    total = {m: None for m in modes}
    zero_noise = torch.zeros(1, wide, device="cuda")  # IS ratio exactly 1
    for g in groups:
        for i in range(8 * g, 8 * g + 8):
            for m, (trim, width) in modes.items():
                w = s.other_width(i) if width == "narrow" else width
                grad = s.train_pass(i, trim, adv_all[i:i + 1], zero_noise, w)[1]
                total[m] = grad if total[m] is None else total[m] + grad
        print(f"  group {g} done", flush=True)
    ref = total["TRL"]
    print(f"{'':<20} {'norm':>10} {'rel diff vs TRL':>16} {'cosine':>9}")
    for m, gsum in total.items():
        print(f"{m:<20} {gsum.norm().item():>10.4e} {_rel(gsum, ref):>16.2e} {_cos(gsum, ref):>9.6f}")
    if "trim" in total and "TRL narrower width" in total:
        a, b = total["trim"], total["TRL narrower width"]
        print(f"trim vs TRL narrower width: rel diff {_rel(a, b):.2e}, cosine {_cos(a, b):.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("completions", help="completions_0000N.parquet of a training run")
    ap.add_argument("--config", default="configs/grpo-vanilla-run3.yaml")
    ap.add_argument("--n-logps", type=int, default=8, help="answers checked in part 1")
    ap.add_argument("--n-train", type=int, default=3, help="answers checked in part 2")
    ap.add_argument("--n-exact", type=int, default=4, help="answers checked in part 3")
    ap.add_argument("--n-groups", type=int, default=2, help="task groups in part 4b")
    ap.add_argument("--parts", default="1,2,3", help="parts to run: 1,2,3,4a,4b")
    ap.add_argument("--modes-4b", default="all", help="comma list of part-4b modes (TRL always needed)")
    args = ap.parse_args()
    parts = set(args.parts.split(","))

    s = Step(args.completions, args.config)
    # warm-up (kernel autotuning / compilation) on the shortest answer, both modes
    s.old_logps(s.order[0], False)
    s.old_logps(s.order[0], True)
    if "1" in parts:
        part1(s, args.n_logps)
    if "2" in parts:
        part2(s, args.n_train)
    if "3" in parts:
        part3(s, args.n_exact)
    if parts & {"4a", "4b"}:
        part4(s, args.n_groups, parts, args.modes_4b)
    print(f"\npeak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
