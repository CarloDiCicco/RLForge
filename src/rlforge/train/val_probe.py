"""
train/val_probe.py
──────────────────
Held-out validation probe run DURING training. Shared by every algorithm script.

Why it exists. `train/reward` is not a learning curve: each optimizer step draws
different tasks, so the number moves with task difficulty as much as with the
policy (RESEARCH_LOG 2026-09-19 and 2026-09-26). This probe scores the SAME
validation tasks, with the SAME random seed, at fixed step intervals. Tasks and
sampling noise are held constant, so a change between two probes comes from the
weights.

What it does at each probe step:
  1. samples `samples_per_task` answers per validation task from the current
     policy, in chat format, at the frozen baseline sampling settings;
  2. grades each answer with the same verifier the reward and the benchmarks use;
  3. writes one line per answer to `val_executions.jsonl`, one summary line per
     probe to `val_summary.jsonl`, and the summary to TensorBoard under `eval/`.

It never touches training state: the model's train/eval mode and the torch RNG
streams are restored afterwards, so a run with the probe draws the same training
rollouts as the same run without it.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import torch
from transformers import TrainerCallback
from transformers.generation.logits_process import LogitsProcessorList

from rlforge.models.qwen import _PresencePenaltyLogitsProcessor
from rlforge.train.grpo_config import ValidationBlock
from rlforge.train.reward import VerifierReward

# (text, completion_tokens, hit_token_cap) for one sampled answer.
Sample = tuple[str, int, bool]
GenerateFn = Callable[[object, list[str]], list[Sample]]


def generate_chat_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    cfg: ValidationBlock,
    enable_thinking: bool,
    max_new_tokens: int,
    stopping_criteria=None,
) -> list[Sample]:
    """Sample one answer per prompt, batched, in the eval pipeline's input format.

    `stopping_criteria` (optional) is passed straight to generate(); used by the
    thinking length study to report progress and save each answer as it ends.

    Mirrors models/qwen.py: one user message, generation prompt, the same
    `enable_thinking` switch and the same presence-penalty processor. The
    difference is batching, which needs LEFT padding: a decoder-only model continues
    from the last position, so padding on the right would put pad tokens between
    the prompt and the answer.
    """
    samples: list[Sample] = []
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), cfg.batch_size):
            batch = prompts[start : start + cfg.batch_size]
            enc = tokenizer.apply_chat_template(
                [[{"role": "user", "content": p}] for p in batch],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                enable_thinking=enable_thinking,
            ).to(model.device)
            # Left padding makes every row's prompt end at the same index, so one
            # prompt length is valid for the whole batch.
            prompt_len = enc["input_ids"].shape[1]

            processors = LogitsProcessorList()
            if cfg.presence_penalty > 0.0:
                processors.append(
                    _PresencePenaltyLogitsProcessor(cfg.presence_penalty, prompt_len)
                )
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    logits_processor=processors,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    # Gradient checkpointing can leave use_cache off on the model
                    # config; without the cache, generation is quadratic in length.
                    use_cache=True,
                    # Optional per-step hook (progress, saving answers as they
                    # finish). It only observes; it never stops generation early.
                    stopping_criteria=stopping_criteria,
                )
            for row in out[:, prompt_len:]:
                ids = row.tolist()
                # No EOS in the new tokens = cut off at max_new_tokens, the same
                # definition of "truncated" the eval pipeline uses.
                hit_cap = tokenizer.eos_token_id not in ids
                n_tokens = len(ids) if hit_cap else ids.index(tokenizer.eos_token_id) + 1
                text = tokenizer.decode(row, skip_special_tokens=True)
                samples.append((text, n_tokens, hit_cap))
    finally:
        tokenizer.padding_side = old_side
    return samples


def make_vllm_generate_fn(get_trainer, cfg: ValidationBlock, *, enable_thinking: bool,
                          max_new_tokens: int) -> GenerateFn:
    """Probe generation on the trainer's own colocated vLLM engine.

    Why: at thinking-mode lengths (tens of thousands of tokens) HF generate() takes
    hours per probe, vLLM about ten times less (length study 2026-10-14). Reusing
    the trainer's engine costs no extra memory.

    Correctness points, each easy to get silently wrong:
      - The engine must hold the CURRENT policy. TRL pushes weights into vLLM only
        when it generates rollouts, i.e. at the START of a step, so at the end of a
        step vLLM still holds the weights from before that step's update. The
        weights are pushed here first (and always after a sleep, which discards
        them).
      - Same seed scheme in every probe (seed + 1000 * round + task index), so
        sampling noise is held constant across probes, as in the HF path.
      - Sleep mode is restored afterwards, so the training loop finds the engine
        in the state it left it.
    """
    from vllm import SamplingParams  # only importable in the vLLM environment

    calls = {"n": 0}

    def generate(model, prompts: list[str]) -> list[Sample]:  # noqa: ANN001
        trainer = get_trainer()
        vg = trainer.vllm_generation
        step = trainer.state.global_step
        if vg._llm_weights_sleeping or trainer._last_loaded_step != step:
            vg.sync_weights()
            trainer._last_loaded_step = step
        if vg.enable_sleep_mode:
            vg.llm.wake_up(tags=["kv_cache"])
        rnd = calls["n"] % cfg.samples_per_task
        calls["n"] += 1
        try:
            params = [
                SamplingParams(
                    max_tokens=max_new_tokens, temperature=cfg.temperature, top_p=cfg.top_p,
                    top_k=cfg.top_k, min_p=cfg.min_p, presence_penalty=cfg.presence_penalty,
                    seed=cfg.seed + 1000 * rnd + i,
                )
                for i in range(len(prompts))
            ]
            outputs = vg.llm.chat(
                [[{"role": "user", "content": p}] for p in prompts], params,
                chat_template_kwargs={"enable_thinking": enable_thinking}, use_tqdm=False,
            )
        finally:
            if vg.enable_sleep_mode:
                vg.llm.sleep(level=2)
                vg._llm_weights_sleeping = True
        return [
            (o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason == "length")
            for o in outputs
        ]

    return generate


def summarize(records: list[dict], samples_per_task: int) -> dict[str, float]:
    """Aggregate one probe's per-answer records into the logged numbers."""
    n = len(records)
    by_task: dict[str, list[float]] = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r["correctness"])
    n_tasks = len(by_task)
    passes = [sum(v) for v in by_task.values()]
    return {
        # Mean pass rate over all sampled answers: the headline number.
        "pass_rate": sum(r["correctness"] for r in records) / n,
        # Share of tasks with at least one passing answer.
        "tasks_solved_any": sum(p > 0 for p in passes) / n_tasks,
        # Share of tasks solved by every sampled answer.
        "tasks_solved_all": sum(p == samples_per_task for p in passes) / n_tasks,
        "truncated_rate": sum(r["hit_token_cap"] for r in records) / n,
        "extraction_fail_rate": sum(r["failure_type"] == "extraction" for r in records) / n,
        # Thinking mode: answers cut off before "</think>" (no answer at all).
        "no_answer_rate": sum(r["failure_type"] == "no_answer" for r in records) / n,
        "mean_completion_tokens": sum(r["completion_tokens"] for r in records) / n,
    }


class ValidationProbe(TrainerCallback):
    """Run the probe at step 0, every `every_steps`, and at the final step."""

    def __init__(
        self,
        *,
        prompts: dict[str, str],
        grader: VerifierReward,
        tokenizer,
        cfg: ValidationBlock,
        enable_thinking: bool,
        max_new_tokens: int,
        out_dir: Path,
        generate_fn: GenerateFn | None = None,
    ) -> None:
        """
        Args:
            prompts: validation task_id -> final prompt text (already templated).
            grader: a VerifierReward built on the validation tasks, with no log
                path of its own, so probe answers never mix into the training
                execution log.
            generate_fn: test seam. Defaults to real batched generation.
        """
        if not prompts:
            raise ValueError("validation probe needs at least one task")
        self._task_ids = sorted(prompts)
        self._prompts = prompts
        self._grader = grader
        self._cfg = cfg
        self._exec_path = Path(out_dir) / "val_executions.jsonl"
        self._summary_path = Path(out_dir) / "val_summary.jsonl"
        self._last_probed_step: int | None = None
        # Set by the training script once the trainer exists, to reach its
        # TensorBoard writer. Optional: the JSONL files are the record either way.
        self.trainer = None
        self._generate: GenerateFn = generate_fn or (
            lambda model, batch: generate_chat_batch(
                model, tokenizer, batch, cfg=cfg,
                enable_thinking=enable_thinking, max_new_tokens=max_new_tokens,
            )
        )

    # ── schedule ──────────────────────────────────────────────────────────────

    def on_train_begin(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
        # Step 0 = the untrained policy, measured by the same procedure as every
        # later point. This is the reference the curve is read against.
        self.run(state.global_step, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
        step = state.global_step
        if step % self._cfg.every_steps == 0 or step >= state.max_steps:
            self.run(step, model)

    # ── one probe ─────────────────────────────────────────────────────────────

    def run(self, step: int, model) -> dict[str, float] | None:  # noqa: ANN001
        if step == self._last_probed_step:
            return None  # the final step can also be a multiple of every_steps
        self._last_probed_step = step
        k = self._cfg.samples_per_task
        n_tasks = len(self._task_ids)
        print(f"\n[val] step {step}: {k} rounds, each = one answer for each of the "
              f"{n_tasks} validation tasks", flush=True)
        t0 = time.time()
        t_gen = 0.0
        records: list[dict] = []
        prompts = [self._prompts[tid] for tid in self._task_ids]

        was_training = model.training
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            model.eval()
            torch.manual_seed(self._cfg.seed)
            # One ROUND = one answer per task. Rounds are generated and graded one
            # at a time so progress is visible and written to disk as it happens:
            # a full probe takes tens of minutes, and silence that long is
            # indistinguishable from a hang.
            for rnd in range(k):
                t_round = time.time()
                samples = self._generate(model, prompts)
                t_gen += time.time() - t_round
                with ThreadPoolExecutor(max_workers=self._grader.max_workers) as pool:
                    graded = list(pool.map(
                        self._grader.grade, self._task_ids, [s[0] for s in samples]))
                round_records = [
                    {
                        "step": step, "task_id": tid, "sample": rnd,
                        "correctness": g["correctness"], "failure_type": g["failure_type"],
                        "extraction_status": g["extraction_status"],
                        "exec_time_s": g["exec_time_s"], "code_chars": g["code_chars"],
                        "completion_tokens": n_tokens, "hit_token_cap": hit_cap,
                        "stderr": (g["stderr"] or "")[-2000:],
                    }
                    for tid, (_text, n_tokens, hit_cap), g in zip(
                        self._task_ids, samples, graded, strict=True)
                ]
                records.extend(round_records)
                self._append(self._exec_path, round_records)
                n_pass = sum(r["correctness"] for r in round_records)
                n_cut = sum(r["hit_token_cap"] for r in round_records)
                print(f"[val] step {step}: round {rnd + 1}/{k}  {n_pass:.0f}/{n_tasks} passed  "
                      f"{n_cut} cut off at the token limit  "
                      f"({(time.time() - t_round) / 60:.1f} min)", flush=True)
        finally:
            # Hand the training loop back exactly the state it had.
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            model.train(was_training)

        metrics = summarize(records, k)
        summary = {"step": step, "n_tasks": len(self._task_ids), "samples_per_task": k,
                   "gen_seconds": round(t_gen, 1),
                   "total_seconds": round(time.time() - t0, 1), **metrics}
        self._append(self._summary_path, [summary])

        self._to_tensorboard(step, metrics)
        # The peak-memory counters are deliberately NOT reset here. The training
        # script reads them right after this callback, so a reset would make that
        # step's VRAM reading miss the training step's own peak (seen in the wiring
        # check, 2026-10-10). Left alone, a probe step reports the true maximum of
        # the step and the probe together.
        print(f"[val] step {step} RESULT: pass rate {metrics['pass_rate']:.3f} "
              f"({sum(r['correctness'] for r in records):.0f}/{len(records)} answers)  "
              f"tasks solved at least once {metrics['tasks_solved_any']:.0%}  "
              f"cut off {metrics['truncated_rate']:.0%}  "
              f"({summary['total_seconds'] / 60:.1f} min)", flush=True)
        return metrics

    @staticmethod
    def _append(path: Path, rows: list[dict]) -> None:
        """Append JSON lines. Logging must never kill a training run."""
        try:
            with path.open("a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
        except OSError:
            pass

    def _to_tensorboard(self, step: int, metrics: dict[str, float]) -> None:
        """Write under `eval/` with the trainer's own TensorBoard writer.

        Deliberately not `trainer.log(...)`: TRL's log() merges and then CLEARS the
        training metrics it has accumulated for the current step, so calling it
        from here would silently drop that step's reward and length statistics.
        """
        if self.trainer is None:
            return
        for cb in self.trainer.callback_handler.callbacks:
            writer = getattr(cb, "tb_writer", None)
            if writer is not None:
                for key, value in metrics.items():
                    writer.add_scalar(f"eval/{key}", value, step)
                writer.flush()
                return
