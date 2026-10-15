"""Thinking-mode length study: how much room does Qwen3.5-4B need to reason?

Runs the UNTRAINED model in thinking mode with a generous cap (default 32k) on
validation tasks (never test), several answers per task, and records for every
answer how long it was, whether it finished, and whether it passed.

Why one long run is enough: an answer that finished within C tokens would have
finished identically under a cap of C (same seed, same tokens up to that point).
So "pass rate at cap C" = share of answers that passed AND used <= C tokens, for
every C below the run's cap, with no reruns. That curve picks the training cap.

Thinking-mode trap handled here: with enable_thinking=True the chat template ends
the PROMPT with "<think>\\n", so the completion contains only
"reasoning...</think>\\n\\nanswer". The extractor's <think>...</think> stripping
never matches, and a code block written while reasoning would be graded as the
answer. So we split on "</think>" ourselves: only the text after it is graded,
and an answer with no "</think>" (cut off mid-reasoning) has no answer at all.

Usage (from the repo root, on the DGX, inside tmux):
    python scripts/thinking_length_study.py --rounds 4 --max-new-tokens 32768
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

# Must be set before torch initialises CUDA. Growing per-token caches make the
# default allocator fragment; expandable segments let freed blocks be reused.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from transformers import StoppingCriteria, StoppingCriteriaList  # noqa: E402

from rlforge.adapters.task_loader import load_tasks  # noqa: E402
from rlforge.models.qwen import QwenModel  # noqa: E402
from rlforge.prompt_template import build_prompt_for_task  # noqa: E402
from rlforge.sandbox.base import SandboxConfig  # noqa: E402
from rlforge.train.reward import VerifierReward  # noqa: E402
from rlforge.train.val_probe import generate_chat_batch  # noqa: E402

THINK_END = "</think>"
# Caps at which the pass-rate-vs-cap curve is reported (must be <= --max-new-tokens).
CURVE_CAPS = [2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768]


def split_thinking(text: str) -> tuple[str, str, bool]:
    """Return (reasoning, answer, closed). closed=False: no answer was produced."""
    if THINK_END not in text:
        return text, "", False
    reasoning, answer = text.split(THINK_END, 1)
    return reasoning, answer, True


def curve(records: list[dict], max_cap: int) -> list[dict]:
    """Pass rate and finished share at each cap, from one long run (see docstring)."""
    n = len(records)
    rows = []
    for cap in [c for c in CURVE_CAPS if c <= max_cap]:
        within = [r for r in records if not r["hit_token_cap"] and r["completion_tokens"] <= cap]
        rows.append({
            "cap": cap,
            "pass_rate": sum(r["passed"] for r in within) / n,
            "finished": len(within) / n,
        })
    return rows


class LiveLog(StoppingCriteria):
    """Called by generate() after every new token. It never stops generation
    (always returns False); it only watches, so the run is no longer blind:

    - every `every` steps it prints the step, how many answers have ended, and speed;
    - the moment an answer ends (emits the end-of-turn token) its raw text is
      appended to finished_raw.jsonl, so a crash loses only unfinished answers.
    Grading happens after the batch, as before; these raw rows are the backup."""

    def __init__(self, tok, prompt_len: int, task_ids: list[str], rnd: int,
                 raw_path: Path, every: int = 500) -> None:
        self.tok, self.prompt_len, self.task_ids, self.rnd = tok, prompt_len, task_ids, rnd
        self.raw_path, self.every = raw_path, every
        self.done: set[int] = set()
        self.t0 = time.time()

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        step = input_ids.shape[1] - self.prompt_len
        last = input_ids[:, -1].tolist()
        for i, token in enumerate(last):
            if i not in self.done and token == self.tok.eos_token_id:
                self.done.add(i)
                text = self.tok.decode(input_ids[i, self.prompt_len:], skip_special_tokens=True)
                with open(self.raw_path, "a") as f:
                    f.write(json.dumps({"round": self.rnd, "task_id": self.task_ids[i],
                                        "completion_tokens": step, "text": text}) + "\n")
                print(f"[finished] round {self.rnd + 1} {self.task_ids[i]} at {step} tokens "
                      f"({len(self.done)}/{len(last)} done)", flush=True)
        if step % self.every == 0:
            secs = time.time() - self.t0
            print(f"[progress] round {self.rnd + 1} step {step}: {len(self.done)}/{len(last)} done, "
                  f"{step / secs:.2f} steps/s, {secs / 60:.0f} min", flush=True)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def _start_memory_monitor(every_s: int = 60) -> None:
    """Print GPU memory once a minute, so growth is visible before any failure and
    the run shows signs of life during rounds that last hours."""
    t0 = time.time()

    def loop() -> None:
        while True:
            time.sleep(every_s)
            gb = 1024 ** 3
            print(f"[mem] {(time.time() - t0) / 60:6.1f} min  "
                  f"allocated {torch.cuda.memory_allocated() / gb:5.1f} GB  "
                  f"reserved {torch.cuda.memory_reserved() / gb:5.1f} GB  "
                  f"peak {torch.cuda.max_memory_allocated() / gb:5.1f} GB", flush=True)

    threading.Thread(target=loop, daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks-file", default="data/livecodebench_leetcode.jsonl")
    ap.add_argument("--split", default="data/splits/lcb_leetcode_v2_val.json")
    ap.add_argument("--rounds", type=int, default=4, help="answers per task")
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--batch-size", type=int, default=22)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    ap.add_argument("--mem-fraction", type=float, default=0.6,
                    help="max share of the device this process may take")
    args = ap.parse_args()

    # The DGX Spark's GPU memory IS system memory. A run that takes all of it does
    # not fail alone: the kernel's OOM killer then kills the user session (tmux,
    # shells, everything). Capping PyTorch here turns that into a plain CUDA OOM
    # error in this process only. Happened on 2026-10-14 12:39 without the cap.
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    _start_memory_monitor()

    # Qwen's recommended thinking-mode settings for coding (model card:
    # temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence 0, repetition 1).
    sampling = SimpleNamespace(
        batch_size=args.batch_size, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0
    )

    split_ids = json.load(open(args.split))
    if "test" in Path(args.split).name:
        sys.exit("refusing to run on the test split: the cap must be chosen without it")
    tasks = {t.task_id: t for t in load_tasks(args.tasks_file) if t.task_id in set(split_ids)}
    ids = sorted(tasks)
    prompts = [build_prompt_for_task(tasks[t])[0] for t in ids]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path("runs/eval") / f"{stamp}-thinking-length-study"
    out.mkdir(parents=True, exist_ok=False)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "scripts"],
                                capture_output=True, text=True).stdout.strip())
    meta = {"args": vars(args), "sampling": vars(sampling), "enable_thinking": True,
            "n_tasks": len(ids), "git_commit": git, "git_dirty": dirty, "started_at": stamp,
            "torch": torch.__version__}
    try:
        import fla  # noqa: F401
        meta["flash_linear_attention"] = getattr(fla, "__version__", "installed")
    except ImportError:
        meta["flash_linear_attention"] = None
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[length-study] {len(ids)} tasks x {args.rounds} rounds, cap {args.max_new_tokens}, -> {out}")

    qm = QwenModel("Qwen/Qwen3.5-4B", revision=args.revision)
    tok = qm.tokenizer
    grader = VerifierReward(tasks, SandboxConfig(backend="bwrap", timeout_s=60.0, cpu_s=60, mem_mb=8192))

    records: list[dict] = []
    torch.manual_seed(args.seed)
    t0 = time.time()
    for rnd in range(args.rounds):
        t_round = time.time()
        round_rows = []
        # One batch at a time (instead of the whole round in one call) so graded
        # answers are written after every batch, not only at the end of a round.
        for start in range(0, len(ids), args.batch_size):
            b_ids, b_prompts = ids[start:start + args.batch_size], prompts[start:start + args.batch_size]
            # Left padding: every prompt in the batch ends at the same position.
            tok.padding_side = "left"
            prompt_len = tok.apply_chat_template(
                [[{"role": "user", "content": p}] for p in b_prompts], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt", padding=True,
                enable_thinking=True)["input_ids"].shape[1]
            live = LiveLog(tok, prompt_len, b_ids, rnd, out / "finished_raw.jsonl")
            samples = generate_chat_batch(qm.model, tok, b_prompts, cfg=sampling,
                                          enable_thinking=True, max_new_tokens=args.max_new_tokens,
                                          stopping_criteria=StoppingCriteriaList([live]))
            rows = grade_batch(b_ids, samples, rnd, tasks, tok, grader)
            with open(out / "answers.jsonl", "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"[batch] round {rnd + 1} tasks {start + 1}-{start + len(b_ids)}: "
                  f"{sum(r['passed'] for r in rows)}/{len(rows)} passed, "
                  f"{sum(r['hit_token_cap'] for r in rows)} cut off", flush=True)
            round_rows += rows
        rows = round_rows
        records += rows
        mins = (time.time() - t_round) / 60
        toks = sum(r["completion_tokens"] for r in rows)
        print(f"[length-study] round {rnd + 1}/{args.rounds}: "
              f"{sum(r['passed'] for r in rows)}/{len(rows)} passed, "
              f"{sum(r['hit_token_cap'] for r in rows)} cut off, "
              f"{mins:.1f} min, {toks / (mins * 60):.0f} tokens/s overall")

    write_summary(records, args, out, t0)


def grade_batch(b_ids, samples, rnd, tasks, tok, grader) -> list[dict]:
    """Grade one batch: only the text after </think> counts as the answer."""
    rows = []
    for tid, (text, n_tokens, hit_cap) in zip(b_ids, samples):
        reasoning, answer, closed = split_thinking(text)
        rec = {"round": rnd, "task_id": tid,
               "difficulty": tasks[tid].metadata.get("difficulty"),
               "completion_tokens": n_tokens, "hit_token_cap": hit_cap,
               "closed_thinking": closed,
               "answer_tokens": len(tok.encode(answer, add_special_tokens=False)) if closed else 0}
        if closed:
            g = grader.grade(tid, answer)
            rec.update(passed=g["correctness"] == 1.0, failure_type=g["failure_type"])
        else:
            rec.update(passed=False, failure_type="no_answer")
        rec["text"] = text
        rows.append(rec)
    return rows


def write_summary(records: list[dict], args, out: Path, t0: float) -> None:
    """Pass rate, cut-off share, finished-length percentiles and the cap curve."""
    lengths = sorted(r["completion_tokens"] for r in records if not r["hit_token_cap"])
    pct = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))] if lengths else None  # noqa: E731
    summary = {
        "answers": len(records),
        "pass_rate": sum(r["passed"] for r in records) / len(records),
        "cut_off": sum(r["hit_token_cap"] for r in records) / len(records),
        "finished_length_p50_p90_p99": [pct(0.5), pct(0.9), pct(0.99)],
        "curve": curve(records, args.max_new_tokens),
        "hours": (time.time() - t0) / 3600,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
