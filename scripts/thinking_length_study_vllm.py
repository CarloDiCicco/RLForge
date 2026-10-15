"""Thinking-mode length study on vLLM: same study as thinking_length_study.py,
generated with vLLM instead of Hugging Face generate().

Why a second version: HF generate() runs a batch in lockstep (a finished answer
keeps its slot until the longest one ends) and slows down as answers grow, so a
32k thinking study takes most of a day. vLLM gives a freed slot to the next
answer immediately and keeps per-token cost flat, which is the same speed-up it
would bring to GRPO rollouts and to thinking-mode evaluation — so this run also
measures that speed on this machine.

What is the same as the HF version (so the two are comparable):
- model and pinned revision, the training prompt (build_prompt_for_task), chat
  template with enable_thinking=True, no system prompt;
- sampling: temperature 0.6, top_p 0.95, top_k 20, min_p 0, no penalties;
- grading: only the text after </think>; no </think> = no answer;
- record format and summary (pass rate, cut-off share, lengths, cap curve).

What is new: EVERY answer's text is saved, including cut-off ones, plus a
`repeated` score (share of 30-word windows seen earlier in the same answer), so
long-but-real reasoning can be told apart from loops.

Must run in the separate vLLM environment (vLLM needs torch 2.13; the training
environment stays on 2.12):
    ~/data/venvs/vllm-test/bin/python scripts/thinking_length_study_vllm.py \
        --rounds 4 --max-new-tokens 32768
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlforge.adapters.task_loader import load_tasks  # noqa: E402
from rlforge.prompt_template import build_prompt_for_task  # noqa: E402
from rlforge.sandbox.base import SandboxConfig  # noqa: E402
from rlforge.train.reward import VerifierReward  # noqa: E402

THINK_END = "</think>"
CURVE_CAPS = [2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768]


def split_thinking(text: str) -> tuple[str, str, bool]:
    """Return (reasoning, answer, closed). closed=False: no answer was produced."""
    if THINK_END not in text:
        return text, "", False
    reasoning, answer = text.split(THINK_END, 1)
    return reasoning, answer, True


def repeated_share(text: str, n: int = 30, stride: int = 10) -> float:
    """Share of n-word windows that already appeared earlier in the text.
    ~0 for normal reasoning; high when the model loops on the same passage."""
    words = text.split()
    grams = [" ".join(words[i:i + n]) for i in range(0, max(0, len(words) - n), stride)]
    seen: set[str] = set()
    dup = 0
    for g in grams:
        dup += g in seen
        seen.add(g)
    return dup / max(1, len(grams))


def curve(records: list[dict], max_cap: int) -> list[dict]:
    """Pass rate and finished share at each cap, from one long run: an answer that
    finished within C tokens would have been identical under a cap of C."""
    n = len(records)
    rows = []
    for cap in [c for c in CURVE_CAPS if c <= max_cap]:
        within = [r for r in records if not r["hit_token_cap"] and r["completion_tokens"] <= cap]
        rows.append({"cap": cap, "pass_rate": sum(r["passed"] for r in within) / n,
                     "finished": len(within) / n})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks-file", default="data/livecodebench_leetcode.jsonl")
    ap.add_argument("--split", default="data/splits/lcb_leetcode_v2_val.json")
    ap.add_argument("--rounds", type=int, default=4, help="answers per task")
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    # Sampling knobs, to compare settings against looping. Defaults = Qwen's
    # thinking-mode "precise coding" row; the "general" row is temperature 1.0,
    # presence_penalty 1.5 (model card: 0-2 reduces endless repetitions).
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks (quick tests)")
    ap.add_argument("--tag", default="", help="suffix for the output folder name")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5,
                    help="share of the 121 GB unified memory vLLM may reserve")
    args = ap.parse_args()

    # vLLM compiles some GPU code on first use with `ninja`, which lives in this
    # environment's bin/ folder. Started by full path (no `activate`), that folder
    # is not on PATH and start-up fails with "No such file: ninja"; add it here.
    import os
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    from vllm import LLM, SamplingParams  # imported late: only the vLLM env has it

    split_ids = json.load(open(args.split))
    if "test" in Path(args.split).name:
        sys.exit("refusing to run on the test split: the cap must be chosen without it")
    tasks = {t.task_id: t for t in load_tasks(args.tasks_file) if t.task_id in set(split_ids)}
    ids = sorted(tasks)[: args.limit or None]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path("runs/eval") / f"{stamp}-thinking-length-study-vllm{('-' + args.tag) if args.tag else ''}"
    out.mkdir(parents=True, exist_ok=False)
    import torch
    import vllm
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "scripts"],
                                capture_output=True, text=True).stdout.strip())
    sampling = {"temperature": args.temperature, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                "presence_penalty": args.presence_penalty, "repetition_penalty": 1.0}
    meta = {"args": vars(args), "sampling": sampling, "enable_thinking": True, "engine": "vllm",
            "vllm": vllm.__version__, "torch": torch.__version__, "n_tasks": len(ids),
            "git_commit": git, "git_dirty": dirty, "started_at": stamp}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[length-study-vllm] {len(ids)} tasks x {args.rounds} rounds, cap {args.max_new_tokens}, -> {out}",
          flush=True)

    llm = LLM("Qwen/Qwen3.5-4B", revision=args.revision, seed=args.seed,
              gpu_memory_utilization=args.gpu_memory_utilization,
              # prompt (<1k tokens) + the full answer must fit in one sequence
              max_model_len=args.max_new_tokens + 2048)
    grader = VerifierReward(tasks, SandboxConfig(backend="bwrap", timeout_s=60.0, cpu_s=60, mem_mb=8192))
    tok = llm.get_tokenizer()

    messages = [[{"role": "user", "content": build_prompt_for_task(tasks[tid])[0]}] for tid in ids]
    records = []
    gen_s = 0.0
    # One vLLM call per round, saved right after it: a crash loses at most the
    # round in progress. (One call for all rounds would be slightly faster, but
    # would lose everything on a crash.)
    for rnd in range(args.rounds):
        # Own seed per answer so rounds are different samples, and reproducible.
        params = [SamplingParams(max_tokens=args.max_new_tokens, seed=args.seed + 1000 * rnd + i,
                                 **sampling)
                  for i in range(len(ids))]
        t0 = time.time()
        outputs = llm.chat(messages, params, chat_template_kwargs={"enable_thinking": True}, use_tqdm=True)
        gen_s += time.time() - t0
        rows = grade_round(rnd, ids, outputs, tasks, tok, grader)
        with open(out / "answers.jsonl", "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        records += rows
        print(f"[length-study-vllm] round {rnd + 1}/{args.rounds}: "
              f"{sum(r['passed'] for r in rows)}/{len(rows)} passed, "
              f"{sum(r['hit_token_cap'] for r in rows)} cut off, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    write_summary(records, gen_s, args, out)


def grade_round(rnd, ids, outputs, tasks, tok, grader) -> list[dict]:
    """Grade one round: only the text after </think> counts as the answer."""
    rows = []
    for tid, o in zip(ids, outputs):
        c = o.outputs[0]
        text = c.text
        reasoning, answer, closed = split_thinking(text)
        rec = {"round": rnd, "task_id": tid, "difficulty": tasks[tid].metadata.get("difficulty"),
               "completion_tokens": len(c.token_ids), "hit_token_cap": c.finish_reason == "length",
               "closed_thinking": closed,
               "answer_tokens": len(tok.encode(answer, add_special_tokens=False)) if closed else 0,
               "repeated": round(repeated_share(text), 4)}
        if closed:
            g = grader.grade(tid, answer)
            rec.update(passed=g["correctness"] == 1.0, failure_type=g["failure_type"])
        else:
            rec.update(passed=False, failure_type="no_answer")
        rec["text"] = text
        rows.append(rec)
    return rows


def write_summary(records: list[dict], gen_s: float, args, out: Path) -> None:
    """Pass rate, cut-offs (and how many of them loop), lengths, cap curve, speed."""
    lengths = sorted(r["completion_tokens"] for r in records if not r["hit_token_cap"])
    pct = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))] if lengths else None  # noqa: E731
    cut = [r for r in records if r["hit_token_cap"]]
    summary = {
        "answers": len(records),
        "pass_rate": sum(r["passed"] for r in records) / len(records),
        "cut_off": len(cut) / len(records),
        "cut_off_looping": sum(r["repeated"] > 0.2 for r in cut),
        "finished_length_p50_p90_p99": [pct(0.5), pct(0.9), pct(0.99)],
        "curve": curve(records, args.max_new_tokens),
        "generation_minutes": gen_s / 60,
        "tokens_per_s": sum(r["completion_tokens"] for r in records) / gen_s,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
