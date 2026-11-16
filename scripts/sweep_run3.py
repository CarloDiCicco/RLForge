#!/usr/bin/env python3
"""Run #3 sweep: find clean mixed training tasks, one task at a time.

For each pool task (one fixed random order, seed 0) the untrained model writes
4 answers with the TRAINING setting (thinking mode, temperature 1.0, top_p 0.95,
top_k 20, no presence penalty, 32k cap). The 4 answers are graded with the same
grader as training, and the task is appended to data/run3_sweep.jsonl the moment
it is done (all 4 answers, full text, verdicts).

"Clean mixed" = reference solution passes every test (gold check) AND 1-3 of the
4 answers pass. The sweep stops at --target clean mixed tasks.

The GPU is kept busy by keeping --in-flight tasks generating at once; each task
is still saved on its own as soon as its 4 answers are graded. Stop any time
(Ctrl-C); rerunning resumes, skipping tasks already in the file.

Usage: .venv/bin/python scripts/sweep_run3.py [--target 1000]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thinking_length_study_vllm import split_thinking  # noqa: E402  same answer split as the eval

from rlforge.adapters.task_loader import load_tasks  # noqa: E402
from rlforge.prompt_template import build_prompt_for_task  # noqa: E402
from rlforge.sandbox.base import SandboxConfig  # noqa: E402
from rlforge.train.reward import VerifierReward  # noqa: E402

POOL = "data/run3_pool.jsonl"
OUT = Path("data/run3_sweep.jsonl")
GOLD = Path("data/run3_gold.jsonl")
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # same pinned model as all evals
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                presence_penalty=0.0, repetition_penalty=1.0)


def gold_ok() -> dict[str, bool]:
    out = {}
    if GOLD.exists():
        for line in GOLD.open(encoding="utf-8"):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line while the gold check is running
            out[o["task_id"]] = o["status"] == "verified" and 0 < o["n_tests_probed"] == o["n_tests_passed"]
    return out


def counts(done: dict[str, int], rounds: int) -> dict[str, int]:
    gold = gold_ok()
    c = {"mixed": 0, "always_solved": 0, "never_solved": 0, "gold_failed": 0, "gold_pending": 0}
    for t, k in done.items():
        g = gold.get(t)
        key = ("gold_pending" if g is None else "gold_failed" if not g else
               "mixed" if 0 < k < rounds else "always_solved" if k == rounds else "never_solved")
        c[key] += 1
    return c


async def main_async(args) -> None:
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    tasks = {t.task_id: t for t in load_tasks(POOL)}
    test_ids = set(json.load(open("data/splits/run3_test.json")))
    assert not test_ids & set(tasks), "a test task is in the training pool"
    order = sorted(tasks)
    random.Random(args.seed).shuffle(order)

    done: dict[str, int] = {}  # task_id -> passes, from the file (resume)
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[r["task_id"]] = r["n_passed"]
    print(f"[sweep] resume: {len(done)} tasks already saved; {counts(done, args.rounds)}", flush=True)

    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model="Qwen/Qwen3.5-4B", revision=REVISION, seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_new_tokens + 2048))
    tok = engine.get_tokenizer()  # synchronous in vLLM 0.31
    grader = VerifierReward(tasks, SandboxConfig(backend="bwrap", timeout_s=60.0, cpu_s=60, mem_mb=8192))
    pool = ThreadPoolExecutor(max_workers=8)   # grading runs beside generation
    lock = asyncio.Lock()
    queue = [(i, t) for i, t in enumerate(order) if t not in done]
    gold_cache = {"t": 0.0, "v": gold_ok()}
    stop = asyncio.Event()

    def skip(t: str) -> bool:  # known gold failure: no GPU time spent on it
        if time.time() - gold_cache["t"] > 300:
            gold_cache.update(t=time.time(), v=gold_ok())
        return gold_cache["v"].get(t) is False

    async def one_task(idx: int, tid: str) -> None:
        msgs = [{"role": "user", "content": build_prompt_for_task(tasks[tid])[0]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True)
        # seed per task (its position in the fixed order): reproducible
        # FINAL_ONLY: one result holding all n answers. The default streaming
        # mode returned partial outputs (smoke test: 1 of 4 answers saved).
        params = SamplingParams(n=args.rounds, max_tokens=args.max_new_tokens, seed=args.seed * 10**6 + idx,
                                output_kind=RequestOutputKind.FINAL_ONLY, **SAMPLING)
        final = None
        async for out in engine.generate(prompt, params, request_id=uuid.uuid4().hex):
            final = out
        if final is None or len(final.outputs) != args.rounds:
            raise RuntimeError(f"{tid}: got {0 if final is None else len(final.outputs)} answers, "
                               f"expected {args.rounds}")
        answers = []
        for c in sorted(final.outputs, key=lambda c: c.index):
            _, answer, closed = split_thinking(c.text)
            g = (await asyncio.get_running_loop().run_in_executor(pool, grader.grade, tid, answer)) \
                if closed else {"correctness": 0.0, "failure_type": "no_answer"}
            answers.append({"text": c.text, "completion_tokens": len(c.token_ids),
                            "hit_token_cap": c.finish_reason == "length", "closed_thinking": closed,
                            "passed": g["correctness"] == 1.0, "failure_type": g["failure_type"]})
        rec = {"task_id": tid, "order_index": idx, "n_passed": sum(a["passed"] for a in answers),
               "rounds": args.rounds, "sampling": SAMPLING, "max_new_tokens": args.max_new_tokens,
               "answers": answers}
        async with lock:  # one line per task, written the moment it is graded
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            done[tid] = rec["n_passed"]
            c = counts(done, args.rounds)
            print(f"[sweep] {tid} {rec['n_passed']}/{args.rounds} | saved {len(done)} | {c}", flush=True)
            if c["mixed"] >= args.target:
                stop.set()

    async def worker() -> None:
        while queue and not stop.is_set():
            idx, tid = queue.pop(0)
            if not skip(tid):
                await one_task(idx, tid)

    await asyncio.gather(*(worker() for _ in range(args.in_flight)))
    print(f"[sweep] finished: {counts(done, args.rounds)}", flush=True)
    engine.shutdown()


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=4, help="answers per task")
    ap.add_argument("--in-flight", type=int, default=64, help="tasks generating at once")
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/run3_sweep.jsonl")
    args = ap.parse_args()
    OUT = Path(args.out)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
