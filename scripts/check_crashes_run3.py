"""Re-check the sweep answers that CRASHED or TIMED OUT, to find tasks whose 0
reward may come from our setup instead of the model's code.

Why: the sweep stores only the failure type, not the error. A crash caused by the
sandbox (memory cap, blocked call, thread limit) would give a correct answer 0 and
bias training. The grader is FROZEN (RESEARCH_LOG 2026-11-21: changing it now would
mean re-grading the sweep and the base-model test score), so the fix for a
setup-caused failure is to DROP the task, never to change a limit.

Steps per answer:
  1. re-grade it EXACTLY as the sweep did: only the text after </think>
     (split_thinking), same VerifierReward, same SandboxConfig;
  2. if the result differs from the sweep's → "flaky" (unreliable reward) → drop;
  3. classify the error (last stderr line): a normal code bug → keep; a
     signature that our setup can cause (memory, signal kill, recursion, thread
     limit, blocked os/socket call, permission, missing module) or a timeout →
  4. re-run in the same isolation with LOOSER limits (more memory, time, processes):
     passes → "setup_caused" → drop; still fails → keep.
Generated code only ever runs inside the bwrap sandbox, never on the host.

Output: data/run3_crash_check.json with every answer's verdict and the task ids to
drop; scripts/build_run3_splits.py reads it.

Usage (sweep stopped, GPU free):
    .venv/bin/python scripts/check_crashes_run3.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_run3_splits import read_pool  # noqa: E402
from thinking_length_study_vllm import split_thinking  # noqa: E402  same split as the sweep

from rlforge.sandbox.base import SandboxConfig  # noqa: E402
from rlforge.train.reward import VerifierReward  # noqa: E402

# The sweep's grader settings, copied from scripts/sweep_run3.py (must stay equal).
SWEEP_CFG = dict(backend="bwrap", timeout_s=60.0, cpu_s=60, mem_mb=8192)
# Looser limits for step 4: 4x memory, 3x time, processes/threads allowed.
LOOSE_CFG = dict(backend="bwrap", timeout_s=180.0, cpu_s=180, mem_mb=32768, nproc=256)

# Error signatures our setup can cause (see src/rlforge/sandbox/preamble.py: it
# sets disabled os/socket functions to None, so calling one raises
# "'NoneType' object is not callable"; RLIMIT_NPROC also limits threads).
_SUSPICIOUS = re.compile(
    r"MemoryError|Killed|signal|SIGKILL|SIGSEGV|Segmentation|RecursionError|maximum recursion"
    r"|can't start new thread|Resource temporarily unavailable|BlockingIOError"
    r"|'NoneType' object is not callable|PermissionError|OSError|ModuleNotFoundError|ImportError",
)


def last_error_line(stderr: str) -> str:
    lines = [ln for ln in (stderr or "").strip().splitlines() if ln.strip()]
    return lines[-1][:200] if lines else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="data/run3_sweep.jsonl")
    ap.add_argument("--pool", default="data/run3_pool.jsonl")
    ap.add_argument("--out", default="data/run3_crash_check.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    # every crashed / timed-out answer, with its task's outcome for the report
    items = []
    for line in open(args.sweep, encoding="utf-8"):
        r = json.loads(line)
        n = r["n_passed"]
        out = "mixed" if 0 < n < r["rounds"] else ("never" if n == 0 else "always")
        for k, a in enumerate(r["answers"]):
            if a["failure_type"] in ("runtime", "timeout"):
                items.append({"task_id": r["task_id"], "answer": k, "outcome": out,
                              "sweep_failure": a["failure_type"], "text": a["text"]})
    tasks = read_pool(Path(args.pool), {it["task_id"] for it in items})
    strict = VerifierReward(tasks, SandboxConfig(**SWEEP_CFG))
    loose = VerifierReward(tasks, SandboxConfig(**LOOSE_CFG))
    print(f"{len(items)} answers to re-check ({Counter(i['sweep_failure'] for i in items)})", flush=True)

    def check(it: dict) -> dict:
        _, answer, closed = split_thinking(it["text"])
        assert closed, f"{it['task_id']}: a graded answer must have closed thinking"
        g = strict.grade(it["task_id"], answer)
        res = {k: it[k] for k in ("task_id", "answer", "outcome", "sweep_failure")}
        res |= {"regrade_failure": g["failure_type"], "error": last_error_line(g.get("stderr", "")),
                "exec_time_s": g.get("exec_time_s")}
        if g["failure_type"] != it["sweep_failure"]:
            res["verdict"] = "flaky"  # same input, different result → reward not reliable
        elif it["sweep_failure"] == "timeout" or _SUSPICIOUS.search(res["error"]):
            gl = loose.grade(it["task_id"], answer)
            res["loose_failure"] = gl["failure_type"]
            res["loose_error"] = last_error_line(gl.get("stderr", ""))
            res["verdict"] = "setup_caused" if gl["correctness"] == 1.0 else "still_fails"
        else:
            res["verdict"] = "code_bug"
        return res

    results = []
    with ThreadPoolExecutor(args.workers) as ex:
        for res in ex.map(check, items):
            results.append(res)
            print(f"{res['task_id']} #{res['answer']} {res['outcome']:6s} {res['sweep_failure']:7s} -> "
                  f"{res['verdict']:12s} | {res['error'][:110]}", flush=True)

    # error type = text before the first ":" of the last stderr line
    by_verdict = Counter((r["outcome"], r["verdict"]) for r in results)
    by_error = Counter(r["error"].split(":")[0] or f"<{r['regrade_failure']}>" for r in results)
    drop = {}
    for r in results:
        if r["verdict"] in ("flaky", "setup_caused"):
            drop.setdefault(r["task_id"], []).append(r["verdict"])
    print("\nverdicts (outcome, verdict):", dict(sorted(by_verdict.items())))
    print("error types:", dict(by_error.most_common()))
    print(f"tasks to drop: {len(drop)} ({Counter(v for vs in drop.values() for v in set(vs))})")
    for t, v in sorted(drop.items()):
        print(f"   {t}: {sorted(set(v))}")
    Path(args.out).write_text(json.dumps(
        {"sweep_cfg": SWEEP_CFG, "loose_cfg": LOOSE_CFG, "drop": drop, "results": results}, indent=1) + "\n")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
