#!/usr/bin/env python3
"""Gold check for the run #3 training pool: run each problem's reference
solution against ALL of its tests in the bwrap sandbox.

Why all tests: the training reward (train/reward.py) grades every test in the
suite, so a problem is only safe to train on if a known-correct solution passes
every one of them. The verifier's default cap (30 tests) is lifted here.

Output: data/run3_gold.jsonl, an append-only ledger (resumable: re-running skips
task ids already present). A task is "gold" iff n_tests_passed == n_tests_probed
> 0. Only the first stored reference solution is tried; a problem whose first
solution fails is dropped (a cost in recall, never in correctness).

CPU only. Workers default to 8 so the sweep's own grading on the same machine
is not starved when both run at once.

Usage:
    .venv/bin/python scripts/gold_check_run3.py [--workers 8] [--limit N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlforge.sandbox.base import SandboxConfig
from rlforge.schema.task import CodeTask
from rlforge.verify.reference_verifier import verify_tasks

POOL = Path("data/run3_pool.jsonl")
LEDGER = Path("data/run3_gold.jsonl")


def iter_pool(limit: int | None):
    with POOL.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            yield CodeTask.model_validate_json(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N tasks only (timing sample)")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="seconds for the whole suite; same as the eval sandbox")
    args = ap.parse_args()

    cfg = SandboxConfig(backend="bwrap", timeout_s=args.timeout, cpu_s=int(args.timeout), mem_mb=8192)
    stats = verify_tasks(iter_pool(args.limit), out_path=LEDGER, sandbox_config=cfg,
                         workers=args.workers, max_tests=10**9, progress_every=250)
    print("[gold] final:", stats.summary())

    # Recount from the ledger itself, so the numbers cover resumed runs too.
    gold = total = 0
    for line in LEDGER.open(encoding="utf-8"):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        gold += o["status"] == "verified" and 0 < o["n_tests_probed"] == o["n_tests_passed"]
    print(f"[gold] ledger: {gold}/{total} tasks pass every test with their reference solution")


if __name__ == "__main__":
    main()
