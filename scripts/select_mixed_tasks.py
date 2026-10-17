"""Keep only training tasks with MIXED results: the ones GRPO can learn from.

Why: GRPO learns from the spread of rewards inside a task's group of answers. A
task the model always solves (or never solves) gives every answer the same
reward, so it teaches nothing and only costs generation and update time. In the
thinking-mode smoke test (2026-10-15/16) 7 of 8 task groups were all-pass.

Input: a length-study folder made on the TRAIN split, e.g.
    .venv/bin/python scripts/thinking_length_study_vllm.py \
        --split data/splits/lcb_leetcode_v2_train.json --rounds 4 \
        --max-new-tokens <cap> --temperature 0.6 --presence-penalty 1.5 --tag train-filter
Output: a split file with the task ids whose pass count is strictly between 0 and
the number of rounds, plus a printed breakdown.

The filter only uses the training split; validation and test stay untouched, so
the held-out numbers are not affected by it.

Usage:
    python3 scripts/select_mixed_tasks.py runs/eval/<study folder> data/splits/<out>.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    meta = json.loads((folder / "meta.json").read_text())
    if "train" not in Path(meta["args"]["split"]).name:
        sys.exit(f"refusing: study was not run on a train split ({meta['args']['split']})")
    rounds = meta["args"]["rounds"]
    passes: dict[str, int] = defaultdict(int)
    seen: Counter = Counter()
    for line in open(folder / "answers.jsonl"):
        r = json.loads(line)
        seen[r["task_id"]] += 1
        passes[r["task_id"]] += bool(r["passed"])
    incomplete = [t for t, n in seen.items() if n != rounds]
    if incomplete:
        sys.exit(f"{len(incomplete)} tasks do not have {rounds} answers; is the study finished?")

    keep = sorted(t for t in seen if 0 < passes[t] < rounds)
    hist = Counter(passes[t] for t in seen)
    print(f"{len(seen)} tasks, {rounds} answers each. Tasks by number of passing answers:")
    for k in range(rounds + 1):
        print(f"  {k}/{rounds}: {hist.get(k, 0)}")
    print(f"kept (mixed results): {len(keep)}")
    out.write_text(json.dumps(keep, indent=0) + "\n")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
