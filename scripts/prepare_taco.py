#!/usr/bin/env python3
"""
scripts/prepare_taco.py
────────────────────────
Stage 1 of the TACO training-pool pipeline: materialize BAAI/TACO as a plain,
inspectable JSONL in RLForge's canonical CodeTask/TestSuite schema.

TACO is TRAINING data (SFT / RL), never a benchmark — no TACO number is ever
reported as a result. Both TACO splits (train and test) are converted and merged
here: we do not evaluate on TACO, so its "test" split carries no special status
and is simply 1000 more training problems.

The four stages are deliberately separate so each one's numbers can be logged
independently in RESEARCH_LOG.md:

    1. prepare_taco.py         -> data/taco_converted.jsonl        (this script)
    2. decontaminate_taco.py   -> data/taco_decontaminated.jsonl
    3. verify_taco_solutions.py-> data/taco_verification.jsonl
    4. filter_taco.py          -> data/taco_train_pool.jsonl

Output is loadable by the EXISTING adapters.task_loader.load_tasks(). Unlike the
LiveCodeBench/HumanEval+ outputs, it is NOT committed: .gitignore's `data/*`
already denies it and no `!data/taco_*` exception is added.

Usage:
    python scripts/prepare_taco.py                       # train + test
    python scripts/prepare_taco.py --splits train        # one split
    python scripts/prepare_taco.py --limit 50            # smoke test
    python scripts/prepare_taco.py --output /custom.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rlforge.adapters.taco import (
    DEFAULT_SPLITS,
    _MAX_TESTCASE_DROP_RATE,
    _TACO_REVISION,
    TacoConversionStats,
    iter_taco_tasks,
)

DEFAULT_OUTPUT = Path("data") / "taco_converted.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits", nargs="+", default=list(DEFAULT_SPLITS),
        choices=["train", "test"],
        help="TACO splits to convert (default: both; both are training data).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--revision", type=str, default=_TACO_REVISION,
        help="Pinned HuggingFace dataset revision (default: the recorded sha).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after N tasks (smoke tests; disables the attrition guard).",
    )
    parser.add_argument(
        "--max-testcase-drop-rate", type=float, default=_MAX_TESTCASE_DROP_RATE,
        help=f"Abort if per-testcase attrition exceeds this "
             f"(default: {_MAX_TESTCASE_DROP_RATE}).",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    stats = TacoConversionStats()
    written = 0

    # Streamed straight to disk: the whole point is never to hold ~26k tasks
    # (with their inline test payloads) in memory at once.
    with args.output.open("w", encoding="utf-8") as handle:
        for task in iter_taco_tasks(
            splits=args.splits, limit=args.limit, revision=args.revision, stats=stats
        ):
            handle.write(task.model_dump_json() + "\n")
            written += 1
            if written % 1000 == 0:
                print(f"  ... {written} tasks written", flush=True)

    print(stats.summary())
    print(stats.composition())

    # The guard is only meaningful on a full pass; --limit stops mid-stream by
    # design, so its attrition figure is not comparable.
    rate = stats.testcase_drop_rate()
    if args.limit is None and rate > args.max_testcase_drop_rate:
        raise SystemExit(
            f"ABORT: test-case drop rate {rate:.4f} exceeds "
            f"{args.max_testcase_drop_rate:.4f}. Likely an upstream format change. "
            f"Downstream pass rates would be computed against weakened suites."
        )

    print(f"Saved {written} tasks to {args.output} (revision {args.revision})")


if __name__ == "__main__":
    main()
