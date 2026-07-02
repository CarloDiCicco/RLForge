#!/usr/bin/env python3
"""
scripts/prepare_humaneval_plus.py
──────────────────────────────────
Materialize HumanEval+ as a plain, inspectable JSONL file in our canonical
CodeTask/TestSuite schema (PLAN §2.2).

adapters.humaneval.load_humaneval_plus() re-derives CodeTasks from evalplus's
own cache on every call — there is no plain file anywhere in RLForge's own
schema to `cat`/grep/open directly. This script runs that conversion once and
writes the result to disk, matching ReasoningLab's scripts/prepare_*.py ->
data/*.jsonl pattern exactly (in-repo, committed — see .gitignore's
data/* + !data/humaneval_plus.jsonl exception).

The output is loadable by the EXISTING adapters.task_loader.load_tasks() —
no new loader needed; a config can point tasks_file at this file instead of
using dataset: humaneval_plus (the always-fresh, un-materialized path).

Usage:
    python scripts/prepare_humaneval_plus.py
    python scripts/prepare_humaneval_plus.py --output /custom/path.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rlforge.adapters.humaneval import load_humaneval_plus

# In-repo, matching ReasoningLab's convention (data/humaneval_plus.jsonl,
# committed to git — see .gitignore's explicit exception for this file).
DEFAULT_OUTPUT = Path("data") / "humaneval_plus.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_humaneval_plus()
    with args.output.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")

    print(f"Saved {len(tasks)} tasks to {args.output}")


if __name__ == "__main__":
    main()
