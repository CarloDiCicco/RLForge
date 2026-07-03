#!/usr/bin/env python3
"""
scripts/prepare_livecodebench.py
──────────────────────────────────
Materialize LiveCodeBench's LeetCode-platform problems as a plain, inspectable
JSONL file in our canonical CodeTask/TestSuite schema (PLAN §2.2).

adapters.livecodebench.load_livecodebench_leetcode() re-downloads/re-derives
CodeTasks from HuggingFace's cache on every call — there is no plain file
anywhere in RLForge's own schema to `cat`/grep/open directly. This script runs
that conversion once and writes the result to disk, matching
scripts/prepare_humaneval_plus.py's pattern exactly (in-repo, committed — see
.gitignore's data/* + !data/livecodebench_leetcode.jsonl exception).

The output is loadable by the EXISTING adapters.task_loader.load_tasks() — no
new loader needed; a config can point tasks_file at this file instead of
calling the adapter directly.

Only the LeetCode-platform subset is included (assert-style tests, runnable
with our existing sandbox); see adapters/livecodebench.py's module docstring
for why AtCoder/Codeforces (stdin/stdout judged) are out of scope for now.

Usage:
    python scripts/prepare_livecodebench.py
    python scripts/prepare_livecodebench.py --release release_v5
    python scripts/prepare_livecodebench.py --output /custom/path.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rlforge.adapters.livecodebench import load_livecodebench_leetcode

# In-repo, matching adapters/humaneval.py's convention (data/*.jsonl,
# committed to git — see .gitignore's explicit exception for this file).
DEFAULT_OUTPUT = Path("data") / "livecodebench_leetcode.jsonl"
DEFAULT_RELEASE = "release_v6"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--release", type=str, default=DEFAULT_RELEASE,
        help=f"LiveCodeBench release tag to load (default: {DEFAULT_RELEASE})",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_livecodebench_leetcode(release=args.release)
    with args.output.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")

    print(f"Saved {len(tasks)} tasks to {args.output}")


if __name__ == "__main__":
    main()