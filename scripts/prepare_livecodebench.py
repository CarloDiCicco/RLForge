#!/usr/bin/env python3
"""
scripts/prepare_livecodebench.py
──────────────────────────────────
Materialize LiveCodeBench problems as a plain, inspectable JSONL file in our
canonical CodeTask/TestSuite schema (PLAN §2.2).

adapters.livecodebench's loaders re-download/re-derive CodeTasks from
HuggingFace's cache on every call — there is no plain file anywhere in
RLForge's own schema to `cat`/grep/open directly. This script runs that
conversion once and writes the result to disk, matching
scripts/prepare_humaneval_plus.py's pattern exactly (in-repo, committed — see
.gitignore's data/* + the per-file exceptions).

The output is loadable by the EXISTING adapters.task_loader.load_tasks() — no
new loader needed; a config can point tasks_file at this file instead of
calling the adapter directly.

Two `--kind`s, mirroring the two adapter loaders and the two judging styles:
  - "leetcode": function-call/return-value problems, graded assert-style.
  - "stdio":    AtCoder/Codeforces problems, whole programs graded by the
                stdin/stdout harness (verify/compiler.py).

Usage:
    python scripts/prepare_livecodebench.py                    # leetcode (default)
    python scripts/prepare_livecodebench.py --kind stdio       # stdin/stdout
    python scripts/prepare_livecodebench.py --release release_v5
    python scripts/prepare_livecodebench.py --output /custom/path.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rlforge.adapters.livecodebench import (
    load_livecodebench_leetcode,
    load_livecodebench_stdio,
)

# In-repo, matching adapters/humaneval.py's convention (data/*.jsonl,
# committed to git — see .gitignore's explicit exceptions for these files).
DEFAULT_OUTPUTS = {
    "leetcode": Path("data") / "livecodebench_leetcode.jsonl",
    "stdio": Path("data") / "livecodebench_stdin_stdout.jsonl",
}
_LOADERS = {
    "leetcode": load_livecodebench_leetcode,
    "stdio": load_livecodebench_stdio,
}
DEFAULT_RELEASE = "release_v6"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=sorted(_LOADERS), default="leetcode",
        help="Which judging style to materialize (default: leetcode).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSONL path (default: data/livecodebench_<kind>.jsonl).",
    )
    parser.add_argument(
        "--release", type=str, default=DEFAULT_RELEASE,
        help=f"LiveCodeBench release tag to load (default: {DEFAULT_RELEASE})",
    )
    args = parser.parse_args()
    output = args.output or DEFAULT_OUTPUTS[args.kind]
    output.parent.mkdir(parents=True, exist_ok=True)

    tasks = _LOADERS[args.kind](release=args.release)
    with output.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")

    print(f"Saved {len(tasks)} tasks to {output}")


if __name__ == "__main__":
    main()