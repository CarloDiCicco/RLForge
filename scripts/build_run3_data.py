#!/usr/bin/env python3
"""Build the run #3 data: the LiveCodeBench test set and the DeepCoder training pool.

Outputs (task files are gitignored, split files are committed):
    data/run3_test.jsonl          test tasks, one file so the eval loads only these
    data/splits/run3_test.json    test task ids
    data/run3_pool.jsonl          training pool BEFORE the gold check and sweep
    data/splits/run3_build_report.json   every count, for RESEARCH_LOG

Test = LiveCodeBench release_v6 problems dated >= 2024-08-01 (LeetCode + AtCoder/
Codeforces), minus problems the audit rejects (min_tests=1, see curation.py).
Pool = DeepCoder TACO + SYNTHETIC-1, deduplicated, audited (min_tests=5), and
with any problem overlapping a test problem removed.

The step aborts if any test task id or overlapping statement survives into the
pool, so a leak cannot be written silently.

Usage:
    .venv/bin/python scripts/build_run3_data.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from rlforge.adapters.deepcoder import DeepCoderStats, iter_deepcoder_tasks
from rlforge.curation import MIN_TESTS_TRAIN, OVERLAP_THRESHOLD, audit_task, overlap_pairs
from rlforge.schema.task import CodeTask

TEST_START = "2024-08"
LCB_FILES = ("data/livecodebench_leetcode.jsonl", "data/livecodebench_stdin_stdout.jsonl")


def load_test_window() -> list[CodeTask]:
    # Line by line: the stdio file is 8.4 GB, too big to load whole next to the
    # pool. Only rows in the date window are parsed into CodeTasks.
    tasks = []
    for f in LCB_FILES:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if (rec["metadata"].get("contest_date") or "")[:7] >= TEST_START:
                    tasks.append(CodeTask.model_validate(rec))
    return tasks


def main() -> None:
    report: dict = {}

    # ── test ──
    window = load_test_window()
    test, test_drop = [], Counter()
    for t in window:
        reasons = audit_task(t, min_tests=1)
        test_drop.update(reasons)
        if not reasons:
            test.append(t)
        else:
            report.setdefault("test_dropped_ids", {})[t.task_id] = reasons
    report["test"] = {"window_from": TEST_START, "in_window": len(window), "kept": len(test),
                      "dropped_by_reason": dict(test_drop),
                      "by_platform": dict(Counter(t.metadata.get("platform") or "leetcode" for t in test)),
                      "by_difficulty": dict(Counter(t.metadata.get("difficulty") for t in test))}
    print("[test]", report["test"])

    # ── pool ──
    st = DeepCoderStats()
    pool, pool_drop = [], Counter()
    for t in iter_deepcoder_tasks(stats=st):
        reasons = audit_task(t, min_tests=MIN_TESTS_TRAIN)
        pool_drop.update(reasons)
        if not reasons:
            pool.append(t)
    print("[pool] conversion:", st.summary())

    # Overlap is checked against the WHOLE window, not just kept test tasks: a
    # dropped test problem could come back in a later test version.
    pairs = overlap_pairs(pool, window)
    leaked = {tid for frac, tid, _ in pairs if frac >= OVERLAP_THRESHOLD}
    pool = [t for t in pool if t.task_id not in leaked]
    report["pool"] = {"raw_rows": st.rows_seen, "unique_after_dedup": st.tasks_built,
                      "duplicates": st.dropped_duplicate, "dropped_by_reason": dict(pool_drop),
                      "overlap_removed": len(leaked),
                      "max_overlap_fraction": round(pairs[0][0], 3) if pairs else 0.0,
                      "top_overlap_pairs": [p for p in pairs[:5]],
                      "kept": len(pool),
                      "by_subset": dict(Counter(t.metadata["dc_subset"] for t in pool)),
                      "by_kind": dict(Counter(t.source for t in pool))}
    print("[pool]", {k: v for k, v in report["pool"].items() if k != "top_overlap_pairs"})

    # ── leak guard ──
    test_ids = {t.task_id for t in window}
    if test_ids & {t.task_id for t in pool}:
        sys.exit("ABORT: a test task id is in the training pool")
    if any(frac >= OVERLAP_THRESHOLD for frac, *_ in overlap_pairs(pool, window)):
        sys.exit("ABORT: overlap survived the filter")

    # ── write ──
    Path("data/splits").mkdir(parents=True, exist_ok=True)
    with open("data/run3_test.jsonl", "w") as f:
        for t in test:
            f.write(t.model_dump_json() + "\n")
    json.dump(sorted(t.task_id for t in test), open("data/splits/run3_test.json", "w"), indent=1)
    with open("data/run3_pool.jsonl", "w") as f:
        for t in pool:
            f.write(t.model_dump_json() + "\n")
    json.dump(report, open("data/splits/run3_build_report.json", "w"), indent=1)
    print("[done] data/run3_test.jsonl, data/splits/run3_test.json, data/run3_pool.jsonl, "
          "data/splits/run3_build_report.json")


if __name__ == "__main__":
    main()
