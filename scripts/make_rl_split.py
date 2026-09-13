"""
scripts/make_rl_split.py
────────────────────────
Build the train/eval split for the RL runs and write it to `data/splits/`.

Why this is a committed artifact and not an inline `random.sample`
─────────────────────────────────────────────────────────────────
The split *defines the experiment*. If the held-out set is not fixed and
inspectable, "pass@1 on 100 held-out tasks" is not a reproducible claim: a future
re-run with a different shuffle would silently measure something else, and a
reviewer has no way to check that the eval tasks were never trained on. So the
ids are generated once, deterministically, and committed.

Only task_ids are written — the source `data/livecodebench_leetcode.jsonl` is
~1 GB because LiveCodeBench inlines its test payloads, so the split file
references tasks rather than copying them.

Stratified by `metadata["difficulty"]` (easy/medium/hard). Random sampling of 100
from 444 would leave the eval set's difficulty mix to chance, and since difficulty
is the dominant driver of pass@1 on this benchmark, an unlucky draw would shift
the eval number for reasons unrelated to the policy.

Usage:
    python scripts/make_rl_split.py
    python scripts/make_rl_split.py --n-eval 100 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlforge.adapters.task_loader import load_tasks  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_TASKS = _REPO / "data" / "livecodebench_leetcode.jsonl"
_DEFAULT_OUT = _REPO / "data" / "splits"


def make_split(
    tasks_file: str | Path = _DEFAULT_TASKS,
    n_eval: int = 100,
    seed: int = 0,
) -> tuple[list[str], list[str], dict]:
    """Return (train_ids, eval_ids, stats), stratified by difficulty.

    The eval set takes a per-difficulty quota proportional to that difficulty's
    share of the population, so its mix mirrors the full benchmark.
    """
    tasks = load_tasks(tasks_file)
    if n_eval >= len(tasks):
        raise ValueError(f"n_eval={n_eval} must be < {len(tasks)} tasks")

    # Group by difficulty. "unknown" keeps a task with missing metadata in the
    # split rather than dropping it — silent attrition is the failure mode this
    # project has already been bitten by (RESEARCH_LOG 2026-07-17).
    by_difficulty: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        by_difficulty[task.metadata.get("difficulty") or "unknown"].append(task.task_id)

    rng = Random(seed)
    eval_ids: list[str] = []
    # Sort strata by name so the allocation order does not depend on dict
    # insertion order (i.e. on the order rows appear in the data file).
    for difficulty in sorted(by_difficulty):
        ids = sorted(by_difficulty[difficulty])  # sort first: load order must not matter
        rng.shuffle(ids)
        quota = round(n_eval * len(ids) / len(tasks))
        eval_ids.extend(ids[:quota])

    # Rounding per stratum can land a task or two off the target; correct against
    # the remaining pool, largest stratum first, so the mix stays closest to the
    # population.
    remaining = [t.task_id for t in tasks if t.task_id not in set(eval_ids)]
    rng.shuffle(remaining)
    while len(eval_ids) < n_eval and remaining:
        eval_ids.append(remaining.pop())
    while len(eval_ids) > n_eval:
        eval_ids.pop()

    eval_set = set(eval_ids)
    train_ids = [t.task_id for t in tasks if t.task_id not in eval_set]

    # Invariants. These are assertions, not logging: a malformed split silently
    # invalidates every number produced downstream.
    assert len(eval_ids) == n_eval, f"{len(eval_ids)} != {n_eval}"
    assert not (set(train_ids) & eval_set), "train and eval overlap"
    assert len(train_ids) + len(eval_ids) == len(tasks), "split does not cover all tasks"
    assert len(set(train_ids)) == len(train_ids), "duplicate train ids"
    assert len(eval_set) == len(eval_ids), "duplicate eval ids"

    diff_of = {t.task_id: (t.metadata.get("difficulty") or "unknown") for t in tasks}
    stats = {
        "n_tasks": len(tasks),
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "seed": seed,
        "tasks_file": str(tasks_file),
        "difficulty_population": dict(Counter(diff_of.values())),
        "difficulty_eval": dict(Counter(diff_of[i] for i in eval_ids)),
        "difficulty_train": dict(Counter(diff_of[i] for i in train_ids)),
    }
    return sorted(train_ids), sorted(eval_ids), stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-file", default=str(_DEFAULT_TASKS))
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(_DEFAULT_OUT))
    ap.add_argument("--prefix", default="lcb_leetcode")
    args = ap.parse_args()

    train_ids, eval_ids, stats = make_split(args.tasks_file, args.n_eval, args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.prefix}_train.json").write_text(json.dumps(train_ids, indent=1) + "\n")
    (out / f"{args.prefix}_eval.json").write_text(json.dumps(eval_ids, indent=1) + "\n")
    (out / f"{args.prefix}_split_stats.json").write_text(json.dumps(stats, indent=1) + "\n")

    print(f"[make_rl_split] {stats['n_train']} train / {stats['n_eval']} eval "
          f"of {stats['n_tasks']} (seed {stats['seed']})")
    pop, ev = stats["difficulty_population"], stats["difficulty_eval"]
    for d in sorted(pop):
        share_pop = 100 * pop[d] / stats["n_tasks"]
        share_ev = 100 * ev.get(d, 0) / stats["n_eval"]
        print(f"  {d:<8} population {pop[d]:>3} ({share_pop:4.1f}%)   "
              f"eval {ev.get(d, 0):>3} ({share_ev:4.1f}%)")
    print(f"[make_rl_split] wrote {out}/{args.prefix}_{{train,eval,split_stats}}.json")


if __name__ == "__main__":
    main()
