"""Build the run #3 train/val splits from the informativeness sweep.

Run AFTER the sweep is stopped (it is safe to run while it runs with --dry-run:
the sweep only appends, and a half-written last line is skipped and reported).

What it does, every step counted in the printed report:
  1. reads data/run3_sweep.jsonl and checks each record is complete;
  2. re-checks every swept task against ALL quality rules:
       - audit_task (empty statement, image, multi-answer, <5 tests, no entry
         point) — already applied when the pool was built, expected to drop 0;
       - grading risks added 2026-11-21: "[Image]" placeholders (now part of the
         image rule), statements pointing at a figure, decimal answers graded with
         a tolerance (our grader compares text exactly);
       - gold check passed (reference solution passes every test);
       - not in the test split;
  3. keeps clean tasks with MIXED results (0 < passes < rounds) and labels
     each by why its failing answers failed (cutoff / code / both); two sides:
     code side = code + both, cut-off side = cutoff;
  4. validation (PAPER_PLAN.md, "Validation (fixed)"): 25 code side (code and
     both in proportion) + 25 cut-off + 30 always-solved + 20 never-solved;
  5. train = the remaining mixed tasks, ORDERED in batches of 8 = 4 code side
     + 4 cut-off side (option A), seeded. The shorter side sets the number of
     batches (= max_steps); the longer side's leftover tasks are spare.
     Checked against the run config (max_steps, tasks per step).

It also writes the "clean" test subset (full test minus figure / decimal tasks),
reported beside the full-test score; the full test split is not changed.

run3_train.json is ORDERED (not sorted): training reads it in file order
(shuffle_dataset false), 8 consecutive tasks per step. Do not sort it.

Outputs (in --out-dir, default data/splits): run3_train.json, run3_val.json,
run3_test_clean.json, run3_splits_report.json. Existing files are not
overwritten without --force.

Usage:
    .venv/bin/python scripts/build_run3_splits.py --dry-run     # preview only
    .venv/bin/python scripts/build_run3_splits.py               # write splits
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlforge.curation import MIN_TESTS_TRAIN, audit_task, grading_risk_reasons, n_tests  # noqa: E402
from rlforge.run3_splits import (  # noqa: E402
    failure_kind,
    interleave_batches,
    label_counts,
    outcome,
    random_draw,
    side,
    stratified_val,
    validate_record,
)
from rlforge.schema.task import CodeTask  # noqa: E402

# task_id is the 2nd key of every pool row; reading it from the line start avoids
# parsing ~1.8 GB of JSON just to find the few thousand swept tasks
_ID_RE = re.compile(r'"task_id":\s*"([^"]+)"')


def read_sweep(path: Path, rounds: int) -> tuple[dict[str, dict], list[str]]:
    """Return ({task_id: record}, problems). A truncated LAST line (the sweep is
    mid-write) is a note, not an error; any other bad line is an error."""
    records: dict[str, dict] = {}
    problems: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, start=1):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            where = "last line (sweep writing it)" if i == len(lines) else f"line {i}"
            problems.append(f"unreadable {where}")
            continue
        bad = validate_record(r, rounds)
        if bad:
            problems.append(f"{r.get('task_id')}: {bad}")
            continue
        if r["task_id"] in records:
            problems.append(f"{r['task_id']}: duplicate record")
            continue
        records[r["task_id"]] = r
    return records, problems


def read_pool(path: Path, wanted: set[str]) -> dict[str, CodeTask]:
    """Load and validate only the wanted tasks from the (large) pool file."""
    tasks: dict[str, CodeTask] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            m = _ID_RE.search(line[:400])
            if m and m.group(1) in wanted:
                t = CodeTask.model_validate(json.loads(line))
                tasks[t.task_id] = t
    return tasks


def read_gold(path: Path) -> dict[str, bool]:
    """Same rule as scripts/sweep_run3.py: verified and every probed test passed."""
    out = {}
    for line in path.open(encoding="utf-8"):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[o["task_id"]] = o["status"] == "verified" and 0 < o["n_tests_probed"] == o["n_tests_passed"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="data/run3_sweep.jsonl")
    ap.add_argument("--pool", default="data/run3_pool.jsonl")
    ap.add_argument("--gold", default="data/run3_gold.jsonl")
    ap.add_argument("--test-tasks", default="data/run3_test.jsonl")
    ap.add_argument("--test-split", default="data/splits/run3_test.json")
    ap.add_argument("--rounds", type=int, default=4)
    # validation sizes, fixed in PAPER_PLAN.md ("Validation (fixed)")
    ap.add_argument("--val-code", type=int, default=25, help="code-side val tasks (code + both, in proportion)")
    ap.add_argument("--val-cutoff", type=int, default=25, help="cut-off-only val tasks")
    ap.add_argument("--val-always", type=int, default=30, help="always-solved val tasks")
    ap.add_argument("--val-never", type=int, default=20, help="never-solved val tasks")
    ap.add_argument("--per-side", type=int, default=4, help="code-side (and cut-off) tasks per batch")
    ap.add_argument("--config", default="configs/grpo-vanilla-run3.yaml",
                    help="run config checked against the split (max_steps, tasks per step)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--dry-run", action="store_true", help="print the report, write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = ap.parse_args()

    # ── 1. sweep records ──
    records, problems = read_sweep(Path(args.sweep), args.rounds)
    hard = [p for p in problems if "sweep writing it" not in p]
    print(f"[1] sweep: {len(records)} complete records; notes: {problems or 'none'}")
    if hard:
        sys.exit(f"refusing: {len(hard)} bad sweep records (see above)")

    # ── 2. re-check every swept task ──
    test_ids = set(json.load(open(args.test_split)))
    gold = read_gold(Path(args.gold))
    tasks = read_pool(Path(args.pool), set(records))
    missing = set(records) - set(tasks)
    if missing:
        sys.exit(f"refusing: {len(missing)} swept tasks not in the pool, e.g. {sorted(missing)[:3]}")

    reasons: dict[str, list[str]] = {}
    for tid, task in tasks.items():
        r = audit_task(task, min_tests=MIN_TESTS_TRAIN) + grading_risk_reasons(task)
        if not gold.get(tid, False):
            r.append("gold_not_passed")
        if tid in test_ids:
            r.append("in_test_split")
        reasons[tid] = r

    outcomes = {tid: outcome(rec["n_passed"], args.rounds) for tid, rec in records.items()}
    by_out_reason: dict[str, Counter] = defaultdict(Counter)
    for tid, r in reasons.items():
        for x in r:
            by_out_reason[outcomes[tid]][x] += 1
    print(f"[2] quality re-check on {len(tasks)} swept tasks (a task can have several reasons):")
    for out in ("mixed", "always", "never"):
        n_all = sum(1 for o in outcomes.values() if o == out)
        n_drop = sum(1 for t, o in outcomes.items() if o == out and reasons[t])
        print(f"    {out:6s}: {n_all} swept, {n_drop} dropped, by reason {dict(by_out_reason[out]) or '{}'}")

    # ── 3-4. clean mixed tasks and their failure kind ──
    clean = {t for t, r in reasons.items() if not r}
    mixed = sorted(t for t in clean if outcomes[t] == "mixed")
    labels = {t: failure_kind(records[t]) for t in mixed}
    print(f"[3] clean mixed: {len(mixed)}  by failure kind {label_counts(mixed, labels)}")

    code_side = [t for t in mixed if side(labels[t]) == "code"]
    cut_side = [t for t in mixed if side(labels[t]) == "cutoff"]
    always = sorted(t for t in clean if outcomes[t] == "always")
    never = sorted(t for t in clean if outcomes[t] == "never")
    print(f"    code side {len(code_side)}, cut-off side {len(cut_side)}; "
          f"clean always {len(always)}, clean never {len(never)}")

    # ── 4. validation: 25 code side (code/both in proportion) + 25 cut-off +
    #       30 always + 20 never. Each draw is seeded and independent of input order.
    for name, pool, n in (("code side", code_side, args.val_code), ("cut-off", cut_side, args.val_cutoff),
                          ("always", always, args.val_always), ("never", never, args.val_never)):
        if n > len(pool):
            sys.exit(f"refusing: {n} {name} val tasks asked, only {len(pool)} clean ones")
    code_rest, val_code = stratified_val({t: labels[t] for t in code_side}, args.val_code, args.seed)
    cut_rest, val_cut = random_draw(cut_side, args.val_cutoff, args.seed)
    _, val_always = random_draw(always, args.val_always, args.seed)
    _, val_never = random_draw(never, args.val_never, args.seed)
    val = sorted(val_code + val_cut + val_always + val_never)
    # the group each val task is scored in (kept in the report for the analysis)
    val_group = ({t: f"code_side/{labels[t]}" for t in val_code} | {t: "cutoff" for t in val_cut}
                 | {t: "always" for t in val_always} | {t: "never" for t in val_never})
    print(f"[4] val {len(val)}: {dict(Counter(val_group.values()))}")

    # ── 5. ordered train list: every batch = per_side code side + per_side cut-off ──
    batch = 2 * args.per_side
    train, spare = interleave_batches(code_rest, cut_rest, args.per_side, args.seed)
    n_batches = len(train) // batch
    # guard: re-count every batch from the labels, so a bug in the interleave
    # cannot silently give a run a drifting mix
    for b in range(n_batches):
        sides = Counter(side(labels[t]) for t in train[b * batch:(b + 1) * batch])
        assert sides == {"code": args.per_side, "cutoff": args.per_side}, f"batch {b}: {dict(sides)}"
    print(f"[5] train {len(train)} tasks = {n_batches} batches of {batch} "
          f"({args.per_side} code side + {args.per_side} cut-off), by kind {label_counts(train, labels)}")
    print(f"    spare (longer side's leftover, not trained on): {len(spare)} {label_counts(spare, labels)}")
    tests_train = sorted(n_tests(tasks[t]) for t in train)
    print(f"    train tests per task: min {tests_train[0]}, median {tests_train[len(tests_train) // 2]}")
    print(f"    train by source: {dict(Counter(tasks[t].source for t in train))}")

    # run config must match: one step = one batch, and the run covers every batch once
    cfg_problems = []
    if Path(args.config).exists():
        import yaml  # local import: only needed for this check
        grpo = yaml.safe_load(open(args.config))["grpo"]
        if grpo.get("tasks_per_update") != batch:
            cfg_problems.append(f"tasks_per_update is {grpo.get('tasks_per_update')}, the split needs {batch}")
        if grpo.get("max_steps") != n_batches:
            cfg_problems.append(f"max_steps is {grpo.get('max_steps')}, the split has {n_batches} batches")
        if grpo.get("shuffle_dataset", True):
            cfg_problems.append("shuffle_dataset is not false: TRL would reshuffle the order")
        print(f"    config {args.config}: {cfg_problems or 'matches'}")
    else:
        cfg_problems.append(f"{args.config} not found, not checked")
        print(f"    WARNING: {cfg_problems[-1]}")

    # leak guard: the three splits must be disjoint
    assert not (set(train) & set(val)) and not (set(train) | set(val)) & test_ids, "split overlap"

    # ── test set: clean subset (report only; the full split is unchanged) ──
    test_risk = {}
    for line in open(args.test_tasks, encoding="utf-8"):
        t = CodeTask.model_validate(json.loads(line))
        if t.task_id in test_ids:
            test_risk[t.task_id] = grading_risk_reasons(t) + [x for x in audit_task(t, min_tests=1) if x == "image"]
    if set(test_risk) != test_ids:
        sys.exit(f"refusing: {len(test_ids - set(test_risk))} test ids not found in {args.test_tasks}")
    test_clean = sorted(t for t, r in test_risk.items() if not r)
    test_drop = {t: r for t, r in sorted(test_risk.items()) if r}
    print(f"[5] test: full {len(test_ids)}, clean {len(test_clean)}, "
          f"by reason {dict(Counter(x for r in test_drop.values() for x in r))}")
    for t, r in test_drop.items():
        print(f"      {t}: {r}")

    report = {
        "args": vars(args),
        "sweep_records": len(records), "sweep_notes": problems,
        "dropped": {t: r for t, r in sorted(reasons.items()) if r},
        "dropped_by_outcome_reason": {k: dict(v) for k, v in by_out_reason.items()},
        "clean_mixed": len(mixed), "failure_kind": labels,
        "clean_always": len(always), "clean_never": len(never),
        "train": len(train), "batches": n_batches, "max_steps": n_batches,
        "train_by_kind": label_counts(train, labels),
        # position of each task in the run: batch b = optimizer step b + 1
        "train_order": [{"batch": i // batch, "task_id": t, "kind": labels[t]} for i, t in enumerate(train)],
        "spare": spare, "spare_by_kind": label_counts(spare, labels),
        "val": len(val), "val_group": val_group,
        "val_by_group": dict(Counter(val_group.values())),
        "config_check": cfg_problems or "matches",
        "test_full": len(test_ids), "test_clean": len(test_clean), "test_dropped": test_drop,
    }
    if args.dry_run:
        print("dry run: nothing written")
        return
    out = Path(args.out_dir)
    files = {"run3_train.json": train, "run3_val.json": val, "run3_test_clean.json": test_clean,
             "run3_splits_report.json": report}
    existing = [n for n in files if (out / n).exists()]
    if existing and not args.force:
        sys.exit(f"refusing to overwrite {existing}; pass --force")
    for name, obj in files.items():
        (out / name).write_text(json.dumps(obj, indent=1) + "\n")
    print(f"written to {out}: {', '.join(files)}")


if __name__ == "__main__":
    main()
