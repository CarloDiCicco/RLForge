#!/usr/bin/env python3
"""
scripts/run_experiment.py
──────────────────────────
Run one experiment from a YAML/JSON ExperimentConfig (PLAN §2.4).

Translates a config file into the pipeline call that already exists —
pipeline.py itself is untouched; this script is the thin layer above it.

Usage:
    python scripts/run_experiment.py configs/qwen3.5-4b-humaneval-smoke.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rlforge.adapters.humaneval import load_humaneval_plus
from rlforge.adapters.task_loader import load_tasks
from rlforge.config.schema import ExperimentConfig, load_experiment_config
from rlforge.models.generate_fn import make_qwen_generate_fn
from rlforge.models.qwen import QwenModel
from rlforge.pipeline import run_pipeline
from rlforge.schema.task import CodeTask


def _load_tasks(config: ExperimentConfig) -> list[CodeTask]:
    """Load tasks from whichever source the config selected.

    _exactly_one_task_source (config/schema.py) already guarantees exactly one
    of these is set.
    """
    if config.dataset == "humaneval_plus":
        tasks = load_humaneval_plus(limit=config.max_tasks)
    elif config.task_ids_file is None:
        tasks = load_tasks(config.tasks_file, limit=config.max_tasks)
    else:
        # Filter to an explicit task_id list (e.g. the 100 held-out RL eval
        # tasks). `limit` is applied AFTER filtering, not by the loader, or a cap
        # would truncate the source file before the wanted ids were reached.
        wanted = json.loads(Path(config.task_ids_file).read_text())
        by_id = {t.task_id: t for t in load_tasks(config.tasks_file)}
        missing = [t for t in wanted if t not in by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} task_id(s) from {config.task_ids_file} are not in "
                f"{config.tasks_file} (first few: {missing[:5]})"
            )
        # Preserve the id file's order so runs over the same split are aligned.
        tasks = [by_id[t] for t in wanted]
        if config.max_tasks is not None:
            tasks = tasks[: config.max_tasks]
        print(f"[run_experiment] restricted to {len(tasks)} tasks from {config.task_ids_file}")

    return tasks


def main(config_path: str) -> None:
    config = load_experiment_config(config_path)
    print(f"[run_experiment] {config.experiment_id}: loading tasks...")
    tasks = _load_tasks(config)
    print(f"[run_experiment] {len(tasks)} tasks loaded; loading {config.model_id}...")

    model = QwenModel(
        config.model_id,
        revision=config.model_revision,
        adapter_path=config.adapter_path,
    )
    gen_fn = make_qwen_generate_fn(
        model,
        max_new_tokens=config.generation.max_new_tokens,
        do_sample=config.generation.do_sample,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
        top_k=config.generation.top_k,
        min_p=config.generation.min_p,
        repetition_penalty=config.generation.repetition_penalty,
        presence_penalty=config.generation.presence_penalty,
        seed=config.seed,
        enable_thinking=config.generation.enable_thinking,
    )

    # Stamp where the tasks came from into the run summary — exactly one of
    # tasks_file / dataset is set (enforced by ExperimentConfig's validator).
    config.run.tasks_source = config.tasks_file or config.dataset
    run_pipeline(tasks, generate_fn=gen_fn, config=config.run)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/run_experiment.py <config.yaml>", file=sys.stderr)
        raise SystemExit(1)
    main(sys.argv[1])
