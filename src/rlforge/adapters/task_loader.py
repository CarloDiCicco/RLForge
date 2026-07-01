"""
adapters/task_loader.py
────────────────────────
load_tasks — read CodeTasks from disk (PLAN §1.1 ingestion boundary).

Named for what it does (load task files), not the format it defaults to:
despite the historical "jsonl" framing, it dispatches on file extension and
also reads .json/.yaml/.yml — JSONL is just what the rest of the pipeline
happens to write (registry output, fixtures), not the only input shape a
hand-authored task file may use.

Adapted from ReasoningLab `tasks/loaders.py`: the same per-line error context and
extension dispatch (.jsonl / .json / .yaml), but validating against CodeTask (not
TaskRecord) and adding a duplicate-task_id guard.

Why the duplicate guard is new, not cosmetic: registry/runinfo.py mints
deterministic ids from (run_id, task_id, sample_idx) — see Step 7. That scheme is
unique only if a run never sees the same task_id twice; a task file with a
repeated id would silently collide two different tasks' sample_ids onto one
another. Rejecting duplicates here makes that precondition guaranteed, not
assumed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rlforge.schema.task import CodeTask


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON task records."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            # Ignore blank lines to make hand-edited files friendlier.
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: expected object")
        records.append(record)
    return records


def _read_yaml(path: Path) -> list[dict[str, Any]]:
    """Read YAML task records from either list or {'tasks': [...]} format."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read YAML task files. Install `pyyaml`."
        ) from exc

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return []

    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and isinstance(raw.get("tasks"), list):
        records = raw["tasks"]
    else:
        raise ValueError(
            f"Invalid YAML task format in {path}. Use a list or {{'tasks': [...]}}."
        )

    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid task record at {path}:{idx}: expected object")
    return records


def _read_task_records(path: Path) -> list[dict[str, Any]]:
    """Dispatch task loading based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Invalid JSON task format in {path}: expected list")
        return raw
    if suffix in {".yaml", ".yml"}:
        return _read_yaml(path)
    raise ValueError(
        f"Unsupported task extension: {suffix}. Use .jsonl, .json, .yaml or .yml."
    )


def load_tasks(tasks_path: str | Path, limit: int | None = None) -> list[CodeTask]:
    """Load and validate CodeTasks from disk with stable input order.

    Each row is validated against CodeTask (nested TestSuite payloads are
    validated recursively by Pydantic). A duplicate task_id across rows raises
    immediately — the registry's deterministic sample_id scheme (Step 7) assumes
    each task_id appears at most once per run.

    Args:
        tasks_path: path to a .jsonl / .json / .yaml / .yml task file.
        limit: if given, keep only the first `limit` tasks (input order).

    Raises:
        FileNotFoundError: tasks_path does not exist.
        ValueError: malformed file, invalid CodeTask row, or a duplicate task_id.
    """
    path = Path(tasks_path)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided")

    raw_records = _read_task_records(path)
    tasks: list[CodeTask] = []
    seen_task_ids: set[str] = set()

    for idx, raw in enumerate(raw_records, start=1):
        try:
            task = CodeTask.model_validate(raw)
        except ValidationError as exc:
            task_hint = raw.get("task_id") if isinstance(raw, dict) else None
            suffix = f" ({task_hint})" if task_hint else ""
            raise ValueError(f"Invalid task record at {path}:{idx}{suffix}: {exc}") from exc

        if task.task_id in seen_task_ids:
            raise ValueError(
                f"Duplicate task_id {task.task_id!r} at {path}:{idx} — task_ids "
                "must be unique within a task file (the registry derives ids "
                "from task_id and would silently collide otherwise)."
            )
        seen_task_ids.add(task.task_id)
        tasks.append(task)

    if limit is not None:
        tasks = tasks[:limit]
    return tasks
