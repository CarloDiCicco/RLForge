"""
registry/jsonl_logger.py
────────────────────────
Append-only JSONL persistence for run records (PLAN §1.5).

JSONL (one JSON object per line) is the right format for an experiment log:
  - Append-only: each record is written the moment it is produced, so a crash
    mid-run never loses the records already on disk.
  - Streamable: read back line-by-line without loading the whole file.
  - Inspectable: plain text; grep/jq from the shell.

Adapted from ReasoningLab `io/jsonl_logger.py`. The crash-safe `_append_line`
helper is copied as-is; the difference is serialization — ReasoningLab dumped a
frozen dataclass via `dataclasses.asdict`, whereas RLForge records are Pydantic,
so we use `record.model_dump(mode="json")`. mode="json" renders the str-enum
`FailureType` and the nested `ExecutionResult` to JSON-native types with no custom
encoder. We also add a `read_records` reader (the dataclass version had none) that
re-validates every line back into its model — the registry round-trips, so a
written record is guaranteed reloadable.

Run layout: runs/<run_id>/{generations,executions,summary}.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

from rlforge.metrics.metrics import RunMetrics
from rlforge.schema.execution import ExecutionRecord
from rlforge.schema.records import GenerationRecord

M = TypeVar("M", bound=BaseModel)


def _append_line(path: Path, obj: dict) -> None:
    """Serialize obj to one JSON line and append it to path (crash-safe).

    Creates parent directories on first write. Append mode means existing lines
    are never rewritten, so a crash leaves all prior lines intact.
    ensure_ascii=False preserves Unicode in prompts / model output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def log_generation(path: Path, record: GenerationRecord) -> None:
    """Append one GenerationRecord as a JSON line."""
    _append_line(path, record.model_dump(mode="json"))


def log_execution(path: Path, record: ExecutionRecord) -> None:
    """Append one ExecutionRecord as a JSON line (nested ExecutionResult included)."""
    _append_line(path, record.model_dump(mode="json"))


def log_summary(
    path: Path, run_id: str, metrics: RunMetrics, *, extra: dict | None = None
) -> None:
    """Append one run's aggregate RunMetrics as a JSON line.

    Called once per pipeline run, after compute_metrics(). `extra` carries
    provenance that belongs with the summary but is not part of RunMetrics
    itself (e.g. model_name, git_commit, hardware_info, sandbox_config_id,
    n_samples) — merged in after the metrics fields, mirroring ReasoningLab's
    log_summary(policy, metrics, extra).
    """
    obj: dict = {"run_id": run_id, **metrics.model_dump(mode="json")}
    if extra:
        obj.update(extra)
    _append_line(path, obj)


def read_records(path: Path, model_cls: Type[M]) -> list[M]:
    """Read a JSONL file back into validated model instances.

    Each non-blank line is parsed and validated against model_cls; a malformed
    line raises with its line number rather than being silently skipped (fail
    fast in a research harness). Returns [] if the file does not exist.
    """
    if not path.exists():
        return []
    records: list[M] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.strip()
        if not text:
            continue
        try:
            records.append(model_cls.model_validate_json(text))
        except Exception as exc:  # pydantic ValidationError or JSON error
            raise ValueError(f"Invalid record at {path}:{line_number}: {exc}") from exc
    return records
