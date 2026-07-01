"""Tests for adapters/task_loader.py — the CodeTask loader (PLAN §1.1 ingestion)."""

from __future__ import annotations

import json

import pytest

from rlforge.adapters.task_loader import load_tasks

_TASK_A = {
    "task_id": "add",
    "source": "fixture",
    "prompt": "Write add(a, b).",
    "test_suites": {
        "visible": {"kind": "assert", "asserts": ["assert add(2, 3) == 5"]},
        "hidden": {"kind": "assert", "asserts": ["assert add(-1, 1) == 0"]},
    },
    "reference_solution": "def add(a, b):\n    return a + b",
}

_TASK_B = {
    "task_id": "is_even",
    "source": "fixture",
    "prompt": "Write is_even(n).",
    "test_suites": {
        "hidden": {"kind": "assert", "asserts": ["assert is_even(4) is True"]},
    },
    "reference_solution": "def is_even(n):\n    return n % 2 == 0",
}


def _write_jsonl(tmp_path, rows) -> str:
    path = tmp_path / "tasks.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(path)


def test_loads_valid_tasks_in_order(tmp_path):
    path = _write_jsonl(tmp_path, [_TASK_A, _TASK_B])
    tasks = load_tasks(path)
    assert [t.task_id for t in tasks] == ["add", "is_even"]
    assert tasks[0].test_suites["hidden"].asserts == ["assert add(-1, 1) == 0"]


def test_limit_keeps_input_order_prefix(tmp_path):
    path = _write_jsonl(tmp_path, [_TASK_A, _TASK_B])
    tasks = load_tasks(path, limit=1)
    assert [t.task_id for t in tasks] == ["add"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tasks(tmp_path / "nope.jsonl")


def test_duplicate_task_id_raises(tmp_path):
    dup = dict(_TASK_B, task_id="add")  # collides with _TASK_A's id
    path = _write_jsonl(tmp_path, [_TASK_A, dup])
    with pytest.raises(ValueError, match="Duplicate task_id"):
        load_tasks(path)


def test_invalid_row_raises_with_context(tmp_path):
    bad = dict(_TASK_A)
    del bad["prompt"]  # required field
    path = _write_jsonl(tmp_path, [bad])
    with pytest.raises(ValueError, match="Invalid task record"):
        load_tasks(path)


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(_TASK_A) + "\n\n" + json.dumps(_TASK_B) + "\n")
    tasks = load_tasks(path)
    assert len(tasks) == 2


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "tasks.csv"
    path.write_text("task_id,prompt\n")
    with pytest.raises(ValueError, match="Unsupported task extension"):
        load_tasks(path)
