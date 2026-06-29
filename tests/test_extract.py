"""Tests for the extraction layer (PLAN §1.2): one per decision branch."""

from __future__ import annotations

from rlforge.extract.extractor import (
    STATUS_EMPTY,
    STATUS_FENCED_OTHER,
    STATUS_FENCED_PYTHON,
    STATUS_RAW_FALLBACK,
    STATUS_SYNTAX_ERROR,
    extract_code,
)

ADD = "def add(a, b):\n    return a + b"


def test_prefers_python_fence_over_earlier_other_fence():
    raw = (
        "Here is some bash first:\n"
        "```bash\necho nope\n```\n"
        "and the real answer:\n"
        f"```python\n{ADD}\n```\n"
    )
    r = extract_code(raw)
    assert r.status == STATUS_FENCED_PYTHON
    assert r.code == ADD
    assert r.had_fences
    assert r.reason


def test_first_fence_when_no_python_tag():
    raw = f"```\n{ADD}\n```\nthen another\n```text\nignored\n```"
    r = extract_code(raw)
    assert r.status == STATUS_FENCED_OTHER
    assert r.code == ADD
    assert r.reason


def test_raw_fallback_when_no_fences():
    r = extract_code(ADD)
    assert r.status == STATUS_RAW_FALLBACK
    assert r.code == ADD
    assert not r.had_fences


def test_think_block_is_ignored():
    raw = (
        "<think>\nmaybe use ```python\nreturn 1/0\n``` ? no.\n</think>\n"
        f"```python\n{ADD}\n```"
    )
    r = extract_code(raw)
    assert r.status == STATUS_FENCED_PYTHON
    assert r.code == ADD


def test_empty_extraction():
    r = extract_code("```python\n   \n```")
    assert r.status == STATUS_EMPTY
    assert r.code == ""
    assert not r.ok


def test_syntax_error_is_caught_before_sandbox():
    raw = "```python\ndef add(a, b)\n    return a + b\n```"  # missing colon
    r = extract_code(raw)
    assert r.status == STATUS_SYNTAX_ERROR
    assert r.code  # kept for the record
    assert not r.ok
    assert "parse" in r.reason


def test_ok_property_true_for_runnable_code():
    assert extract_code(ADD).ok
