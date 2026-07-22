"""Tests for prompt_template/default_prompt_template.py — build_prompt()."""

from __future__ import annotations

from rlforge.extract.extractor import extract_code
from rlforge.prompt_template.default_prompt_template import PROMPT_TEMPLATE_ID, build_prompt
from rlforge.schema.task import CodeTask

_STUB = "def add(a: int, b: int) -> int:\n    \"\"\"Return a + b.\"\"\"\n"


def _task(**overrides) -> CodeTask:
    base = dict(task_id="t1", source="fixture", prompt=_STUB)
    base.update(overrides)
    return CodeTask(**base)


def test_build_prompt_preserves_task_prompt_verbatim():
    task = _task()
    built = build_prompt(task)
    assert _STUB.rstrip() in built


def test_build_prompt_instructs_single_fenced_python_block():
    built = build_prompt(_task())
    assert "```python" in built
    assert "Return only one Markdown fenced Python block" in built


def test_build_prompt_places_stub_inside_a_fence():
    built = build_prompt(_task())
    # The stub itself should be wrapped in its own fence, not bare text.
    fence_start = built.index("```python")
    assert built[fence_start:].rstrip().endswith("```")


def test_build_prompt_does_not_double_fence_a_prefenced_prompt():
    """Regression: the LiveCodeBench-LeetCode adapter stores a prose statement
    that already ends with its own ```python stub fence. Wrapping it again put
    the English statement inside a ```python fence and left a doubled ``` at the
    end (found by reading final_prompt back out of a run's generations.jsonl)."""
    prefenced = (
        "You are given an array.\n\nReturn its length.\n\n"
        "```python\nclass Solution:\n    def f(self, a): ...\n```"
    )
    built = build_prompt(_task(prompt=prefenced))
    # Only the stub's own fence, plus the two in the instruction sentence.
    assert built.count("```") == 4
    assert not built.rstrip().endswith("```\n```")
    # The prose must NOT be inside a fence: statement comes after the
    # instruction as plain text.
    assert "explanation.\n\nYou are given an array." in built


def test_build_prompt_still_fences_a_bare_stub():
    """HumanEval+ stores a bare stub (no fence) — that must still get wrapped,
    unchanged from the behaviour its baseline was produced with."""
    built = build_prompt(_task())
    assert built.count("```") == 4
    assert "```python\ndef add" in built


def test_prompt_template_id_is_versioned_string():
    assert PROMPT_TEMPLATE_ID == "generic_instruct_v1"


def test_built_prompt_and_extractor_agree_on_response_format():
    # Integration check: a fake "model response" shaped the way build_prompt()
    # asked for it (one ```python fence) must extract cleanly as fenced_python —
    # proving the instruction text and the extractor's expectations agree.
    fake_response = f"```python\n{_STUB}    return a + b\n```"
    result = extract_code(fake_response)
    assert result.status == "fenced_python"
    assert result.ok
