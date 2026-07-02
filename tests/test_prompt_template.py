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
