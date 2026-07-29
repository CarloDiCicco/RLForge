"""Tests for the stdin/stdout prompt template and the per-task dispatch.

The dispatch is the load-bearing part: sending a function-completion
instruction to a whole-program task makes the model answer with a bare `def`,
which prints nothing when the stdin_stdout harness execs it — every IO pair
mismatches and the task scores 0. These tests pin the routing so that
regression cannot happen silently.
"""

from __future__ import annotations

from rlforge.extract.extractor import extract_code
from rlforge.prompt_template import (
    PROMPT_TEMPLATE_ID,
    STDIO_PROMPT_TEMPLATE_ID,
    build_prompt_for_task,
    build_stdio_prompt,
    is_stdio_task,
)
from rlforge.schema.task import CodeTask
from rlforge.schema.testsuite import IOPair, TestSuite

_STATEMENT = (
    "Given two integers A and B, print A+B.\n\n"
    "Input\n\nA B\n\nOutput\n\nPrint the sum.\n\n"
    "Sample Input 1\n\n2 3\n\nSample Output 1\n\n5"
)
_STUB = 'def add(a: int, b: int) -> int:\n    """Return a + b."""\n'


def _stdio_task(**overrides) -> CodeTask:
    base = dict(
        task_id="LCB/abc999_a",
        source="livecodebench_stdio",
        prompt=_STATEMENT,
        test_suites={
            "hidden": TestSuite(
                kind="stdin_stdout",
                io_pairs=[IOPair(stdin="2 3\n", stdout="5\n")],
            )
        },
    )
    base.update(overrides)
    return CodeTask(**base)


def _assert_task() -> CodeTask:
    return CodeTask(
        task_id="t1",
        source="fixture",
        prompt=_STUB,
        entry_point="add",
        test_suites={"hidden": TestSuite(kind="assert", asserts=["assert add(1,1)==2"])},
    )


def test_is_stdio_task_true_for_stdin_stdout_suite():
    assert is_stdio_task(_stdio_task()) is True


def test_is_stdio_task_false_for_assert_suite():
    assert is_stdio_task(_assert_task()) is False


def test_dispatch_routes_stdio_task_to_stdio_template():
    prompt, template_id = build_prompt_for_task(_stdio_task())
    assert template_id == STDIO_PROMPT_TEMPLATE_ID
    assert "complete Python program" in prompt
    assert "standard input" in prompt


def test_dispatch_routes_assert_task_to_default_template():
    prompt, template_id = build_prompt_for_task(_assert_task())
    assert template_id == PROMPT_TEMPLATE_ID
    assert "Complete the following Python function" in prompt


def test_stdio_prompt_preserves_statement_verbatim():
    built = build_stdio_prompt(_stdio_task())
    assert _STATEMENT.rstrip() in built


def test_stdio_prompt_does_not_fence_the_prose_statement():
    """The statement is prose, not code: the only fence mentioned must be the
    one the ANSWER is requested in, never one wrapping the problem text."""
    built = build_stdio_prompt(_stdio_task())
    # exactly one ```python occurrence: the instruction about the answer format
    assert built.count("```python") == 1
    # and it appears before the statement (in the instruction), not around it
    assert built.index("```python") < built.index("Given two integers")


def test_stdio_prompt_carries_the_official_sample_inputs_guard():
    """The guard is LiveCodeBench's exact parenthetical, adopted verbatim: it
    stops models hardcoding the statement's sample answers instead of writing a
    general stdin-reading algorithm. Pinned verbatim so a rewording (which would
    silently deviate from the official string) fails this test."""
    built = build_stdio_prompt(_stdio_task())
    assert "(do not directly test on the sample inputs)" in built


def test_stdio_prompt_requests_single_fenced_python_block():
    built = build_stdio_prompt(_stdio_task())
    assert "Return only one Markdown fenced Python block" in built


def test_stdio_template_id_is_versioned_string():
    assert STDIO_PROMPT_TEMPLATE_ID == "stdio_instruct_v1"


def test_stdio_prompt_and_extractor_agree_on_response_format():
    # A response shaped the way the stdio template asks for it (one ```python
    # fence containing a whole program) must extract cleanly.
    fake_response = "```python\na,b=map(int,input().split())\nprint(a+b)\n```"
    result = extract_code(fake_response)
    assert result.status == "fenced_python"
    assert result.ok
