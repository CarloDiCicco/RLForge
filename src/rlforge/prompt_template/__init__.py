"""Prompt-building: wrap a CodeTask's canonical prompt for a model call.

Two templates exist because the benchmarks pose two structurally different
tasks, and an instruction written for one produces a broken prompt for the other
(see stdio_prompt_template's module docstring for how the mismatch silently
zeroes a run):

  - "assert" suites (HumanEval+, LiveCodeBench-LeetCode): complete a FUNCTION
    from a starter stub  -> default_prompt_template.build_prompt
  - "stdin_stdout" suites (LiveCodeBench AtCoder/Codeforces): write a whole
    PROGRAM that reads stdin and prints stdout
    -> stdio_prompt_template.build_stdio_prompt

`build_prompt_for_task` is the single dispatch point: callers pass a CodeTask
and get the right prompt plus the id of the template that produced it, so
GenerationRecord.prompt_template_id always names what actually ran. Adding a
future task shape is a new branch here, not a change at every call site.
"""

from __future__ import annotations

from rlforge.prompt_template.default_prompt_template import (
    PROMPT_TEMPLATE_ID,
    build_prompt,
)
from rlforge.prompt_template.stdio_prompt_template import (
    STDIO_PROMPT_TEMPLATE_ID,
    build_stdio_prompt,
)
from rlforge.schema.task import CodeTask

__all__ = [
    "PROMPT_TEMPLATE_ID",
    "STDIO_PROMPT_TEMPLATE_ID",
    "build_prompt",
    "build_stdio_prompt",
    "build_prompt_for_task",
    "is_stdio_task",
]


def is_stdio_task(task: CodeTask) -> bool:
    """True iff this task is graded by the stdin/stdout harness.

    Routes on the task's own grading suites (any suite of kind "stdin_stdout"),
    not on `entry_point is None` — the suite kind is what the verifier actually
    dispatches on (verify/compiler.py), so selecting the prompt the same way
    keeps the two ends of the pipeline from ever disagreeing about what shape of
    answer a task wants.
    """
    return any(
        suite.kind == "stdin_stdout" for suite in task.test_suites.values()
    )


def build_prompt_for_task(task: CodeTask) -> tuple[str, str]:
    """Build the prompt for `task` with the template matching its task shape.

    Returns:
        (final_prompt, prompt_template_id) — the id belongs with the prompt so a
        caller cannot record one template while having sent another.
    """
    if is_stdio_task(task):
        return build_stdio_prompt(task), STDIO_PROMPT_TEMPLATE_ID
    return build_prompt(task), PROMPT_TEMPLATE_ID
