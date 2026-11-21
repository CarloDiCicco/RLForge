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
    "KNOWN_TEMPLATES",
    "check_prompt_templates",
]

# Every template id a run can name in its config, with the task shape it serves.
# A run lists the ids it expects (YAML `prompt_templates` for training,
# --prompt-templates for evaluation); check_prompt_templates enforces the list.
KNOWN_TEMPLATES = {
    STDIO_PROMPT_TEMPLATE_ID: "stdin/stdout program",
    PROMPT_TEMPLATE_ID: "complete a function (starter stub)",
}


def check_prompt_templates(counts: dict[str, int], allowed: list[str]) -> None:
    """Fail if a run's prompts used a template the run did not declare.

    `counts` = {template id: number of prompts built with it}, `allowed` = the ids
    the config declares. Two mistakes are caught before any GPU work: a typo in
    the config (an id that does not exist), and a task set that contains a shape
    the run was not meant to include (e.g. function-call tasks slipping into a
    stdin/stdout-only run). A declared template with no prompts is allowed (a
    smoke run on the first few tasks may not reach both shapes)."""
    unknown = sorted(set(allowed) - set(KNOWN_TEMPLATES))
    if unknown:
        raise ValueError(f"unknown prompt template id(s) {unknown}; known: {sorted(KNOWN_TEMPLATES)}")
    undeclared = {t: n for t, n in counts.items() if t not in allowed}
    if undeclared:
        raise ValueError(
            f"prompts built with template(s) not declared in the config: {undeclared} "
            f"(declared: {sorted(allowed)})"
        )


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
