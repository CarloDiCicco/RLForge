"""
prompt_template/stdio_prompt_template.py
─────────────────────────────────────────
build_stdio_prompt — the instruction wrapper for stdin/stdout tasks
(AtCoder/Codeforces, TestSuite kind="stdin_stdout").

Why a second template instead of reusing default_prompt_template.build_prompt:
that one says "Complete the following Python function" and places the task's
prompt INSIDE a ```python fence. Both are wrong here, and not cosmetically —
they would silently produce a meaningless baseline:

  - There is no function to complete. A stdin/stdout task's prompt is a prose
    problem statement with no starter stub (entry_point is None), so "complete
    the function" describes nothing in the input.
  - Wrapping a prose statement in a ```python fence presents English text as if
    it were code.
  - Worst: told to "complete a function", the model tends to answer with a bare
    `def solve(...)`. The stdin_stdout harness execs the candidate as a PROGRAM
    and compares what it prints — a module that only defines a function prints
    nothing, so every IO pair mismatches and the task scores 0. The run would
    report a near-zero pass@1 that measures the prompt, not the model.

So this template instructs a complete, runnable program that reads stdin and
writes stdout, and passes the statement through as plain text (it is prose, and
it already contains the problem's sample cases — see the adapter: LiveCodeBench
statements carry their own "Sample Input/Output" sections, which is exactly what
a human competitor sees; we inject no test cases of our own, and the private
tests never appear in any prompt).

What is kept identical to the default template, deliberately:
  - task.prompt is included VERBATIM, never rewritten, so benchmark numbers stay
    comparable to the literature.
  - The answer is requested as one ```python fenced block, matching
    extract.extractor.extract_code's "prefer the first ```python fence"
    priority, so both ends of the pipeline still agree on the format.
  - The id is versioned, so a future wording change becomes a new id rather than
    silently redefining this one.
"""

from __future__ import annotations

from rlforge.schema.task import CodeTask

# Stamped onto GenerationRecord.prompt_template_id for stdin/stdout tasks, so a
# record always names the template that actually produced its prompt.
STDIO_PROMPT_TEMPLATE_ID = "stdio_instruct_v1"


def build_stdio_prompt(task: CodeTask) -> str:
    """Wrap a stdin/stdout task's statement in a whole-program instruction.

    task.prompt is included verbatim as plain text (not fenced — it is a prose
    problem statement, and it already carries the problem's own sample cases).
    Only the surrounding instruction is added.
    """
    return (
        "Write a complete Python program that solves the following problem. "
        "The program must read its input from standard input and write its "
        "answer to standard output "
        # Guard taken VERBATIM from LiveCodeBench's official template (their
        # exact parenthetical, unreworded on purpose — it is the string the LCB
        # ecosystem runs with). It stops models from hardcoding the statement's
        # sample answers or embedding self-test scaffolding against them,
        # instead of writing a general algorithm that reads stdin.
        "(do not directly test on the sample inputs). "
        "Return only one Markdown fenced Python block (```python ... ```), "
        "with no extra explanation.\n\n"
        f"{task.prompt.rstrip()}"
    )
