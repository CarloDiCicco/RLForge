"""
prompt_template/default_prompt_template.py
────────────────────────────────────────────
build_prompt — wrap a CodeTask's canonical prompt in an instruction, for
sending to an instruct-tuned model (PLAN §2.3 generation).

CodeTask.prompt is stored verbatim (see adapters/humaneval.py and the Data &
Evaluation Methodology in PLAN.md) — never rewritten, so benchmark numbers stay
comparable to the literature. But a bare function stub with no instruction is a
degenerate prompt for a chat model: nothing tells it to complete the function,
or how to format the answer so extract.extractor.extract_code (which prefers a
```python fence) can find it.

This is a harness-level concern, not a dataset one — the same instruction
wrapping every HumanEval leaderboard (BigCode, EvalPlus, ...) applies for
instruct models, without altering the benchmark itself. It is its own pipeline
responsibility (building the model's input text), separate from calling a model
(the actual generation step) even though the two run back-to-back — the same
separation of concerns that already keeps extract/ (parse text) and verify/
(compile a test suite) as distinct packages despite always running in sequence.
Not fused into CodeTask.prompt: GenerationRecord.prompt_template_id exists
precisely so which template produced a given prompt stays a recorded, swappable
fact, not a silent assumption baked into the stored data.

Structure adopted from ReasoningLab's prepare_humaneval_plus.py /
prepare_livecodebench.py (both scripts independently converged on the same
shape): one instruction paragraph BEFORE the code, and the task's own prompt
placed inside a ```python fence too — not just the instruction about the
expected output format. Showing the model "your input is a python block" cues
it to continue that exact formatting convention in its answer. This is the
structure ReasoningLab actually ran experiments with, not an untested guess;
generalized here from a hardcoded HumanEval+ stub to any CodeTask.prompt.
"""

from __future__ import annotations

from rlforge.schema.task import CodeTask

# Stamped onto GenerationRecord.prompt_template_id. Versioned (_v1) so a future
# wording change becomes a new id rather than silently redefining this one —
# old generation records stay correctly attributed to the template that
# actually produced them.
PROMPT_TEMPLATE_ID = "generic_instruct_v1"


def build_prompt(task: CodeTask) -> str:
    """Wrap task.prompt in a generic instruction for an instruct-tuned model.

    task.prompt is included verbatim, untouched, inside a ```python fence —
    only the surrounding instruction is added. Matches extract_code()'s
    "prefer the first ```python fence" priority, so the two ends of the
    pipeline agree on the expected format.
    """
    return (
        "Complete the following Python function. "
        "Return only one Markdown fenced Python block (```python ... ```), "
        "with no extra explanation.\n\n"
        f"```python\n{task.prompt.rstrip()}\n```"
    )
