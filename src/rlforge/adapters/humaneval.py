"""
adapters/humaneval.py
──────────────────────
load_humaneval_plus — convert the HumanEval+ benchmark (EvalPlus's corrected
HumanEval) into RLForge's canonical CodeTask (PLAN §2.2 ingestion).

Adapted from ReasoningLab `scripts/prepare_humaneval_plus.py`: same dataset
source (`evalplus.data.get_human_eval_plus()`), but the output shape and two
policy decisions differ:

  - The raw `test` field is a `def check(candidate): assert ...; assert ...`
    function body, not a bare list of asserts. `_parse_check_asserts` (new —
    ReasoningLab kept the whole function body as one opaque test_code string)
    splits it into individual assert STATEMENTS matching TestSuite(kind=
    "assert", asserts=[...]) exactly, so `ExecutionRecord.test_suite_size`
    reflects the real number of assertions instead of "1" for an unparsed blob.
    Uses Python's `ast` module, not line-splitting: some HumanEval+ asserts
    span multiple physical lines (e.g. a list literal broken across lines),
    which a `^assert...$`-per-line regex truncates into invalid syntax — caught
    by test_parsed_asserts_actually_run_against_the_reference_solution running
    the real 164-task set, not a hypothetical edge case.
  - The PROMPT IS KEPT VERBATIM. ReasoningLab wrapped the raw stub in its own
    instruction template ("Complete the following Python function. Return only
    one Markdown fenced Python block..."); we deliberately do NOT do that here.
    Rewriting a benchmark's canonical prompt is exactly the homogenization the
    Data & Evaluation Methodology (PLAN.md) rejects for benchmark data — it
    would sever comparability with published HumanEval+ pass@1 numbers. Any
    instruction framing belongs in a later, separate prompt-template layer,
    tagged via GenerationRecord.prompt_template_id so its effect stays
    auditable rather than baked silently into the task data.

HumanEval+ has no separate structured "visible" test data (the `>>>` doctest
examples live only as text inside the prompt's docstring), so converted tasks
have no "visible" key in test_suites — CodeTask.prompt_tests() already handles
that as a valid state (returns None), no special-casing needed here.
"""

from __future__ import annotations

import ast
import re

from rlforge.schema.task import CodeTask
from rlforge.schema.testsuite import TestSuite

# HumanEval+'s check() body calls the candidate as `candidate(...)`. We must
# rewrite this to the real function name so each assert is standalone-runnable
# against the actual generated function, matching what suite_to_test_code joins.
_CANDIDATE_CALL_RE = re.compile(r"\bcandidate\b")


def _source_of(test_code: str, node: ast.AST) -> str:
    """ast.get_source_segment, asserting the node's source was recovered.

    Only None when the node's line/col offsets fall outside test_code, which
    cannot happen here — node came from ast.parse(test_code) itself. Asserting
    (not silently coalescing to "") turns an impossible-in-practice case into a
    loud failure instead of a silently truncated test suite.
    """
    segment = ast.get_source_segment(test_code, node)
    assert segment is not None, "ast.get_source_segment unexpectedly returned None"
    return segment


def _load_humaneval_plus() -> dict:
    """Load the raw HumanEval+ problem set via the evalplus package.

    Returns a dict keyed by task_id (e.g. "HumanEval/0"); each value has keys
    prompt, entry_point, canonical_solution, test (and more).
    """
    try:
        from evalplus.data import get_human_eval_plus
    except ImportError as exc:
        raise RuntimeError(
            "evalplus is required to load HumanEval+. Install it with: "
            "pip install evalplus  (or: uv sync --extra data)"
        ) from exc
    return get_human_eval_plus()


def _parse_check_asserts(test_code: str, entry_point: str) -> list[str]:
    """Extract standalone assert statements from a HumanEval+ check() body.

    The raw `test` field looks like:
        METADATA = {...}

        def check(candidate):
            assert candidate(...) == ...
            assert candidate(...) == ...

        check(has_close_elements)

    Parses `test_code` with the `ast` module rather than regex/line matching —
    some HumanEval+ asserts span multiple physical lines (e.g. a list literal
    broken across lines), and a line-based `^assert...$` match truncates those
    into invalid syntax. `ast.get_source_segment` recovers the exact original
    source text for each node (whitespace and all).

    Two shapes occur in practice (checked against the real 164-task set):
      - Common case (163/164 tasks): check()'s body contains ONLY Assert
        statements — nothing else (no loops, no variable bindings one assert
        depends on another to set up). Each is genuinely independent, so they
        split into N separate entries and ExecutionRecord.test_suite_size
        reflects the real assertion count.
      - Rare case (1/164, e.g. HumanEval/32 "find_zero"; also seen: an assert
        preceded by sibling `lst = ...` / `odd_sum = ...` bindings it depends
        on, HumanEval/151): the body contains any non-Assert statement (loop,
        assignment, import, ...) alongside the asserts. Splitting individual
        assert lines would silently drop the bindings/control-flow they rely
        on — caught for real by running the full 164-task set, not a
        hypothetical (an assert with no syntax error can still silently
        reference an undefined name once split). The WHOLE function body is
        then kept as ONE asserts=[...] entry — honest, since it really is one
        procedural test, not N independent assertions.

    Raises:
        ValueError: the code does not parse, no `check` function is found, or
            it contains no assert statements — an empty/broken grading suite
            for a real benchmark task is a data/parsing bug, not a valid state
            to silently pass through.
    """
    tree = ast.parse(test_code)
    check_fn = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "check"),
        None,
    )
    if check_fn is None:
        raise ValueError(
            f"no `check` function found while parsing test code for "
            f"entry_point={entry_point!r}; the raw test field format may have "
            "changed upstream."
        )

    body = check_fn.body
    # "any assert anywhere" (for the empty-check guard) vs. "top-level asserts
    # only" (for the split decision below) are deliberately different queries:
    # find_zero's assert is nested inside a for-loop, so it is present in the
    # function but absent from `body` directly.
    any_assert_in_fn = any(isinstance(n, ast.Assert) for n in ast.walk(check_fn))
    if not any_assert_in_fn:
        raise ValueError(
            f"no assert statements found inside check() for "
            f"entry_point={entry_point!r}; the raw test field format may have "
            "changed upstream."
        )
    top_level_asserts = [n for n in body if isinstance(n, ast.Assert)]

    # Splittable ONLY if every top-level statement is itself an Assert — i.e.
    # no assert depends on a preceding sibling statement (an assignment, an
    # import, a loop it lives inside). Checking "is every assert independent"
    # by node-count comparison is not enough: HumanEval/151 has 7 top-level
    # Asserts and nothing else BUT one of those 7 references `lst`/`odd_sum`
    # bound by two ordinary Assign statements a few lines earlier — splitting
    # loses those bindings even though every assert node itself is top-level.
    # Requiring the WHOLE body to be Asserts-only rules this out correctly.
    if all(isinstance(n, ast.Assert) for n in body):
        return [
            _CANDIDATE_CALL_RE.sub(entry_point, _source_of(test_code, node))
            for node in top_level_asserts
        ]

    # Some non-Assert statement is present (import/assign/loop/...) that a
    # later assert may depend on — keep the whole body as ONE runnable block
    # rather than risk silently dropping a dependency.
    whole_body = "\n".join(_source_of(test_code, node) for node in body)
    return [_CANDIDATE_CALL_RE.sub(entry_point, whole_body)]


def _convert_problem(task_id: str, problem: dict) -> CodeTask:
    """Convert one raw HumanEval+ problem dict into a CodeTask.

    `canonical_solution` is only the function BODY (already indented as if
    inside a def) — evalplus expects it concatenated after `prompt`, which is
    the bare function stub (imports + signature + docstring, no body). Storing
    canonical_solution alone in reference_solution would be unrunnable
    (IndentationError: a body with no enclosing def). reference_solution must
    be the complete, standalone-runnable function, so it is prompt + solution.
    """
    asserts = _parse_check_asserts(problem["test"], problem["entry_point"])
    return CodeTask(
        task_id=task_id,
        source="humaneval_plus",
        prompt=problem["prompt"],  # verbatim — see module docstring
        entry_point=problem["entry_point"],
        reference_solution=problem["prompt"] + problem["canonical_solution"],
        test_suites={"hidden": TestSuite(kind="assert", asserts=asserts)},
        timeout_s=8.0,
    )


def load_humaneval_plus(limit: int | None = None) -> list[CodeTask]:
    """Load and convert the HumanEval+ benchmark into CodeTasks.

    Args:
        limit: if given, keep only the first `limit` tasks in task_id order
            (e.g. HumanEval/0, HumanEval/1, ...) — useful for a quick smoke
            slice before running the full 164-task set.

    Raises:
        RuntimeError: evalplus is not installed.
        ValueError: limit < 1, or a problem's test field could not be parsed.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided")

    raw = _load_humaneval_plus()
    # HumanEval/<n> task_ids sort correctly as strings only up to single
    # digits; sort by the numeric suffix to get stable, human-expected order.
    ordered_ids = sorted(raw.keys(), key=lambda tid: int(tid.rsplit("/", 1)[-1]))
    tasks = [_convert_problem(tid, raw[tid]) for tid in ordered_ids]

    if limit is not None:
        tasks = tasks[:limit]
    return tasks
