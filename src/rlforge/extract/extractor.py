"""
extract/extractor.py
────────────────────
Turn raw model output into runnable candidate code (PLAN §1.2).

A code model rarely emits a clean function. Its output may be wrapped in one or
more markdown fences, prefixed with prose, or carry a `<think>…</think>` reasoning
trace. This module reduces that to a single code string, and — crucially for a
research harness — records *why* it chose what it chose.

Design (adapted from ReasoningLab `policies/_utils.py::_extract_candidate_code`):
strip `<think>` blocks, then prefer the first python-tagged fence, else the first
fence of any kind, else fall back to the raw text. RLForge adds three things on
top of that core:

  1. A structured `ExtractionResult` (not a bare string) carrying a `status` and
     a human-readable `reason`, so every decision is auditable in the registry.
  2. A syntax pre-check (`compile`): code that cannot even parse is caught here,
     before the sandbox, and flagged so the pipeline labels it FailureType.
     EXTRACTION rather than wasting a sandboxed run on it.
  3. An empty guard: whitespace-only extraction is its own analysable state.

The pipeline maps `status in {"empty", "syntax_error"}` to a pre-execution
EXTRACTION failure and skips the sandbox; every other status proceeds to run.
This module does NOT import the taxonomy — it stays a leaf that only knows about
extraction, and the pipeline owns the status→FailureType mapping.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from rlforge import FORMAT_VERSION

# Remove chain-of-thought reasoning before scanning for code: a model may emit
# illustrative fenced blocks *inside* <think>…</think> that are not the answer,
# and selecting one of those over the real solution would be a silent bug.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# A markdown fenced block: ```<lang>\n<code>``` . `lang` is the optional info
# string after the opening backticks (e.g. "python"); `code` is the body.
_FENCED_BLOCK_RE = re.compile(
    r"```(?P<lang>[^\n`]*)\n(?P<code>.*?)```",
    re.DOTALL,
)

# Allowed status values, in rough "happy path → failure" order. Kept as a tuple
# (not an Enum) because these are extraction-local labels stored verbatim in
# GenerationRecord.extraction_status; the pipeline maps them to FailureType.
STATUS_FENCED_PYTHON = "fenced_python"   # chose a ```python fence
STATUS_FENCED_OTHER = "fenced_other"     # chose a fence with no/other lang tag
STATUS_RAW_FALLBACK = "raw_fallback"     # no fences; used the whole text
STATUS_EMPTY = "empty"                   # nothing runnable after stripping
STATUS_SYNTAX_ERROR = "syntax_error"     # extracted, but does not parse

# Statuses the pipeline must treat as a failure *before* the sandbox.
PRE_EXECUTION_FAILURES: frozenset[str] = frozenset({STATUS_EMPTY, STATUS_SYNTAX_ERROR})


class ExtractionResult(BaseModel):
    """The outcome of extracting code from one raw model output (PLAN §1.2)."""

    format_version: str = Field(default=FORMAT_VERSION)

    # The extracted code. Populated even on syntax_error (so the record shows
    # what failed to parse); empty string only when status == "empty".
    code: str
    # One of the STATUS_* constants above. Stored in
    # GenerationRecord.extraction_status verbatim.
    status: str
    # Human-readable log of the decision taken, for auditing in the registry.
    reason: str
    # Whether any markdown fence was present in the (think-stripped) output.
    had_fences: bool

    @property
    def ok(self) -> bool:
        """True iff the extracted code is runnable (should reach the sandbox)."""
        return self.status not in PRE_EXECUTION_FAILURES


def _syntax_ok(code: str) -> tuple[bool, str | None]:
    """Compile-check `code`; return (ok, error_message).

    Uses compile(..., "exec") — the same parse the interpreter does at import —
    without executing anything, so it is safe to run on the host. We only care
    whether it parses; the message is kept for the decision log.
    """
    try:
        compile(code, "<extracted>", "exec")
        return True, None
    except SyntaxError as exc:
        # exc carries the line/offset; str(exc) is the compact human message.
        return False, str(exc)


def extract_code(raw_output: str) -> ExtractionResult:
    """Extract runnable code from one raw model output, logging the decision.

    Priority (after stripping <think> blocks):
      1. the first ```python fence anywhere in the output,
      2. else the first fence of any language tag,
      3. else the whole text (no fences present).
    The chosen code is then guarded for emptiness and pre-checked for syntax.
    """
    # 1. Strip reasoning traces so fences inside them cannot be selected.
    text = _THINK_RE.sub("", raw_output)
    matches = list(_FENCED_BLOCK_RE.finditer(text))
    had_fences = bool(matches)

    # 2. Choose the code string + provisional status/reason. The three cases
    #    grab DIFFERENT text on purpose:
    #
    #      - no fences         → the WHOLE (think-stripped) output is the code.
    #                            Surrounding prose is unlikely (the model gave
    #                            bare code), and there is nothing better to take.
    #      - a python fence    → ONLY that fence's body. The prose around it
    #                            ("Here is my solution:") is not code and would
    #                            break the syntax pre-check, so we discard it.
    #      - fences, no python → ONLY the first fence's body, same reasoning.
    #
    #    `m.group("code")` is the named capture group in _FENCED_BLOCK_RE: the
    #    text between the opening ```<lang>\n and the closing ``` of one fence.
    if not matches:
        code = text.strip()
        status = STATUS_RAW_FALLBACK
        reason = "no markdown fence found; used the whole output as code"
    else:
        # Prefer a python-tagged fence wherever it appears, taking only its body.
        python_match = next(
            (m for m in matches if m.group("lang").strip().lower() == "python"),
            None,
        )
        if python_match is not None:
            code = python_match.group("code").strip()  # body of the python fence
            status = STATUS_FENCED_PYTHON
            reason = "selected the first ```python fence"
        else:
            # No python fence: take the body of the FIRST fence of any tag.
            code = matches[0].group("code").strip()
            lang = matches[0].group("lang").strip() or "<none>"
            status = STATUS_FENCED_OTHER
            reason = (
                f"no python fence; selected the first fence (lang={lang!r}) "
                f"of {len(matches)} present"
            )

    # 3. Empty guard: nothing runnable was produced. Overrides the status above
    #    because an empty raw_fallback / empty fence is the same failure mode.
    if not code.strip():
        return ExtractionResult(
            code="",
            status=STATUS_EMPTY,
            reason="extraction yielded only whitespace; no runnable code",
            had_fences=had_fences,
        )

    # 4. Syntax pre-check: catch unparseable code before the sandbox. Keep the
    #    code (for the record) but flag it so the pipeline labels it EXTRACTION.
    parses, err = _syntax_ok(code)
    if not parses:
        return ExtractionResult(
            code=code,
            status=STATUS_SYNTAX_ERROR,
            reason=f"extracted code failed to parse: {err}",
            had_fences=had_fences,
        )

    return ExtractionResult(
        code=code, status=status, reason=reason, had_fences=had_fences
    )
