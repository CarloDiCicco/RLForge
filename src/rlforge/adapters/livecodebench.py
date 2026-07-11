"""
adapters/livecodebench.py
──────────────────────────
load_livecodebench_leetcode — convert LiveCodeBench's LeetCode-platform
problems into RLForge's canonical CodeTask (PLAN §2.2 ingestion).

SCOPE (deliberate, not a full LiveCodeBench adapter): LiveCodeBench mixes three
judging styles by source platform — LeetCode (function-call, assert-style),
AtCoder and Codeforces (stdin/stdout, competitive-programming style). Our
sandbox only runs `TestSuite(kind="assert", ...)` today (verify/compiler.py
raises NotImplementedError for kind="stdin_stdout"). Rather than build a
stdin/stdout harness now, this adapter converts ONLY the LeetCode subset, whose
test cases are function-call/return-value pairs that map onto our existing
assert infra with zero sandbox changes. Full LiveCodeBench coverage
(AtCoder/Codeforces) is deferred until stdin_stdout support is built.

RAW DATA SOURCE: HuggingFace dataset "livecodebench/code_generation_lite",
downloaded file-by-file via huggingface_hub.hf_hub_download (not
datasets.load_dataset — LCB's custom loading script is not reliably supported
by newer `datasets` releases). Caches to ~/.cache/huggingface/hub/, outside the
repo, same convention as evalplus's own ~/.cache/evalplus/ cache for HumanEval+.

TEST-CASE ENCODING: LiveCodeBench's `public_test_cases` field is always a
plain JSON string. `private_test_cases` is NOT consistently plain JSON —
inspecting real rows shows it is sometimes base64(zlib(pickle(...))) instead.
_decode_test_cases() below tries plain JSON first and falls back to the
base64/zlib/pickle decoding, so private tests are actually included rather
than silently parsed as empty on the rows using the second encoding.

CONTAMINATION SAFETY: deliberately NOT attempted. Qwen3.5's training cutoff is
credibly ~2026 (year-level precision only; DashScope API metadata, not an
official documented date from Alibaba); LiveCodeBench's latest available data
(test6.jsonl) only extends to ~April 2025. Every problem in this dataset is
therefore inside Qwen3.5's training window — there is no genuinely
contamination-safe slice available from LiveCodeBench for this model family,
so no date-based filtering is offered here (an "--after" cutoff would create a
false sense of safety). See RESEARCH_LOG.md for the full reasoning. This
adapter is used for benchmark diversity/difficulty (different problem style
than HumanEval+), not novelty-safety.

PROMPT WRAPPING: unlike adapters/humaneval.py (which keeps HumanEval+'s prompt
verbatim, since it IS the benchmark's canonical prompt), LiveCodeBench's raw
`question_content` is a natural-language problem statement with no code stub of
its own — there is no "canonical prompt" to preserve verbatim the way
HumanEval's docstring-as-prompt is. We still keep instruction-wrapping OUT of
this adapter's stored prompt, for the same reason as HumanEval+: wrapping
happens at generation time via prompt_template.build_prompt(), tagged by
prompt_template_id, so its effect stays auditable rather than baked into
stored data. The adapter concatenates the raw problem statement and starter
code as plain text; build_prompt() adds the "return one fenced code block"
instruction on top at generation time, same as it does for HumanEval+.
"""

from __future__ import annotations

import ast
import base64
import json
import pickle
import re
import zlib
from dataclasses import dataclass, field

from rlforge.schema.task import CodeTask
from rlforge.schema.testsuite import TestSuite


@dataclass
class ConversionStats:
    """Counts every place the conversion drops data.

    The whole point of this struct is that attrition is impossible to miss.
    Silent per-testcase drops are a nasty failure mode: a change in the upstream
    encoding could quietly shrink a ~28-test task down to a handful of tests with
    no error at all, leaving every downstream pass@1 number computed against a
    far weaker verifier than intended. The fix is not to stop dropping genuinely
    malformed rows (one bad row must not kill a 444-task load) but to COUNT every
    drop and report it, so an upstream format change surfaces loudly instead of
    quietly eating tests. load_livecodebench_leetcode() raises if the aggregate
    drop rate crosses a threshold, and always prints the breakdown.
    """

    leetcode_rows: int = 0
    tasks_built: int = 0
    rows_dropped_no_starter: int = 0
    rows_dropped_no_tests: int = 0
    rows_dropped_no_asserts: int = 0
    raw_testcases: int = 0
    asserts_built: int = 0
    testcase_decode_failures: int = 0
    dropped_task_ids: list[str] = field(default_factory=list)

    @property
    def testcases_dropped(self) -> int:
        return self.raw_testcases - self.asserts_built

    def summary(self) -> str:
        rows_dropped = (
            self.rows_dropped_no_starter
            + self.rows_dropped_no_tests
            + self.rows_dropped_no_asserts
        )
        tc_pct = (
            100 * self.testcases_dropped / self.raw_testcases
            if self.raw_testcases
            else 0.0
        )
        return (
            f"[adapters.livecodebench] leetcode_rows={self.leetcode_rows} "
            f"tasks_built={self.tasks_built} rows_dropped={rows_dropped} "
            f"(no_starter={self.rows_dropped_no_starter}, "
            f"no_tests={self.rows_dropped_no_tests}, "
            f"no_asserts={self.rows_dropped_no_asserts}) | "
            f"raw_testcases={self.raw_testcases} asserts_built={self.asserts_built} "
            f"testcases_dropped={self.testcases_dropped} ({tc_pct:.2f}%) "
            f"decode_failures={self.testcase_decode_failures}"
        )

# Maps a release tag to its cumulative set of raw split files — LiveCodeBench
# ships new problems as additional files, not a replaced/renamed single file.
_RELEASE_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": [
        "test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl",
        "test5.jsonl", "test6.jsonl",
    ],
}
_DEFAULT_RELEASE = "release_v6"

_HF_DATASET_ID = "livecodebench/code_generation_lite"

# Matches `class Solution:` / `def <method>(self, ...)` in a LeetCode starter
# stub, e.g. "class Solution:\n    def isValid(self, s: str) -> bool:\n".
_CLASS_RE = re.compile(r"class\s+(\w+)")
_METHOD_RE = re.compile(r"def\s+(\w+)\s*\(\s*self")


def _load_raw_rows(release: str) -> list[dict]:
    """Download and concatenate every split file for `release` via hf_hub_download.

    Raises:
        RuntimeError: huggingface_hub is not installed.
        ValueError: `release` is not a known release tag.
    """
    if release not in _RELEASE_FILES:
        raise ValueError(
            f"unknown release {release!r}; known releases: "
            f"{sorted(_RELEASE_FILES)}"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to load LiveCodeBench. Install it "
            "with: pip install huggingface_hub  (or: uv sync --extra data)"
        ) from exc

    rows: list[dict] = []
    for filename in _RELEASE_FILES[release]:
        path = hf_hub_download(_HF_DATASET_ID, filename=filename, repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _decode_test_cases(raw: str, stats: ConversionStats | None = None) -> list[dict]:
    """Parse a LiveCodeBench test-case field into a list of {input, output} dicts.

    Tries plain JSON first (the common case, and always true for
    `public_test_cases`); falls back to base64(zlib(pickle(...))) decoding,
    which some `private_test_cases` values actually use instead of plain JSON.

    Both forms may decode to a single {"input":..., "output":...} dict or a
    list of such dicts — normalized to a list either way.
    """
    if not raw or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            decompressed = zlib.decompress(base64.b64decode(raw))
            parsed = json.loads(pickle.loads(decompressed))
        except Exception:
            # A genuinely malformed test payload for this one problem — skip
            # it (mirrors the rest of this adapter's per-problem best-effort
            # policy: an unparseable row is dropped, not a hard failure for
            # the whole dataset load). Counted so the drop is never silent.
            if stats is not None:
                stats.testcase_decode_failures += 1
            return []

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def _parse_starter_code(starter_code: str) -> tuple[str, str] | None:
    """Extract (class_name, method_name) from a LeetCode starter stub.

    Returns None if either the class or the (self-taking) method cannot be
    found — e.g. an empty starter_code (non-LeetCode rows have "").
    """
    class_match = _CLASS_RE.search(starter_code)
    method_match = _METHOD_RE.search(starter_code)
    if class_match is None or method_match is None:
        return None
    return class_match.group(1), method_match.group(1)


def _build_assertion(class_var: str, method_name: str, tc: dict) -> str | None:
    """Build one `assert <class_var>.<method>(...) == <expected>` line.

    Returns None if the test case's input/output cannot be parsed into real
    Python values (ast.literal_eval failure) — skipped, not fatal for the
    whole problem, matching this adapter's per-testcase best-effort policy.

    LeetCode inputs are one literal per line, in parameter order; outputs are
    a single literal. LCB encodes booleans as the JSON-style string literals
    "true"/"false" rather than Python's True/False — normalized explicitly,
    since ast.literal_eval('true') raises (it is not a Python literal).
    """
    tc_input = tc.get("input", "")
    tc_output = tc.get("output", tc.get("expected_output", ""))
    input_lines = [line.strip() for line in tc_input.strip().split("\n") if line.strip()]
    try:
        args = [ast.literal_eval(line) for line in input_lines]
    except (ValueError, SyntaxError):
        return None

    output_str = str(tc_output).strip()
    if output_str == "true":
        expected: object = True
    elif output_str == "false":
        expected = False
    else:
        try:
            expected = ast.literal_eval(output_str)
        except (ValueError, SyntaxError):
            expected = output_str

    args_str = ", ".join(repr(a) for a in args)
    return f"assert {class_var}.{method_name}({args_str}) == {expected!r}"


def _convert_problem(
    problem: dict, stats: ConversionStats | None = None
) -> CodeTask | None:
    """Convert one raw LeetCode-platform LiveCodeBench row into a CodeTask.

    Returns None if the problem cannot be converted (unparseable starter
    code, no usable test cases) — the caller drops these and reports a count
    rather than failing the whole dataset load on one bad problem.

    When `stats` is provided, every drop (whole row or individual test case)
    is tallied there so the loader can report attrition instead of it being
    silent.
    """
    task_id_for_log = str(
        problem.get("question_id") or problem.get("question_title") or "unknown"
    )
    parsed_stub = _parse_starter_code(problem.get("starter_code", ""))
    if parsed_stub is None:
        if stats is not None:
            stats.rows_dropped_no_starter += 1
            stats.dropped_task_ids.append(task_id_for_log)
        return None
    class_name, method_name = parsed_stub

    all_tests = _decode_test_cases(
        problem.get("public_test_cases", ""), stats
    ) + _decode_test_cases(problem.get("private_test_cases", ""), stats)
    if stats is not None:
        stats.raw_testcases += len(all_tests)
    if not all_tests:
        if stats is not None:
            stats.rows_dropped_no_tests += 1
            stats.dropped_task_ids.append(task_id_for_log)
        return None

    class_var = "sol"
    assertions = [
        a for tc in all_tests if (a := _build_assertion(class_var, method_name, tc)) is not None
    ]
    if stats is not None:
        stats.asserts_built += len(assertions)
    if not assertions:
        if stats is not None:
            stats.rows_dropped_no_asserts += 1
            stats.dropped_task_ids.append(task_id_for_log)
        return None

    starter_code = problem["starter_code"].rstrip()
    question_content = problem.get("question_content", "").strip()
    prompt = f"{question_content}\n\n```python\n{starter_code}\n```"

    question_id = problem.get("question_id") or problem.get("question_title") or "unknown"
    return CodeTask(
        task_id=f"LCB/{question_id}",
        source="livecodebench_leetcode",
        prompt=prompt,
        entry_point=method_name,
        test_suites={
            "hidden": TestSuite(
                kind="assert",
                asserts=[f"{class_var} = {class_name}()"] + assertions,
            )
        },
        # LCB problems (competitive-programming-adjacent) run longer than
        # HumanEval+'s simple function bodies — bumped above the schema
        # default (8.0s) for the same reason ReasoningLab's script did.
        timeout_s=10.0,
        metadata={
            "difficulty": problem.get("difficulty"),
            "contest_date": problem.get("contest_date"),
        },
    )


# If more than this fraction of decoded test cases fail to become asserts, the
# loader raises rather than returning a silently-weakened dataset. 0.0 raw data
# today (measured: 12766/12766 asserts built, 0% dropped); a small nonzero
# tolerance leaves room for a genuinely malformed future row without masking the
# kind of large-scale attrition that would indicate an upstream format change.
_MAX_TESTCASE_DROP_RATE = 0.00


def load_livecodebench_leetcode(
    release: str = _DEFAULT_RELEASE,
    limit: int | None = None,
    max_testcase_drop_rate: float = _MAX_TESTCASE_DROP_RATE,
) -> list[CodeTask]:
    """Load and convert LiveCodeBench's LeetCode-platform problems into CodeTasks.

    Only `platform == "leetcode"` rows are considered (see module docstring
    for why AtCoder/Codeforces are out of scope for now). Rows whose starter
    code or test cases could not be parsed are silently dropped; a count is
    printed so a large drop rate is visible, not just invisible attrition.

    Args:
        release: which cumulative LiveCodeBench release to load (default:
            the latest, "release_v6"). See _RELEASE_FILES.
        limit: if given, keep only the first `limit` converted tasks (in
            dataset iteration order) — useful for a quick smoke slice.

    Raises:
        RuntimeError: huggingface_hub is not installed, or test-case attrition
            exceeds `max_testcase_drop_rate` (a signal the upstream encoding
            changed and asserts are being silently discarded).
        ValueError: unknown release tag, or limit < 1 when provided.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided")

    raw_rows = _load_raw_rows(release)
    leetcode_rows = [r for r in raw_rows if r.get("platform") == "leetcode"]

    stats = ConversionStats(leetcode_rows=len(leetcode_rows))
    tasks: list[CodeTask] = []
    for row in leetcode_rows:
        task = _convert_problem(row, stats)
        if task is None:
            continue
        tasks.append(task)
    stats.tasks_built = len(tasks)

    # Always report the full attrition breakdown — never let a drop be silent.
    print(stats.summary())

    # Guardrail: a large-scale test-case drop almost certainly means the upstream
    # encoding changed and we are about to grade against a gutted verifier. Fail
    # loudly rather than return a dataset that looks fine but is not.
    if stats.raw_testcases:
        drop_rate = stats.testcases_dropped / stats.raw_testcases
        if drop_rate > max_testcase_drop_rate:
            raise RuntimeError(
                f"LiveCodeBench test-case attrition {drop_rate:.2%} exceeds the "
                f"allowed {max_testcase_drop_rate:.2%} "
                f"({stats.testcases_dropped}/{stats.raw_testcases} test cases dropped). "
                "This usually means the upstream test-case encoding changed and "
                "asserts are being silently discarded. Inspect the adapter before "
                "trusting any pass@1 computed against this data.\n" + stats.summary()
            )

    if limit is not None:
        tasks = tasks[:limit]
    return tasks