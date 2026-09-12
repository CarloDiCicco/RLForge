"""
adapters/taco.py
────────────────
load_taco — convert BAAI/TACO into RLForge's canonical CodeTask, for use as a
TRAINING pool (SFT / RL), never as a benchmark.

WHY TACO. Training needs a task pool disjoint from the three frozen benchmarks
(HumanEval+, LiveCodeBench-LeetCode, LiveCodeBench-stdio). TACO is the standard
choice in the RLVR-for-code literature (DeepCoder used TACO-Verified; Code-R1
used a ~10k TACO subset) and, decisively, it ships REFERENCE SOLUTIONS. Those
let us verify that each problem's *tests* are themselves correct by running the
official solution against them (verify/reference_verifier.py). Without that, a
broken verifier silently teaches an RL policy that correct code is wrong, and
DeepCoder documented models learning to print memorised answers when tests are
too weak. A dataset without reference solutions cannot be trusted as a reward.

SCOPE: both judging styles, in one kind-tagged output. TACO's `input_output`
carries an optional "fn_name" key: present => function-call/assert style,
absent => stdin/stdout. These map exactly onto TestSuite kinds "assert" and
"stdin_stdout", both of which the sandbox and prompt layers already dispatch on
(prompt_template.build_prompt_for_task switches on suite kind), so no change is
needed anywhere downstream. Measured on train shard 0: 2467 stdio / 360 fn_name.

RAW DATA SOURCE: HuggingFace "BAAI/TACO", read from the `ALL/*.parquet` shards
via huggingface_hub.hf_hub_download + pyarrow — NOT datasets.load_dataset (TACO
ships a custom loading script, the same reason adapters/livecodebench.py avoids
it). Caches under HF_HOME (~/data/cache/huggingface), outside the repo.

THE CENTRAL DESIGN ARGUMENT — why the heuristics below are allowed to be
imperfect. Converting TACO's function-call rows into assert strings requires
guesses about an under-documented format (how args are packed, whether an
expected value is wrapped in a list). Those guesses do not have to be right,
because STAGE 3 RUNS EACH PROBLEM'S OWN OFFICIAL SOLUTION AGAINST THE ASSERTS
THIS MODULE BUILT and drops every problem that fails. A wrong heuristic costs
recall (some good problems are discarded), never correctness (a broken task
cannot reach the training pool). Measured on a 25-row sample of fn_name rows:
21/25 pass, all 4 failures being non-class rows with variable-arity signatures.

FORMAT FACTS, all measured on the real data rather than assumed (train shard 0,
1000-row test split; see RESEARCH_LOG.md):
  - `inputs[i]` is a LIST OF POSITIONAL ARGUMENTS for fn_name rows (unlike
    LiveCodeBench, whose inputs were newline-joined literal *strings*).
  - `outputs[i]` wraps the return value in a 1-element list 2354/2354 times in
    the test split, but is a bare scalar on ~91 rows of train shard 0. Hence
    _normalize_expected unwraps length-1 lists and passes everything else
    through.
  - String arguments are used VERBATIM, including any surrounding quote
    characters. This was tested both ways on a real row (train/38,
    minFlipsMonoIncr, whose input is the JSON string '"00110"'): passing the
    raw value PASSES its reference solution and stripping the quotes FAILS it.
    So no unquoting is attempted — the quotes are part of the string.
  - 358/360 fn_name rows in train shard 0 carry `class Solution:` starter code,
    so the class-wrapped shape is the COMMON case here, not a rarity.
  - TACO's stored solutions carry their own imports (`from typing import List`
    etc.), so no typing preamble is injected; verified by running a real
    class-Solution row end to end through the sandbox.

TASK IDs ARE CONTENT-ADDRESSED. `TACO/<sha256(normalized question)[:12]>`.
TACO rows have no reliable native id (`name` is blank or duplicated across
platforms), and a positional id (split/row_index) would silently re-point if
upstream re-shards or re-uploads: row 10422 would then name a DIFFERENT problem,
and the Stage-2 contamination list plus the Stage-3 verification ledger would
apply to the wrong tasks with no error raised anywhere. Hashing the normalized
statement gives the same problem the same id on every version of the dataset, so
those artifacts survive a re-download. The HF revision sha and row index are
still recorded, as provenance only — nothing depends on them for identity.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from rlforge.schema.task import CodeTask
from rlforge.schema.testsuite import IOPair, TestSuite

# ── Dataset coordinates ──────────────────────────────────────────────────────

_HF_DATASET_ID = "BAAI/TACO"

# Pinned so that a re-download reproduces byte-identical data. Recorded in every
# task's metadata as provenance (identity comes from the content hash instead).
_TACO_REVISION = "d593ed0a2becbbc952230bb89be09189bf1056dc"

# The parquet shards under ALL/. Listed explicitly rather than globbed so a
# change in upstream layout fails loudly instead of silently reading less data.
_SHARDS: dict[str, tuple[str, ...]] = {
    "test": ("ALL/test-00000-of-00001.parquet",),
    "train": tuple(f"ALL/train-0000{i}-of-00009.parquet" for i in range(9)),
}

DEFAULT_SPLITS: tuple[str, ...] = ("train", "test")

# Per-testcase attrition above this fraction aborts the load. LiveCodeBench uses
# 0.0 because its encoding is uniform; TACO is a nine-platform scrape with known
# format variance, so an honest non-zero bound is set up front rather than
# quietly relaxed later when the guard first fires.
_MAX_TESTCASE_DROP_RATE = 0.05

# A single assert line longer than this is dropped. Guards against TACO rows
# with enormous inlined literals: LiveCodeBench's worst task (36.7 MB of assert
# source) had to be discovered the hard way when it OOM-killed the sandbox
# during COMPILATION, before any candidate code ran (see RESEARCH_LOG.md).
_MAX_ASSERT_CHARS = 100_000

# Above this, ast.literal_eval is not attempted on a malformed JSON payload —
# literal_eval on a huge string is pathologically slow and these rows are junk.
_MAX_LITERAL_EVAL_CHARS = 50_000_000


@dataclass
class TacoConversionStats:
    """Counts every place the conversion drops or reshapes data.

    Same discipline as adapters/livecodebench.py's ConversionStats: silent
    attrition is the failure mode that quietly invalidates every downstream
    number, so each drop point gets its own counter and `summary()` is always
    printed. The histograms exist because the Stage-1 RESEARCH_LOG entry needs
    the composition of the pool, not just its size.
    """

    rows_seen: int = 0
    tasks_built: int = 0

    # Routing.
    rows_assert: int = 0
    rows_stdio: int = 0

    # Whole-row drops.
    rows_dropped_empty_question: int = 0
    rows_dropped_bad_input_output: int = 0
    rows_dropped_no_tests: int = 0
    rows_dropped_no_asserts: int = 0
    rows_dropped_no_io_pairs: int = 0
    rows_dropped_duplicate_id: int = 0

    # Observations (NOT drops) — a row with no solution is still converted; it
    # is Stage 3 that removes it, since it has nothing to verify against.
    rows_without_solutions: int = 0
    rows_with_pictures: int = 0

    # Per-testcase attrition.
    raw_testcases: int = 0
    asserts_built: int = 0
    io_pairs_built: int = 0
    testcases_dropped_length_mismatch: int = 0
    testcases_dropped_unreprable: int = 0
    testcases_dropped_oversized: int = 0
    testcases_dropped_bad_stdio: int = 0

    # Parse failures.
    solution_decode_failures: int = 0
    input_output_decode_failures: int = 0

    # Composition, for the log entry.
    by_platform: dict[str, int] = field(default_factory=dict)
    by_difficulty: dict[str, int] = field(default_factory=dict)
    dropped_task_ids: list[str] = field(default_factory=list)

    def testcase_drop_rate(self) -> float:
        """Fraction of raw test cases that did not survive into a suite."""
        if self.raw_testcases == 0:
            return 0.0
        kept = self.asserts_built + self.io_pairs_built
        return 1.0 - (kept / self.raw_testcases)

    def summary(self) -> str:
        return (
            f"TACO: {self.rows_seen} rows -> {self.tasks_built} tasks "
            f"({self.rows_assert} assert / {self.rows_stdio} stdio); "
            f"testcases {self.raw_testcases} -> "
            f"{self.asserts_built + self.io_pairs_built} "
            f"(drop rate {self.testcase_drop_rate():.4f}); "
            f"dropped rows: empty_question={self.rows_dropped_empty_question}, "
            f"bad_io={self.rows_dropped_bad_input_output}, "
            f"no_tests={self.rows_dropped_no_tests}, "
            f"no_asserts={self.rows_dropped_no_asserts}, "
            f"no_io_pairs={self.rows_dropped_no_io_pairs}, "
            f"duplicate_id={self.rows_dropped_duplicate_id}; "
            f"observations: no_solutions={self.rows_without_solutions}, "
            f"pictures={self.rows_with_pictures}"
        )

    def composition(self) -> str:
        """Second summary line: what the surviving pool is made of."""
        plats = ", ".join(f"{k}={v}" for k, v in sorted(self.by_platform.items()))
        diffs = ", ".join(f"{k}={v}" for k, v in sorted(self.by_difficulty.items()))
        return f"TACO composition: platforms [{plats}]; difficulty [{diffs}]"


# ── Statement normalization + content-addressed ids ──────────────────────────


def normalize_statement(text: str) -> str:
    """Canonical form of a problem statement, used for ids and for dedup.

    FROZEN once Stage 1 has run: every task id is a hash of this output, so
    changing it renames every task and orphans the Stage-2/Stage-3 artifacts.
    Treat it like the prompt templates — a change is a data migration, not an
    edit. (The same function is reused by decontaminate/statement.py so ids and
    similarity always see identical text.)
    """
    return " ".join(text.split()).lower()


def content_task_id(question: str) -> str:
    """`TACO/<12 hex>` — stable across dataset re-uploads. See module docstring."""
    digest = hashlib.sha256(normalize_statement(question).encode("utf-8")).hexdigest()
    return f"TACO/{digest[:12]}"


# ── Parsing helpers (each independently unit-tested) ─────────────────────────


def _parse_json_field(raw: Any) -> Any | None:
    """Decode a TACO JSON-string column, tolerating Python-repr'd rows.

    TACO stores `solutions` and `input_output` as JSON *strings*, but the scrape
    is not uniformly strict JSON — some rows are Python reprs (single quotes,
    True/None). Try json first, then ast.literal_eval, then give up. Returns
    None on failure; the caller counts it.
    """
    if raw is None:
        return None
    if isinstance(raw, (list, dict)):
        return raw  # some readers pre-parse
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    if len(raw) > _MAX_LITERAL_EVAL_CHARS:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def _parse_solutions(raw: Any, stats: TacoConversionStats) -> list[str]:
    """Decoded list of reference-solution sources (may be empty)."""
    parsed = _parse_json_field(raw)
    if parsed is None:
        if raw:  # a non-empty field that failed to decode is a real failure
            stats.solution_decode_failures += 1
        return []
    if not isinstance(parsed, list):
        stats.solution_decode_failures += 1
        return []
    return [s for s in parsed if isinstance(s, str) and s.strip()]


def _parse_input_output(
    raw: Any, stats: TacoConversionStats
) -> tuple[list[Any], list[Any], str | None] | None:
    """Return (inputs, outputs, fn_name), or None if the row is unusable.

    A length mismatch between inputs and outputs is a genuine TACO defect;
    zipping silently would misalign every subsequent test case, so the excess is
    truncated and counted instead.
    """
    parsed = _parse_json_field(raw)
    if parsed is None:
        stats.input_output_decode_failures += 1
        return None
    if not isinstance(parsed, dict):
        return None
    inputs, outputs = parsed.get("inputs"), parsed.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return None

    if len(inputs) != len(outputs):
        keep = min(len(inputs), len(outputs))
        stats.testcases_dropped_length_mismatch += max(len(inputs), len(outputs)) - keep
        inputs, outputs = inputs[:keep], outputs[:keep]

    fn_name = parsed.get("fn_name")
    if not isinstance(fn_name, str) or not fn_name.strip():
        fn_name = None
    return inputs, outputs, fn_name


def _is_assert_row(fn_name: str | None) -> bool:
    """Routing predicate: fn_name present => function-call/assert judging."""
    return fn_name is not None


def _normalize_expected(raw_output: Any) -> Any:
    """Unwrap TACO's 1-element output wrapper, pass everything else through.

    Measured: 2354/2354 outputs in the test split are 1-element lists, but train
    shard 0 also contains bare scalars and a handful of genuine multi-element
    lists. Unwrapping exactly length-1 covers all three correctly.
    """
    if isinstance(raw_output, (list, tuple)) and len(raw_output) == 1:
        return raw_output[0]
    return raw_output


def _is_reprable(value: Any) -> bool:
    """True iff `repr(value)` round-trips through ast.literal_eval.

    Filters values whose repr is not a valid literal — `inf`/`nan` repr as bare
    names (NameError at run time) and nan breaks equality asserts outright
    (nan != nan makes the assert unsatisfiable regardless of correctness).
    """
    try:
        ast.literal_eval(repr(value))
        return True
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return False


_CLASS_MARKER = "class Solution"


def _is_class_wrapped(starter_code: str | None) -> bool:
    """True for LeetCode-style rows whose solution is a method on `Solution`.

    358/360 fn_name rows in train shard 0 are class-wrapped, so this is the
    common shape. Detected from starter code rather than guessed, because the
    call form differs: `sol.method(...)` vs a bare `fn(...)`.
    """
    return bool(starter_code) and _CLASS_MARKER in starter_code


def _build_assertion(
    fn_name: str,
    args: Any,
    expected: Any,
    *,
    class_wrapped: bool,
    stats: TacoConversionStats,
) -> str | None:
    """One `assert <call>(<args>) == <expected>` line, or None if unusable.

    Argument packing: TACO stores `inputs[i]` as a list of POSITIONAL arguments,
    so a list is splatted. A non-list is passed as a single argument. Dict
    inputs are deliberately treated as one positional argument rather than
    keyword-expanded: TACO dict inputs are ambiguous (sometimes genuinely a
    single dict argument), and guessing wrong here is exactly what Stage 3 is
    there to catch and discard.

    Values are emitted via repr() with NO unquoting of strings — verified
    against a real row, see the module docstring.
    """
    values = list(args) if isinstance(args, list) else [args]
    if not all(_is_reprable(v) for v in values) or not _is_reprable(expected):
        stats.testcases_dropped_unreprable += 1
        return None

    call = f"sol.{fn_name}" if class_wrapped else fn_name
    line = f"assert {call}({', '.join(repr(v) for v in values)}) == {expected!r}"
    if len(line) > _MAX_ASSERT_CHARS:
        stats.testcases_dropped_oversized += 1
        return None
    return line


def _to_stdio_text(value: Any) -> str | None:
    """Normalize one stdin/stdout payload to a single string.

    Most rows store a plain string, but 664 payloads in the test split are
    lists of lines. The compiler's comparison right-strips each line and drops
    trailing blanks, so the exact joining convention is forgiving; what matters
    is not silently producing "['5\\n']" via str() on a list.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        if any(x.endswith("\n") for x in value):
            return "".join(value)
        return "\n".join(value) + "\n" if value else ""
    if isinstance(value, (int, float, bool)):
        return f"{value}\n"
    return None


# ── Row → CodeTask ───────────────────────────────────────────────────────────


def _build_metadata(
    row: dict[str, Any], split: str, row_index: int, n_solutions: int, n_tests: int
) -> dict[str, Any]:
    """Metadata carried on every TACO task.

    `platform` deliberately holds TACO's own `source` column (codeforces,
    leetcode, ...): CodeTask.source is reserved for RLForge provenance
    ("taco_assert"/"taco_stdio"), and naming the platform `platform` matches
    what the LiveCodeBench stdio adapter already stores, so decontamination can
    treat both datasets through one code path.

    `url` and `date` are load-bearing for Stage 2 (exact-key matching and the
    temporal argument); the rest is for filtering and paper-time analysis.
    """
    picture_num = row.get("picture_num")
    return {
        "url": (row.get("url") or None),
        "date": (row.get("date") or None),
        "difficulty": (row.get("difficulty") or None),
        "platform": (row.get("source") or None),
        "name": (row.get("name") or None),
        "tags": (row.get("tags") or None),
        "skill_types": (row.get("skill_types") or None),
        "time_limit": (row.get("time_limit") or None),
        "memory_limit": (row.get("memory_limit") or None),
        "picture_num": picture_num,
        "n_reference_solutions": n_solutions,
        "n_tests": n_tests,
        # Provenance only — identity is the content hash.
        "taco_split": split,
        "taco_row_index": row_index,
        "taco_revision": _TACO_REVISION,
    }


def _convert_row(
    row: dict[str, Any], split: str, row_index: int, stats: TacoConversionStats
) -> CodeTask | None:
    """Convert one raw TACO row into a CodeTask, or None if unusable."""
    question = (row.get("question") or "").strip()
    if not question:
        stats.rows_dropped_empty_question += 1
        return None

    parsed = _parse_input_output(row.get("input_output"), stats)
    if parsed is None:
        stats.rows_dropped_bad_input_output += 1
        return None
    inputs, outputs, fn_name = parsed
    if not inputs:
        stats.rows_dropped_no_tests += 1
        return None

    stats.raw_testcases += len(inputs)
    solutions = _parse_solutions(row.get("solutions"), stats)
    starter_code = row.get("starter_code") or ""

    if _is_assert_row(fn_name):
        assert fn_name is not None  # narrowed by _is_assert_row
        class_wrapped = _is_class_wrapped(starter_code)
        asserts: list[str] = []
        for raw_args, raw_expected in zip(inputs, outputs):
            line = _build_assertion(
                fn_name,
                raw_args,
                _normalize_expected(raw_expected),
                class_wrapped=class_wrapped,
                stats=stats,
            )
            if line is not None:
                asserts.append(line)
        if not asserts:
            stats.rows_dropped_no_asserts += 1
            return None
        # The instantiation line must precede the asserts, exactly as the
        # LiveCodeBench adapter does for its LeetCode rows.
        if class_wrapped:
            asserts.insert(0, "sol = Solution()")
        stats.asserts_built += len(asserts) - (1 if class_wrapped else 0)
        suite = TestSuite(kind="assert", asserts=asserts)
        source = "taco_assert"
        entry_point: str | None = fn_name
    else:
        class_wrapped = False
        io_pairs: list[IOPair] = []
        for raw_in, raw_out in zip(inputs, outputs):
            stdin, stdout = _to_stdio_text(raw_in), _to_stdio_text(raw_out)
            if stdin is None or stdout is None:
                stats.testcases_dropped_bad_stdio += 1
                continue
            io_pairs.append(IOPair(stdin=stdin, stdout=stdout))
        if not io_pairs:
            stats.rows_dropped_no_io_pairs += 1
            return None
        stats.io_pairs_built += len(io_pairs)
        suite = TestSuite(kind="stdin_stdout", io_pairs=io_pairs)
        source = "taco_stdio"
        entry_point = None

    # Prompt: statement plus starter code when present, fenced the same way the
    # LiveCodeBench adapter does. Instruction wrapping stays OUT of stored data
    # — prompt_template.build_prompt_for_task adds it at generation time so its
    # effect remains auditable via prompt_template_id.
    prompt = (
        f"{question}\n\n```python\n{starter_code.strip()}\n```"
        if starter_code.strip()
        else question
    )

    metadata = _build_metadata(row, split, row_index, len(solutions), suite.size())
    metadata["class_wrapped"] = class_wrapped

    if not solutions:
        stats.rows_without_solutions += 1
    try:
        if int(row.get("picture_num") or 0) > 0:
            stats.rows_with_pictures += 1
    except (TypeError, ValueError):
        pass

    stats.rows_assert += 1 if source == "taco_assert" else 0
    stats.rows_stdio += 1 if source == "taco_stdio" else 0
    platform = metadata["platform"] or "unknown"
    difficulty = metadata["difficulty"] or "unknown"
    stats.by_platform[platform] = stats.by_platform.get(platform, 0) + 1
    stats.by_difficulty[difficulty] = stats.by_difficulty.get(difficulty, 0) + 1

    return CodeTask(
        task_id=content_task_id(question),
        source=source,
        prompt=prompt,
        entry_point=entry_point,
        test_suites={"hidden": suite},
        # Only a hidden suite: PROMPTABLE_SUITE is "visible", and any test
        # reaching a prompt in an RL pool is a direct reward-hacking channel.
        reference_solution=(solutions[0] if solutions else None),
        timeout_s=10.0,
        metadata=metadata,
    )


# ── Raw row iteration ────────────────────────────────────────────────────────


def _iter_raw_rows(
    splits: Sequence[str] = DEFAULT_SPLITS, revision: str = _TACO_REVISION
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """Stream (split, row_index, row) from the parquet shards.

    Streaming is mandatory, not an optimisation: TACO's `solutions` and
    `input_output` columns are large inline blobs, and materialising all ~26k
    rows at once needs tens of GB.
    """
    from huggingface_hub import hf_hub_download  # imported late: optional dep
    import pyarrow.parquet as pq

    for split in splits:
        if split not in _SHARDS:
            raise ValueError(f"unknown TACO split {split!r}; expected {sorted(_SHARDS)}")
        row_index = 0
        for shard in _SHARDS[split]:
            path = hf_hub_download(
                _HF_DATASET_ID, shard, repo_type="dataset", revision=revision
            )
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    yield split, row_index, row
                    row_index += 1


# ── Public API ───────────────────────────────────────────────────────────────


def iter_taco_tasks(
    splits: Sequence[str] = DEFAULT_SPLITS,
    limit: int | None = None,
    revision: str = _TACO_REVISION,
    stats: TacoConversionStats | None = None,
) -> Iterator[CodeTask]:
    """Stream converted TACO tasks. Deduplicates ids as it goes.

    Two rows hashing to the same id are the same problem (identical normalized
    statement), which happens because TACO aggregates mirrored problems across
    platforms. The first occurrence wins here; Stage 4 does the richer
    near-duplicate pass.
    """
    stats = stats if stats is not None else TacoConversionStats()
    seen: set[str] = set()
    emitted = 0

    for split, row_index, row in _iter_raw_rows(splits, revision=revision):
        stats.rows_seen += 1
        task = _convert_row(row, split, row_index, stats)
        if task is None:
            continue
        if task.task_id in seen:
            stats.rows_dropped_duplicate_id += 1
            if len(stats.dropped_task_ids) < 100:
                stats.dropped_task_ids.append(task.task_id)
            continue
        seen.add(task.task_id)
        stats.tasks_built += 1
        yield task
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def load_taco(
    splits: Sequence[str] = DEFAULT_SPLITS,
    limit: int | None = None,
    max_testcase_drop_rate: float = _MAX_TESTCASE_DROP_RATE,
    revision: str = _TACO_REVISION,
    stats: TacoConversionStats | None = None,
) -> list[CodeTask]:
    """Convert TACO into CodeTasks, raising if per-testcase attrition is high.

    The guard exists for the same reason as LiveCodeBench's: a change in the
    upstream encoding could quietly shrink rich suites down to a handful of
    tests, leaving every downstream number computed against a weaker verifier
    than intended, with no error anywhere.
    """
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    stats = stats if stats is not None else TacoConversionStats()
    tasks = list(iter_taco_tasks(splits, limit=limit, revision=revision, stats=stats))

    # Only meaningful on a full pass; a `limit` stops mid-stream by design.
    if limit is None:
        rate = stats.testcase_drop_rate()
        if rate > max_testcase_drop_rate:
            raise RuntimeError(
                f"TACO test-case drop rate {rate:.4f} exceeds "
                f"{max_testcase_drop_rate:.4f}. Likely an upstream format change "
                f"(input_output encoding or output wrapping). Downstream pass "
                f"rates would be computed against weakened suites — investigate "
                f"before trusting them. Stats: {stats.summary()}"
            )
    return tasks
