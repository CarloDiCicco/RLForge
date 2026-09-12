"""
tests/test_adapter_taco.py
───────────────────────────
Tests for adapters/taco.py, split the same way as test_adapter_livecodebench.py:
pure-function tests over REAL captured rows (no network), then a small number of
network smoke tests gated behind an importorskip.

Every row constant below is copied verbatim from the real dataset (BAAI/TACO at
revision d593ed0a..., ALL/train-00000-of-00009.parquet) with the test-case lists
truncated for readability — never hand-fabricated, so the shapes the parser sees
here are the shapes it sees in production.

The two round-trip tests are the load-bearing ones: they convert a captured row
and then run that row's OWN reference solution against the asserts/io_pairs the
adapter built, through the real sandbox. Those are what prove the conversion
heuristics documented in the adapter's module docstring.
"""

from __future__ import annotations

import json

from rlforge.adapters.taco import (
    TacoConversionStats,
    _build_assertion,
    _convert_row,
    _is_assert_row,
    _is_class_wrapped,
    _is_reprable,
    _normalize_expected,
    _parse_input_output,
    _parse_solutions,
    _to_stdio_text,
    content_task_id,
    normalize_statement,
)
from rlforge.schema.testsuite import TestSuite
from rlforge.sandbox.subprocess_sandbox import SubprocessSandbox
from rlforge.verify.compiler import suite_to_test_code

# ── Real captured rows ───────────────────────────────────────────────────────

# LeetCode "Teemo Attacking" (train row 30) — the class-wrapped shape, which is
# the COMMON case among fn_name rows (358/360 in this shard). Note the starter
# code's `List[int]` annotation: TACO's own solutions carry their imports, so no
# typing preamble is injected.
_REAL_CLASS_ROW: dict[str, object] = {
    "question": (
        "In LOL world, there is a hero called Teemo whose attacking can make his "
        "enemy Ashe be in poisoned condition. Now, given the Teemo's attacking "
        "ascending time series towards Ashe and the poisoning time duration per "
        "Teemo's attacking, you need to output the total time that Ashe is in "
        "poisoned condition."
    ),
    "starter_code": (
        "class Solution:\n"
        "    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:\n"
        "        "
    ),
    "input_output": json.dumps(
        {
            "inputs": [[[1, 4], 2]],
            "outputs": [4],
            "fn_name": "findPoisonedDuration",
        }
    ),
    "solutions": json.dumps(
        [
            "class Solution:\n"
            "\n"
            "\tdef findPoisonedDuration(self, timeSeries, duration):\n"
            "\t\tif not timeSeries:\n"
            "\t\t\treturn 0\n"
            "\t\tprev = timeSeries[0]\n"
            "\t\tret = 0\n"
            "\t\tcount = 0\n"
            "\t\tfor t in timeSeries[1:]:\n"
            "\t\t\tdiff = t - prev\n"
            "\t\t\tif diff > duration:\n"
            "\t\t\t\tcount += 1\n"
            "\t\t\telse:\n"
            "\t\t\t\tret += diff\n"
            "\t\t\tprev = t\n"
            "\t\tret += (count + 1) * duration\n"
            "\t\treturn ret\n"
        ]
    ),
    "url": "https://leetcode.com/problems/teemo-attacking/",
    "date": None,
    "difficulty": "MEDIUM_HARD",
    "source": "leetcode",
    "name": None,
    "picture_num": None,
    "tags": None,
    "skill_types": None,
    "time_limit": None,
    "memory_limit": None,
}

# Codeforces 1057/C (train shard 0) — the stdin/stdout shape, which is the bulk
# of TACO (2467/2827 rows in this shard).
_REAL_STDIO_ROW: dict[str, object] = {
    "question": (
        "There are $n$ candy boxes in front of Tania. The boxes are arranged in a "
        "row from left to right, numbered from $1$ to $n$."
    ),
    "starter_code": "",
    "input_output": json.dumps(
        {
            "inputs": ["5 3 10\n1 2 3 4 5\nRGBRR\n", "2 1 15\n5 6\nRG\n"],
            "outputs": ["4\n", "-1\n"],
        }
    ),
    "solutions": json.dumps(["import sys\nprint(len(sys.stdin.read().split()))\n"]),
    "url": "https://codeforces.com/problemset/problem/1057/C",
    "date": "2019-12-31",
    "difficulty": "HARD",
    "source": "codeforces",
    "name": None,
    "picture_num": None,
    "tags": None,
    "skill_types": None,
    "time_limit": None,
    "memory_limit": None,
}


def _stdio_row(**overrides: object) -> dict[str, object]:
    """Build a stdio row variant, mirroring the LCB tests' factory style."""
    row = dict(_REAL_STDIO_ROW)
    row.update(overrides)
    return row


# ── normalize_statement / content_task_id ────────────────────────────────────


def test_normalize_statement_collapses_whitespace_and_case():
    assert normalize_statement("  The  FIRST\nline\t ") == "the first line"


def test_content_task_id_is_stable_and_prefixed():
    a = content_task_id("Given an array, return the sum.")
    b = content_task_id("  Given an array,   return the sum.  ")
    assert a == b, "ids must survive incidental whitespace differences"
    assert a.startswith("TACO/") and len(a) == len("TACO/") + 12


def test_content_task_id_differs_for_different_problems():
    assert content_task_id("problem one") != content_task_id("problem two")


# ── _parse_input_output ──────────────────────────────────────────────────────


def test_parse_input_output_extracts_fn_name():
    stats = TacoConversionStats()
    parsed = _parse_input_output(_REAL_CLASS_ROW["input_output"], stats)
    assert parsed is not None
    inputs, outputs, fn_name = parsed
    assert fn_name == "findPoisonedDuration"
    assert inputs == [[[1, 4], 2]]
    assert outputs == [4]


def test_parse_input_output_returns_none_fn_name_for_stdio_row():
    stats = TacoConversionStats()
    parsed = _parse_input_output(_REAL_STDIO_ROW["input_output"], stats)
    assert parsed is not None
    assert parsed[2] is None


def test_parse_input_output_truncates_length_mismatch_and_counts():
    stats = TacoConversionStats()
    raw = json.dumps({"inputs": ["a", "b", "c"], "outputs": ["1"]})
    inputs, outputs, _ = _parse_input_output(raw, stats)
    assert len(inputs) == len(outputs) == 1
    assert stats.testcases_dropped_length_mismatch == 2


def test_parse_input_output_returns_none_for_garbage():
    stats = TacoConversionStats()
    assert _parse_input_output("not json at all {", stats) is None
    assert stats.input_output_decode_failures == 1


def test_parse_input_output_accepts_python_repr_fallback():
    """TACO is a scrape; some rows are Python reprs rather than strict JSON."""
    stats = TacoConversionStats()
    parsed = _parse_input_output("{'inputs': ['x'], 'outputs': ['y']}", stats)
    assert parsed is not None
    assert parsed[0] == ["x"]


# ── _parse_solutions ─────────────────────────────────────────────────────────


def test_parse_solutions_decodes_json_list():
    stats = TacoConversionStats()
    sols = _parse_solutions(_REAL_CLASS_ROW["solutions"], stats)
    assert len(sols) == 1
    assert "findPoisonedDuration" in sols[0]


def test_parse_solutions_returns_empty_for_missing_field():
    stats = TacoConversionStats()
    assert _parse_solutions("", stats) == []
    assert stats.solution_decode_failures == 0, "empty is not a decode failure"


def test_parse_solutions_drops_blank_entries():
    stats = TacoConversionStats()
    assert _parse_solutions(json.dumps(["ok", "", "   "]), stats) == ["ok"]


# ── Routing / shape helpers ──────────────────────────────────────────────────


def test_is_assert_row_true_when_fn_name_present():
    assert _is_assert_row("foo") is True


def test_is_assert_row_false_when_absent():
    assert _is_assert_row(None) is False


def test_is_class_wrapped_detects_starter_class():
    assert _is_class_wrapped(_REAL_CLASS_ROW["starter_code"]) is True
    assert _is_class_wrapped("") is False
    assert _is_class_wrapped(None) is False


def test_normalize_expected_unwraps_single_element_list():
    assert _normalize_expected([4]) == 4


def test_normalize_expected_passes_scalars_and_longer_lists_through():
    # Both shapes occur in the real data: bare scalars on ~91 rows of train
    # shard 0, and genuine multi-element expected values on a handful.
    assert _normalize_expected(4) == 4
    assert _normalize_expected([1, 2]) == [1, 2]


def test_is_reprable_rejects_inf_and_nan():
    # inf/nan repr as bare names (NameError at run time), and nan != nan makes
    # an equality assert unsatisfiable regardless of the candidate's behaviour.
    assert _is_reprable(float("inf")) is False
    assert _is_reprable(float("nan")) is False
    assert _is_reprable([1, "a", {"k": 2}]) is True


# ── _build_assertion ─────────────────────────────────────────────────────────


def test_build_assertion_splats_positional_args_for_class_row():
    stats = TacoConversionStats()
    line = _build_assertion(
        "findPoisonedDuration", [[1, 4], 2], 4, class_wrapped=True, stats=stats
    )
    assert line == "assert sol.findPoisonedDuration([1, 4], 2) == 4"


def test_build_assertion_uses_bare_call_when_not_class_wrapped():
    stats = TacoConversionStats()
    line = _build_assertion("converter", [13], "103", class_wrapped=False, stats=stats)
    assert line == "assert converter(13) == '103'"


def test_build_assertion_keeps_string_quotes_verbatim():
    """Quoted strings are used AS-IS; this was verified against a real row.

    train/38 (minFlipsMonoIncr) stores its input as the JSON string '"00110"'.
    Passing it raw PASSES that row's reference solution; stripping the quotes
    FAILS it — the quote characters are genuinely part of the string.
    """
    stats = TacoConversionStats()
    line = _build_assertion("f", ['"00110"'], 2, class_wrapped=False, stats=stats)
    assert line == "assert f('\"00110\"') == 2"


def test_build_assertion_returns_none_for_unreprable_value():
    stats = TacoConversionStats()
    assert (
        _build_assertion("f", [float("nan")], 1, class_wrapped=False, stats=stats)
        is None
    )
    assert stats.testcases_dropped_unreprable == 1


def test_build_assertion_returns_none_for_oversized_line():
    stats = TacoConversionStats()
    huge = "x" * 200_000
    assert _build_assertion("f", [huge], 1, class_wrapped=False, stats=stats) is None
    assert stats.testcases_dropped_oversized == 1


# ── _to_stdio_text ───────────────────────────────────────────────────────────


def test_to_stdio_text_passes_string_through():
    assert _to_stdio_text("5 3\n") == "5 3\n"


def test_to_stdio_text_joins_list_of_lines():
    # 664 payloads in the test split arrive as lists of lines rather than one
    # string; str() on the list would produce "['5\\n']" and fail every pair.
    assert _to_stdio_text(["1\n", "2\n"]) == "1\n2\n"
    assert _to_stdio_text(["1", "2"]) == "1\n2\n"


def test_to_stdio_text_returns_none_for_unusable():
    assert _to_stdio_text({"k": "v"}) is None


# ── _convert_row ─────────────────────────────────────────────────────────────


def test_convert_class_row_builds_assert_suite_with_instantiation():
    stats = TacoConversionStats()
    task = _convert_row(_REAL_CLASS_ROW, "train", 30, stats)
    assert task is not None
    suite = task.test_suites["hidden"]
    assert suite.kind == "assert"
    assert suite.asserts[0] == "sol = Solution()"
    assert suite.asserts[1] == "assert sol.findPoisonedDuration([1, 4], 2) == 4"
    assert task.source == "taco_assert"
    assert task.entry_point == "findPoisonedDuration"
    assert task.metadata["class_wrapped"] is True


def test_convert_row_preserves_decontamination_metadata():
    """url/date/platform are load-bearing for Stage 2 — assert they survive."""
    stats = TacoConversionStats()
    task = _convert_row(_REAL_CLASS_ROW, "train", 30, stats)
    assert task is not None
    assert task.metadata["url"] == "https://leetcode.com/problems/teemo-attacking/"
    assert task.metadata["platform"] == "leetcode"
    assert task.metadata["difficulty"] == "MEDIUM_HARD"
    assert task.metadata["taco_split"] == "train"
    assert task.metadata["taco_row_index"] == 30


def test_convert_class_row_fences_starter_code_into_prompt():
    stats = TacoConversionStats()
    task = _convert_row(_REAL_CLASS_ROW, "train", 30, stats)
    assert task is not None
    assert task.prompt.count("```") == 2
    assert "class Solution:" in task.prompt


def test_convert_stdio_row_builds_stdin_stdout_suite():
    stats = TacoConversionStats()
    task = _convert_row(_REAL_STDIO_ROW, "train", 0, stats)
    assert task is not None
    suite = task.test_suites["hidden"]
    assert suite.kind == "stdin_stdout"
    assert len(suite.io_pairs) == 2
    assert suite.io_pairs[0].stdin == "5 3 10\n1 2 3 4 5\nRGBRR\n"
    assert task.source == "taco_stdio"
    assert task.entry_point is None, "stdio dispatch depends on entry_point being None"
    assert "```" not in task.prompt, "no starter code => statement stays unfenced"


def test_convert_row_stores_only_a_hidden_suite():
    """A visible suite would be promptable — a direct reward-hacking channel."""
    stats = TacoConversionStats()
    task = _convert_row(_REAL_STDIO_ROW, "train", 0, stats)
    assert task is not None
    assert set(task.test_suites) == {"hidden"}


def test_convert_row_drops_empty_question():
    stats = TacoConversionStats()
    assert _convert_row(_stdio_row(question="   "), "train", 1, stats) is None
    assert stats.rows_dropped_empty_question == 1


def test_convert_row_drops_row_without_tests():
    stats = TacoConversionStats()
    row = _stdio_row(input_output=json.dumps({"inputs": [], "outputs": []}))
    assert _convert_row(row, "train", 1, stats) is None
    assert stats.rows_dropped_no_tests == 1


def test_convert_row_keeps_row_without_solutions_as_observation():
    """No solution is not a drop — Stage 3 removes it, and it must be counted."""
    stats = TacoConversionStats()
    task = _convert_row(_stdio_row(solutions=""), "train", 1, stats)
    assert task is not None
    assert task.reference_solution is None
    assert stats.rows_without_solutions == 1
    assert stats.tasks_built == 0, "tasks_built is incremented by the iterator"


def test_convert_row_records_composition_histograms():
    stats = TacoConversionStats()
    _convert_row(_REAL_STDIO_ROW, "train", 0, stats)
    _convert_row(_REAL_CLASS_ROW, "train", 30, stats)
    assert stats.by_platform == {"codeforces": 1, "leetcode": 1}
    assert stats.by_difficulty == {"HARD": 1, "MEDIUM_HARD": 1}
    assert stats.rows_stdio == 1 and stats.rows_assert == 1


def test_conversion_stats_accounts_every_testcase():
    stats = TacoConversionStats()
    _convert_row(_REAL_STDIO_ROW, "train", 0, stats)
    assert stats.raw_testcases == 2
    assert stats.io_pairs_built == 2
    assert stats.testcase_drop_rate() == 0.0


# ── Round-trip: the tests that prove the heuristics on real data ─────────────


def test_converted_class_task_runs_against_its_reference_solution():
    """Convert a real class-wrapped row, then run ITS OWN solution in the sandbox.

    This is the repo's signature adapter test and the in-miniature version of
    Stage 3: if the argument splatting or the expected-value unwrapping were
    wrong, the official solution would fail its own tests here.
    """
    stats = TacoConversionStats()
    task = _convert_row(_REAL_CLASS_ROW, "train", 30, stats)
    assert task is not None
    suite = task.test_suites["hidden"]
    test_code = suite_to_test_code(suite, task.entry_point)
    result = SubprocessSandbox().run(
        task.reference_solution, test_code, mode=suite.kind
    )
    assert result.passed, f"reference solution failed: {result.stderr}"


def test_converted_stdio_task_runs_against_a_real_solution():
    """Same round trip for the stdin/stdout path.

    The solution here counts whitespace-separated tokens, and the io_pairs are
    replaced with matching expectations, so the test exercises the adapter's
    IOPair construction and the compiler's stdin/stdout mode rather than the
    original problem's algorithm.
    """
    row = _stdio_row(
        input_output=json.dumps(
            {"inputs": ["a b c\n", "x y\n"], "outputs": ["3\n", "2\n"]}
        ),
        solutions=json.dumps(["import sys\nprint(len(sys.stdin.read().split()))\n"]),
    )
    stats = TacoConversionStats()
    task = _convert_row(row, "train", 0, stats)
    assert task is not None
    suite = task.test_suites["hidden"]
    test_code = suite_to_test_code(suite, task.entry_point)
    result = SubprocessSandbox().run(
        task.reference_solution, test_code, mode=suite.kind
    )
    assert result.passed, f"stdio round trip failed: {result.stderr}"


def test_converted_task_survives_schema_round_trip():
    """model_dump_json -> model_validate_json, the path prepare_taco.py writes."""
    from rlforge.schema.task import CodeTask

    stats = TacoConversionStats()
    task = _convert_row(_REAL_CLASS_ROW, "train", 30, stats)
    assert task is not None
    restored = CodeTask.model_validate_json(task.model_dump_json())
    assert restored.task_id == task.task_id
    assert restored.test_suites["hidden"].asserts == task.test_suites["hidden"].asserts


# ── Network smoke tests ──────────────────────────────────────────────────────

import pytest  # noqa: E402

pytest.importorskip("huggingface_hub")
pytest.importorskip("pyarrow")

from rlforge.adapters.taco import load_taco  # noqa: E402


@pytest.mark.slow
def test_load_taco_returns_valid_codetasks():
    tasks = load_taco(splits=["test"], limit=5)
    assert len(tasks) == 5
    for task in tasks:
        assert task.task_id.startswith("TACO/")
        assert task.source in {"taco_assert", "taco_stdio"}
        assert not task.test_suites["hidden"].is_empty()


@pytest.mark.slow
def test_load_taco_task_ids_are_unique():
    tasks = load_taco(splits=["test"], limit=50)
    assert len({t.task_id for t in tasks}) == len(tasks)


def test_load_taco_rejects_bad_limit():
    with pytest.raises(ValueError):
        load_taco(splits=["test"], limit=0)


def test_load_taco_rejects_unknown_split():
    with pytest.raises(ValueError, match="unknown TACO split"):
        load_taco(splits=["nope"], limit=1)
