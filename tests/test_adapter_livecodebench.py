"""Tests for adapters/livecodebench.py — the LiveCodeBench (LeetCode-only) adapter.

Pure-function tests (decoding, parsing, conversion) use real captured rows —
no network needed, always run. A separate small test exercises the real
huggingface_hub download path and is skipped if huggingface_hub is not
installed (optional [data] extra), same pattern as test_adapter_humaneval.py
skipping when evalplus is unavailable.
"""

from __future__ import annotations

import ast
import base64
import json
import pickle
import zlib

import pytest

from rlforge.adapters.livecodebench import (
    _build_assertion,
    _convert_problem,
    _decode_test_cases,
    _parse_starter_code,
)
from rlforge.sandbox.subprocess_sandbox import SubprocessSandbox
from rlforge.verify.compiler import suite_to_test_code

# Real LeetCode-platform row shape, captured this session from
# livecodebench/code_generation_lite (release_v1, question_id=2727,
# "countSeniors") — public_test_cases is plain JSON; private_test_cases is
# the base64(zlib(pickle(...)))-encoded form actually seen on this row.
_REAL_STARTER_CODE = (
    "class Solution:\n"
    "    def countSeniors(self, details: List[str]) -> int:\n"
    "        "
)

_REAL_PUBLIC_TEST_CASES = json.dumps(
    [
        {"input": '["7868190130M7522", "5303914400F9211", "9273338290F4010"]', "output": "2"},
        {"input": '["1313579440F2036", "2921522980M5644"]', "output": "0"},
    ]
)


def _encode_private(payload: list[dict]) -> str:
    """Build a base64(zlib(pickle(json.dumps(payload)))) blob, matching the
    real encoding _decode_test_cases must handle for some private_test_cases
    rows (confirmed by decoding real LiveCodeBench data this session)."""
    pickled = pickle.dumps(json.dumps(payload))
    return base64.b64encode(zlib.compress(pickled)).decode("ascii")


_REAL_PRIVATE_TEST_CASES = _encode_private(
    [{"input": '["0121327775O7247"]', "output": "1"}]
)


def test_decode_test_cases_handles_plain_json():
    parsed = _decode_test_cases(_REAL_PUBLIC_TEST_CASES)
    assert len(parsed) == 2
    assert parsed[0]["output"] == "2"


def test_decode_test_cases_handles_base64_zlib_pickle():
    parsed = _decode_test_cases(_REAL_PRIVATE_TEST_CASES)
    assert len(parsed) == 1
    assert parsed[0]["output"] == "1"


def test_decode_test_cases_handles_single_dict_not_list():
    raw = json.dumps({"input": "[1]", "output": "1"})
    parsed = _decode_test_cases(raw)
    assert parsed == [{"input": "[1]", "output": "1"}]


def test_decode_test_cases_empty_string_returns_empty_list():
    assert _decode_test_cases("") == []
    assert _decode_test_cases("   ") == []


def test_decode_test_cases_malformed_payload_returns_empty_list():
    assert _decode_test_cases("not json and not base64 either!!") == []


def test_parse_starter_code_extracts_class_and_method():
    result = _parse_starter_code(_REAL_STARTER_CODE)
    assert result == ("Solution", "countSeniors")


def test_parse_starter_code_returns_none_for_non_leetcode_rows():
    # AtCoder/Codeforces rows have empty starter_code — must be skipped, not
    # crash the whole load.
    assert _parse_starter_code("") is None


def test_build_assertion_parses_list_input_and_int_output():
    tc = {"input": '["7868190130M7522", "5303914400F9211", "9273338290F4010"]', "output": "2"}
    line = _build_assertion("sol", "countSeniors", tc)
    assert line == "assert sol.countSeniors(['7868190130M7522', '5303914400F9211', '9273338290F4010']) == 2"
    compile(line, "<test>", "exec")


def test_build_assertion_normalizes_json_style_booleans():
    tc = {"input": "'abc'", "output": "true"}
    line = _build_assertion("sol", "isValid", tc)
    assert line == "assert sol.isValid('abc') == True"


def test_build_assertion_returns_none_on_unparseable_input():
    tc = {"input": "not a literal (((", "output": "1"}
    assert _build_assertion("sol", "f", tc) is None


def test_convert_problem_builds_runnable_task():
    problem = {
        "question_id": 2727,
        "platform": "leetcode",
        "difficulty": "easy",
        "contest_date": "2023-05-13T00:00:00",
        "starter_code": _REAL_STARTER_CODE,
        "question_content": "Count seniors among the given details.",
        "public_test_cases": _REAL_PUBLIC_TEST_CASES,
        "private_test_cases": _REAL_PRIVATE_TEST_CASES,
    }
    task = _convert_problem(problem)
    assert task is not None
    assert task.task_id == "LCB/2727"
    assert task.source == "livecodebench_leetcode"
    assert task.entry_point == "countSeniors"
    assert task.metadata["difficulty"] == "easy"

    suite = task.test_suites["hidden"]
    assert suite.kind == "assert"
    # 1 instantiation line + 2 public + 1 private assert.
    assert len(suite.asserts) == 4
    for line in suite.asserts:
        compile(line, "<test>", "exec")


def test_convert_problem_returns_none_for_unparseable_starter_code():
    problem = {
        "question_id": "1",
        "starter_code": "",  # e.g. an AtCoder/Codeforces row
        "public_test_cases": _REAL_PUBLIC_TEST_CASES,
        "private_test_cases": "",
    }
    assert _convert_problem(problem) is None


def test_convert_problem_returns_none_when_no_tests_available():
    problem = {
        "question_id": "1",
        "starter_code": _REAL_STARTER_CODE,
        "public_test_cases": "",
        "private_test_cases": "",
    }
    assert _convert_problem(problem) is None


def test_converted_task_asserts_actually_run_against_a_real_solution():
    problem = {
        "question_id": 2727,
        "platform": "leetcode",
        "difficulty": "easy",
        "contest_date": "2023-05-13T00:00:00",
        "starter_code": _REAL_STARTER_CODE,
        "question_content": "Count seniors among the given details.",
        "public_test_cases": _REAL_PUBLIC_TEST_CASES,
        "private_test_cases": _REAL_PRIVATE_TEST_CASES,
    }
    task = _convert_problem(problem)
    suite = task.test_suites["hidden"]
    test_code = suite_to_test_code(suite, task.entry_point)
    real_solution = (
        "class Solution:\n"
        "    def countSeniors(self, details):\n"
        "        return sum(1 for d in details if int(d[-4:-2]) > 60)\n"
    )
    result = SubprocessSandbox().run(real_solution, test_code)
    assert result.passed


# --- Network-touching tests: real huggingface_hub download path ---

pytest.importorskip("huggingface_hub")

from rlforge.adapters.livecodebench import load_livecodebench_leetcode  # noqa: E402


def test_load_livecodebench_leetcode_returns_valid_codetasks():
    tasks = load_livecodebench_leetcode(release="release_v1", limit=5)
    assert 0 < len(tasks) <= 5
    for task in tasks:
        assert task.source == "livecodebench_leetcode"
        assert task.task_id.startswith("LCB/")
        assert "hidden" in task.test_suites
        assert task.test_suites["hidden"].asserts


def test_load_livecodebench_leetcode_rejects_unknown_release():
    with pytest.raises(ValueError, match="unknown release"):
        load_livecodebench_leetcode(release="release_v99")


def test_load_livecodebench_leetcode_rejects_bad_limit():
    with pytest.raises(ValueError, match="limit must be"):
        load_livecodebench_leetcode(release="release_v1", limit=0)
