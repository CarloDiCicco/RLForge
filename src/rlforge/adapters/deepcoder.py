"""
adapters/deepcoder.py
─────────────────────
load_deepcoder — the run #3 TRAINING pool: the TACO-Verified and SYNTHETIC-1
(PrimeIntellect) subsets of agentica-org/DeepCoder-Preview-Dataset, converted to
CodeTask. Its lcbv5 and codeforces subsets are never read: LiveCodeBench is the
run #3 test source.

FACTS MEASURED ON THE RAW DATA (2026-11-14), each driving a design choice:
  - The parquet has only three columns: problem, tests, solutions. No url,
    platform, date or starter code.
  - TACO subset (7436 rows): its 1049 function-call rows do NOT name the
    function in the statement (DeepCoder dropped TACO's starter_code column), so
    a model could not know what to call it. Every row matches an original
    BAAI/TACO row by normalized statement (7436/7436), so starter code, url,
    platform and date are recovered from there. Tests and solutions are kept
    from DeepCoder (its verified versions).
  - SYNTHETIC-1 subset (16252 rows): every statement is wrapped in one fixed
    header and one of three fixed trailers; function-call rows carry their
    starter code inside the trailer. The wrapper is stripped (our own prompt
    templates add the instruction) and the starter code kept. Solutions are
    ```python-fenced and are unfenced here.
  - The same problem can appear in both subsets (e.g. is_anagram). Ids are a
    hash of the normalized statement, so a duplicate gets the same id and only
    the first copy is kept (TACO first, as it carries metadata).

CONVERSION REUSES adapters/taco.py. Each row is rebuilt in BAAI/TACO's row shape
and passed to taco._convert_row, so the assert/stdio building, size guards and
attrition counters are the tested TACO code, not a second copy.
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlforge.adapters.taco import (
    TacoConversionStats,
    _convert_row,
    normalize_statement,
)
from rlforge.schema.task import CodeTask

_HF_DATASET_ID = "agentica-org/DeepCoder-Preview-Dataset"
SUBSETS: tuple[str, ...] = ("taco", "primeintellect")
DEFAULT_DIR = Path(os.path.expanduser("~/data/cache/deepcoder"))

_PI_HEADER = "Solve the following coding problem using the programming language python:"
_PI_TRAILERS = (
    "\n\nThe input will be stdin and you should print your solution to stdout\n\n\nNow solve the problem and return the code.",
    "\n\nThe input will be given via stdin and the output should be printed to stdout by your code.\n\nNow solve the problem by providing the code.",
)
_PI_FN_MARKER = "\n\nWrite your solution by modifying this code:\n\n"
_PI_FN_TAIL_RE = re.compile(
    r"\n\nYour solution should implemented in the function \"[^\"]+\"\. The inputs will be "
    r"passed to it and it should return the correct solution\.\n\n\nNow solve the problem "
    r"and return the code\.$"
)
_FENCE_RE = re.compile(r"^\s*```(?:python|py)?\s*\n(.*?)\n?```\s*$", re.DOTALL)


@dataclass
class DeepCoderStats:
    rows_seen: dict[str, int] = field(default_factory=dict)
    tasks_built: dict[str, int] = field(default_factory=dict)
    dropped_no_taco_match: int = 0
    dropped_unknown_wrapper: int = 0
    dropped_mixed_test_types: int = 0
    dropped_convert: dict[str, int] = field(default_factory=dict)
    dropped_duplicate: dict[str, int] = field(default_factory=dict)
    taco: TacoConversionStats = field(default_factory=TacoConversionStats)

    def summary(self) -> str:
        return (
            f"rows {self.rows_seen} -> tasks {self.tasks_built}; dropped: "
            f"no TACO match {self.dropped_no_taco_match}, unknown wrapper "
            f"{self.dropped_unknown_wrapper}, mixed test types "
            f"{self.dropped_mixed_test_types}, conversion {self.dropped_convert}, "
            f"duplicate {self.dropped_duplicate}; test cases raw "
            f"{self.taco.raw_testcases}, drop rate {self.taco.testcase_drop_rate():.4f}"
        )


def task_id_for(statement: str) -> str:
    import hashlib

    digest = hashlib.sha256(normalize_statement(statement).encode("utf-8")).hexdigest()
    return f"DC/{digest[:12]}"


def unfence(code: str) -> str:
    m = _FENCE_RE.match(code)
    return m.group(1) if m else code


def strip_pi_wrapper(problem: str) -> tuple[str, str] | None:
    """(statement, starter_code) from a SYNTHETIC-1 prompt, or None if the
    wrapper is not one of the known shapes (the row is then dropped, never
    guessed at)."""
    text = problem.rstrip()
    if not text.startswith(_PI_HEADER):
        return None
    text = text[len(_PI_HEADER):].lstrip("\n")
    for trailer in _PI_TRAILERS:
        if text.endswith(trailer):
            return text[: -len(trailer)].strip(), ""
    m = _PI_FN_TAIL_RE.search(text)
    if m and _PI_FN_MARKER in text[: m.start()]:
        body, starter = text[: m.start()].rsplit(_PI_FN_MARKER, 1)
        return body.strip(), unfence(starter).strip()
    return None


def _pi_tests_to_io(tests: list[dict]) -> dict[str, Any] | None:
    types = {t.get("type") for t in tests}
    if types == {"stdin_stdout"}:
        return {"inputs": [t["input"] for t in tests], "outputs": [t["output"] for t in tests]}
    if types == {"function_call"}:
        names = {t.get("fn_name") for t in tests}
        if len(names) != 1:
            return None
        return {
            "fn_name": names.pop(),
            "inputs": [t["input"] for t in tests],
            "outputs": [t["output"] for t in tests],
        }
    return None


def _as_list(value: Any) -> list:
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def _taco_index() -> dict[str, dict[str, Any]]:
    """normalized statement -> original BAAI/TACO row (metadata columns only)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from rlforge.adapters.taco import _HF_DATASET_ID as TACO_ID
    from rlforge.adapters.taco import _SHARDS, _TACO_REVISION

    cols = ["question", "starter_code", "url", "source", "date", "difficulty", "name",
            "tags", "skill_types", "time_limit", "memory_limit", "picture_num"]
    index: dict[str, dict[str, Any]] = {}
    for shards in _SHARDS.values():
        for shard in shards:
            path = hf_hub_download(TACO_ID, shard, repo_type="dataset", revision=_TACO_REVISION)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=cols):
                for row in batch.to_pylist():
                    index.setdefault(normalize_statement(row["question"] or ""), row)
    return index


def _iter_rows(subset: str, data_dir: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(data_dir / subset / "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"no parquet under {data_dir / subset}; download {_HF_DATASET_ID} first"
        )
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
            yield from batch.to_pylist()


def iter_deepcoder_tasks(
    subsets: tuple[str, ...] = SUBSETS,
    data_dir: Path = DEFAULT_DIR,
    stats: DeepCoderStats | None = None,
    limit_per_subset: int | None = None,
) -> Iterator[CodeTask]:
    stats = stats if stats is not None else DeepCoderStats()
    for s in subsets:
        if s not in SUBSETS:
            raise ValueError(f"subset {s!r} is not allowed in the training pool: {SUBSETS}")
    index = _taco_index() if "taco" in subsets else {}
    seen: set[str] = set()

    for subset in subsets:
        n = 0
        for i, raw in enumerate(_iter_rows(subset, data_dir)):
            if limit_per_subset is not None and n >= limit_per_subset:
                break
            stats.rows_seen[subset] = stats.rows_seen.get(subset, 0) + 1
            solutions = [unfence(s) for s in _as_list(raw["solutions"])]

            if subset == "taco":
                meta = index.get(normalize_statement(raw["problem"]))
                if meta is None:
                    stats.dropped_no_taco_match += 1
                    continue
                row = dict(meta)
                row["question"] = raw["problem"]
                row["input_output"] = raw["tests"]
            else:
                parsed = strip_pi_wrapper(raw["problem"])
                if parsed is None:
                    stats.dropped_unknown_wrapper += 1
                    continue
                statement, starter = parsed
                io = _pi_tests_to_io(_as_list(raw["tests"]))
                if io is None:
                    stats.dropped_mixed_test_types += 1
                    continue
                row = {"question": statement, "starter_code": starter,
                       "input_output": json.dumps(io), "source": None}
            row["solutions"] = json.dumps(solutions)

            task = _convert_row(row, subset, i, stats.taco)
            if task is None:
                stats.dropped_convert[subset] = stats.dropped_convert.get(subset, 0) + 1
                continue
            tid = task_id_for(row["question"])
            if tid in seen:
                stats.dropped_duplicate[subset] = stats.dropped_duplicate.get(subset, 0) + 1
                continue
            seen.add(tid)
            md = dict(task.metadata)
            md.update({"dc_subset": subset, "dc_row_index": i, "dc_dataset": _HF_DATASET_ID,
                       "starter_code": row.get("starter_code") or ""})
            for k in ("taco_split", "taco_row_index"):
                md.pop(k, None)
            if subset != "taco":
                md.pop("taco_revision", None)
            task = task.model_copy(update={
                "task_id": tid,
                "source": task.source.replace("taco_", f"dc_{subset}_"),
                "metadata": md,
            })
            stats.tasks_built[subset] = stats.tasks_built.get(subset, 0) + 1
            n += 1
            yield task


def load_deepcoder(**kwargs: Any) -> list[CodeTask]:
    return list(iter_deepcoder_tasks(**kwargs))
