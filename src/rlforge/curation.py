"""
curation.py
───────────
Task-quality checks and train/test overlap detection, shared by the run #3 data
build (scripts/build_run3_data.py) and its tests.

Two jobs, both measured on the real data before being written down
(RESEARCH_LOG 2026-11-14):

1. audit_task — reasons a task cannot give a trustworthy 0/1 reward.
   - "multi_answer": the statement accepts any of several outputs ("print any of
     them"), but we grade against ONE stored output, so a correct, different
     answer scores 0. The patterns are deliberately narrow: a looser regex hit
     benign text like "any of the four directions". On the LiveCodeBench window
     they flag exactly the 10 genuine cases, all AtCoder.
   - "image": the statement refers to a picture the model never sees.
   - "few_tests" / "empty_statement" / "no_entry_point": structurally broken.
   An empty expected output is NOT a reason: it is legitimate for some problems
   (e.g. "print nothing if no query of type 2"); the gold check decides those.

2. overlap pairs — train statements that share text with a test statement.
   A plain "any shared 13-gram" rule over-filters badly here: on the run #3 data
   it flagged 1,410 training problems, all through AtCoder boilerplate
   ("All input values are integers ..."). So a pair is scored by the FRACTION of
   the shorter statement's 13-grams that the two share; a real copy scores near
   1.0, boilerplate stays under 0.2 (measured max: 0.17).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from rlforge.schema.task import CodeTask

# ── audit ────────────────────────────────────────────────────────────────────

_MULTI_ANSWER_RE = re.compile(
    r"(print|output|printing|return|find)\s+any\s+(one\s+)?(of\s+them|of\s+these|such|valid|solution|answer)"
    r"|any\s+(one\s+)?of\s+them\s+(will\s+be|is|are)\s+(accepted|considered|correct)"
    r"|any\s+(valid|correct)\s+(answer|solution|output)\s+(will\s+be|is)\s+accepted"
    r"|if\s+(there\s+are|multiple)\s+[^.]{0,40}(answers|solutions|ways)[^.]{0,40}(any|either)",
    re.IGNORECASE,
)
_IMAGE_RE = re.compile(r"<image>|!\[|\.png\b|\.jpe?g\b|\.gif\b|<img", re.IGNORECASE)

MIN_TESTS_TRAIN = 5  # training only; see audit_task's `min_tests`


def n_tests(task: CodeTask) -> int:
    """Gradable cases across all suites (assert lines exclude the setup line)."""
    total = 0
    for suite in task.test_suites.values():
        total += len(suite.io_pairs or [])
        total += sum(1 for a in (suite.asserts or []) if a.lstrip().startswith("assert"))
    return total


def audit_task(task: CodeTask, *, min_tests: int) -> list[str]:
    """Return the reasons `task` must be dropped (empty list = keep).

    `min_tests` differs by role: 5 for training (a weak suite rewards wrong code),
    1 for the test set, where ten hard AtCoder problems ship only 1-2 very large
    hidden cases upstream; dropping them would remove mostly hard problems and
    skew the benchmark (RESEARCH_LOG 2026-07-17 confirmed they match the raw data).
    """
    reasons = []
    prompt = task.prompt or ""
    if len(prompt.strip()) < 80:
        reasons.append("empty_statement")
    if _IMAGE_RE.search(prompt):
        reasons.append("image")
    if _MULTI_ANSWER_RE.search(prompt):
        reasons.append("multi_answer")
    if n_tests(task) < min_tests:
        reasons.append("few_tests")
    kinds = {s.kind for s in task.test_suites.values()}
    if "assert" in kinds and not task.entry_point:
        reasons.append("no_entry_point")
    return reasons


# ── overlap ──────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\w+")
NGRAM = 13
OVERLAP_THRESHOLD = 0.5  # fraction of the shorter statement's 13-grams shared


def ngrams(text: str, n: int = NGRAM) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def overlap_pairs(
    train: Iterable[CodeTask], test: Iterable[CodeTask]
) -> list[tuple[float, str, str]]:
    """(fraction, train_id, test_id) for every pair sharing at least one 13-gram,
    highest first. Indexed on the (small) test side so the cost is one pass over
    the training pool."""
    test = list(test)
    test_grams = [ngrams(t.prompt) for t in test]
    index: dict[str, list[int]] = defaultdict(list)
    for j, grams in enumerate(test_grams):
        for g in grams:
            index[g].append(j)
    pairs = []
    for task in train:
        grams = ngrams(task.prompt)
        shared: Counter[int] = Counter()
        for g in grams:
            for j in index.get(g, ()):
                shared[j] += 1
        for j, count in shared.items():
            frac = count / max(1, min(len(grams), len(test_grams[j])))
            pairs.append((frac, task.task_id, test[j].task_id))
    pairs.sort(reverse=True)
    return pairs
