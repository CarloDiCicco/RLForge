"""
verify/runner.py
────────────────
run_suite — THE single grading function: run candidate code against a TestSuite
and decide pass/fail. Used by the training reward (train/reward.py), the gold
check (verify/reference_verifier.py) and the benchmark pipeline (pipeline.py),
so every number in this project comes from the same grader.

WHY THIS REPLACED THE OLD IN-PROCESS HARNESS (audit 2026-11-15). The old design
pasted the expected outputs into the same Python process as the candidate code,
then compared inside it. Confirmed with test programs:
  - FALSE ONE (reward hacking): a wrong program could read the expected outputs
    (`import __main__; print(__main__._expected)`, `sys._getframe`, `gc`) or
    replace the comparison function, and pass. In function-call mode, returning
    an object whose `__eq__` always returns True passed every assert.
  - FALSE ZERO: `open(0).read()` and `os.read(sys.stdin.fileno(), ...)` -- the
    usual fast-input idioms in AtCoder/Codeforces Python -- failed, because stdin
    was a Python object, not a real file descriptor.
  - Tests shared one process, so state could leak from one test to the next.

THE NEW CONTRACT: expected outputs never enter the sandbox.
  stdin_stdout: the cage receives the candidate source and the test INPUTS only.
    A small runner forks one fresh child per test with real file descriptors
    (fd 0 = the input file, fd 1 = an output file), so the program sees exactly
    what it would see on a judge: `open(0)`, `sys.stdin.buffer`, `exit()`,
    threads all work. The runner sends each raw output back; the comparison
    happens here, in the host process.
  assert (function-call): each `assert CALL == EXPECTED` line is split; the cage
    evaluates only CALL and sends back the value as plain data (repr of builtin
    types). EXPECTED is compared here. A value that is not plain data (custom
    class, object with a rigged __eq__) cannot be sent back and fails.

Grading rules kept identical to before, so numbers stay comparable:
  - stdout comparison: each line right-stripped, leading/trailing blank lines
    dropped, then exact match.
  - function-call comparison: Python `==` between the returned plain value and
    the expected literal (so 2 == 2.0, (1, 2) != [1, 2], as before).
  - time: ONE wall-clock budget (config.timeout_s) for the whole suite, as
    before; CPU-seconds and memory limits apply to every process.
  - pass = every test passes.
"""

from __future__ import annotations

import ast
import copy
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from rlforge.classify.taxonomy import FailureType, classify_failure
from rlforge.sandbox.preamble import LIVECODEBENCH_IMPORTS, SAFETY_PREAMBLE
from rlforge.schema.testsuite import TestSuite

# Printed by runner.py's in-cage scripts after all tests, like the old sentinel:
# exit 0 without it means the runner itself died.
_DONE = "__RLFORGE_RUNNER_DONE_9b2e__"


@dataclass
class SuiteResult:
    passed: bool
    failure_type: FailureType
    n_tests: int
    passed_indices: list[int] = field(default_factory=list)
    stderr: str = ""
    execution_time_s: float = 0.0
    exit_code: int | None = None
    detail: str = ""


def _norm(text: str) -> str:
    """Output normalization: right-strip lines, drop leading/trailing blanks."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    while lines and lines[0] == "":
        lines.pop(0)
    return "\n".join(lines)


# ── stdin/stdout ─────────────────────────────────────────────────────────────

# The in-cage runner. It is OUR code, not the candidate's: it runs before any
# safety preamble, keeps the real os.fork/os.kill/os._exit, and holds only the
# test inputs. Per test it forks; the child redirects fds 0/1/2 to files, applies
# the safety preamble to itself, re-enables exit()/sys.exit() (a real process may
# exit; os._exit stays disabled for the candidate), joins non-daemon threads like
# a real interpreter shutdown, and exits. The parent enforces the shared deadline
# with SIGKILL and reports {status, output} for each test on its own stdout.
_STDIO_RUNNER = r'''
import json as _rj, os as _ro, sys as _rs, time as _rt, signal as _rsig, base64 as _rb
_NONCE = %(nonce)r
_SRC = %(src)r
_STOP_ON_FAIL = %(stop)r
_DEADLINE = _rt.monotonic() + %(budget)r
_inputs = _rj.loads(_rs.stdin.buffer.read().decode("utf-8"))
_real_fork, _real_kill, _real_exit = _ro.fork, _ro.kill, _ro._exit
_report = _ro.fdopen(_ro.dup(1), "w")   # our channel; the child's fd 1 is a file
# Import the standard modules once here; forked children inherit them, so the
# per-test exec of the import block only binds names (no re-import cost).
exec(_IMPORTS, {})
try:
    _code = compile(_SRC, "<candidate>", "exec")
    _compile_error = None
except BaseException as _e:  # SyntaxError and friends: every test fails the same way
    _code, _compile_error = None, "%%s: %%s" %% (type(_e).__name__, _e)

def _child(_k):
    _fd = _ro.open("_in_%%d" %% _k, _ro.O_RDONLY); _ro.dup2(_fd, 0); _ro.close(_fd)
    _fd = _ro.open("_out_%%d" %% _k, _ro.O_WRONLY | _ro.O_CREAT | _ro.O_TRUNC); _ro.dup2(_fd, 1); _ro.close(_fd)
    _fd = _ro.open("_err_%%d" %% _k, _ro.O_WRONLY | _ro.O_CREAT | _ro.O_TRUNC); _ro.dup2(_fd, 2); _ro.close(_fd)
    # Fresh Python-level streams on the redirected fds (the inherited sys.stdin
    # already hit EOF reading the runner's inputs).
    _rs.stdin = open(0, "r", encoding="utf-8", errors="replace", closefd=False)
    _rs.stdout = open(1, "w", encoding="utf-8", closefd=False)
    _rs.stderr = open(2, "w", encoding="utf-8", closefd=False)
    _rs.__stdin__, _rs.__stdout__, _rs.__stderr__ = _rs.stdin, _rs.stdout, _rs.stderr
    _ns = {"__name__": "__main__", "__builtins__": __builtins__}
    exec(_IMPORTS, _ns)
    exec(_PREAMBLE, _ns)
    import builtins as _bi, threading as _th, traceback as _tb
    def _raise_exit(*_a):
        raise SystemExit(*_a)
    _bi.exit = _bi.quit = _rs.exit = _ns["exit"] = _ns["quit"] = _raise_exit
    _rc = 0
    try:
        exec(_code, _ns)
    except SystemExit as _e:
        _c = _e.code
        _rc = 0 if _c is None else (_c if isinstance(_c, int) else 1)
        if not isinstance(_c, (int, type(None))):
            try: print(_c, file=_rs.stderr)
            except BaseException: pass
    except BaseException:
        _tb.print_exc()
        _rc = 1
    try:
        _th._shutdown()   # wait for non-daemon threads, as interpreter exit does
    except BaseException:
        pass
    for _s in (_rs.stdout, _rs.stderr):
        try: _s.flush()
        except BaseException: pass
    _real_exit(_rc & 0xFF)

def _tail(_p, _n=2000):
    try:
        with open(_p, "rb") as _f:
            return _f.read()[-_n:].decode("utf-8", "replace")
    except OSError:
        return ""

for _k, _inp in enumerate(_inputs):
    if _compile_error is not None:
        _report.write(_NONCE + _rj.dumps({"i": _k, "status": "error", "err": _compile_error}) + "\n")
        break
    if _rt.monotonic() >= _DEADLINE:
        _report.write(_NONCE + _rj.dumps({"i": _k, "status": "timeout"}) + "\n")
        break
    with open("_in_%%d" %% _k, "wb") as _f:
        _f.write(_inp.encode("utf-8"))
    _report.flush()
    _pid = _real_fork()
    if _pid == 0:
        try:
            _child(_k)
        finally:
            _real_exit(1)
    _status = None
    while True:
        _w, _st = _ro.waitpid(_pid, _ro.WNOHANG)
        if _w:
            _status = _st
            break
        if _rt.monotonic() >= _DEADLINE:
            _real_kill(_pid, _rsig.SIGKILL)
            _ro.waitpid(_pid, 0)
            break
        _rt.sleep(0.0005)
    if _status is None:
        _rec = {"i": _k, "status": "timeout", "err": _tail("_err_%%d" %% _k)}
    elif _ro.WIFSIGNALED(_status):
        _sig = _ro.WTERMSIG(_status)
        # SIGXCPU/SIGKILL with nothing on stderr = CPU-seconds limit hit
        _rec = {"i": _k, "status": "timeout" if _sig in (_rsig.SIGXCPU, _rsig.SIGKILL) else "error",
                "err": "killed by signal %%d\n" %% _sig + _tail("_err_%%d" %% _k)}
    else:
        _rc = _ro.WEXITSTATUS(_status)
        with open("_out_%%d" %% _k, "rb") as _f:
            _out = _f.read()
        _rec = {"i": _k, "status": "ok" if _rc == 0 else "error", "rc": _rc,
                "out": _out.decode("utf-8", "replace"), "err": _tail("_err_%%d" %% _k)}
    for _p in ("_in_%%d", "_out_%%d", "_err_%%d"):
        try: _ro.unlink(_p %% _k)
        except OSError: pass
    _report.write(_NONCE + _rj.dumps(_rec) + "\n")
    _report.flush()
    if _STOP_ON_FAIL and _rec["status"] != "ok":
        break
_report.write(%(done)r + "\n")
_report.flush()
'''


def _runner_script(src: str, *, stop_on_fail: bool, budget_s: float, nonce: str) -> str:
    body = _STDIO_RUNNER % {"nonce": nonce, "src": src, "stop": stop_on_fail,
                            "budget": float(budget_s), "done": _DONE}
    # The import block and the preamble are handed to the child as strings and
    # exec'd in the candidate's own namespace, so the candidate sees the same
    # implicit names (LiveCodeBench's import_string) as before.
    return f"_IMPORTS = {LIVECODEBENCH_IMPORTS!r}\n_PREAMBLE = {SAFETY_PREAMBLE!r}\n" + body


def _parse_records(stdout: str, nonce: str) -> dict[int, dict]:
    """First record per test index from lines carrying this run's nonce."""
    recs: dict[int, dict] = {}
    for line in stdout.splitlines():
        if line.startswith(nonce):
            try:
                rec = json.loads(line[len(nonce):])
            except json.JSONDecodeError:
                continue
            recs.setdefault(int(rec.get("i", -1)), rec)
    return recs


def _run_stdio(sandbox, code: str, suite: TestSuite, *, max_tests: int | None,
               stop_on_fail: bool) -> SuiteResult:
    pairs = suite.io_pairs[: max_tests] if max_tests else suite.io_pairs
    nonce = "__R" + secrets.token_hex(8) + "__"
    budget = sandbox.config.timeout_s
    script = _runner_script(code, stop_on_fail=stop_on_fail, budget_s=budget, nonce=nonce)
    stdin = json.dumps([p.stdin for p in pairs]).encode("utf-8")
    t0 = time.perf_counter()
    # Outer timeout: the runner's own deadline plus slack for start-up/reporting.
    res = sandbox.run_script(script, stdin, allow_fork=True, timeout_s=budget + 15.0)
    elapsed = time.perf_counter() - t0
    recs = _parse_records(res.stdout or "", nonce)

    ok, first_bad, err_text = [], None, ""
    for i, pair in enumerate(pairs):
        rec = recs.get(i)
        good = rec is not None and rec.get("status") == "ok" and _norm(rec.get("out", "")) == _norm(pair.stdout)
        if good:
            ok.append(i)
        elif first_bad is None:
            first_bad = (i, rec)
            if rec is None and _DONE not in (res.stdout or ""):
                err_text = res.stderr or ""
    n = len(pairs)
    if len(ok) == n and n > 0:
        return SuiteResult(True, FailureType.PASS, n, ok, execution_time_s=elapsed, exit_code=0)

    i, rec = first_bad if first_bad else (0, None)
    if rec is None:
        # The first failing test has no report: the runner itself died or the
        # outer timeout fired (a candidate's own failure always gets a record).
        if res.failure_type == FailureType.TIMEOUT:
            ft = FailureType.TIMEOUT
        elif err_text.strip():
            ft = classify_failure(passed=False, extracted=True, timed_out=False, stderr=err_text)
        else:
            ft = FailureType.UNKNOWN
        return SuiteResult(False, ft, n, ok, stderr=err_text[-2000:], execution_time_s=elapsed,
                           exit_code=res.exit_code, detail=f"no report for test {i}")
    status, err = rec.get("status"), rec.get("err", "") or ""
    if status == "timeout":
        ft = FailureType.TIMEOUT
    elif status == "error":
        ft = classify_failure(passed=False, extracted=True, timed_out=False, stderr=err or "error")
        if ft in (FailureType.UNKNOWN, FailureType.ASSERTION):
            ft = FailureType.RUNTIME  # a program's own AssertionError is a crash here
    else:
        ft = FailureType.ASSERTION  # ran fine, wrong output
        err = (f"IO pair {i} failed: stdin={pairs[i].stdin[:200]!r} "
               f"expected={pairs[i].stdout[:200]!r} got={rec.get('out', '')[:200]!r}")
    return SuiteResult(False, ft, n, ok, stderr=err[-2000:], execution_time_s=elapsed,
                       exit_code=res.exit_code)


# ── function-call (assert) ───────────────────────────────────────────────────


_CONST_NODES = (ast.Expression, ast.Constant, ast.Tuple, ast.List, ast.Set, ast.Dict,
                ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop, ast.Load)


# Built-ins a TEST SUITE may use to compute expected values on the host. Only
# test-suite text is ever evaluated with these (never candidate code), and any
# other name -- e.g. the candidate's own function -- raises NameError, which
# marks the expression as "computed by the candidate" instead.
_SAFE_BUILTINS = {
    "tuple": tuple, "list": list, "set": set, "frozenset": frozenset, "dict": dict,
    "range": range, "sum": sum, "str": str, "len": len, "int": int, "float": float,
    "abs": abs, "min": min, "max": max, "sorted": sorted, "bool": bool, "round": round,
    "enumerate": enumerate, "zip": zip, "reversed": reversed, "any": any, "all": all,
}
_HOST_NODES = _CONST_NODES + (ast.Name, ast.Call, ast.ListComp, ast.SetComp, ast.GeneratorExp,
                              ast.DictComp, ast.comprehension, ast.Compare, ast.cmpop,
                              ast.BoolOp, ast.boolop, ast.IfExp, ast.Store, ast.Subscript, ast.Slice)


def _host_eval(node: ast.AST, env: dict[str, Any] | None = None) -> tuple[bool, Any]:
    """Evaluate a test-suite expression on the host (literals, arithmetic,
    comprehensions, safe built-ins, and names bound earlier in the suite)."""
    expr = ast.Expression(body=node)
    if not all(isinstance(n, _HOST_NODES) for n in ast.walk(expr)):
        return False, None
    try:
        return True, eval(compile(expr, "<expected>", "eval"),
                          {"__builtins__": {}, **_SAFE_BUILTINS}, dict(env or {}))
    except Exception:
        return False, None


def _const_value(node: ast.AST) -> tuple[bool, Any]:
    return _host_eval(node, None)


class _Bind(ast.NodeTransformer):
    """Replace names bound by the suite (loop variables, helper assignments)
    with their literal values, so the call sent to the cage is self-contained."""

    def __init__(self, env: dict[str, Any]):
        self.env = env

    def visit_Name(self, node: ast.Name):
        if node.id in self.env and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.parse(repr(self.env[node.id]), mode="eval").body, node)
        return node


# A check is (kind, expected, tol):
#   ("eq", v, None)    CALL == v           ("is", v, None)   CALL is v (True/False/None)
#   ("truthy", None, None)  assert CALL    ("falsy", None, None)  assert not CALL
#   ("approx", v, tol) abs(CALL - v) < tol
# Shapes measured on the real suites (2026-11-15): LiveCodeBench, TACO and
# SYNTHETIC-1 use only `CALL == literal`; HumanEval+ also uses the other four.
Check = tuple[str, Any, Any]


def split_assert(line: str) -> tuple[str, Check] | None | str:
    """Split ONE assert statement into (CALL source, check).

    Returns "skip" for a constant assert that checks nothing (`assert True, ...`,
    48 such lines in HumanEval+), and None for a shape we do not support. Lines
    holding several statements are split by _parse_suite first.
    """
    try:
        tree = ast.parse(line.strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None
    return _split_assert_node(tree.body[0])


def _split_assert_node(node: ast.Assert):
    test = node.test
    if isinstance(test, ast.Constant):
        return "skip" if test.value else None
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return ast.unparse(test.operand), ("falsy", None, None)
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op, right = test.ops[0], test.comparators[0]
        ok, value = _const_value(right)
        if not ok:
            # Both sides are computed by the candidate (HumanEval/33 compares
            # sort_third(x) with itself): evaluate both in the cage, compare here.
            if isinstance(op, ast.Eq):
                return ast.unparse(test.left), ("eq_expr", ast.unparse(right), None)
            return None
        if isinstance(op, ast.Eq):
            return ast.unparse(test.left), ("eq", value, None)
        if isinstance(op, ast.Is) and value in (True, False, None) and type(value) in (bool, type(None)):
            return ast.unparse(test.left), ("is", value, None)
        # abs(CALL - EXPECTED) < TOL
        left = test.left
        if (isinstance(op, ast.Lt) and isinstance(left, ast.Call)
                and isinstance(left.func, ast.Name) and left.func.id == "abs"
                and len(left.args) == 1 and isinstance(left.args[0], ast.BinOp)
                and isinstance(left.args[0].op, ast.Sub)):
            ok2, exp = _const_value(left.args[0].right)
            if ok2:
                return ast.unparse(left.args[0].left), ("approx", exp, value)
        return None
    if isinstance(test, (ast.Call, ast.Attribute, ast.Subscript, ast.Name)):
        return ast.unparse(test), ("truthy", None, None)
    return None


# In-cage evaluator: candidate code and setup lines run first (as before), then
# each CALL is evaluated and its value sent back as repr of plain data. The
# expected values are not in this script. numpy values are converted to Python
# ones (a correct numpy answer used to pass `==` and must still pass).
_ASSERT_TAIL = r'''
import json as _rj2
def _rl_plain(_v, _d=0):
    if _d > 60:
        raise ValueError("nested too deep")
    _t = type(_v)
    if _v is None or _t in (bool, int, float, str, bytes, complex):
        return _v
    if _t in (list, tuple, set, frozenset):
        _items = [_rl_plain(_x, _d + 1) for _x in _v]
        return {list: list, tuple: tuple, set: set, frozenset: set}[_t](_items)
    if _t is dict:
        return {_rl_plain(_k, _d + 1): _rl_plain(_x, _d + 1) for _k, _x in _v.items()}
    _mod = getattr(_t, "__module__", "") or ""
    if _mod.startswith("numpy"):
        return _rl_plain(_v.tolist() if hasattr(_v, "tolist") else _v.item(), _d + 1)
    raise TypeError("not plain data: %%s" %% _t.__name__)
for _rl_i, _rl_src in enumerate(%(calls)r):
    try:
        _rl_val = repr(_rl_plain(eval(_rl_src)))
        _rl_rec = {"i": _rl_i, "status": "ok", "val": _rl_val}
    except BaseException as _rl_e:
        _rl_rec = {"i": _rl_i, "status": "error", "err": "%%s: %%s" %% (type(_rl_e).__name__, str(_rl_e)[:300])}
    _rl_out.write(%(nonce)r + _rj2.dumps(_rl_rec) + "\n")
    _rl_out.flush()
_rl_out.write(%(done)r + "\n")
_rl_out.flush()
'''


def _assert_script(code: str, setup: list[str], calls: list[str], nonce: str) -> str:
    # Our report channel is a dup of fd 1 taken BEFORE the candidate runs, so the
    # candidate replacing sys.stdout cannot redirect it (os.dup is not disabled
    # by the preamble).
    head = "import os as _rl_os\n_rl_out = _rl_os.fdopen(_rl_os.dup(1), 'w')\n"
    tail = _ASSERT_TAIL % {"calls": calls, "nonce": nonce, "done": _DONE}
    return (LIVECODEBENCH_IMPORTS + "\n" + head + SAFETY_PREAMBLE + "\n\n" + code.rstrip()
            + "\n\n" + "\n".join(setup) + "\n" + tail)


def _check_value(got_repr: str, check: Check) -> bool:
    """Apply one check to the plain value the cage sent back (host side)."""
    kind, expected, tol = check

    def _decode(text: str):
        try:
            return True, ast.literal_eval(text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            # repr of plain data is always literal, except float inf/nan, which
            # repr as bare names; map them back so `== float('inf')` still works.
            try:
                return True, eval(text, {"__builtins__": {}}, {"inf": float("inf"), "nan": float("nan")})
            except Exception:
                return False, None

    ok, got = _decode(got_repr)
    if not ok:
        return False
    if kind == "eq_lit_repr":  # compare with another value sent back by the cage
        ok, expected = _decode(expected)
        if not ok:
            return False
        kind = "eq"
    try:
        if kind == "eq":
            return bool(got == expected)
        if kind == "is":
            return got is expected
        if kind == "truthy":
            return bool(got)
        if kind == "falsy":
            return not bool(got)
        if kind == "approx":
            return bool(abs(got - expected) < tol)
    except Exception:
        return False
    return False


# Parsed suites, keyed by id(suite) and holding the suite itself so a recycled
# id can never return another suite's parse. Splitting is costly for the
# multi-MB LiveCodeBench asserts, and the same suite is graded 8 times per step.
_PARSE_CACHE: dict[int, tuple[TestSuite, Any]] = {}


def _add_assert(node: ast.Assert, env: dict[str, Any], calls: list, checks: list) -> str | None:
    if env:
        # deepcopy: NodeTransformer edits in place, and a loop body is reused
        node = ast.fix_missing_locations(_Bind(env).visit(copy.deepcopy(node)))
    parsed = _split_assert_node(node)
    if parsed == "skip":
        return None
    if parsed is None:
        return "unsupported assert shape"
    if parsed[1][0] == "eq_expr":
        # Both sides computed by the candidate: two calls, compared pairwise.
        calls.append(parsed[0]); checks.append(("eq_next", None, None))
        calls.append(parsed[1][1]); checks.append(("rhs", None, None))
    else:
        calls.append(parsed[0]); checks.append(parsed[1])
    return None


def _expand(stmts: list, env: dict[str, Any], calls: list, checks: list) -> str | None:
    """Turn suite statements into calls + checks. Supports asserts, helper
    assignments computable on the host, and `for` loops over host-computable
    iterables whose body is asserts (unrolled). Anything else is refused."""
    for st in stmts:
        if isinstance(st, ast.Assert):
            err = _add_assert(st, env, calls, checks)
        elif (isinstance(st, ast.Assign) and len(st.targets) == 1
              and isinstance(st.targets[0], ast.Name)):
            ok, val = _host_eval(st.value, env)
            if not ok:
                return "assignment not computable on the host"
            env[st.targets[0].id] = val
            err = None
        elif isinstance(st, ast.For) and isinstance(st.target, ast.Name) and not st.orelse:
            ok, it = _host_eval(st.iter, env)
            if not ok:
                return "loop not computable on the host"
            for v in list(it):
                err = _expand(st.body, {**env, st.target.id: v}, calls, checks)
                if err:
                    return err
            err = None
        else:
            return f"unsupported statement {type(st).__name__}"
        if err:
            return err
    return None


def _parse_suite(suite: TestSuite):
    hit = _PARSE_CACHE.get(id(suite))
    if hit is not None and hit[0] is suite:
        return hit[1]
    setup, calls, checks = [], [], []
    env: dict[str, Any] = {}  # helper values the suite computes (HumanEval/151)
    parsed_all: Any = None
    for line in suite.asserts or []:
        if not line.lstrip().startswith("assert"):
            setup.append(line)
            continue
        try:
            stmts = ast.parse(line.strip()).body
        except SyntaxError:
            stmts = []
        if not stmts:
            parsed_all = f"unsplittable assert: {line[:120]}"
            break
        err = _expand(stmts, env, calls, checks)
        if err:
            parsed_all = f"unsplittable assert: {err}: {line[:120]}"
        if parsed_all is not None:
            break
    if parsed_all is None:
        parsed_all = (setup, calls, checks)
    if len(_PARSE_CACHE) > 50_000:
        _PARSE_CACHE.clear()
    _PARSE_CACHE[id(suite)] = (suite, parsed_all)
    return parsed_all


def _legacy_inprocess(sandbox, code: str, suite: TestSuite, why: str) -> SuiteResult:
    """Old in-process assert grading. ONLY for evaluation suites that cannot be
    split into call/expected pairs (5 HumanEval+ tasks with randomized or
    round-trip tests). It is hackable in principle (expected values live in the
    candidate's process), so training rewards never use it (allow_legacy=False)
    and every result is marked in `detail`."""
    from rlforge.verify.compiler import suite_to_test_code

    r = sandbox.run(code, suite_to_test_code(suite), mode="assert")
    n = sum(1 for a in suite.asserts or [] if a.lstrip())
    return SuiteResult(r.passed, r.failure_type, n, list(range(n)) if r.passed else [],
                       stderr=r.stderr or "", execution_time_s=r.execution_time_s,
                       exit_code=r.exit_code, detail=f"legacy_inprocess ({why[:80]})")


def _run_assert(sandbox, code: str, suite: TestSuite, *, max_tests: int | None,
                allow_legacy: bool = False) -> SuiteResult:
    parsed_all = _parse_suite(suite)
    if isinstance(parsed_all, str) or not parsed_all[1]:
        why = parsed_all if isinstance(parsed_all, str) else "no checkable asserts"
        if allow_legacy:
            return _legacy_inprocess(sandbox, code, suite, why)
        # Training never falls back to in-process asserts (hackable). A suite we
        # cannot split is reported, not silently graded another way.
        return SuiteResult(False, FailureType.UNKNOWN, 0, detail=why)
    setup, calls, expected = parsed_all
    if max_tests:
        calls, expected = calls[:max_tests], expected[:max_tests]
    n = len(calls)
    if n == 0:
        return SuiteResult(False, FailureType.UNKNOWN, 0, detail="no checkable asserts")
    nonce = "__R" + secrets.token_hex(8) + "__"
    t0 = time.perf_counter()
    res = sandbox.run_script(_assert_script(code, setup, calls, nonce))
    elapsed = time.perf_counter() - t0
    recs = _parse_records(res.stdout or "", nonce)
    def _passes(i: int) -> bool:
        rec = recs.get(i, {})
        if rec.get("status") != "ok":
            return False
        kind = expected[i][0]
        if kind == "eq_next":  # left side of a candidate-vs-candidate comparison
            nxt = recs.get(i + 1, {})
            return nxt.get("status") == "ok" and _check_value(rec["val"], ("eq_lit_repr", nxt["val"], None))
        if kind == "rhs":
            return _passes(i - 1)
        return _check_value(rec["val"], expected[i])

    ok = [i for i in range(n) if _passes(i)]
    if n and len(ok) == n and _DONE in (res.stdout or ""):
        return SuiteResult(True, FailureType.PASS, n, ok, execution_time_s=elapsed, exit_code=0)

    if res.failure_type == FailureType.TIMEOUT:
        ft, err = FailureType.TIMEOUT, res.stderr or ""
    elif not recs:
        # Died before the first call: syntax error, crash at import or in setup.
        err = res.stderr or ""
        ft = classify_failure(passed=False, extracted=True, timed_out=False, stderr=err) \
            if err.strip() else FailureType.UNKNOWN
    else:
        bad = next(i for i in range(n) if i not in set(ok))
        rec = recs.get(bad)
        if rec is None:
            err = res.stderr or ""
            ft = classify_failure(passed=False, extracted=True, timed_out=False, stderr=err) \
                if err.strip() else FailureType.UNKNOWN
        elif rec["status"] == "error":
            err = rec.get("err", "")
            ft = FailureType.RUNTIME
        else:
            err = f"call {bad}: {calls[bad][:200]} returned {rec['val'][:200]} expected {expected[bad]!r}"[:2000]
            ft = FailureType.ASSERTION
    return SuiteResult(False, ft, n, ok, stderr=err[-2000:], execution_time_s=elapsed,
                       exit_code=res.exit_code)


# ── public API ───────────────────────────────────────────────────────────────


def run_suite(sandbox, code: str, suite: TestSuite, *, max_tests: int | None = None,
              stop_on_fail: bool = True, allow_legacy: bool = False) -> SuiteResult:
    """Grade `code` against `suite`. Never raises for candidate behaviour.

    stop_on_fail: stdin/stdout runs stop at the first crash/timeout (the result
    is already a fail). The gold check passes False to count passing tests.
    """
    if suite.kind == "stdin_stdout":
        if not suite.io_pairs:
            return SuiteResult(False, FailureType.UNKNOWN, 0, detail="empty suite")
        return _run_stdio(sandbox, code, suite, max_tests=max_tests, stop_on_fail=stop_on_fail)
    if suite.kind == "assert":
        if not any(a.strip() for a in suite.asserts or []):
            return SuiteResult(False, FailureType.UNKNOWN, 0, detail="empty suite")
        return _run_assert(sandbox, code, suite, max_tests=max_tests, allow_legacy=allow_legacy)
    raise NotImplementedError(f"suite kind {suite.kind!r} has no grader")
