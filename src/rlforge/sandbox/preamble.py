"""
sandbox/preamble.py
────────────────────
The safety preamble — a string prepended to every candidate script before it
runs (ported from ReasoningLab's verify/executor.py).

Role in RLForge's layered defense:
  - It is NOT the security boundary. The real cage is OS-enforced:
    `resource.setrlimit` (subprocess_sandbox.py, layer B) and `bubblewrap`
    (bwrap_sandbox.py, layer C). The preamble is *defense-in-depth* — it
    neutralizes common accidental destructive calls inside the Python process
    so a hallucinated `os.system("rm -rf ...")` is a no-op even before the cage
    would have contained it.
  - It also serves correctness, not just safety: it disables faulthandler (keeps
    stderr clean for failure classification), caps thread counts (stable timing),
    and — critically — nullifies early-exit paths so a candidate cannot fake a
    "passed" result by exiting 0 before the tests run. The exception that cannot
    be blocked this way (`raise SystemExit(0)`) is caught by the sentinel
    protocol in subprocess_sandbox.py instead.

We do NOT block `import resource` here (ReasoningLab did): RLForge applies
setrlimit in the parent via preexec_fn before the interpreter starts, so the
limits are already in force and immutable from the child's perspective on the
caps that matter (a child cannot *raise* a hard limit).
"""

SAFETY_PREAMBLE = '''\
# ── RLForge safety preamble (defense-in-depth; the real cage is rlimit + bwrap) ─
import builtins as _builtins
import faulthandler as _faulthandler
import os as _os
import shutil as _shutil
import socket as _socket
import subprocess as _sp
import sys as _sys

# Disable the crash reporter so faulthandler noise cannot corrupt the stderr we
# parse for failure classification.
_faulthandler.disable()

# Block early-exit paths so a candidate cannot exit 0 before the tests run and
# fake a pass. (raise SystemExit(0) can't be blocked here — the sentinel handles
# it: passed=True requires the sentinel token in stdout, proving tests ran.)
_builtins.exit = None
_builtins.quit = None
_sys.exit      = None
_os._exit      = None   # C-level exit; bypasses Python exception handling

# Cap implicit threading so wall-clock timings are comparable across runs.
_os.environ["OMP_NUM_THREADS"] = "1"

# Process control.
_os.kill   = None
_os.system = None
_os.putenv = None

# File deletion.
_os.remove      = None
_os.unlink      = None
_os.removedirs  = None
_os.rmdir       = None
_shutil.rmtree  = None

# File modification.
_os.rename    = None
_os.renames   = None
_os.replace   = None
_os.truncate  = None
_shutil.move  = None

# Directory navigation (cwd is a throwaway dir; block escaping/discovering it).
_os.chdir  = None
_os.getcwd = None

# Directory listing (no need to enumerate the filesystem for algorithm tasks).
_os.listdir = None
_os.scandir = None

# File permissions.
_os.chmod = None
if hasattr(_shutil, "chown"):
    _shutil.chown = None

# Subprocess / process spawning (Popen is the base of all the wrappers).
_sp.Popen           = None
_sp.run             = None
_sp.call            = None
_sp.check_call      = None
_sp.check_output    = None
_sp.getoutput       = None
_sp.getstatusoutput = None

# Network. socket.socket constructs every connection type; nullifying it blocks
# outbound connections at the Python level (bwrap also removes the network
# namespace entirely in layer C).
_socket.socket = None

# Dangerous third-party imports (each becomes ImportError).
#   joblib  – spawns worker processes, bypassing isolation
#   psutil  – inspect/kill arbitrary processes
#   ipdb    – interactive debugger; hangs waiting for input
#   tkinter – GUI toolkit; hangs looking for a display
_sys.modules["joblib"]  = None
_sys.modules["psutil"]  = None
_sys.modules["ipdb"]    = None
_sys.modules["tkinter"] = None

# Linux-only os attributes (we run on Linux/aarch64). fork/forkpty are a
# fork-bomb vector (RLIMIT_NPROC=0 also blocks this at the OS level); the
# set*id / chroot family are privilege-escalation vectors.
_LINUX_ONLY_OS_ATTRS = [
    "fork", "forkpty", "killpg",
    "setuid", "setgid", "seteuid", "setegid",
    "fchdir", "fchmod", "fchown", "chown", "chroot",
    "lchflags", "lchmod", "lchown",
]
for _attr in _LINUX_ONLY_OS_ATTRS:
    if hasattr(_os, _attr):
        setattr(_os, _attr, None)

# Close the wildcard-import hole: the stdlib import block prepended BEFORE this
# preamble (LIVECODEBENCH_IMPORTS) runs `from sys import *` / `from builtins
# import *`, binding bare globals `exit`/`quit` to the real functions. Nullifying
# _sys.exit / _os._exit above does not touch those bare names, so re-nullify them
# here, after the wildcard imports, or a candidate could fake an early clean exit.
exit = None
quit = None
# ── End of safety preamble ──────────────────────────────────────────────────
'''

# LiveCodeBench's exact import preamble (lcb_runner/evaluation/testing_util.py,
# variable `import_string`), prepended verbatim before SAFETY_PREAMBLE so
# candidate code sees the same implicitly-available stdlib names the LiveCodeBench
# judge provides (bare List, deque, math.*, ...) plus the same recursion limit.
# Without this, otherwise-correct LeetCode-style solutions fail with NameError /
# RecursionError on names the platform supplies for free. Kept character-for-
# character faithful to upstream for benchmark comparability.
LIVECODEBENCH_IMPORTS = '''\
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(50000)
'''
