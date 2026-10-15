"""Compare thinking length studies run with different sampling settings.

Reads each study folder (meta.json + answers.jsonl) and prints one row per
setting: pass rate, cut-off share, how many cut-offs are loops, and the pass
rate at 16k / 24k / 32k. Only the standard library is used, so it runs with any
Python.

Usage:
    python3 scripts/compare_length_studies.py runs/eval/<study1> runs/eval/<study2> ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LOOP = 0.2  # repeated share above this = a loop (finished answers stay below ~0.05)
CAPS = [16384, 24576, 32768]


def row(folder: Path) -> dict:
    meta = json.loads((folder / "meta.json").read_text())
    recs = [json.loads(line) for line in open(folder / "answers.jsonl")]
    n = len(recs)
    cut = [r for r in recs if r["hit_token_cap"]]
    s = meta["sampling"]
    out = {
        "setting": f"temp {s['temperature']}, presence {s['presence_penalty']}",
        "answers": n,
        "pass": sum(r["passed"] for r in recs) / n,
        "cut_off": len(cut) / n,
        # older studies (HF version) have no `repeated` field
        "loops": sum(r.get("repeated", 0) > LOOP for r in cut),
    }
    for cap in CAPS:
        out[f"pass@{cap // 1024}k"] = sum(
            r["passed"] for r in recs if not r["hit_token_cap"] and r["completion_tokens"] <= cap
        ) / n
    return out


def main() -> None:
    rows = []
    for arg in sys.argv[1:]:
        folder = Path(arg)
        if not (folder / "answers.jsonl").exists():
            print(f"skipped (no answers): {folder}")
            continue
        rows.append((folder.name, row(folder)))
    if not rows:
        sys.exit("no studies with answers found")
    print(f"{'setting':<26}{'answers':>8}{'pass':>7}{'cut off':>9}{'loops':>7}"
          + "".join(f"{f'pass@{c // 1024}k':>11}" for c in CAPS))
    for name, r in rows:
        print(f"{r['setting']:<26}{r['answers']:>8}{r['pass']:>7.0%}{r['cut_off']:>9.0%}{r['loops']:>7}"
              + "".join(f"{r[f'pass@{c // 1024}k']:>11.0%}" for c in CAPS) + f"   {name}")


if __name__ == "__main__":
    main()
