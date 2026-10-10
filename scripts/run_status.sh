#!/usr/bin/env bash
# Status of a training run. Read-only: never kills, never deletes.
#   scripts/run_status.sh                     # newest run in runs/train
#   scripts/run_status.sh runs/train/<run>    # a specific run
#   watch -n 600 scripts/run_status.sh        # refresh every 10 min
set -uo pipefail

# Default: the most recently created train run folder.
RUN_DIR="${1:-$(ls -td "$(dirname "$0")"/../runs/train/*/ 2>/dev/null | head -1)}"
RUN_DIR="${RUN_DIR%/}"
echo "run: $(basename "$RUN_DIR")"

# Match the python process itself, not any shell whose command line merely
# mentions the script name.
alive=$(pgrep -f "python[0-9.]* scripts/train_grpo\.py" >/dev/null && echo RUNNING || echo "NOT RUNNING")
# Step count and rollouts per step come from the run's own config snapshot, so the
# numbers are right for any run length (older runs without a snapshot show "?").
read -r max_steps per_step < <(python3 - "$RUN_DIR" <<'PY'
import json, sys
try:
    g = json.load(open(sys.argv[1] + "/config_resolved.json"))["grpo"]
    print(g["max_steps"], g["tasks_per_update"] * g["num_generations"])
except Exception:
    print("? 64")
PY
)
# The file does not exist until the first rollouts are graded.
rollouts=0
[ -f "$RUN_DIR/executions.jsonl" ] && rollouts=$(wc -l < "$RUN_DIR/executions.jsonl")
status=$(grep -o '"status": "[a-z]*"' "$RUN_DIR/run_meta.json" 2>/dev/null | cut -d'"' -f4)
vram=$(tail -1 "$RUN_DIR/vram.jsonl" 2>/dev/null | grep -o '"reserved_gb": [0-9.]*' | cut -d' ' -f2)

printf '%s  %s (%s)  step ~%s/%s  %s rollouts  vram %sGB\n' \
  "$(date +%H:%M:%S)" "$alive" "${status:-no run_meta}" "$(( rollouts / per_step ))" \
  "$max_steps" "$rollouts" "${vram:-?}"

# Validation curve so far: one line per probe (step, pass rate, truncated share).
if [ -f "$RUN_DIR/val_summary.jsonl" ]; then
  echo "validation probes:"
  python3 - "$RUN_DIR/val_summary.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    s = json.loads(line)
    print(f"  step {s['step']:>3}  pass {s['pass_rate']:.3f}  solved-any {s['tasks_solved_any']:.0%}"
          f"  truncated {s['truncated_rate']:.0%}  ({s['total_seconds']/60:.0f} min)")
PY
fi
