#!/usr/bin/env bash
# One-line status for a GRPO training run. Read-only: never kills, never deletes.
#   watch -n 600 scripts/run_status.sh        # refresh every 10 min
#   scripts/run_status.sh                     # one shot
set -uo pipefail

# Default: the most recently created train run folder.
RUN_DIR="${1:-$(ls -td "$(dirname "$0")"/../runs/train/*/ 2>/dev/null | head -1)}"
echo "run: $(basename "$RUN_DIR")"

alive=$(pgrep -f "train_grpo.py" >/dev/null && echo RUNNING || echo "NOT RUNNING")
rollouts=$(wc -l < "$RUN_DIR/executions.jsonl" 2>/dev/null || echo 0)
rollouts=${rollouts:-0}
step=$(( rollouts / 64 ))
vram=$(tail -1 "$RUN_DIR/vram.jsonl" 2>/dev/null | grep -o '"reserved_gb": [0-9.]*' | cut -d' ' -f2)
events=$(find "$RUN_DIR" -name 'events.out.tfevents*' 2>/dev/null | wc -l)

printf '%s  %s  step ~%s/12  %s rollouts  vram %sGB  tb-events %s\n' \
  "$(date +%H:%M:%S)" "$alive" "$step" "$rollouts" "${vram:-?}" "$events"

# last two reward groups, so you see per-task pass rates as they land
grep -o '"task_id": "[^"]*"' "$RUN_DIR/executions.jsonl" 2>/dev/null | tail -1 | sed 's/^/  last graded: /'
