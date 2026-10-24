#!/bin/bash
# Overnight sampling sweep for thinking mode: does temperature and/or presence
# penalty reduce the endless-repetition loops without lowering the pass rate?
#
# Runs the vLLM length study once per setting, one after the other (each in its
# own process, so GPU memory is fully released between them). The baseline
# (temperature 0.6, presence 0) already exists: runs/eval/20261014T155332Z-...
# Together they form a 2x2: temperature {0.6, 1.0} x presence penalty {0, 1.5}.
#
# - A failed setting does NOT stop the sweep; the next one still runs.
# - Everything printed goes to runs/eval/<stamp>-sampling-sweep/sweep.log
#   (plus one log per setting), and the comparison table is written at the end
#   to compare.txt in the same folder.
#
# Usage (from the repo root, inside tmux):
#     bash scripts/run_sampling_sweep.sh
# Optional: ROUNDS=2 (answers per task per setting), PY=<python of vLLM env>.

set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# WORKDIR only exists for testing (a scratch folder with its own data/ link);
# real runs work from the repo root, where the study writes runs/eval/.
cd "${WORKDIR:-$REPO}"

PY="${PY:-$HOME/repos/RLForge/.venv/bin/python}"
STUDY="${STUDY:-$REPO/scripts/thinking_length_study_vllm.py}"
ROUNDS="${ROUNDS:-2}"
BASELINE="$REPO/runs/eval/20261014T155332Z-thinking-length-study-vllm"

# "temperature presence_penalty tag"
SETTINGS=(
  "1.0 1.5 t1.0-pp1.5"   # Qwen's official thinking-mode "general" setting
  "0.6 1.5 t0.6-pp1.5"   # penalty only: same temperature as the baseline
  "1.0 0.0 t1.0-pp0"     # temperature only: same penalty as the baseline
)

SWEEP="runs/eval/$(date -u +%Y%m%dT%H%M%SZ)-sampling-sweep"
mkdir -p "$SWEEP"
LOG="$SWEEP/sweep.log"
say() { echo "[sweep $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "start: ${#SETTINGS[@]} settings x $ROUNDS rounds, python $PY"
FOLDERS=("$BASELINE")
for s in "${SETTINGS[@]}"; do
  read -r temp pp tag <<< "$s"
  say "setting $tag: temperature $temp, presence penalty $pp"
  t0=$(date +%s)
  "$PY" "$STUDY" --rounds "$ROUNDS" --max-new-tokens 32768 \
      --temperature "$temp" --presence-penalty "$pp" --tag "$tag" \
      2>&1 | tee "$SWEEP/$tag.log"
  code=${PIPESTATUS[0]}
  # The study prints "... -> runs/eval/<folder>" at start; take the folder from
  # THIS run's log, so an older folder with the same tag can never be picked.
  folder=$(grep -oE -- "-> runs/eval/[^ ]+" "$SWEEP/$tag.log" | head -1 | cut -c4-)
  say "setting $tag: exit code $code, $(( ($(date +%s) - t0) / 60 )) min, folder ${folder:-NONE}"
  [ -n "$folder" ] && FOLDERS+=("$folder")
done

say "comparison:"
python3 "$REPO/scripts/compare_length_studies.py" "${FOLDERS[@]}" 2>&1 | tee "$SWEEP/compare.txt" | tee -a "$LOG"
say "done"
