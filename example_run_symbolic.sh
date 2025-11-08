#!/usr/bin/env bash
set -euo pipefail

# One-shot runner: generate instances, then solve all three problems
# using the symbolic baselines (problem-specific, random, random-average).

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

CFG_PATH="configs/problems.json"

if [[ ! -f "$CFG_PATH" ]]; then
  echo "[Error] Config file not found: $CFG_PATH" >&2
  exit 2
fi

echo "[Step 1] Generating instances from $CFG_PATH ..."
python example00_generate.py "$CFG_PATH"

OUT_ROOT="outputs/examples/01_generate"

declare -a PROBLEMS=(
  "personal_assistant"
  "meeting_scheduling"
  "smart_grid"
)

echo "[Step 2] Running symbolic baselines for each problem ..."
for P in "${PROBLEMS[@]}"; do
  INST_DIR="${OUT_ROOT}/${P}"
  if [[ ! -d "$INST_DIR" ]]; then
    echo "[Warning] Instance directory not found: $INST_DIR (skipping)" >&2
    continue
  fi
  echo "--> ${P}: $INST_DIR"
  python example01_symbolic.py "$INST_DIR" || {
    echo "[Warning] Symbolic run failed for $P" >&2
  }
done

echo "[Done]"

