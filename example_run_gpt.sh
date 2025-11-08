#!/usr/bin/env bash
set -euo pipefail

# Example runner:
# 1) Generate instances via example00 (config-driven)
# 2) Run ICC_DM for all three problems using OpenAI Chat, reading model/key from configs/gpt_api.json

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PROBLEMS_CFG="configs/problems.json"
GPT_CFG="configs/gpt_api.json"

if [[ ! -f "$PROBLEMS_CFG" ]]; then
  echo "[Error] Problems config not found: $PROBLEMS_CFG" >&2
  exit 2
fi
if [[ ! -f "$GPT_CFG" ]]; then
  echo "[Error] GPT config not found: $GPT_CFG" >&2
  exit 2
fi

echo "[Step 1] Generating instances from $PROBLEMS_CFG ..."
python example00_generate.py "$PROBLEMS_CFG"

OUT_ROOT="outputs/examples/01_generate"
declare -a PROBLEMS=(
  "personal_assistant"
  "meeting_scheduling"
  "smart_grid"
)

echo "[Step 2] Running ICC_DM for all problems via GPT ..."
for P in "${PROBLEMS[@]}"; do
  INST_DIR="${OUT_ROOT}/${P}"
  if [[ ! -d "$INST_DIR" ]]; then
    echo "[Warning] Instance dir not found: $INST_DIR (skipping)" >&2
    continue
  fi
  echo "--> ${P}: $INST_DIR"
  python example03_icc_dm.py "$INST_DIR" --provider gpt --provider-config "$GPT_CFG" --rounds 5 || {
    echo "[Warning] ICC_DM failed for $P using GPT." >&2
  }
done

echo "[Done]"
