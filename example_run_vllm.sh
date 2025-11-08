#!/usr/bin/env bash
set -euo pipefail

# Example runner:
# 1) Generate instances via example00_generate.py (config-driven)
# 2) Start vLLM server using example02_run_vllm.py and configs/vllm_server.json
# 3) Run ICC_DM for all three problems
# 4) Shut down the server gracefully

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PROBLEMS_CFG="configs/problems.json"
VLLM_CFG="configs/vllm_server.json"

if [[ ! -f "$PROBLEMS_CFG" ]]; then
  echo "[Error] Problems config not found: $PROBLEMS_CFG" >&2
  exit 2
fi
if [[ ! -f "$VLLM_CFG" ]]; then
  echo "[Error] vLLM config not found: $VLLM_CFG" >&2
  exit 2
fi

echo "[Step 1] Generating instances from $PROBLEMS_CFG ..."
python example00_generate.py "$PROBLEMS_CFG"

echo "[Step 2] Starting vLLM server via example02_run_vllm.py (text mode) ..."
# Best-effort: clear any straggler servers from previous runs to release ports
pkill -f 'api_layer/vllm_server.py' >/dev/null 2>&1 || true
pkill -f 'api_layer/vllm_server_image.py' >/dev/null 2>&1 || true

"$(which python)" example02_run_vllm.py --mode text --config "$VLLM_CFG" >/tmp/vllm_server.log 2>&1 &
WRAPPER_PID=$!
trap 'echo "[Info] Stopping vLLM wrapper (PID ${WRAPPER_PID})"; kill -INT ${WRAPPER_PID} >/dev/null 2>&1 || true' EXIT

echo "  -> PID=${WRAPPER_PID}; sleeping 120s for server warm-up (large models take time) ..."
sleep 120
echo "  -> Continuing. If issues arise, check /tmp/vllm_server.log"

OUT_ROOT="outputs/examples/01_generate"
declare -a PROBLEMS=(
  "personal_assistant"
  "meeting_scheduling"
  "smart_grid"
)

echo "[Step 3] Running ICC_DM for all problems via vLLM ..."
for P in "${PROBLEMS[@]}"; do
  INST_DIR="${OUT_ROOT}/${P}"
  if [[ ! -d "$INST_DIR" ]]; then
    echo "[Warning] Instance dir not found: $INST_DIR (skipping)" >&2
    continue
  fi
  echo "--> ${P}: $INST_DIR"
  python example03_icc_dm.py "$INST_DIR" --provider vllm --provider-config "$VLLM_CFG" --config-mode text --rounds 5 || {
    echo "[Warning] ICC_DM failed for $P; see /tmp/vllm_server.log" >&2
  }
done

echo "[Step 4] Shutting down vLLM server ..."
kill -INT "${WRAPPER_PID}" >/dev/null 2>&1 || true
# Wait up to ~8s for a graceful stop, then escalate
for _ in $(seq 1 16); do
  if ! kill -0 "${WRAPPER_PID}" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
if kill -0 "${WRAPPER_PID}" >/dev/null 2>&1; then
  echo "  -> Wrapper still running; sending SIGTERM ..."
  kill -TERM "${WRAPPER_PID}" >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do
    if ! kill -0 "${WRAPPER_PID}" >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
fi
if kill -0 "${WRAPPER_PID}" >/dev/null 2>&1; then
  echo "  -> Forcing SIGKILL ..."
  kill -KILL "${WRAPPER_PID}" >/dev/null 2>&1 || true
fi
wait "${WRAPPER_PID}" 2>/dev/null || true
# Extra safety: ensure no stray server remains bound
pkill -f 'api_layer/vllm_server.py' >/dev/null 2>&1 || true
trap - EXIT
echo "[Done]"
