#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
INSTALLER="$ROOT_DIR/3-install-model-inference.sh"
ENTRYPOINT="$ROOT_DIR/model-inference/examples/run_llm_with_flaggems.py"

[[ -f "$INSTALLER" ]] || { echo "missing installer: $INSTALLER" >&2; exit 1; }
[[ -f "$ENTRYPOINT" ]] || { echo "missing inference entrypoint: $ENTRYPOINT" >&2; exit 1; }

bash -n "$INSTALLER"
HELP_OUTPUT=$(bash "$INSTALLER" --help)
grep -F -- '--pytorch-mode' <<<"$HELP_OUTPUT"
grep -F -- '--model-id' <<<"$HELP_OUTPUT"
grep -F -- '--skip-inference' <<<"$HELP_OUTPUT"

python3 -m py_compile "$ENTRYPOINT"
grep -F -- 'flag_gems.use_gems' "$ENTRYPOINT"
grep -F -- 'AutoModelForCausalLM' "$ENTRYPOINT"

if [[ "${RUN_MODEL_INFERENCE_INTEGRATION:-0}" == 1 ]]; then
  PREFIX="${MODEL_INFERENCE_TEST_PREFIX:-$ROOT_DIR/../flagOS-installed/model-inference}"
  MODEL_ID="${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
  INSTALL_ARGS=(
    --prefix "$PREFIX"
    --model-id "$MODEL_ID"
    --max-new-tokens "${MAX_NEW_TOKENS:-8}"
  )
  if [[ -n "${MODEL_PATH:-}" ]]; then
    INSTALL_ARGS+=(--model-path "$MODEL_PATH")
  fi
  if [[ -n "${PYTORCH_MODE:-}" ]]; then
    INSTALL_ARGS+=(--pytorch-mode "$PYTORCH_MODE")
  fi
  bash "$INSTALLER" "${INSTALL_ARGS[@]}"

  LOG_FILE=$(find "$PREFIX/logs" -maxdepth 1 -type f -name 'inference-*.log' \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
  [[ -n "$LOG_FILE" && -s "$LOG_FILE" ]] || {
    echo "missing non-empty inference log" >&2
    exit 1
  }
  grep -F -- 'flag_gems:' "$LOG_FILE"
  grep -F -- 'flagtree:' "$LOG_FILE"
  grep -F -- 'triton_import:' "$LOG_FILE"
  grep -F -- 'torch:' "$LOG_FILE"
  grep -F -- 'flaggems_text:' "$LOG_FILE"
  awk '/flaggems_text:/{seen=1; next} seen && NF {found=1; exit} END{exit !found}' "$LOG_FILE"
  find "$PREFIX/artifacts/triton-dumps" -type f -size +0c -print -quit | grep -q .
fi

printf 'model-inference static checks passed\n'
