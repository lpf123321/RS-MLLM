#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:?stage required}"
PARENT="${2:?parent model required}"
LORA="${3:?LoRA output required}"
MERGED="${4:?merged output required}"
GPUS="${5:?GPU count required}"
MAX_UPDATES="${6:?max updates required}"
SMOKE="${7:?smoke flag required}"

runner_args=(
  "$STAGE" --model "$PARENT" --output "$LORA"
  --gpus "$GPUS" --max-updates "$MAX_UPDATES"
)
((SMOKE == 1)) && runner_args+=(--smoke)

bash "$REPO_ROOT/scripts/training/run_stage35.sh" "${runner_args[@]}"
bash "$REPO_ROOT/scripts/merge_checkpoint.sh" \
  --base "$PARENT" --lora "$LORA" --output "$MERGED"
python "$REPO_ROOT/scripts/verify_training35_artifacts.py" \
  --lora "$LORA" --merged "$MERGED"
