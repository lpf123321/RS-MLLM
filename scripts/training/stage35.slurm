#!/usr/bin/env bash
set -euo pipefail

# Slurm copies the submitted script into /var/spool/slurm before executing it,
# so BASH_SOURCE no longer points into the checkout. sbatch is launched from
# the repository root by scripts/train.sh; use that stable submit directory.
REPO_ROOT="${RS_MLLM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$REPO_ROOT" && -f "$REPO_ROOT/scripts/training/run_stage35.sh" ]] || {
  echo "cannot resolve repository root from SLURM_SUBMIT_DIR=$REPO_ROOT" >&2
  exit 2
}
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
