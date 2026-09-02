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
REMAINING="${8:-}"

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

if [[ -n "$REMAINING" ]]; then
  NEXT_STAGE="${REMAINING%%,*}"
  if [[ "$REMAINING" == *,* ]]; then NEXT_REMAINING="${REMAINING#*,}"; else NEXT_REMAINING=""; fi
  RUN_ROOT="$(cd "$(dirname "$LORA")/.." && pwd)"
  LORA_ROOT="$RUN_ROOT/lora"
  MERGED_ROOT="$RUN_ROOT/merged"
  case "$NEXT_STAGE" in
    ga2_general|a1_grounding|a2b_change) NEXT_PARENT="$MERGED_ROOT/stage1_clean" ;;
    caption) NEXT_PARENT="$MERGED_ROOT/ga2_general" ;;
    *) echo "invalid successor stage: $NEXT_STAGE" >&2; exit 2 ;;
  esac
  NEXT_LORA="$LORA_ROOT/$NEXT_STAGE"
  NEXT_MERGED="$MERGED_ROOT/$NEXT_STAGE"
  command=(
    sbatch --parsable --job-name="training35-$NEXT_STAGE"
    --nodes=1 --ntasks=1
    --cpus-per-task="${TRAINING35_SLURM_CPUS:-12}"
    --gres="gpu:$GPUS"
    --output="$REPO_ROOT/outputs/logs/training35_${NEXT_STAGE}_%j.log"
    --dependency="afterok:$SLURM_JOB_ID"
  )
  [[ -n "${TRAINING35_SLURM_PARTITION:-}" ]] && command+=(--partition="$TRAINING35_SLURM_PARTITION")
  [[ -n "${TRAINING35_SLURM_TIME:-}" ]] && command+=(--time="$TRAINING35_SLURM_TIME")
  command+=(
    "$REPO_ROOT/scripts/training/stage35.slurm"
    "$NEXT_STAGE" "$NEXT_PARENT" "$NEXT_LORA" "$NEXT_MERGED"
    "$GPUS" "$MAX_UPDATES" "$SMOKE" "$NEXT_REMAINING"
  )
  submitted_raw="$("${command[@]}")"
  submitted_id="${submitted_raw%%;*}"
  [[ "$submitted_id" =~ ^[0-9]+$ ]] || { echo "sbatch returned invalid successor id: $submitted_id" >&2; exit 1; }
  echo "[training35 slurm] submitted successor stage=$NEXT_STAGE job=$submitted_id dependency=$SLURM_JOB_ID"
fi
