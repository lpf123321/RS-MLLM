#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMAND=
GPUS=1 MAX_UPDATES=0 RUN_ID="${RS_MLLM_RUN_ID:-}" DRY=0 SMOKE=0 SLURM=0
while (($#)); do
  case "$1" in
    --gpus) GPUS="${2:?}"; shift 2;;
    --max-updates) MAX_UPDATES="${2:?}"; shift 2;;
    --run-id) RUN_ID="${2:?}"; shift 2;;
    --dry-run) DRY=1; shift;;
    --slurm) SLURM=1; shift;;
    --local) SLURM=0; shift;;
    *)
      [[ -z "$COMMAND" ]] || { echo "unexpected argument: $1" >&2; exit 2; }
      COMMAND="$1"; shift;;
  esac
done
if [[ "$COMMAND" == smoke-all ]]; then
  SMOKE=1
  ((MAX_UPDATES > 0)) || MAX_UPDATES=1
  COMMAND=all
fi
case "$COMMAND" in all|stage1_clean|ga2_general|a1_grounding|a2b_change|caption) ;; *)
  echo "Usage: bash scripts/train.sh [--slurm] smoke-all|all|STAGE [--gpus N] [--max-updates N] [--run-id ID] [--dry-run]" >&2; exit 2;;
esac
if [[ -z "$RUN_ID" ]]; then
  if ((SMOKE == 1)); then RUN_ID="smoke-$(date +%Y%m%d-%H%M%S)"; else RUN_ID=five-stage; fi
fi
RUN_ROOT="$REPO_ROOT/models/training35/runs/$RUN_ID"
LORA_ROOT="$RUN_ROOT/lora"; MERGED_ROOT="$RUN_ROOT/merged"
RUNNER="$REPO_ROOT/scripts/training/run_stage35.sh"
MERGER="$REPO_ROOT/scripts/merge_checkpoint.sh"
VERIFY="$REPO_ROOT/scripts/verify_training35_artifacts.py"
runner_extra=(); merge_extra=(); ((DRY == 1)) && runner_extra+=(--dry-run) && merge_extra+=(--dry-run)
((SMOKE == 1)) && runner_extra+=(--smoke)

run_one() {
  local stage="$1" parent="$2"
  local lora="$LORA_ROOT/$stage"
  local merged="$MERGED_ROOT/$stage"
  if [[ -f "$lora/adapter_config.json" && -f "$merged/config.json" ]]; then
    python "$VERIFY" --lora "$lora" --merged "$merged"
    echo "[training35] reuse completed stage: $stage"
    return
  fi
  bash "$RUNNER" "$stage" --model "$parent" --output "$lora" --gpus "$GPUS" --max-updates "$MAX_UPDATES" "${runner_extra[@]}"
  bash "$MERGER" --base "$parent" --lora "$lora" --output "$merged" "${merge_extra[@]}"
  ((DRY == 1)) || python "$VERIFY" --lora "$lora" --merged "$merged"
}

stage1_parent="$REPO_ROOT/models/Qwen3.5-4B"
submit_stage() {
  local stage="$1" parent="$2" dependency="$3"
  local lora="$LORA_ROOT/$stage" merged="$MERGED_ROOT/$stage"
  local -a command=(
    sbatch --parsable
    --job-name="training35-$stage"
    --nodes=1 --ntasks=1
    --cpus-per-task="${TRAINING35_SLURM_CPUS:-12}"
    --gres="gpu:$GPUS"
    --output="$REPO_ROOT/outputs/logs/training35_${stage}_%j.log"
  )
  [[ -n "${TRAINING35_SLURM_PARTITION:-}" ]] && command+=(--partition="$TRAINING35_SLURM_PARTITION")
  [[ -n "${TRAINING35_SLURM_TIME:-}" ]] && command+=(--time="$TRAINING35_SLURM_TIME")
  [[ -n "$dependency" ]] && command+=(--dependency="afterok:$dependency")
  command+=(
    "$REPO_ROOT/scripts/training/stage35.slurm"
    "$stage" "$parent" "$lora" "$merged" "$GPUS" "$MAX_UPDATES" "$SMOKE"
  )
  if ((DRY == 1)); then
    printf '[training35 slurm dry-run]'; printf ' %q' "${command[@]}"; printf '\n'
    SUBMITTED_JOB_ID="DRY_$stage"
    return
  fi
  mkdir -p "$REPO_ROOT/outputs/logs"
  local submitted_raw
  submitted_raw="$("${command[@]}")"
  SUBMITTED_JOB_ID="${submitted_raw%%;*}"
  [[ "$SUBMITTED_JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "sbatch returned an invalid job id: $SUBMITTED_JOB_ID" >&2; exit 1;
  }
  echo "[training35 slurm] submitted stage=$stage job=$SUBMITTED_JOB_ID dependency=${dependency:-none}"
}

if ((SLURM == 1)); then
  stages=("$COMMAND")
  [[ "$COMMAND" == all ]] && stages=(stage1_clean ga2_general a1_grounding a2b_change caption)
  previous_job=
  for stage in "${stages[@]}"; do
    case "$stage" in
      stage1_clean) parent="$stage1_parent";;
      ga2_general|a1_grounding|a2b_change) parent="$MERGED_ROOT/stage1_clean";;
      caption) parent="$MERGED_ROOT/ga2_general";;
    esac
    submit_stage "$stage" "$parent" "$previous_job"
    previous_job="$SUBMITTED_JOB_ID"
  done
  echo "[training35 slurm] final job: $previous_job; run root: $RUN_ROOT"
  exit 0
fi

if [[ "$COMMAND" == all ]]; then
  run_one stage1_clean "$stage1_parent"
  run_one ga2_general "$MERGED_ROOT/stage1_clean"
  run_one a1_grounding "$MERGED_ROOT/stage1_clean"
  run_one a2b_change "$MERGED_ROOT/stage1_clean"
  run_one caption "$MERGED_ROOT/ga2_general"
  if ((DRY == 1)); then echo "[training35] five-stage dry-run complete: $RUN_ROOT";
  else echo "[training35] five-stage chain passed: $RUN_ROOT"; fi
else
  case "$COMMAND" in
    stage1_clean) parent="$stage1_parent";;
    ga2_general|a1_grounding|a2b_change) parent="$MERGED_ROOT/stage1_clean";;
    caption) parent="$MERGED_ROOT/ga2_general";;
  esac
  run_one "$COMMAND" "$parent"
fi
