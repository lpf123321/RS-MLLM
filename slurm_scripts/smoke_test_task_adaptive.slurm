#!/bin/bash
# GPU smoke test for task-adaptive pruning (RouterPrunedAdapter + task_keep_ratio).
#   sbatch slurm_scripts/smoke_test_task_adaptive.slurm
#SBATCH --job-name=task_adapt
#SBATCH --output=logs/task_adapt_%j.out
#SBATCH --error=logs/task_adapt_%j.err
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/u2024311149/RS-MLLM
cd "$REPO_ROOT"

export DATA_ROOT=/users/u2024311136/shared/shared_datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:$PATH"

MODEL_PATH="${MODEL_PATH:-/users/u2024311136/shared/shared_models/lora_expert/base_model}"
GENERAL="${GENERAL_LORA:-/home/u2024311149/RS-MLLM/prune/output/new_experts/general_exp7_delta.pt}"
GROUNDING="${GROUNDING_LORA:-/home/u2024311149/RS-MLLM/prune/output/new_experts/ground_expert_update_delta.pt}"
CHANGE="${CHANGE_LORA:-/users/u2024311136/shared/shared_models/lora_expert/lora/change/delta_model.pt}"
CAPTION="${CAPTION_LORA:-/users/u2024311136/shared/shared_models/lora_expert/lora/caption/delta_model.pt}"

mkdir -p logs

python scripts/smoke_test_task_adaptive.py \
    --model_path "$MODEL_PATH" \
    --general_lora "$GENERAL" \
    --grounding_lora "$GROUNDING" \
    --change_lora "$CHANGE" \
    --caption_lora "$CAPTION"

echo "=== DONE ==="
