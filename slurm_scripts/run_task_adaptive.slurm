#!/bin/bash
# Task-adaptive pruning evaluation (RouterPrunedAdapter + task_keep_ratio).
#   sbatch slurm_scripts/run_task_adaptive.slurm
#SBATCH --job-name=task_adapt
#SBATCH --output=logs/task_adapt_eval_%j.out
#SBATCH --error=logs/task_adapt_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/u2024311149/RS-MLLM
cd "$REPO_ROOT"

export DATA_ROOT=/users/u2024311136/shared/shared_datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:$PATH"

MODEL_PATH="${MODEL_PATH:-/users/u2024311136/shared/shared_models/lora_expert/base_model}"
GENERAL="${GENERAL_LORA:-/users/u2024311136/shared/shared_models/lora_expert/lora/general}"
GROUNDING="${GROUNDING_LORA:-/users/u2024311136/shared/shared_models/lora_expert/lora/grounding}"
CHANGE="${CHANGE_LORA:-/users/u2024311136/shared/shared_models/lora_expert/lora/change}"

mkdir -p logs prune/output/prune_sweep

python -m evaluation.run_task_adaptive \
    --model_path "$MODEL_PATH" \
    --general_lora "$GENERAL" \
    --grounding_lora "$GROUNDING" \
    --change_lora "$CHANGE" \
    --prune_method l2 \
    --random_samples "${N_SAMPLES:-1000}" \
    --sample_seed "${SAMPLE_SEED:-2026}" \
    --eval_batch_size "${BATCH_SIZE:-32}"

echo "=== DONE ==="
