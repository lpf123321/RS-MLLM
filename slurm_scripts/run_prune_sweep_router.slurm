#!/bin/bash
# L2 / DivPrune keep-ratio sweep on the multi-expert router model.
# Submits 2 parallel jobs (one per method; each loads the model once and loops
# 4 datasets x 8 keep-ratios). Results go to prune/output/prune_sweep/.
# (QOS limit is 2 jobs/user, so array is 0-1.)
#
#   sbatch slurm_scripts/run_prune_sweep_router.slurm
#
# Overrides (env): MODEL_PATH, GENERAL_LORA, GROUNDING_LORA, CHANGE_LORA,
#                  N_SAMPLES, SAMPLE_SEED, BATCH_SIZE, RATIOS.
#SBATCH --job-name=prune_sweep
#SBATCH --output=logs/prune_sweep_%A_%a.out
#SBATCH --error=logs/prune_sweep_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-1

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/u2024311149/RS-MLLM
cd "$REPO_ROOT"

export DATA_ROOT=/users/u2024311136/shared/shared_datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:$PATH"

MODEL_PATH="${MODEL_PATH:-/users/u2024311136/shared/shared_models/lora_expert/base_model}"
GENERAL="${GENERAL_LORA:-/users/u2024311136/shared/shared_models/lora_expert/lora/general/delta_model.pt}"
GROUNDING="${GROUNDING_LORA:-/users/u2024311136/shared/shared_models/lora_expert/lora/grounding/delta_model.pt}"
CHANGE="${CHANGE_LORA:-/users/u2024311136/shared/shared_models/lora_expert/lora/change/delta_model.pt}"
N_SAMPLES="${N_SAMPLES:-1000}"
SAMPLE_SEED="${SAMPLE_SEED:-2026}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RATIOS="${RATIOS:-0.1 0.25 0.35 0.5 0.65 0.75 0.9 1.0}"

METHODS=(l2 divprune)
DATASETS_ARGS="vrsbench mme xlrs levircc"

TASK="$SLURM_ARRAY_TASK_ID"
METHOD="${METHODS[$TASK]}"

mkdir -p logs prune/output/prune_sweep

echo "task=$TASK method=$METHOD datasets='$DATASETS_ARGS'" \
     "samples=$N_SAMPLES seed=$SAMPLE_SEED batch=$BATCH_SIZE"

python -m evaluation.run_prune_sweep \
    --model_path "$MODEL_PATH" \
    --general_lora "$GENERAL" \
    --grounding_lora "$GROUNDING" \
    --change_lora "$CHANGE" \
    --methods "$METHOD" \
    --datasets $DATASETS_ARGS \
    --ratios $RATIOS \
    --random_samples "$N_SAMPLES" \
    --sample_seed "$SAMPLE_SEED" \
    --eval_batch_size "$BATCH_SIZE" \
    --output_dir prune/output/prune_sweep

echo "=== DONE task=$TASK ==="
