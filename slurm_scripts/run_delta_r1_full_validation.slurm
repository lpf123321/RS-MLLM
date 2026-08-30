#!/bin/bash
# Full-data R=1 validation against the shared multi-expert evaluation protocol.
# This is intentionally separate from the 1000-sample R sweep.
#
#SBATCH --job-name=delta_r1_full
#SBATCH --output=logs/delta_r1_full_%j.out
#SBATCH --error=logs/delta_r1_full_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /home/u2024311149/RS-MLLM
export DATA_ROOT=/users/u2024311136/shared/shared_datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:$PATH"

python -m evaluation.run_delta_prune_eval \
    --model_path /users/u2024311136/shared/shared_models/lora_expert/base_model \
    --general_lora /users/u2024311136/shared/shared_models/lora_expert/lora/general/delta_model.pt \
    --grounding_lora /users/u2024311136/shared/shared_models/lora_expert/lora/grounding/delta_model.pt \
    --change_lora /users/u2024311136/shared/shared_models/lora_expert/lora/change/delta_model.pt \
    --caption_lora /users/u2024311136/shared/shared_models/lora_expert/lora/caption/delta_model.pt \
    --methods l2norm \
    --keep_ratios 1.0 \
    --datasets vrsbench mme xlrs levircc \
    --random_samples 0 \
    --sample_seed 2026 \
    --eval_batch_size 4 \
    --force_think \
    --output_dir prune/output/delta_r1_full_validation

echo "=== DELTA R1 FULL VALIDATION DONE ==="
