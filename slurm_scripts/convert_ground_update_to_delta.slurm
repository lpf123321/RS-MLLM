#!/bin/bash
# Convert merged Grounding Expert + update to a Delta relative to raw W0.
#
#SBATCH --job-name=ground_update_delta
#SBATCH --output=logs/ground_update_delta_%j.out
#SBATCH --error=logs/ground_update_delta_%j.err
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /home/u2024311149/RS-MLLM
export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:$PATH"

python scripts/convert_merged_model_to_delta.py \
    --base_model /users/u2024311136/shared/shared_models/lora_expert/base_model \
    --merged_model prune/output/new_experts/ground_expert_update_merged \
    --reference_delta /users/u2024311136/shared/shared_models/lora_expert/lora/grounding/delta_model.pt \
    --output prune/output/new_experts/ground_expert_update_delta.pt

echo "=== GROUND EXPERT UPDATE TO DELTA DONE ==="
