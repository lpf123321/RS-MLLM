#!/bin/bash
# Convert merged General Expert + exp7 to a Delta relative to raw W0.
#
#SBATCH --job-name=gen_exp7_to_delta
#SBATCH --output=logs/gen_exp7_to_delta_%j.out
#SBATCH --error=logs/gen_exp7_to_delta_%j.err
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
    --merged_model prune/output/new_experts/general_exp7_merged \
    --reference_delta /users/u2024311136/shared/shared_models/lora_expert/lora/general/delta_model.pt \
    --output prune/output/new_experts/general_exp7_delta.pt

echo "=== MERGED MODEL TO DELTA DONE ==="
