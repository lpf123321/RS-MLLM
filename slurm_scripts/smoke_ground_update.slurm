#!/bin/bash
# Smoke test the new Grounding Expert Delta (ground_expert_update).
#
#SBATCH --job-name=ground_upd_smoke
#SBATCH --output=logs/ground_upd_smoke_%j.out
#SBATCH --error=logs/ground_upd_smoke_%j.err
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /home/u2024311149/RS-MLLM
export DATA_ROOT=/users/u2024311136/shared/shared_datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:$PATH"

python -m evaluation.run_delta_prune_eval \
    --model_path /users/u2024311136/shared/shared_models/lora_expert/base_model \
    --general_lora prune/output/new_experts/general_exp7_delta.pt \
    --grounding_lora prune/output/new_experts/ground_expert_update_delta.pt \
    --change_lora /users/u2024311136/shared/shared_models/lora_expert/lora/change/delta_model.pt \
    --caption_lora /users/u2024311136/shared/shared_models/lora_expert/lora/caption/delta_model.pt \
    --methods l2norm \
    --keep_ratios 1.0 \
    --datasets vrsbench xlrs_grounding \
    --random_samples 8 \
    --sample_seed 2026 \
    --eval_batch_size 4 \
    --grounding_max_new_tokens 64 \
    --force_think \
    --output_dir prune/output/ground_update_smoke

echo "=== GROUND UPDATE SMOKE DONE ==="
