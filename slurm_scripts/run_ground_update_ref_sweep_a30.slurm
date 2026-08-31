#!/bin/bash
# Referring R sweep, single method chunk, on A30 (2h limit).
# Usage: sbatch slurm_scripts/run_ground_update_ref_sweep_a30.slurm "uniform random"
# The eval script skips existing JSONs, so a timed-out job can be resubmitted.
#
#SBATCH --job-name=ref_sweep
#SBATCH --output=logs/ref_sweep_%A_%j.out
#SBATCH --error=logs/ref_sweep_%A_%j.err
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /home/u2024311149/RS-MLLM
export DATA_ROOT=/users/u2024311136/shared/shared_datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:$PATH"

METHOD="${1:?usage: sbatch ... \"method1 method2 ...\"}"

python -m evaluation.run_delta_prune_eval \
    --model_path /users/u2024311136/shared/shared_models/lora_expert/base_model \
    --general_lora prune/output/new_experts/general_exp7_delta.pt \
    --grounding_lora prune/output/new_experts/ground_expert_update_delta.pt \
    --change_lora /users/u2024311136/shared/shared_models/lora_expert/lora/change/delta_model.pt \
    --caption_lora /users/u2024311136/shared/shared_models/lora_expert/lora/caption/delta_model.pt \
    --methods $METHOD \
    --keep_ratios 0.1 0.25 0.35 0.5 0.65 0.75 0.9 1.0 \
    --datasets vrsbench_referring xlrs_grounding \
    --random_samples 1000 \
    --sample_seed 2026 \
    --eval_batch_size 32 \
    --grounding_max_new_tokens 64 \
    --image_min_pixels 65536 \
    --image_max_pixels 1048576 \
    --force_think \
    --output_dir prune/output/delta_ground_update_r_sweep

echo "=== REF SWEEP METHODS=$METHOD DONE ==="
