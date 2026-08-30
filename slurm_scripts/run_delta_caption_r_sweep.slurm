#!/bin/bash
# Caption expert keep-ratio sweep. Submit after fixed-ratio caption ablation.
#
#SBATCH --job-name=caption_r_sweep
#SBATCH --output=logs/caption_r_sweep_%j.out
#SBATCH --error=logs/caption_r_sweep_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00

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
    --methods uniform random mmtok l2norm divprune scope_l2 \
    --keep_ratios 0.1 0.25 0.35 0.5 0.65 0.75 0.9 1.0 \
    --datasets vrsbench_caption xlrs_caption_instruction \
    --random_samples 1000 \
    --sample_seed 2026 \
    --eval_batch_size 32 \
    --caption_max_new_tokens 550 \
    --force_think \
    --output_dir prune/output/delta_caption_r_sweep_v2

echo "=== DELTA CAPTION R SWEEP DONE ==="
