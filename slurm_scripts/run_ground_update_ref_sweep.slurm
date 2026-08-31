#!/bin/bash
# Referring R sweep with the new Grounding Expert Delta (ground_expert_update).
# Covers VRSBench Referring and XLRS visual grounding.
#
#SBATCH --job-name=ground_ref_sweep
#SBATCH --output=logs/ground_ref_sweep_%j.out
#SBATCH --error=logs/ground_ref_sweep_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /home/u2024311149/RS-MLLM
export DATA_ROOT=${RSMLLM_DATA_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/datasets}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:$PATH"

python -m evaluation.run_delta_prune_eval \
    --model_path ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/base_model \
    --general_lora prune/output/new_experts/general_exp7_delta.pt \
    --grounding_lora prune/output/new_experts/ground_expert_update_delta.pt \
    --change_lora ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/lora/change/delta_model.pt \
    --caption_lora ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/lora/caption/delta_model.pt \
    --methods uniform random mmtok l2norm divprune scope_l2 \
    --keep_ratios 0.1 0.25 0.35 0.5 0.65 0.75 0.9 1.0 \
    --datasets vrsbench_referring xlrs_grounding \
    --random_samples 1000 \
    --sample_seed 2026 \
    --eval_batch_size 32 \
    --grounding_max_new_tokens 64 \
    --force_think \
    --output_dir prune/output/delta_ground_update_r_sweep

echo "=== GROUND UPDATE REFERRING R SWEEP DONE ==="
