#!/bin/bash
# Validate the XLRS official visual-grounding path on a small sample.
#
#SBATCH --job-name=xlrs_vg_smoke
#SBATCH --output=logs/xlrs_vg_smoke_%j.out
#SBATCH --error=logs/xlrs_vg_smoke_%j.err
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /home/u2024311149/RS-MLLM
export DATA_ROOT=${RSMLLM_DATA_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/datasets}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:$PATH"

python -m evaluation.run_delta_prune_eval \
    --model_path ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/base_model \
    --general_lora ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/lora/general/delta_model.pt \
    --grounding_lora ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/lora/grounding/delta_model.pt \
    --change_lora ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/lora/change/delta_model.pt \
    --caption_lora ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/lora/caption/delta_model.pt \
    --methods l2norm \
    --keep_ratios 1.0 \
    --datasets xlrs_grounding \
    --random_samples 8 \
    --sample_seed 2026 \
    --eval_batch_size 4 \
    --grounding_max_new_tokens 64 \
    --force_think \
    --output_dir prune/output/xlrs_grounding_smoke_v2

echo "=== XLRS GROUNDING SMOKE DONE ==="
