#!/bin/bash
# Caption expert fixed-ratio evaluation: VRSBench Caption + XLRS Caption.
#
#SBATCH --job-name=delta_caption
#SBATCH --output=logs/delta_caption_%j.out
#SBATCH --error=logs/delta_caption_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

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
    --methods uniform random mmtok l2norm divprune scope_l2 \
    --keep_ratios 0.5 0.25 \
    --datasets vrsbench_caption xlrs_caption_instruction \
    --random_samples 1000 \
    --sample_seed 2026 \
    --eval_batch_size 32 \
    --caption_max_new_tokens 550 \
    --force_think \
    --output_dir prune/output/delta_caption_ablation_batch32_v2

echo "=== DELTA CAPTION ABLATION DONE ==="
