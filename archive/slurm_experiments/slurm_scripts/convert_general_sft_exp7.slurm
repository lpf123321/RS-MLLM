#!/bin/bash
# Build the General Expert Delta after merging the new SFT and exp7 PEFT LoRA.
#
#SBATCH --job-name=gen_exp7_delta
#SBATCH --output=logs/gen_exp7_delta_%j.out
#SBATCH --error=logs/gen_exp7_delta_%j.err
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /home/u2024311149/RS-MLLM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:$PATH"

python scripts/convert_general_sft_exp7_to_delta.py \
    --base_model ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/base_model \
    --expert_delta ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/lora/general/delta_model.pt \
    --peft_lora /users/u2024311164/shared/exp7_old_mme_plus_old_xlrs_plus_vrsbench_vqa_weighted_lora \
    --output prune/output/new_experts/general_exp7_merged

echo "=== GENERAL EXP7 DELTA CONVERSION DONE ==="
