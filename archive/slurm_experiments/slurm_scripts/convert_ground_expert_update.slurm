#!/bin/bash
# Merge the Grounding Expert Delta and ground_expert_update PEFT LoRA.
#
#SBATCH --job-name=ground_update
#SBATCH --output=logs/ground_update_%j.out
#SBATCH --error=logs/ground_update_%j.err
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
    --expert_delta ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/lora/grounding/delta_model.pt \
    --peft_lora /users/u2024311164/shared/ground_expert_update_lora \
    --output prune/output/new_experts/ground_expert_update_merged

echo "=== GROUND EXPERT UPDATE MERGE DONE ==="
