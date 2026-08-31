#!/bin/bash
# GPU smoke test for pre-LLM x in-LLM combination.
#   sbatch slurm_scripts/smoke_test_inllm.slurm
#SBATCH --job-name=inllm
#SBATCH --output=logs/inllm_%j.out
#SBATCH --error=logs/inllm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/u2024311149/RS-MLLM
cd "$REPO_ROOT"

export DATA_ROOT=${RSMLLM_DATA_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/datasets}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:$PATH"

MODEL_PATH="${MODEL_PATH:-${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/base_model}"
GENERAL="${GENERAL_LORA:-${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/lora/general/delta_model.pt}"
GROUNDING="${GROUNDING_LORA:-${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/lora/grounding/delta_model.pt}"
CHANGE="${CHANGE_LORA:-${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/lora/change/delta_model.pt}"

mkdir -p logs

python scripts/smoke_test_inllm.py \
    --model_path "$MODEL_PATH" \
    --general_lora "$GENERAL" \
    --grounding_lora "$GROUNDING" \
    --change_lora "$CHANGE"

echo "=== DONE ==="
