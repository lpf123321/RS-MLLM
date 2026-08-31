#!/bin/bash
#SBATCH --job-name=rerun_caption
#SBATCH --output=evaluation/logs/rerun_caption_%j.out
#SBATCH --error=evaluation/logs/rerun_caption_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -e
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:${RSMLLM_REPO:-REPO}"

cd ${RSMLLM_REPO:-REPO}
source /opt/app/anaconda3/etc/profile.d/conda.sh
conda activate rs_mllm
set +u

python evaluation/rerun_caption.py \
    --adapter router \
    --model_path models/Qwen3.5-4B \
    --general_lora outputs/checkpoints/expert_general_v2 \
    --grounding_lora outputs/checkpoints/expert_grounding \
    --change_lora outputs/checkpoints/expert_change \
    --output outputs/eval_results/rerun_router_vrsbench_caption.json \
    --eval_batch_size 8

echo "=== done $(date) ==="
