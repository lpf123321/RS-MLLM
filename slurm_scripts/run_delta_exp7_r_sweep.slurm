#!/bin/bash
# R sweep with General Expert Delta + exp7 PEFT merged Delta.
# R=0.5 and R=0.25 are included in this single complete sweep.
#
#SBATCH --job-name=exp7_r_sweep
#SBATCH --output=logs/exp7_r_sweep_%j.out
#SBATCH --error=logs/exp7_r_sweep_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /home/u2024311149/RS-MLLM
export DATA_ROOT=${RSMLLM_DATA_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/datasets}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:$PATH"

python -m evaluation.run_delta_prune_eval \
    --model_path ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/base_model \
    --general_lora prune/output/new_experts/general_exp7_delta.pt \
    --grounding_lora ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/lora/grounding/delta_model.pt \
    --change_lora ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/lora/change/delta_model.pt \
    --caption_lora ${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/lora/caption/delta_model.pt \
    --methods uniform random mmtok l2norm divprune scope_l2 \
    --keep_ratios 0.1 0.25 0.35 0.5 0.65 0.75 0.9 1.0 \
    --datasets vrsbench mme xlrs levircc \
    --random_samples 1000 \
    --sample_seed 2026 \
    --eval_batch_size 32 \
    --force_think \
    --output_dir prune/output/delta_exp7_r_sweep

echo "=== EXP7 DELTA R SWEEP DONE ==="
