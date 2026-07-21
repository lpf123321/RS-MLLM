#!/bin/bash
#SBATCH --job-name=vrsbench_grpo_full
#SBATCH --output=finetune_framework/VRSbench/grpo/logs/grpo_full_%j.out
#SBATCH --error=finetune_framework/VRSbench/grpo/logs/grpo_full_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00

set -eo pipefail
export PYTHONUNBUFFERED=1

# 从仓库根目录提交。默认使用全量 referring JSON，并从 merged SFT 模型开始新实验：
#   sbatch finetune_framework/VRSbench/grpo/finetune_grpo_referring.slurm
#
# 已确认 merged model 和全量 JSON 存在时，也可跳过对应检查步骤：
#   sbatch finetune_framework/VRSbench/grpo/finetune_grpo_referring.slurm \
#       --skip_merge --skip_prepare
#
# 可通过环境变量覆盖总训练轮数：
#   NUM_EPOCHS=2 sbatch finetune_framework/VRSbench/grpo/finetune_grpo_referring.slurm

REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/finetune_framework/VRSbench/grpo/finetune_grpo_referring.sh" ]; then
    echo "❌ 请从 RS-MLLM 仓库根目录使用 sbatch 提交该脚本"
    echo "   当前 SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<未设置>}"
    exit 1
fi

SCRIPT_DIR="$REPO_ROOT/finetune_framework/VRSbench/grpo"
TRAIN_SCRIPT="$SCRIPT_DIR/finetune_grpo_referring.sh"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

source /opt/app/anaconda3/etc/profile.d/conda.sh
conda activate rs_mllm

if ! command -v deepspeed >/dev/null 2>&1; then
    echo "❌ rs_mllm 环境中未找到 deepspeed"
    exit 1
fi

export NUM_EPOCHS="${NUM_EPOCHS:-1}"

echo "============================================"
echo "  Task:             VRSBench Full-data GRPO"
echo "  SLURM Job ID:     ${SLURM_JOB_ID:-unknown}"
echo "  SLURM Node:       ${SLURM_NODELIST:-unknown}"
echo "  GPUs Allocated:   ${CUDA_VISIBLE_DEVICES:-unknown}"
echo "  Host:             $(hostname)"
echo "  Repository:       $REPO_ROOT"
echo "  Conda env:        ${CONDA_DEFAULT_ENV:-unknown}"
echo "  Python:           $(command -v python)"
echo "  DeepSpeed:        $(command -v deepspeed)"
echo "  PyTorch CUDA:     $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  Total epochs:     $NUM_EPOCHS"
printf "  Training args:   "
printf " %q" "$@"
printf "\n"
echo "============================================"

cd "$REPO_ROOT"
bash "$TRAIN_SCRIPT" "$@"

echo "============================================"
echo "  Job finished: $(date)"
echo "============================================"
