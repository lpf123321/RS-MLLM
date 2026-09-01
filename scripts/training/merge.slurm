#!/bin/bash
# ============================================================
# SLURM 版 LoRA->merged 合并（供 train.sh 流水线依赖链使用）
#
# 用法（由 train.sh 生成参数或手动）:
#   sbatch scripts/training/merge.slurm <lora_checkpoint> <merged_name>
# ============================================================
#SBATCH --job-name=merge
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LORA_CKPT="${1:?merge.slurm <lora_checkpoint> <merged_name>}"
MERGED_NAME="${2:?merge.slurm <lora_checkpoint> <merged_name>}"

bash "$REPO_ROOT/scripts/merge_checkpoint.sh" "$LORA_CKPT" "$MERGED_NAME"
echo "=== MERGE DONE ==="