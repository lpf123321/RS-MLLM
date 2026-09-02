#!/bin/bash
# ============================================================
# SLURM 版 LoRA->merged 合并（供 train.sh 流水线依赖链使用）
#
# 用法:
#   sbatch scripts/training/merge.slurm <base_model> <lora_checkpoint> <output_dir>
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
BASE_MODEL="${1:?merge.slurm <base_model> <lora_checkpoint> <output_dir>}"
LORA_CKPT="${2:?merge.slurm <base_model> <lora_checkpoint> <output_dir>}"
OUTPUT_DIR="${3:?merge.slurm <base_model> <lora_checkpoint> <output_dir>}"

bash "$REPO_ROOT/scripts/merge_checkpoint.sh" --base "$BASE_MODEL" --lora "$LORA_CKPT" --output "$OUTPUT_DIR"
echo "=== MERGE DONE ==="
