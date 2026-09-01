#!/bin/bash
# ============================================================
# 合并 LoRA checkpoint -> 完整模型（产出 outputs/merged/<name>）
#
# 训练产出的是 LoRA checkpoint（含 adapter_model.safetensors）；
# 后续续训/caption/量化需要完整模型。本脚本把它们合并成
# outputs/merged/<merged_name> 的完整模型目录。
#
# 用法:
#   bash scripts/merge_checkpoint.sh <lora_checkpoint> <merged_name> [base_model]
#   例:
#     bash scripts/merge_checkpoint.sh \
#         outputs/checkpoints/sft_stage1_clean sft_stage1_clean
#     # base 默认 $REPO_ROOT/models/Qwen3.5-4B
#     bash scripts/merge_checkpoint.sh \
#         outputs/checkpoints/ga2_general/checkpoint-451 ga2_general_checkpoint-451
#
#   SLURM 环境也适用（本脚本可直接 sbatch，见 scripts/training/merge.slurm）
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LORA_CKPT="${1:?用法: merge_checkpoint.sh <lora_checkpoint> <merged_name> [base]}"
MERGED_NAME="${2:?用法: merge_checkpoint.sh <lora_checkpoint> <merged_name> [base]}"
BASE_MODEL="${3:-$REPO_ROOT/models/Qwen3.5-4B}"

source "$REPO_ROOT/scripts/training/env.sh"
activate_conda

MERGED_DIR="$REPO_ROOT/outputs/merged/$MERGED_NAME"
mkdir -p "$(dirname "$MERGED_DIR")"

echo "== 合并 LoRA -> 完整模型 =="
echo "  base:   $BASE_MODEL"
echo "  lora:   $LORA_CKPT"
echo "  output: $MERGED_DIR"

# 定位 LoRA checkpoint：允许传入 checkpoint 目录或含 checkpoint-N 的父目录
lora_src="$LORA_CKPT"
if [ ! -f "$lora_src/adapter_model.safetensors" ]; then
  if [ -f "$lora_src/adapter_config.json" ]; then
    lora_src="$LORA_CKPT"
  elif [ -d "$lora_src" ] && compgen -G "$lora_src/checkpoint-*/adapter_model.safetensors" >/dev/null; then
    latest=$(ls -d "$lora_src"/checkpoint-*/ 2>/dev/null | sort -V | tail -1)
    echo "  未指定 sub-checkpoint，使用最新: $latest"
    lora_src="${latest%/}"
  else
    echo "[merge] 无法定位 LoRA checkpoint（缺少 adapter_model.safetensors）: $LORA_CKPT" >&2
    exit 1
  fi
fi

python "$REPO_ROOT/scripts/merge_lora_for_eval.py" \
  --model-base "$BASE_MODEL" \
  --lora-path "$lora_src" \
  --save-path "$MERGED_DIR"

echo "== 合并完成: $MERGED_DIR =="