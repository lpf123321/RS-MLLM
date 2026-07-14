#!/bin/bash
set -e

# ============================================================
# LEVIR-CC LoRA 权重合并脚本
#
# 用法:
#   1. 修改 BASE_MODEL 和 LORA_PATH 为实际路径
#   2. bash merge_lora.sh
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/finetune_framework/LEVIR-CC"

# --- 配置 ---
BASE_MODEL="$REPO_ROOT/models/Qwen3-VL-2B-Instruct"
LORA_PATH="$SCRIPT_DIR/output/finetune/checkpoint-XXXX"   # 替换为实际 checkpoint
OUTPUT_DIR="$SCRIPT_DIR/output/merged_model"

# --- 框架路径 ---
FINETUNE_HOME="$REPO_ROOT/finetune_framework/Qwen-VL-Series-Finetune"
export PYTHONPATH="$FINETUNE_HOME/src:$PYTHONPATH"

echo "============================================================"
echo "LEVIR-CC LoRA 合并"
echo "============================================================"
echo "  Base Model: $BASE_MODEL"
echo "  LoRA Path:  $LORA_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "============================================================"

python "$FINETUNE_HOME/src/merge_lora_weights.py" \
    --version "$BASE_MODEL" \
    --vision_tower "$BASE_MODEL" \
    --weight "$LORA_PATH" \
    --save_path "$OUTPUT_DIR"
