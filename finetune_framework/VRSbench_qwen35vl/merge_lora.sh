#!/bin/bash
set -e

# ============================================================
# VRSBench Qwen3.5VL-4B LoRA 权重合并
#
# 用法:
#   bash merge_lora.sh                          # 合并最终模型
#   bash merge_lora.sh checkpoint-200           # 合并指定 checkpoint
#   bash merge_lora.sh checkpoint-200 /path/to/save  # 指定输出路径
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/finetune_framework/VRSbench_qwen35vl"
FINETUNE_HOME="$REPO_ROOT/finetune_framework/Qwen-VL-Series-Finetune"

MODEL_BASE="/home/u2024311149/models/Qwen3.5-4B"
OUTPUT_DIR="$SCRIPT_DIR/output/finetune"

CHECKPOINT="${1:-}"
if [ -n "$CHECKPOINT" ]; then
    LORA_PATH="$OUTPUT_DIR/$CHECKPOINT"
else
    LORA_PATH="$OUTPUT_DIR"
fi

SAVE_PATH="${2:-$SCRIPT_DIR/output/merged_model}"

export PYTHONPATH="$FINETUNE_HOME/src:$PYTHONPATH"

if [ ! -d "$MODEL_BASE" ]; then
    echo "❌ 基座模型不存在: $MODEL_BASE"
    exit 1
fi
if [ ! -f "$LORA_PATH/adapter_config.json" ] || [ ! -f "$LORA_PATH/adapter_model.safetensors" ]; then
    echo "❌ LoRA 权重不完整: $LORA_PATH"
    echo "  检查: adapter_config.json + adapter_model.safetensors"
    exit 1
fi

echo "============================================================"
echo "Qwen3.5VL-4B LoRA 权重合并"
echo "============================================================"
echo "  基座模型:    $MODEL_BASE"
echo "  LoRA 路径:   $LORA_PATH"
echo "  输出路径:    $SAVE_PATH"
echo "============================================================"

python "$FINETUNE_HOME/src/merge_lora_weights.py" \
    --model-path "$LORA_PATH" \
    --model-base "$MODEL_BASE" \
    --save-model-path "$SAVE_PATH" \
    --safe-serialization

echo ""
echo "✅ 合并完成！模型已保存到: $SAVE_PATH"
