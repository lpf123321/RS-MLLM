#!/bin/bash
set -e

# ============================================================
# VRSBench LoRA 权重合并脚本
# 基于qwen-vl-series-finetune/scripts/merge_lora.sh 修改
# 将 LoRA adapter 合并回原模型，得到完整的微调后模型。
# 支持合并最终输出或指定 checkpoint。
#
# 用法:
#   bash merge_lora.sh                          # 合并最终模型（需训练完成）
#   bash merge_lora.sh checkpoint-200           # 合并指定 checkpoint
#   bash merge_lora.sh checkpoint-200 /path/to/save  # 指定输出路径
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/finetune_framework/VRSbench"
FINETUNE_HOME="$REPO_ROOT/finetune_framework/Qwen-VL-Series-Finetune"

# --- 路径配置 ---
MODEL_BASE="$REPO_ROOT/models/Qwen3-VL-2B-Instruct"      # 原始基座模型
OUTPUT_DIR="$SCRIPT_DIR/output/finetune_test"             # 训练输出目录

# --- 选择 checkpoint ---
CHECKPOINT="${1:-}"                                       # 可选，如 checkpoint-200
if [ -n "$CHECKPOINT" ]; then
    LORA_PATH="$OUTPUT_DIR/$CHECKPOINT"
else
    LORA_PATH="$OUTPUT_DIR"
fi

SAVE_PATH="${2:-$SCRIPT_DIR/output/merged_model}"         # 合并后模型保存路径

# --- 环境 ---
export PYTHONPATH="$FINETUNE_HOME/src:$PYTHONPATH"

# --- 检查前置条件 ---
if [ ! -d "$MODEL_BASE" ]; then
    echo "❌ 基座模型不存在: $MODEL_BASE"
    exit 1
fi
if [ ! -f "$LORA_PATH/adapter_config.json" ] || [ ! -f "$LORA_PATH/adapter_model.safetensors" ]; then
    echo "❌ LoRA 权重不完整: $LORA_PATH"
    echo "   需要 adapter_config.json + adapter_model.safetensors"
    echo ""
    echo "   可用的 checkpoint:"
    ls -d "$OUTPUT_DIR"/checkpoint-*/ 2>/dev/null || echo "   (无)"
    exit 1
fi

echo "============================================================"
echo "VRSBench LoRA 权重合并"
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
echo ""
echo "合并后的模型可直接用于推理或部署，不再依赖 adapter 文件。"
