#!/bin/bash
# ================================================================
# VRSBench 评估脚本
#
# 用法:
#   bash run_vrsbench.sh                         # 使用下方配置直接运行
#   bash run_vrsbench.sh --max_samples 100       # 命令行参数会覆盖脚本内的默认值
#
# 修改方法: 编辑下方 ===== 可配置项 ===== 区域中的变量值即可
# ================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" ## RS-MLLM 仓库根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================================================================
# ===== 可配置项 =====
# ================================================================

# 待测试模型的路径 (支持相对路径和绝对路径)
# 可选模型:
#   - $REPO_ROOT/models/Qwen3-VL-2B-Instruct          (官方预训练模型)
#   - $REPO_ROOT/finetune_framework/VRSbench/output/merged_model   (全量微调合并模型)
#   - $REPO_ROOT/finetune_framework/VRSbench/output/grpo_referring (GRPO 强化学习模型)
#   - $REPO_ROOT/finetune_framework/VRSbench/output/finetune_test  (测试微调模型)
MODEL_PATH="$REPO_ROOT/models/Qwen3-VL-2B-Instruct"

# VRSBench 测试数据集根目录 (需包含 VRSBench_EVAL_*.json 和 images/val/)
TEST_DATA_DIR="/users/u2024311136/shared/shared_datasets/VRSBench"

# 评估任务: all | referring | caption | vqa
TASKS="all"

# 每任务最大样本数 (0 = 全部，调试时建议设为 50-100)
MAX_SAMPLES=100

# 是否使用训练时的 prompt 模板包装问题
#   false → 直接使用数据集原始问题
#   true  → 使用下方 PROMPT_TEMPLATES 中的模板包装 (caption/vqa/referring)
USE_PROMPT_TEMPLATE="true"

# 推理 batch size (多模态模型建议保持 1，因为不同图片尺寸无法 stack)
BATCH_SIZE=1

# 推理设备: cuda | cpu
DEVICE="cuda"

# 输出 JSON 路径 (留空则默认保存到脚本同目录下 vrsbench_results.json)
OUTPUT=""

# ================================================================
# ===== 配置结束，无需修改以下内容 =====
# ================================================================

# --- 命令行参数解析 (会覆盖上方配置) ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)           MODEL_PATH="$2";    shift 2 ;;
        --test_data_dir)        TEST_DATA_DIR="$2"; shift 2 ;;
        --tasks)                TASKS="$2";         shift 2 ;;
        --max_samples)          MAX_SAMPLES="$2";   shift 2 ;;
        --batch_size)           BATCH_SIZE="$2";    shift 2 ;;
        --device)               DEVICE="$2";        shift 2 ;;
        --output)               OUTPUT="$2";        shift 2 ;;
        --use_prompt_template)
            USE_PROMPT_TEMPLATE="true"; shift ;;
        --no_prompt_template)
            USE_PROMPT_TEMPLATE="false"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 构建 --use_prompt_template 参数
if [ "$USE_PROMPT_TEMPLATE" = "true" ]; then
    PROMPT_FLAG="--use_prompt_template"
else
    PROMPT_FLAG=""
fi

echo "============================================"
echo "  VRSBench 评估"
echo "============================================"
echo "  模型:       $MODEL_PATH"
echo "  数据:       $TEST_DATA_DIR"
echo "  任务:       $TASKS"
echo "  最大样本:   $MAX_SAMPLES"
echo "  设备:       $DEVICE"
echo "  Prompt模板: $USE_PROMPT_TEMPLATE"
echo "============================================"

python "$SCRIPT_DIR/test_vrsbench.py" \
    --model_path "$MODEL_PATH" \
    --test_data_dir "$TEST_DATA_DIR" \
    --tasks "$TASKS" \
    --max_samples "$MAX_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    ${OUTPUT:+--output "$OUTPUT"} \
    $PROMPT_FLAG
