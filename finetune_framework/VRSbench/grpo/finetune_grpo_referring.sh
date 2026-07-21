#!/bin/bash
set -e

# ================================================================
# VRSBench Referring 任务 GRPO 强化学习微调
#
# 流程:
#   1. 合并 SFT LoRA → merged model (如果尚未合并)
#   2. 提取全量 referring-only 数据 (如果尚未提取)
#   3. 在 merged model 基础上用 IoU reward 进行 GRPO 微调
#
# 用法:
#   bash finetune_framework/VRSbench/grpo/finetune_grpo_referring.sh
#   bash finetune_framework/VRSbench/grpo/finetune_grpo_referring.sh --skip_merge
#   bash finetune_framework/VRSbench/grpo/finetune_grpo_referring.sh --skip_prepare
#   bash finetune_framework/VRSbench/grpo/finetune_grpo_referring.sh --lr 1e-5 --num_gen 8
# ================================================================

set -eo pipefail

# --- 项目根目录 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRSBENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$VRSBENCH_ROOT/../.." && pwd)"
FINETUNE_HOME="$REPO_ROOT/finetune_framework/Qwen-VL-Series-Finetune"
OUTPUT_ROOT="$REPO_ROOT/finetune_framework/outputs/vrsbench"

# --- 路径配置 ---
BASE_MODEL="$REPO_ROOT/models/Qwen3-VL-2B-Instruct"
SFT_LORA="$OUTPUT_ROOT/sft"                                    # SFT 第一阶段输出
MERGED_MODEL="$OUTPUT_ROOT/merged_model"                       # 合并后的模型
GRPO_OUTPUT="${GRPO_OUTPUT:-$OUTPUT_ROOT/grpo_referring_full}" # 全量 GRPO 新实验输出

# --- 数据路径 ---
REFERRING_DATA="${REFERRING_DATA:-$VRSBENCH_ROOT/data/VRSBench_referring_grpo_full.json}" # 全量 referring 数据
IMAGE_FOLDER="$REPO_ROOT/datasets/shared_datasets/VRSBench/images/Images_train"

# --- 环境 ---
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export PYTHONPATH="$FINETUNE_HOME/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 可覆盖的默认参数 ---
LR="${LR:-5e-6}"                        # GRPO 学习率 (比 SFT 低)
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"  # 每个 prompt 生成的候选数
BETA="${BETA:-0.04}"                    # KL 惩罚系数
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-768}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"         # 默认使用全部 referring 数据
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LORA_RANK="${LORA_RANK:-16}"            # GRPO 阶段使用较小的 rank
LORA_ALPHA="${LORA_ALPHA:-32}"

# --- 参数解析 ---
SKIP_MERGE=false
SKIP_PREPARE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_merge)   SKIP_MERGE=true;   shift ;;
        --skip_prepare) SKIP_PREPARE=true; shift ;;
        --lr)           LR="$2";           shift 2 ;;
        --num_gen)      NUM_GENERATIONS="$2"; shift 2 ;;
        --beta)         BETA="$2";         shift 2 ;;
        --output)       GRPO_OUTPUT="$2";  shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ================================================================
# Step 1: 合并 SFT LoRA (如果需要)
# ================================================================
if [ "$SKIP_MERGE" = false ]; then
    echo "============================================================"
    echo "Step 1/3: 合并 SFT LoRA 权重"
    echo "============================================================"

    if [ -f "$MERGED_MODEL/config.json" ]; then
        echo "  ✅ 合并模型已存在: $MERGED_MODEL"
        echo "     如需重新合并，请删除该目录后重试"
    else
        if [ ! -f "$SFT_LORA/adapter_config.json" ]; then
            echo "  ❌ SFT LoRA 不存在: $SFT_LORA"
            echo "     请先运行 finetune_test.sh 完成第一阶段 SFT"
            exit 1
        fi

        echo "  基座模型: $BASE_MODEL"
        echo "  LoRA 路径: $SFT_LORA"
        echo "  输出路径: $MERGED_MODEL"

        python "$FINETUNE_HOME/src/merge_lora_weights.py" \
            --model-path "$SFT_LORA" \
            --model-base "$BASE_MODEL" \
            --save-model-path "$MERGED_MODEL" \
            --safe-serialization

        echo "  ✅ 合并完成: $MERGED_MODEL"
    fi
else
    echo "Step 1/3: ⏭ 跳过合并 (--skip_merge)"
    if [ ! -d "$MERGED_MODEL" ]; then
        echo "  ❌ 合并模型不存在: $MERGED_MODEL"
        echo "     请先运行 merge_lora.sh 或去掉 --skip_merge"
        exit 1
    fi
fi

# ================================================================
# Step 2: 准备全量 referring-only 数据 (如果需要)
# ================================================================
if [ "$SKIP_PREPARE" = false ]; then
    echo ""
    echo "============================================================"
    echo "Step 2/3: 准备全量 Referring-only 数据"
    echo "============================================================"

    if [ -f "$REFERRING_DATA" ]; then
        echo "  ✅ 数据已存在: $REFERRING_DATA"
        COUNT=$(python3 -c "import json; print(len(json.load(open('$REFERRING_DATA'))))")
        echo "     共 $COUNT 条全量 referring 数据"
    else
        python3 "$SCRIPT_DIR/prepare_referring_grpo_data.py" \
            --input "$VRSBENCH_ROOT/data/VRSBench_train.json" \
            --output "$REFERRING_DATA" \
            --max_samples "$MAX_SAMPLES"
    fi
else
    echo "Step 2/3: ⏭ 跳过数据准备 (--skip_prepare)"
    if [ ! -f "$REFERRING_DATA" ]; then
        echo "  ❌ 数据不存在: $REFERRING_DATA"
        exit 1
    fi
fi

# ================================================================
# Step 3: GRPO 训练
# ================================================================
echo ""
echo "============================================================"
echo "Step 3/3: GRPO 微调 (IoU Reward)"
echo "============================================================"
echo "  模型:         $MERGED_MODEL"
echo "  数据:         $REFERRING_DATA"
echo "  图片目录:     $IMAGE_FOLDER"
echo "  LR:           $LR"
echo "  KL beta:      $BETA"
echo "  Generations:  $NUM_GENERATIONS"
echo "  Max comp len: $MAX_COMPLETION_LENGTH"
echo "  LoRA rank:    $LORA_RANK"
echo "  Batch/卡:     $BATCH_PER_DEVICE × $GRAD_ACCUM acc"
echo "  Epochs:       $NUM_EPOCHS"
echo "  输出:         $GRPO_OUTPUT"
echo "============================================================"

LATEST_CHECKPOINT=""
if [ -d "$GRPO_OUTPUT" ]; then
    LATEST_CHECKPOINT="$(find "$GRPO_OUTPUT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
fi
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "  ↻ 检测到 checkpoint，将自动续训: $LATEST_CHECKPOINT"
fi

# 检查数据
if [ ! -f "$REFERRING_DATA" ]; then
    echo "❌ 训练数据不存在: $REFERRING_DATA"
    exit 1
fi
if [ ! -d "$MERGED_MODEL" ]; then
    echo "❌ 合并模型不存在: $MERGED_MODEL"
    exit 1
fi
if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "❌ 图片目录不存在: $IMAGE_FOLDER"
    exit 1
fi
# 检查 reward 函数是否存在 (在 Python 中验证)
python3 -c "
from train.reward_funcs import iou_reward, iou_format_reward
print('  ✅ reward 函数加载成功: iou_reward, iou_format_reward')
"

deepspeed "$FINETUNE_HOME/src/train/train_grpo.py" \
    --deepspeed "$FINETUNE_HOME/scripts/zero2.json" \
    --use_liger_loss False \
    --model_id "$MERGED_MODEL" \
    --data_path "$REFERRING_DATA" \
    --image_folder "$IMAGE_FOLDER" \
    --disable_flash_attn2 True \
    --lora_enable True \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --num_lora_modules -1 \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --vision_lr 1e-6 \
    --merger_lr 2e-6 \
    --bf16 True \
    --fp16 False \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_PER_DEVICE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_generations "$NUM_GENERATIONS" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --beta "$BETA" \
    --temperature 0.9 \
    --learning_rate "$LR" \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --remove_unused_columns False \
    --save_strategy "steps" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 5 \
    --dataloader_num_workers 4 \
    --output_dir "$GRPO_OUTPUT"

echo ""
echo "============================================================"
echo "✅ GRPO 微调完成！"
echo "   输出目录: $GRPO_OUTPUT"
echo ""
echo "评估命令 (对比 GRPO 前后效果):"
echo "   # 评估 GRPO 模型"
echo "   python $VRSBENCH_ROOT/tools/eval_metrics.py --task referring --base_model $MERGED_MODEL --lora_path $GRPO_OUTPUT"
echo ""
echo "   # 评估 SFT 模型 (对比基线)"
echo "   python $VRSBENCH_ROOT/tools/eval_metrics.py --task referring --base_model $BASE_MODEL --lora_path $SFT_LORA"
echo ""
echo "   # 一次性对比"
echo "   python $VRSBENCH_ROOT/tools/eval_metrics.py --task referring --base_model $MERGED_MODEL --lora_path $GRPO_OUTPUT --output $OUTPUT_ROOT/grpo_full_eval.json"
echo "   python $VRSBENCH_ROOT/tools/compare_results.py $OUTPUT_ROOT/grpo_full_eval.json -t"
echo "============================================================"
