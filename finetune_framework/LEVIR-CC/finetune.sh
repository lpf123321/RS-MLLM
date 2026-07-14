#!/bin/bash
set -e

# ============================================================
# LEVIR-CC LoRA 微调 —— 双图变化描述任务
#
# 用法:
#   1. 预处理数据:     python preprocess_data.py
#      (快速测试):     python preprocess_data.py --max_samples 500
#   2. 激活环境:       conda activate rs_mllm
#   3. 运行:           bash finetune.sh
# ============================================================

# --- 项目根目录 ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/finetune_framework/LEVIR-CC"

# --- 环境：conda GCC ---
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# --- 模型（使用 Qwen3-VL-2B-Instruct，与推理测试一致）---
MODEL_NAME="$REPO_ROOT/models/Qwen3-VL-2B-Instruct"

# --- 数据 ---
DATA_PATH="$SCRIPT_DIR/LEVIR-CC_train.json"
IMAGE_FOLDER="$REPO_ROOT/datasets/shared_datasets/LEVIR-CC/images"

# --- 分布式参数（双图任务显存更高，减小 batch）---
GLOBAL_BATCH_SIZE=16
BATCH_PER_DEVICE=1
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# --- 框架路径 ---
FINETUNE_HOME="$REPO_ROOT/finetune_framework/Qwen-VL-Series-Finetune"
export PYTHONPATH="$FINETUNE_HOME/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 检查前置条件 ---
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ 训练数据不存在: $DATA_PATH"
    echo "   请先运行: python preprocess_data.py"
    exit 1
fi
if [ ! -d "$MODEL_NAME" ]; then
    echo "❌ 模型不存在: $MODEL_NAME"
    exit 1
fi
if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "❌ 图片目录不存在: $IMAGE_FOLDER"
    exit 1
fi

echo "============================================================"
echo "LEVIR-CC LoRA 微调 —— 双图变化描述任务"
echo "============================================================"
echo "  模型:       $MODEL_NAME"
echo "  数据:       $DATA_PATH"
echo "  图片目录:   $IMAGE_FOLDER"
echo "  Batch/卡:   $BATCH_PER_DEVICE × $GRAD_ACCUM_STEPS acc = $GLOBAL_BATCH_SIZE global"
echo "  模式:       LoRA (rank=32, alpha=64)"
echo "  DeepSpeed:  zero2"
echo "============================================================"

deepspeed "$FINETUNE_HOME/src/train/train_sft.py" \
    --use_liger_kernel False \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed "$FINETUNE_HOME/scripts/zero2.json" \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir "$SCRIPT_DIR/output/finetune" \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((512 * 32 * 32)) \
    --image_max_pixels $((1280 * 32 * 32)) \
    --learning_rate 5e-5 \
    --merger_lr 5e-6 \
    --vision_lr 1e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --dataloader_num_workers 4
