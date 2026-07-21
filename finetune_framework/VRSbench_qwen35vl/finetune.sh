#!/bin/bash
set -e

# ============================================================
# VRSBench LoRA 微调 —— Qwen3.5VL-4B
#
# 用法:
#   1. 激活环境:      conda activate rs_mllm
#   2. 运行:          bash finetune.sh
#
# 数据: VRSBench_train.json (142,383 条，LLaVA 格式)
# 任务: [vqa] 问答 / [caption] 描述 / [refer] 定位
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/finetune_framework/VRSbench_qwen35vl"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

MODEL_NAME="/home/u2024311149/models/Qwen3.5-4B"

DATA_PATH="$REPO_ROOT/finetune_framework/VRSbench/VRSBench_train.json"
IMAGE_FOLDER="$REPO_ROOT/datasets/shared_datasets/VRSBench/images/Images_train"

GLOBAL_BATCH_SIZE=32
BATCH_PER_DEVICE=1
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

FINETUNE_HOME="$REPO_ROOT/finetune_framework/Qwen-VL-Series-Finetune"
export PYTHONPATH="$FINETUNE_HOME/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ 训练数据不存在: $DATA_PATH"
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
echo "VRSBench LoRA 微调 —— Qwen3.5VL-4B"
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
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((512 * 32 * 32)) \
    --image_max_pixels $((1280 * 32 * 32)) \
    --learning_rate 5e-5 \
    --merger_lr 5e-6 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4
