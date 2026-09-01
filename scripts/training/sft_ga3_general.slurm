#!/bin/bash
# ============================================================
# G-A3: general 专家对照实验（控制变量）
#   起点: merged/sft_stage1_clean（统一 SFT 主干，与 G-A2 一致）
#   数据: expert_data_v2/general_understanding.json（与 v2 完全一致，85%VQA/10%MCQ/5%Cap）
#   目的: 归因 G-A2 崩盘（短caption配比 vs 统一SFT起点）
#         - VQA≥0.67 且 XLRS-B caption 不崩 → 不需拆分，G-A2 是配比问题
#         - VQA 达标但 XLRS 崩 → 统一SFT起点压制长文，需拆分
#         - VQA 上不去 → v2 优势来自 from-base，续训路线要放弃
# ============================================================
#SBATCH --job-name=ga3_general
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -e
export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"

source "$REPO_ROOT/scripts/training/env.sh"
activate_conda

FINETUNE_HOME="$REPO_ROOT/finetune_framework/Qwen-VL-Series-Finetune"
export PYTHONPATH="$FINETUNE_HOME/src:$PYTHONPATH"

MODEL_PATH="$REPO_ROOT/outputs/merged/sft_stage1_clean"
DATA_PATH="$REPO_ROOT/finetune_framework/VRSbench/expert_data_v2/general_understanding.json"
SHR_DATA="$TRAIN_DATA_ROOT"
OUTPUT_DIR="$REPO_ROOT/outputs/checkpoints/ga3_general"

DATA_ROOT="$(pick_data_root)"

GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=2
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

NUM_SAMPLES=100919
STEPS_PER_EPOCH=$((NUM_SAMPLES / GLOBAL_BATCH_SIZE))
echo "  Model(起点): $MODEL_PATH"
echo "  Data:        $DATA_PATH ($NUM_SAMPLES samples, 85%VQA/10%MCQ/5%Cap)"
echo "  Image root:  $DATA_ROOT"
echo "  Batch/卡:    $BATCH_PER_DEVICE × $GRAD_ACCUM_STEPS acc = $GLOBAL_BATCH_SIZE global"
echo "  1 epoch:     ~$STEPS_PER_EPOCH steps"
echo "  Save steps:  500"
echo "  LR:          1e-4"
echo "  LoRA:        rank=32, alpha=64, dropout=0.05"
echo "  Running from unified-SFT backbone (same as G-A2)"
echo "============================================================"

MASTER_PORT=$((28500 + SLURM_JOB_ID % 1000))

deepspeed --num_gpus 2 --master_port $MASTER_PORT "$FINETUNE_HOME/src/train/train_sft.py" \
    --use_liger_kernel False \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed "$FINETUNE_HOME/scripts/zero2.json" \
    --model_id "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --image_folder "$DATA_ROOT" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels 262144 \
    --image_max_pixels 1048576 \
    --learning_rate 1e-4 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing False \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    2>&1 | tee "$LOG_DIR/ga3_general_${SLURM_JOB_ID}_train.log"

echo "=== GA3 TRAIN DONE ==="
