#!/bin/bash
# ============================================================
# A2b: Change 续训 v2（防遗忘版）
# 起点: merged/sft_stage1_clean（统一 SFT 主干 ckpt-2585 合并）
# 数据: change 33,800 + VRS caption 3,380（change 的 10%，防遗忘）
# ============================================================
#SBATCH --job-name=a2b_change
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --time=18:00:00

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
DATA_PATH="$REPO_ROOT/finetune_framework/VRSbench/a2_change_mix.json"
SHR_DATA="$TRAIN_DATA_ROOT"

OUTPUT_DIR="$REPO_ROOT/outputs/checkpoints/a2b_change"

GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=4
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))  # 8

NUM_SAMPLES=37180
STEPS_PER_EPOCH=$((NUM_SAMPLES / GLOBAL_BATCH_SIZE))
echo "  Model(起点): $MODEL_PATH"
echo "  Data:        $DATA_PATH ($NUM_SAMPLES samples = change 33800 + vrs caption 3380)"
echo "  Image root:  $SHR_DATA"
echo "  Batch/卡:    $BATCH_PER_DEVICE × $GRAD_ACCUM_STEPS acc = $GLOBAL_BATCH_SIZE global"
echo "  1 epoch:     ~$STEPS_PER_EPOCH steps"
echo "  Save steps:  200"
echo "  LR:          1e-4"
echo "  LoRA:        rank=32, alpha=64, dropout=0.05"
echo "  Image px:    262144 min / 1048576 (1024²) max"
echo "  Output:      $OUTPUT_DIR"
echo "============================================================"

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

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
    --image_folder "$SHR_DATA" \
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
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    2>&1 | tee "$REPO_ROOT/outputs/logs/a2b_change_${SLURM_JOB_ID}_train.log"

echo ""
echo "============================================================"
echo "A2b Change 续训 v2 Complete"
echo "============================================================"
