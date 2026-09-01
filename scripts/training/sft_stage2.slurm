#!/bin/bash
# ============================================================
# Stage 2: MCQ-Enhanced LoRA SFT
# Resume from stage1 checkpoint-2758, LR=3e-5, 0.5 epoch
# Data: VRSBench + LEVIR-CC + 5000 synthetic MCQ samples
#
# 注意: 依赖 combined_train_stage2.json（本仓库未分发的衍生数据），
#       正式使用前需用 scripts/generate_mcq_data.py 构建后放入
#       finetune_framework/VRSbench/ 再提交。
# ============================================================
#SBATCH --job-name=sft_stage2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00

set -e
export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"

source "$REPO_ROOT/scripts/training/env.sh"
activate_conda

FINETUNE_HOME="$REPO_ROOT/finetune_framework/Qwen-VL-Series-Finetune"
export PYTHONPATH="$FINETUNE_HOME/src:$PYTHONPATH"

echo "============================================================"
echo "Stage 2: MCQ-Enhanced LoRA SFT"
echo "============================================================"
echo "  Job ID:      $SLURM_JOB_ID"
echo "  Node:        $(hostname)"
echo "  GPU:         $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  PyTorch:     $(python -c 'import torch; print(torch.__version__)')"
echo "============================================================"

MODEL_PATH="$(resolve_model base)"
DATA_PATH="$REPO_ROOT/finetune_framework/VRSbench/combined_train_stage2.json"
IMAGE_FOLDER="/"
OUTPUT_DIR="$REPO_ROOT/outputs/checkpoints/sft_stage2"
STAGE1_CKPT="$REPO_ROOT/outputs/checkpoints/sft_stage1/checkpoint-2758"

# Resume from stage 1 best checkpoint
# Copy checkpoint-2758 into the output dir so trainer auto-resumes
mkdir -p "$OUTPUT_DIR"
if [ -d "$OUTPUT_DIR/checkpoint-2758" ]; then
    echo "  ✅ Resume checkpoint already exists: $OUTPUT_DIR/checkpoint-2758"
else
    echo "  Copying $STAGE1_CKPT → $OUTPUT_DIR/checkpoint-2758 ..."
    cp -r "$STAGE1_CKPT" "$OUTPUT_DIR/checkpoint-2758"
fi

GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=2
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

NUM_EPOCHS=0.5
SAVE_STEPS=250

echo "  Model:       $MODEL_PATH"
echo "  Data:        $DATA_PATH (181,458 samples)"
echo "  Image folder: $IMAGE_FOLDER"
echo "  Batch/卡:    $BATCH_PER_DEVICE × $GRAD_ACCUM_STEPS acc = $GLOBAL_BATCH_SIZE global"
echo "  Epochs:      $NUM_EPOCHS (~$((181458 * 50 / 100 / GLOBAL_BATCH_SIZE)) steps)"
echo "  Save steps:  $SAVE_STEPS"
echo "  LR:          3e-5 (was 1e-4)"
echo "  LoRA:        rank=32, alpha=64, dropout=0.05"
echo "  Frozen:      vision_tower=True, merger=True, LLM=LoRA"
echo "  Resume:      $OUTPUT_DIR/checkpoint-2758"
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
    --image_folder "$IMAGE_FOLDER" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels 262144 \
    --image_max_pixels 1048576 \
    --learning_rate 3e-5 \
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
    --save_steps $SAVE_STEPS \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    2>&1 | tee "$REPO_ROOT/outputs/logs/sft_stage2_${SLURM_JOB_ID}_train.log"

echo ""
echo "============================================================"
echo "Stage 2 SFT Complete"
echo "============================================================"
