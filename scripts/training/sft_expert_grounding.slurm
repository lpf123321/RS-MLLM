#!/bin/bash
# ============================================================
# Expert 2: Grounding LoRA
# Data: VRSBench Referring (25,562 samples)
# Base model: Qwen3.5-4B (from scratch, no resume)
# Output format: [x1, y1, x2, y2] 0-100
# ============================================================
#SBATCH --job-name=expert_grounding
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00

set -e
export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/outputs/logs"
mkdir -p "$LOG_DIR"

source /opt/app/anaconda3/etc/profile.d/conda.sh
conda activate rs_mllm

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FINETUNE_HOME="$REPO_ROOT/finetune_framework/Qwen-VL-Series-Finetune"
export PYTHONPATH="$FINETUNE_HOME/src:$PYTHONPATH"

echo "============================================================"
echo "Expert 2: Grounding LoRA (Referring)"
echo "============================================================"
echo "  Job ID:      $SLURM_JOB_ID"
echo "  Node:        $(hostname)"
echo "  GPU:         $(nvidia-smi -L 2>/dev/null | head -1)"
echo "============================================================"

MODEL_PATH="$REPO_ROOT/models/Qwen3.5-4B"
DATA_PATH="$REPO_ROOT/finetune_framework/VRSbench/expert_data/grounding.json"
SHR_DATA="/users/u2024311009/team_share/rs-mllm-ready-20260725/data"
LOCAL_DATA="$REPO_ROOT/datasets/manifest_data"
OUTPUT_DIR="$REPO_ROOT/outputs/checkpoints/expert_grounding"

if [ -d "$LOCAL_DATA/assets" ] && [ "$(ls -1 $LOCAL_DATA/assets/vrsbench 2>/dev/null | wc -l)" -gt 29000 ]; then
    echo "  Using local assets: $LOCAL_DATA"
    DATA_ROOT="$LOCAL_DATA"
else
    echo "  Local assets incomplete or missing, using NFS: $SHR_DATA"
    DATA_ROOT="$SHR_DATA"
fi

GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=2
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

NUM_SAMPLES=25562
STEPS_PER_EPOCH=$((NUM_SAMPLES / GLOBAL_BATCH_SIZE))
echo "  Model:       $MODEL_PATH"
echo "  Data:        $DATA_PATH ($NUM_SAMPLES samples, VRSBench Referring)"
echo "  Image root:  $DATA_ROOT"
echo "  Batch/卡:    $BATCH_PER_DEVICE × $GRAD_ACCUM_STEPS acc = $GLOBAL_BATCH_SIZE global"
echo "  1 epoch:     ~$STEPS_PER_EPOCH steps"
echo "  Save steps:  200"
echo "  LR:          1e-4"
echo "  LoRA:        rank=32, alpha=64, dropout=0.05"
echo "  Output:      $OUTPUT_DIR"
echo "  Training from scratch (no resume)"
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
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    2>&1 | tee "$REPO_ROOT/outputs/logs/expert_grounding_${SLURM_JOB_ID}_train.log"

echo ""
echo "============================================================"
echo "Expert 2 Grounding SFT Complete"
echo "============================================================"
