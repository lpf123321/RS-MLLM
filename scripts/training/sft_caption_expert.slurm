#!/bin/bash
# ============================================================
# Caption Expert 训练：GA2 (ckpt-451) 为起点，双域数据续训
#   数据: expert_data_caption.jsonl (VRS短域 + XLRS长域，指令在 user 文本)
#   目标: 提升 XLRS caption 长描述，同时不掉 VRS caption
# ============================================================
#SBATCH --job-name=sft_caption_exp
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

MODEL_PATH="$REPO_ROOT/outputs/merged/ga2_general_checkpoint-451"
DATA_PATH="$REPO_ROOT/finetune_framework/VRSbench/expert_data_caption.jsonl"
SHR_DATA="/users/u2024311009/team_share/rs-mllm-ready-20260725/data"

OUTPUT_DIR="$REPO_ROOT/outputs/checkpoints/caption_expert"

GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=2
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))  # 16

NUM_SAMPLES=10064
STEPS_PER_EPOCH=$((NUM_SAMPLES / GLOBAL_BATCH_SIZE))
echo "  Model(起点): $MODEL_PATH"
echo "  Data:        $DATA_PATH ($NUM_SAMPLES samples = VRS短 5032 + XLRS长 5032)"
echo "  Image root:  $SHR_DATA"
echo "  Batch/卡:    $BATCH_PER_DEVICE × $GRAD_ACCUM_STEPS acc = $GLOBAL_BATCH_SIZE global"
echo "  1 epoch:     ~$STEPS_PER_EPOCH steps"
echo "  Save steps:  100"
echo "  Output:      $OUTPUT_DIR"
echo "============================================================"

MASTER_PORT=$((28700 + SLURM_JOB_ID % 1000))

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
    --save_steps 100 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    2>&1 | tee "$LOG_DIR/sft_caption_expert_${SLURM_JOB_ID}_train.log"

echo ""
echo "============================================================"
echo "Caption Expert 训练 Complete"
echo "============================================================"
