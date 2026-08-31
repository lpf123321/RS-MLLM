#!/bin/bash
# ============================================================
# A1: Grounding 域对齐续训（双格式）
# 起点: merged/sft_stage1_clean（统一 SFT 主干 ckpt-2585 合并）
# 数据: VRS refer(0-100 int + Locate) + XLRS train(0-1 float + Given)
# 分辨率: 2048（4096 在 40GB A100 单卡 batch1 仍 OOM，降至 2048）
# ============================================================
#SBATCH --job-name=a1_grounding
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

source /opt/app/anaconda3/etc/profile.d/conda.sh
conda activate rs_mllm

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FINETUNE_HOME="$REPO_ROOT/finetune_framework/Qwen-VL-Series-Finetune"
export PYTHONPATH="$FINETUNE_HOME/src:$PYTHONPATH"

# ---- 起点主干（已 merge 的统一 SFT ckpt-2585）----
MODEL_PATH="$REPO_ROOT/outputs/merged/sft_stage1_clean"
# ---- 混合训练数据（待 4096 导出完成后生成）----
DATA_PATH="$REPO_ROOT/finetune_framework/VRSbench/a1_domainalign.json"
# ---- 高分辨率 XLRS train 图目录 ----
XRS_IMG="${RSMLLM_DATA_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/datasets}/XLRS-Bench_visual_grounding_en/images_exported_train_4096"
# ---- VRS refer 图 root（相对路径 assets/vrsbench/... 拼接）----
SHR_DATA="/users/u2024311009/team_share/rs-mllm-ready-20260725/data"

OUTPUT_DIR="$REPO_ROOT/outputs/checkpoints/a1_grounding"

# 4096 显存约束：低 batch 高累积
GLOBAL_BATCH_SIZE=16
BATCH_PER_DEVICE=1
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))  # 8

NUM_SAMPLES=31871   # VRS 25562 + XLRS 6309
STEPS_PER_EPOCH=$((NUM_SAMPLES / GLOBAL_BATCH_SIZE))
echo "  Model(起点): $MODEL_PATH"
echo "  Data:        $DATA_PATH ($NUM_SAMPLES samples)"
echo "  XLRS img:    $XRS_IMG"
echo "  VRS root:    $SHR_DATA"
echo "  Batch/卡:    $BATCH_PER_DEVICE × $GRAD_ACCUM_STEPS acc = $GLOBAL_BATCH_SIZE global"
echo "  1 epoch:     ~$STEPS_PER_EPOCH steps"
echo "  Save steps:  200"
echo "  LR:          1e-4"
echo "  LoRA:        rank=32, alpha=64, dropout=0.05"
echo "  Image px:    262144 min / 4194304 (2048²) max"
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
    --image_max_pixels 4194304 \
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
    2>&1 | tee "$REPO_ROOT/outputs/logs/a1_grounding_${SLURM_JOB_ID}_train.log"

echo ""
echo "============================================================"
echo "A1 Grounding 域对齐续训 Complete"
echo "============================================================"
