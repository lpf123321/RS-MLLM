#!/bin/bash
#SBATCH --job-name=q35_rs_lora
#SBATCH --output=logs/qwen35_rs_lora_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT=${PROJECT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
ENV_DIR=${PROJECT}/environments/qwen35-finetune
MODEL=${PROJECT}/.ms_cache/models/Qwen--Qwen3.5-4B/snapshots/master
DATA_DIR=${DATA_DIR:-${PROJECT}/finetune_data/qwen35_rs_pilot_v2}
MODE=${MODE:-pilot}

case "${MODE}" in
  smoke)
    OUTPUT_DIR=${PROJECT}/results/qwen35_rs_lora_smoke_${SLURM_JOB_ID}
    MAX_STEPS=2
    EVAL_STEPS=2
    SAVE_STEPS=2
    ;;
  pilot)
    OUTPUT_DIR=${PROJECT}/results/qwen35_rs_lora_pilot_${SLURM_JOB_ID}
    MAX_STEPS=-1
    EVAL_STEPS=125
    SAVE_STEPS=125
    ;;
  *)
    echo "MODE must be smoke or pilot, got: ${MODE}" >&2
    exit 2
    ;;
esac

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export IMAGE_MAX_TOKEN_NUM=1024
export CUDA_VISIBLE_DEVICES=0

cd "${PROJECT}"
mkdir -p "${OUTPUT_DIR}"

echo "=== QWEN3.5 RS LORA START $(date -Is) ==="
echo "mode=${MODE} model=${MODEL} data=${DATA_DIR} output=${OUTPUT_DIR}"
nvidia-smi
(cd "${MODEL}" && sha256sum -c "${PROJECT}/model_manifests/qwen35-4b.sha256")

"${ENV_DIR}/.venv/bin/swift" sft \
  --model "${MODEL}" \
  --attn_impl sdpa \
  --tuner_type lora \
  --target_modules q_proj k_proj v_proj o_proj \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --freeze_vit true \
  --freeze_aligner true \
  --dataset "${DATA_DIR}/train.jsonl" \
  --val_dataset "${DATA_DIR}/validation.jsonl" \
  --split_dataset_ratio 0 \
  --load_from_cache_file true \
  --add_non_thinking_prefix true \
  --loss_scale ignore_empty_think \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --max_length 2048 \
  --gradient_checkpointing true \
  --group_by_length true \
  --dataset_num_proc 8 \
  --dataloader_num_workers 4 \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --save_only_model true \
  --logging_steps 1 \
  --seed 20260715 \
  --data_seed 20260715 \
  --report_to none \
  --output_dir "${OUTPUT_DIR}"

echo "=== QWEN3.5 RS LORA END $(date -Is) ==="
printf '%s\n' "${OUTPUT_DIR}" > "${PROJECT}/results/qwen35_rs_lora_latest_${MODE}.txt"
