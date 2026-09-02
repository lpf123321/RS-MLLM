#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:?usage: run_stage35.sh STAGE --model PATH --output PATH [--gpus N] [--max-updates N]}"; shift
MODEL= OUTPUT= GPUS=1 MAX_UPDATES=0 DRY=0 SMOKE=0
while (($#)); do
  case "$1" in
    --model) MODEL="${2:?}"; shift 2;;
    --output) OUTPUT="${2:?}"; shift 2;;
    --gpus) GPUS="${2:?}"; shift 2;;
    --max-updates) MAX_UPDATES="${2:?}"; shift 2;;
    --dry-run) DRY=1; shift;;
    --smoke) SMOKE=1; shift;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
((DRY == 1)) || [[ -f "$MODEL/config.json" ]] || { echo "missing model: $MODEL" >&2; exit 2; }

case "$STAGE" in
  stage1_clean) DATA=manifest_sft_train.json; GLOBAL_BATCH=64; MAX_PIXELS=1048576;;
  ga2_general) DATA=g_a2_mix.json; GLOBAL_BATCH=64; MAX_PIXELS=1048576;;
  a1_grounding) DATA=a1_domainalign.json; GLOBAL_BATCH=16; MAX_PIXELS=4194304;;
  a2b_change) DATA=a2_change_mix.json; GLOBAL_BATCH=64; MAX_PIXELS=1048576;;
  caption) DATA=expert_data_caption.jsonl; GLOBAL_BATCH=64; MAX_PIXELS=1048576;;
  *) echo "unknown 3.5 stage: $STAGE" >&2; exit 2;;
esac
DATA_PATH="$REPO_ROOT/finetune_framework/VRSbench/$DATA"
IMAGE_ROOT="${TRAIN_DATA_ROOT:-$REPO_ROOT/data}"
[[ -f "$DATA_PATH" ]] || { echo "missing data: $DATA_PATH; run scripts/fetch_training_data.sh" >&2; exit 2; }
[[ -d "$IMAGE_ROOT/assets" ]] || { echo "missing image assets: $IMAGE_ROOT/assets" >&2; exit 2; }
((SMOKE == 1)) && GLOBAL_BATCH=$GPUS
[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "--gpus must be a positive count" >&2; exit 2; }
(( GLOBAL_BATCH % GPUS == 0 )) || { echo "$GLOBAL_BATCH effective batch is not divisible by $GPUS GPUs" >&2; exit 2; }
GRAD_ACCUM=$((GLOBAL_BATCH / GPUS))

source "$REPO_ROOT/scripts/training/env.sh"
if ((DRY == 0)); then activate_conda; fi
FINETUNE_HOME="$REPO_ROOT/finetune_framework/Qwen-VL-Series-Finetune"
export PYTHONPATH="$FINETUNE_HOME/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPUS - 1)))"
mkdir -p "$(dirname "$OUTPUT")" "$REPO_ROOT/outputs/logs"

COMMON=(
  "$FINETUNE_HOME/src/train/train_sft.py"
  --use_liger_kernel False --lora_enable True --use_dora False
  --lora_namespan_exclude "['lm_head', 'embed_tokens']"
  --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --num_lora_modules -1
  --model_id "$MODEL" --data_path "$DATA_PATH" --image_folder "$IMAGE_ROOT"
  --enable_reasoning False --remove_unused_columns False
  --freeze_vision_tower True --freeze_llm True --freeze_merger True
  --bf16 True --fp16 False --disable_flash_attn2 True
  --output_dir "$OUTPUT" --num_train_epochs 1.0 --per_device_train_batch_size 1
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --image_min_pixels 262144 --image_max_pixels "$MAX_PIXELS"
  --learning_rate 1e-4 --merger_lr 1e-5 --vision_lr 2e-6
  --weight_decay 0.1 --warmup_ratio 0.03 --lr_scheduler_type cosine
  --logging_steps 1 --tf32 True --gradient_checkpointing True
  --report_to none --lazy_preprocess True --save_strategy no --dataloader_num_workers 0
)
((MAX_UPDATES > 0)) && COMMON+=(--max_steps "$MAX_UPDATES")
if ((GPUS > 1)); then
  COMMAND=(deepspeed --num_gpus "$GPUS" --master_port "$((28000 + $$ % 1000))" "${COMMON[@]}" --deepspeed "$FINETUNE_HOME/scripts/zero2.json")
else
  COMMAND=(python "${COMMON[@]}")
fi
printf '[training35] stage=%s model=%s data=%s output=%s gpus=%s effective_batch=%s grad_accum=%s think=off smoke=%s\n' "$STAGE" "$MODEL" "$DATA_PATH" "$OUTPUT" "$GPUS" "$GLOBAL_BATCH" "$GRAD_ACCUM" "$SMOKE"
printf '[training35] command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
((DRY == 1)) && exit 0
"${COMMAND[@]}" 2>&1 | tee "$REPO_ROOT/outputs/logs/training35_${STAGE}_$(date +%Y%m%d-%H%M%S).log"
[[ -f "$OUTPUT/adapter_config.json" ]] || { echo "stage did not save a LoRA adapter: $OUTPUT" >&2; exit 1; }
