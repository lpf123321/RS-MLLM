#!/bin/bash
#
# RS-MLLM Evaluation Launcher
#
# 使用方式:
#   1) 交互式运行 (需已在 conda 环境内):
#        bash run_eval.sh
#
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- 数据根目录: 优先用环境变量，回调到 datasets/ ---
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/datasets}"

MODEL_PATH="$REPO_ROOT/models/Qwen3.5-4B"
DATASETS="all"
MAX_SAMPLES=100
DEVICE="cuda"
OUTPUT="$REPO_ROOT/evaluation/results.json"
EVAL_BATCH_SIZE=4
COMPILE_MODEL="--compile_model"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)       MODEL_PATH="$2";        shift 2 ;;
        --datasets)         DATASETS="$2";          shift 2 ;;
        --max_samples)      MAX_SAMPLES="$2";       shift 2 ;;
        --device)           DEVICE="$2";            shift 2 ;;
        --output)           OUTPUT="$2";            shift 2 ;;
        --eval_batch_size)  EVAL_BATCH_SIZE="$2";   shift 2 ;;
        --no_compile)       COMPILE_MODEL="";       shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  RS-MLLM Evaluation"
echo "============================================"
echo "  Model:          $MODEL_PATH"
echo "  Datasets:       $DATASETS"
echo "  Max samples:    $MAX_SAMPLES"
echo "  Device:         $DEVICE"
echo "  Output:         $OUTPUT"
echo "  Eval batch size: $EVAL_BATCH_SIZE"
echo "  Compile model:  $([ -n "$COMPILE_MODEL" ] && echo yes || echo no)"
echo "============================================"

python "$REPO_ROOT/evaluation/main.py" \
    --model_path "$MODEL_PATH" \
    --datasets $DATASETS \
    --max_samples $MAX_SAMPLES \
    --device "$DEVICE" \
    --output "$OUTPUT" \
    --eval_batch_size $EVAL_BATCH_SIZE \
    $COMPILE_MODEL
