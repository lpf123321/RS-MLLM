#!/bin/bash
# GeoEyes Evaluation Launcher
#
# 使用方式:
#   bash evaluation/run_eval_geoeyes.sh
#   bash evaluation/run_eval_geoeyes.sh --datasets xlrs --max_samples 50
#
# 注意: GeoEyes 需要 GPU 运行, 请确保在 GPU 节点上执行
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH="$REPO_ROOT/models/GeoEyes"
DATASETS="all"
MAX_SAMPLES=100
DEVICE="cuda"
OUTPUT="$REPO_ROOT/evaluation/results_geoeyes.json"
EVAL_BATCH_SIZE=1
GPU_MEMORY_UTILIZATION=0.92

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)       MODEL_PATH="$2";        shift 2 ;;
        --datasets)         DATASETS="$2";          shift 2 ;;
        --max_samples)      MAX_SAMPLES="$2";       shift 2 ;;
        --device)           DEVICE="$2";            shift 2 ;;
        --output)           OUTPUT="$2";            shift 2 ;;
        --gpu_memory)       GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  GeoEyes Evaluation"
echo "============================================"
echo "  Model:          $MODEL_PATH"
echo "  Datasets:       $DATASETS"
echo "  Max samples:    $MAX_SAMPLES"
echo "  Device:         $DEVICE"
echo "  Output:         $OUTPUT"
echo "  GPU memory:     $GPU_MEMORY_UTILIZATION"
echo "============================================"

python -m evaluation.main \
    --adapter geoeyes \
    --model_path "$MODEL_PATH" \
    --datasets $DATASETS \
    --max_samples $MAX_SAMPLES \
    --device "$DEVICE" \
    --output "$OUTPUT" \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION
