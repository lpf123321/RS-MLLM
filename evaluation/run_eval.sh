#!/bin/bash
#
# RS-MLLM Evaluation Launcher
#
# 使用方式:
#   1) 交互式运行 (需已在 conda 环境内):
#        bash run_eval.sh
#
#   2) 通过 SLURM 提交 (自动激活 conda 环境):
#        sbatch run_eval.slurm
#        sbatch run_eval.slurm --datasets vrsbench levircc --max_samples 500
#
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="$REPO_ROOT/models/Qwen3-VL-2B-Instruct"
DATASETS="all"
MAX_SAMPLES=100
DEVICE="cuda"
OUTPUT="$REPO_ROOT/evaluation/results.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)   MODEL_PATH="$2";    shift 2 ;;
        --datasets)     DATASETS="$2";      shift 2 ;;
        --max_samples)  MAX_SAMPLES="$2";   shift 2 ;;
        --device)       DEVICE="$2";        shift 2 ;;
        --output)       OUTPUT="$2";        shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  RS-MLLM Evaluation"
echo "============================================"
echo "  Model:      $MODEL_PATH"
echo "  Datasets:   $DATASETS"
echo "  Max samples: $MAX_SAMPLES"
echo "  Device:     $DEVICE"
echo "  Output:     $OUTPUT"
echo "============================================"

python "$REPO_ROOT/evaluation/main.py" \
    --model_path "$MODEL_PATH" \
    --datasets $DATASETS \
    --max_samples $MAX_SAMPLES \
    --device "$DEVICE" \
    --output "$OUTPUT"
