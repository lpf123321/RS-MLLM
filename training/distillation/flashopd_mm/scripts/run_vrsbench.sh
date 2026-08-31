#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/vrsbench_qwen35_4b.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  -m flashopd.cli --config "$CONFIG"
