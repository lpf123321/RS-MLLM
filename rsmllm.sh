#!/usr/bin/env bash
# RS-MLLM 交互式入口: ./rsmllm.sh
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
cd "$(dirname "$0")"
# 用评测环境(有 modelscope/vllm, 评测与推理都依赖它)
PYTHONPATH="${PWD}" evaluation/vllm_eval/.venv/bin/python -m rsmllm.console "$@"
