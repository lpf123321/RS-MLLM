#!/usr/bin/env bash
# RS-MLLM 交互式入口: ./rsmllm.sh
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
cd "$(dirname "$0")"

# 解释器选择: 评测 venv(评测/推理) → 根 .venv(训练/量化) → uv run 兜底
if [ -x "evaluation/vllm_eval/.venv/bin/python" ]; then
  PY="evaluation/vllm_eval/.venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  exec uv run --project . python -m rsmllm.console "$@"
fi

PYTHONPATH="${PWD}" "${PY}" -m rsmllm.console "$@"
