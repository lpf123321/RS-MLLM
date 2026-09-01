#!/bin/bash
# ============================================================
# vLLM 评测环境一键还原 (uv 项目标准流程)
# 用法: bash evaluation/vllm_eval/setup_env.sh
# 产出: evaluation/vllm_eval/.venv  (uv sync --locked 精确还原)
# 已验证: 干净环境 uv sync 后 Qwen3.5 加载+生成成功(无手工 symlink)
# ============================================================
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV="${HOME}/.local/bin/uv"
[ -x "${UV}" ] || UV="$(command -v uv || true)"
[ -x "${UV}" ] || { echo "uv 未安装 (curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 1; }

cd "${EVAL_DIR}"
echo "==> uv sync --locked (依赖锁定 vllm 0.26 url source + torch 2.11 cu129)"
${UV} sync --locked

echo ""
echo "==> 完成。运行评测前:"
echo "    ${EVAL_DIR}/.venv/bin/python ${EVAL_DIR}/vision_opd_vllm_eval.py --help"
echo "    (环境变量由评测器自动设置, 无需手动 export)"
