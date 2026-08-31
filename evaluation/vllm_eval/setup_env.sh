#!/bin/bash
# ============================================================
# vLLM 评测环境一键还原 (uv 项目标准流程)
# 用法: bash evaluation/vllm_eval/setup_env.sh [--skip-download]
# 产出: evaluation/vllm_eval/.venv (uv sync --locked 还原)
# 已验证: 干净环境 uv sync 后 Qwen3.5 加载+生成成功(无手工 symlink)
# ============================================================
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV="${HOME}/.local/bin/uv"
[ -x "${UV}" ] || UV="$(command -v uv || true)"
[ -x "${UV}" ] || { echo "uv 未安装 (curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 1; }

cd "${EVAL_DIR}"

WHL="vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"

# 若 wheel 不在本地，从 GitHub release 下载(约 500MB; 慢可用 ghproxy)
if [ ! -f "${WHL}" ] && [ "${1:-}" != "--skip-download" ]; then
    echo "==> 下载 vllm 0.26 wheel (~500MB)..."
    curl -L -o "${WHL}" \
      "https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
fi

# 若本地已有 wheel, 用 path source 本地还原(避免重新下载)
if [ -f "${WHL}" ]; then
    python3 - <<'PYEOF'
import re, pathlib
p = pathlib.Path("pyproject.toml")
t = p.read_text()
t = re.sub(r'vllm = \{ url = "[^"]+" \}', 'vllm = { path = "./vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" }', t)
p.write_text(t)
PYEOF
    echo "==> 使用本地 wheel: ${WHL}"
fi

echo "==> uv sync --locked"
${UV} sync --locked

echo ""
echo "==> 完成。运行评测前:"
echo "    export VLLM_USE_FLASHINFER_SAMPLER=0   # 必需(flashinfer 0.6.14 与 nvcc12.4 不兼容)"
echo "    export VLLM_WORKER_MULTIPROC_METHOD=spawn"
echo "    ${EVAL_DIR}/.venv/bin/python ${EVAL_DIR}/vision_opd_vllm_eval.py --help"
