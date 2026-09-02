#!/bin/bash
# 量化转换环境(与评测/推理隔离): llmcompressor 0.13 + torch 2.11 cu129
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV="${HOME}/.local/bin/uv"
[ -x "${UV}" ] || UV="$(command -v uv || true)"
[ -x "${UV}" ] || { echo "uv 未安装" >&2; exit 1; }
cd "${SCRIPT_DIR}"
"${UV}" sync --locked
echo "量化环境就绪: $(pwd)/.venv"
