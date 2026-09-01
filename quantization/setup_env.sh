#!/bin/bash
# 量化转换环境(与评测/推理隔离): llmcompressor 0.13 + torch 2.11 cu129
set -euo pipefail
cd "$(dirname "$0")"
uv sync --locked
echo "量化环境就绪: $(pwd)/.venv"
