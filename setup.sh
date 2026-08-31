#!/bin/bash
set -e

# ============================================================
# RS-MLLM 环境安装（懒人回滚路径，自动识别）
#   source setup.sh  -> 自动安装并激活环境（uv 优先，conda 兜底）
#   bash setup.sh    -> 只安装，打印激活指引
# 推荐直接跑: uv sync --locked && source .venv/bin/activate
# ============================================================

ENV_NAME="rs_mllm"

# 判断是否被 source（懒人模式）
if [ -n "${BASH_SOURCE}" ] && [ "${BASH_SOURCE}" != "${0}" ]; then
    _SOURCED=1
else
    _SOURCED=0
fi

# ---------- 主路径: uv ----------
if command -v uv >/dev/null 2>&1; then
    echo "==> uv detected, syncing locked env..."
    uv sync --locked
    if [ "$_SOURCED" = "1" ]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
        echo "==> Done. Activated: $(python -V), $(python -c 'import torch; print("torch", torch.__version__)')"
    else
        echo "==> Done. Activate with: source .venv/bin/activate"
    fi
    return 0 2>/dev/null || exit 0
fi

# ---------- 兜底路径: conda ----------
echo "[WARN] uv not found, falling back to conda (install uv for locked env: https://docs.astral.sh/uv/)"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -z "$CONDA_BASE" ]; then
    for p in /opt/app/anaconda3 ~/miniconda3 ~/anaconda3; do
        [ -f "$p/etc/profile.d/conda.sh" ] && CONDA_BASE="$p" && break
    done
fi
if [ -z "$CONDA_BASE" ]; then
    echo "Conda not found. Install Miniconda first: https://docs.anaconda.com/miniconda/"
    exit 1
fi

. "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Creating conda environment: $ENV_NAME"
    conda create -y -n $ENV_NAME python=3.10
fi

conda activate $ENV_NAME

echo "Installing GCC 12 via conda-forge (for flash-attn build)..."
conda install -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12 sysroot_linux-64

GCC_BIN="$CONDA_PREFIX/bin"
export CC="$GCC_BIN/x86_64-conda-linux-gnu-gcc"
export CXX="$GCC_BIN/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"

echo "Installing pinned dependencies (requirements.txt)..."
pip install -r requirements.txt

if [ "$_SOURCED" = "1" ]; then
    echo "==> Done. Activated: conda $ENV_NAME"
else
    echo "==> Done. Activate with: conda activate $ENV_NAME"
fi
