#!/bin/bash
set -e

# ============================================================
# RS-MLLM 环境安装
# 主路径:  uv 固定版本（uv.lock，推荐——轻量、可复现）
# 兜底路径: conda + pip 兼容老环境（无 uv 时可用）
# 完全隔离: 用 Docker（见 Dockerfile）
# ============================================================

ENV_NAME="rs_mllm"

# ---------- 主路径: uv ----------
if command -v uv >/dev/null 2>&1; then
    echo "Using uv (locked)..."
    uv sync --locked
    echo ""
    echo "============================================================"
    echo "Done! Activate: source .venv/bin/activate"
    echo "============================================================"
    exit 0
fi

# ---------- 兜底路径: conda ----------
echo "[WARN] uv not found, falling back to conda (install uv for locked env: https://docs.astral.sh/uv/)"

# Auto-detect conda
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

echo "Installing pinned dependencies (from pyproject.toml)..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "All done! Activate: conda activate $ENV_NAME"
echo "============================================================"
