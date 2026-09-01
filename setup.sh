#!/bin/bash
set -e

# ============================================================
# RS-MLLM 环境安装（懒人回滚路径，自动识别）
#   bash setup.sh  -> 安装(训练/推理环境, torch 2.8 cu128)
#   vLLM 评测环境: bash evaluation/vllm_eval/setup_env.sh  (vllm 0.26 cu129)
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
if command -v uv >/dev/null 2>&1 || [ -x "${HOME}/.local/bin/uv" ]; then
    export PATH="${HOME}/.local/bin:${PATH}"
    echo "==> uv detected, syncing locked env..."
    echo "    (首次需下载 ~4GB: torch cu128 + nvidia 库。若卡住无进度,"
    echo "     请先 export HTTPS_PROXY/HTTP_PROXY 指向代理再重跑)"
    uv sync --locked
    if [ "$_SOURCED" = "1" ]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
        echo "==> Done. Activated: $(python -V), $(python -c 'import torch; print("torch", torch.__version__)')"
    else
        echo "==> Done. Activate with: source .venv/bin/activate"
    fi
    # source 场景: return 即可(不关闭交互 shell); bash 直接跑: exit
    if [ "$_SOURCED" = "1" ]; then
        return 0
    else
        exit 0
    fi
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
    echo "📌 注意: 若您是直接 source 本脚本, exit 会关闭当前会话。请改运行:  bash setup.sh"
    # 防止 source 场景 exit 关闭交互 shell, 改为 return
    if [ "$_SOURCED" = "1" ]; then
        return 1
    else
        exit 1
    fi
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
