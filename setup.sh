#!/bin/bash
set -e

ENV_NAME="rs_mllm"

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

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install transformers==5.12.1 accelerate==1.14.0 qwen-vl-utils==0.0.14 \
    decord av pillow einops huggingface_hub requests

echo "All done! Activate: conda activate $ENV_NAME"
