# RS-MLLM 稳定运行环境（CUDA 12.8 + torch 2.8.0 + uv.lock 精确锁定）
# 用法:
#   docker build -t rs-mllm .                             # (Docker Hub 直连)
#   docker build --build-arg BASE_IMAGE=镜像代理/nvidia/cuda:12.8.1-devel-ubuntu22.04 -t rs-mllm .
#   docker run --gpus all -it -v $(pwd):/workspace rs-mllm bash
ARG BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

# Python 3.10 + 编译工具（flash-attn 等需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    libgl1 libglib2.0-0 libxcb1 libxcb-shm0 libxcb-xfixes0 \
    curl git build-essential ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# uv（固定 0.10.9）
RUN curl -LsSf https://astral.sh/uv/0.10.9/install.sh | sh

WORKDIR /app
COPY pyproject.toml uv.lock ./
# 按 uv.lock 精确安装（--locked 拒绝任何 drift）
RUN uv sync --locked --no-dev

# 代码放入镜像；评测数据（datasets_data/*.jsonl 等）通过 -v 挂载
COPY . .

CMD ["bash"]
