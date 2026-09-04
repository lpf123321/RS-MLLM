# RS-MLLM 稳定运行环境（CUDA 12.8 + torch 2.8.0 + uv.lock 精确锁定）
# 用法:
#   docker build -t rs-mllm .                             # (Docker Hub 直连)
#   docker build --build-arg BASE_IMAGE=镜像代理/nvidia/cuda:12.8.1-devel-ubuntu22.04 -t rs-mllm .
#   docker run --gpus all -it -p 7860:7860 rs-mllm bash
ARG BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/.venv/bin:/root/.local/bin:$PATH"

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

# 评测环境（vllm 0.26 + torch 2.11 cu129, 独立 uv 项目; uv 自动下载 python 3.11）
COPY evaluation/vllm_eval/pyproject.toml evaluation/vllm_eval/uv.lock evaluation/vllm_eval/
RUN cd evaluation/vllm_eval && uv sync --locked

# 代码和仓库内置清单放入镜像；大体积图片、模型缓存和结果通过 -v 挂载
COPY . .

CMD ["bash"]
