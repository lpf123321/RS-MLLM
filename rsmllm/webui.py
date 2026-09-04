"""RS-MLLM WebUI 推理服务: vLLM OpenAI 兼容 API + Gradio 聊天界面.

用法(评测环境):
    cd evaluation/vllm_eval
    .venv/bin/python -m rsmllm.webui --model <模型路径/别名> [--port 8001]
    .venv/bin/python -m rsmllm.webui --router  # 连接已启动的四专家服务

原理(复用现有库, 不造轮子):
    - 后端: vLLM 自带 OpenAI 兼容 server (vllm.serve), 提供 /v1/chat/completions
    - 前端: Gradio ChatInterface (现成多模态聊天组件, 支持图片上传)
    - 客户端: openai 库 (评测环境自带)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr
from openai import OpenAI

from evaluation.router.rules import route
from rsmllm.config import vllm_mm_processor_kwargs
from rsmllm.router import DEFAULT_PORTS, LazyExpertPool
from rsmllm.pruning_policy import parse_keep_ratio

REPO_ROOT = Path(__file__).resolve().parent.parent  # rsmllm/ 的父目录 = 仓库根
EVAL_DIR = REPO_ROOT / "evaluation" / "vllm_eval"
# webui 由评测环境 python 运行, vllm serve 也用同一 python
PY = Path(sys.executable)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="单模型模式的模型路径或 ModelScope id")
    ap.add_argument("--router", action="store_true", help="连接四专家 Router")
    ap.add_argument("--lazy-router", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--ports-json", help="Router 专家端口 JSON")
    ap.add_argument("--quant", choices=("bf16", "w8a8", "gptq"), default="bf16")
    ap.add_argument("--device", help=argparse.SUPPRESS)
    ap.add_argument("--gpu-mem", type=float, default=0.85, help=argparse.SUPPRESS)
    ap.add_argument(
        "--prune-keep-ratio",
        type=parse_keep_ratio,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--port", type=int, default=8001, help="vLLM API 端口")
    ap.add_argument("--web-port", type=int, default=7860, help="Gradio WebUI 端口")
    ap.add_argument("--max-pixels", type=int, default=2_097_152)
    ap.add_argument("--min-pixels", type=int, default=200_704)
    return ap.parse_args()


def start_vllm_server(args: argparse.Namespace) -> None:
    """后台启动 vLLM OpenAI 兼容服务(复用 vllm serve, 不造轮子)."""
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # 本地 API 调用不经过代理(避免 socks/代理劫持本地回环)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    cmd = [
        str(PY), "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--max-model-len", "65536",
        "--limit-mm-per-prompt", '{"image": 2}',
        "--allowed-local-media-path", "/tmp",
        "--mm-processor-kwargs",
        json.dumps(vllm_mm_processor_kwargs(args.min_pixels, args.max_pixels)),
    ]
    threading.Thread(target=lambda: subprocess.run(cmd), daemon=True).start()


def wait_server(port: int, timeout: int = 600) -> OpenAI:
    """等待 vLLM API 就绪, 返回 OpenAI 客户端."""
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            client.models.list()
            return client
        except Exception:
            time.sleep(5)
    raise RuntimeError(f"vLLM 服务 {port} 端口启动超时")


def build_ui(
    client: OpenAI | dict[str, OpenAI] | LazyExpertPool, args: argparse.Namespace
) -> gr.ChatInterface:
    """Gradio 多模态聊天界面(图片上传 + 文字对话)."""
    thinking_pattern = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>\s*", re.DOTALL)

    def respond(message, history):
        content = []
        prompt = message.get("text") or "请描述这张图片"
        if message.get("files"):
            for f in message["files"]:
                content.append({"type": "image_url", "image_url": {"url": f"file://{f['path'] if isinstance(f, dict) else f}"}})
        content.append({"type": "text", "text": prompt})
        if args.router:
            expert = route(prompt)
            if isinstance(client, LazyExpertPool):
                with client.lock:
                    model, port = client.client_endpoint(expert)
                    resp = wait_server(port).chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": content}],
                        temperature=0,
                        max_tokens=1024,
                    )
            else:
                resp = client[expert].chat.completions.create(
                    model=expert,
                    messages=[{"role": "user", "content": content}],
                    temperature=0,
                    max_tokens=1024,
                )
        else:
            expert = None
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_tokens=1024,
            )
        # 剥离 thinking 块, 只显示最终正文(与评测器口径一致)
        text = resp.choices[0].message.content or ""
        answer = thinking_pattern.sub("", text).strip() or "(无正文, 仅思考过程)"
        return f"[{expert} expert]\n\n{answer}" if expert else answer

    return gr.ChatInterface(
        fn=respond,
        title="RS-MLLM 遥感多模态推理",
        description=(
            "上传遥感图片并输入任务，Router 会自动选择专家"
            if args.router
            else "上传遥感图片或输入文字，进行问答/描述/定位推理"
        ),
        multimodal=True,
    )


def main() -> int:
    args = parse_args()
    if args.router:
        if args.lazy_router:
            print("[webui] 单 GPU 按需 Router；首次请求或切换专家时需要加载模型。")
            client = LazyExpertPool(
                args.quant,
                device=args.device,
                gpu_memory_utilization=args.gpu_mem,
                prune_keep_ratio=args.prune_keep_ratio,
            )
        else:
            ports = DEFAULT_PORTS
            if args.ports_json:
                ports = {
                    key: int(value) for key, value in json.loads(args.ports_json).items()
                }
            print("[webui] 连接四专家 Router ...")
            client = {expert: wait_server(port) for expert, port in ports.items()}
    else:
        if not args.model:
            raise SystemExit("单模型 WebUI 必须提供 --model")
        print(f"[webui] 启动 vLLM 服务: {args.model} @ :{args.port}")
        start_vllm_server(args)
        print("[webui] 等待 vLLM 就绪(模型加载需 1-2 分钟)...")
        client = wait_server(args.port)
    state = "WebUI 就绪，模型将在首个请求时加载" if args.lazy_router else "vLLM 就绪"
    print(f"[webui] {state}. WebUI: http://127.0.0.1:{args.web_port}")
    ui = build_ui(client, args)
    try:
        ui.launch(server_name="0.0.0.0", server_port=args.web_port)
    finally:
        if isinstance(client, LazyExpertPool):
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
