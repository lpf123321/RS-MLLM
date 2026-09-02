"""RS-MLLM vLLM Router: 规则路由 + 多专家实例分发.

架构(与报告 §5.3 一致):
  4 个专家 = 4 个 delta-merged 完整模型(base+delta), general/grounding 叠加 LoRA
  → 每个专家起一个 vLLM OpenAI 兼容服务
  → 本层按 prompt 规则(rules.py)把请求路由到对应专家实例

用法:
  python -m rsmllm.router --config router_config.json    # 启动全部实例+路由入口
  python -m rsmllm.router --probe prompt                 # 仅打印路由结果

路由规则(与评测链路 evaluation/router/rules.py 一致):
  [VQA]/[MCQ]/关键词默认 → general
  [REF]/where/locate    → grounding
  [CD]/change/difference → change
  [CAP]/describe        → caption
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.router.rules import route  # 与评测一致的路由规则

DEFAULT_PORTS = {"general": 8001, "grounding": 8002, "change": 8003, "caption": 8004}
EVAL_PY = REPO_ROOT / "evaluation" / "vllm_eval" / ".venv" / "bin" / "python"
DEFAULT_PYTHON = str(EVAL_PY) if EVAL_PY.is_file() else sys.executable

# 默认专家 -> 模型目录(可被 --config 覆盖)
DEFAULT_MODELS = {
    "general": str(REPO_ROOT / "router_models" / "general_exp7_merged"),
    "grounding": str(REPO_ROOT / "router_models" / "ground_expert_update_merged"),
    "change": str(REPO_ROOT / "router_models" / "change_merged"),
    "caption": str(REPO_ROOT / "router_models" / "caption_merged"),
}
DEFAULT_LORAS = {
    "general": str(REPO_ROOT / "lora_test" / "general"),
    "grounding": str(REPO_ROOT / "lora_test" / "grounding"),
}


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _runtime_path(value: str) -> str:
    expanded = os.path.expanduser(value)
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    candidate = REPO_ROOT / path
    if candidate.exists() or expanded.startswith((".", "~")):
        return str(candidate.resolve())
    # Keep identifiers such as ``Org/model`` intact for vLLM/remote loaders.
    return expanded


def _executable_path(value: str) -> Path:
    expanded = os.path.expanduser(value)
    path = Path(expanded)
    if path.is_absolute():
        return path
    if "/" in expanded or "\\" in expanded:
        return (REPO_ROOT / path).resolve()
    found = shutil.which(expanded)
    return Path(found) if found else path


def start_expert(name: str, model_dir: str, port: int, lora: str | None = None,
                 python: str = DEFAULT_PYTHON) -> None:
    """启动一个专家的 vLLM OpenAI 兼容服务."""
    python_path = _executable_path(python)
    model_path = _runtime_path(model_dir)
    lora_path = _runtime_path(lora) if lora else None
    env = dict(os.environ)
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # flashinfer JIT 与 cub 不兼容
    if python_path.parent != Path("."):
        env["PATH"] = str(python_path.parent) + os.pathsep + env.get("PATH", "")  # ninja
    cmd = [
        str(python_path), "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", "default",
        "--port", str(port),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--max-model-len", "16384",
        "--limit-mm-per-prompt", '{"image": 2}',
    ]
    if lora_path:
        cmd += ["--enable-lora", "--max-lora-rank", "32", "--lora-modules", f"{name}={lora_path}"]
    print(f"[router] 启动 {name}: {model_path} (port {port}{' + LoRA' if lora_path else ''})", flush=True)
    subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_ready(port: int, timeout: int = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3):
                return
        except Exception:
            time.sleep(5)
    raise RuntimeError(f"专家实例 {port} 启动超时")


def route_request(prompt: str, ports: dict[str, int]) -> tuple[str, int]:
    """按规则路由, 返回 (expert_name, port)."""
    expert = route(prompt)
    return expert, ports.get(expert, ports["general"])


def chat(expert_name: str, expert_port: int, messages: list[dict]) -> str:
    """调用专家实例 OpenAI 兼容接口."""
    body = json.dumps({"model": expert_name, "messages": messages, "max_tokens": 1024}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{expert_port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="JSON: {expert: {model, port, lora}}")
    ap.add_argument("--probe", help="仅打印该 prompt 的路由结果")
    ap.add_argument("--serve", action="store_true", help="启动全部专家实例")
    ap.add_argument("--chat", action="store_true", help="交互对话(自动路由)")
    args = ap.parse_args()

    config: dict = {}
    if args.config:
        config = json.loads(_repo_path(args.config).read_text())

    if args.probe:
        expert, port = route_request(args.probe, DEFAULT_PORTS)
        print(f"[router] {args.probe[:50]!r} -> {expert} (port {port})")
        return 0

    if args.serve:
        threads = []
        for name, default_model in DEFAULT_MODELS.items():
            cfg = config.get(name, {})
            model = cfg.get("model", default_model)
            port = cfg.get("port", DEFAULT_PORTS[name])
            lora = cfg.get("lora", DEFAULT_LORAS.get(name))
            t = threading.Thread(target=start_expert, args=(name, model, port, lora))
            t.start()
            threads.append((name, t))
        for name, t in threads:
            t.join()
            print(f"[router] {name} 已提交启动")
        for name, port in DEFAULT_PORTS.items():
            print(f"[router] 等待 {name} (port {port}) 就绪...")
            wait_ready(port)
        print("[router] 全部专家就绪: general:8001 grounding:8002 change:8003 caption:8004")
        return 0

    if args.chat:
        while True:
            try:
                prompt = input("router> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nbye")
                return 0
            if not prompt:
                continue
            expert, port = route_request(prompt, DEFAULT_PORTS)
            print(f"  → {expert} (port {port})")
            try:
                print(chat(expert, port, [{"role": "user", "content": prompt}]))
            except Exception as e:
                print(f"  ! 调用失败: {e}")

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
