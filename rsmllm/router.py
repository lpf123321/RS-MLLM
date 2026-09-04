"""RS-MLLM vLLM Router: 规则路由 + 多专家实例分发.

架构(与报告 §5.3 一致):
  4 个专家 = 当前精度档的 4 个 canonical 完整模型
  → 每个专家起一个 vLLM OpenAI 兼容服务
  → 本层按 prompt 规则(rules.py)把请求路由到对应专家实例

用法:
  python -m rsmllm.router --serve --chat --quant bf16    # 启动全部实例+CLI
  python -m rsmllm.router --serve --webui --quant w8a8   # 启动全部实例+WebUI
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
import math
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
from rsmllm.config import EXPERT_MODEL_ALIASES, LOGS_DIR, REPORT_CONF
from rsmllm.models import get_model

DEFAULT_PORTS = {"general": 8001, "grounding": 8002, "change": 8003, "caption": 8004}
EVAL_PY = REPO_ROOT / "evaluation" / "vllm_eval" / ".venv" / "bin" / "python"
DEFAULT_PYTHON = str(EVAL_PY) if EVAL_PY.is_file() else sys.executable

EXPERT_ORDER = ("general", "grounding", "change", "caption")


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


def start_expert(
    name: str,
    model_dir: str,
    port: int,
    *,
    device: str | None = None,
    gpu_memory_utilization: float = 0.85,
    python: str = DEFAULT_PYTHON,
) -> subprocess.Popen:
    """启动一个专家的 vLLM OpenAI 兼容服务."""
    python_path = _executable_path(python)
    model_path = _runtime_path(model_dir)
    env = dict(os.environ)
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # flashinfer JIT 与 cub 不兼容
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device
    if python_path.parent != Path("."):
        env["PATH"] = str(python_path.parent) + os.pathsep + env.get("PATH", "")  # ninja
    cmd = [
        str(python_path), "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", name,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--max-model-len", "16384",
        "--limit-mm-per-prompt", '{"image": 2}',
        "--allowed-local-media-path", "/tmp",
        "--mm-processor-kwargs",
        json.dumps(
            {
                "min_pixels": REPORT_CONF["min_pixels"],
                "max_pixels": REPORT_CONF["max_pixels"],
            }
        ),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"router_{name}.log"
    log_file = log_path.open("a", encoding="utf-8")
    print(
        f"[router] 启动 {name}: {model_path} "
        f"(port {port}, GPU {device or '默认'}, log {log_path})",
        flush=True,
    )
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    log_file.close()
    return process


def wait_ready(
    port: int, timeout: int = 600, process: subprocess.Popen | None = None
) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"专家实例 {port} 启动失败 (exit={process.returncode})，请查看 logs/router_*.log"
            )
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


def _visible_devices(device_arg: str | None = None) -> list[str]:
    """Return physical GPU identifiers available to expert subprocesses."""
    if device_arg:
        devices = [item.strip() for item in device_arg.split(",") if item.strip()]
        if devices:
            return devices
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if devices:
            return devices
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def expert_device_map(devices: list[str]) -> dict[str, str | None]:
    """Spread four experts across available GPUs, preserving the old 2-GPU layout."""
    if not devices:
        return {expert: None for expert in EXPERT_ORDER}
    if len(devices) == 2:
        assignment = (devices[0], devices[1], devices[1], devices[0])
    else:
        assignment = tuple(devices[i % len(devices)] for i in range(len(EXPERT_ORDER)))
    return dict(zip(EXPERT_ORDER, assignment, strict=True))


def default_gpu_memory_utilization(device_count: int) -> float:
    """Split each GPU's vLLM reservation between experts sharing that GPU."""
    usable_devices = max(1, device_count)
    experts_per_gpu = math.ceil(len(EXPERT_ORDER) / usable_devices)
    return {1: 0.85, 2: 0.42, 3: 0.28, 4: 0.22}[experts_per_gpu]


def start_all_experts(
    quant: str,
    config: dict,
    *,
    devices: list[str],
    gpu_memory_utilization: float | None,
) -> tuple[dict[str, int], list[subprocess.Popen]]:
    """Resolve canonical models and start the four routed vLLM endpoints."""
    ports: dict[str, int] = {}
    processes: list[subprocess.Popen] = []
    device_map = expert_device_map(devices)
    gpu_mem = gpu_memory_utilization or default_gpu_memory_utilization(len(devices))
    try:
        for name in EXPERT_ORDER:
            cfg = config.get(name, {})
            if "model" in cfg:
                model_path = get_model(cfg["model"])
            else:
                # Reuse evaluation's canonical/local-artifact resolution so
                # online and offline Router cannot silently select different weights.
                from rsmllm.router_eval import resolve_model

                model_path = resolve_model(name, quant)
            port = int(cfg.get("port", DEFAULT_PORTS[name]))
            device = str(cfg.get("device", device_map[name])) if cfg.get(
                "device", device_map[name]
            ) is not None else None
            process = start_expert(
                name,
                model_path,
                port,
                device=device,
                gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", gpu_mem)),
            )
            processes.append(process)
            ports[name] = port
            print(f"[router] 等待 {name} (port {port}) 就绪...", flush=True)
            wait_ready(port, process=process)
            print(f"[router] {name} 已就绪", flush=True)
    except Exception:
        stop_experts(processes)
        raise
    return ports, processes


def stop_experts(processes: list[subprocess.Popen]) -> None:
    """Stop only expert processes started by this router invocation."""
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


class LazyExpertPool:
    """Keep one routed expert resident for single-GPU inference."""

    def __init__(
        self,
        quant: str,
        *,
        device: str | None,
        gpu_memory_utilization: float = 0.85,
    ) -> None:
        self.quant = quant
        self.device = device
        self.gpu_memory_utilization = gpu_memory_utilization
        self.current_expert: str | None = None
        self.current_process: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def client_endpoint(self, expert: str) -> tuple[str, int]:
        """Start or reuse the requested expert; caller must hold ``lock``."""
        if (
            expert == self.current_expert
            and self.current_process is not None
            and self.current_process.poll() is None
        ):
            return expert, DEFAULT_PORTS[expert]
        self.close()
        from rsmllm.router_eval import resolve_model

        model_path = resolve_model(expert, self.quant)
        port = DEFAULT_PORTS[expert]
        print(f"[router] prompt 命中 {expert}，加载对应专家...", flush=True)
        process = start_expert(
            expert,
            model_path,
            port,
            device=self.device,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        try:
            wait_ready(port, process=process)
        except Exception:
            stop_experts([process])
            raise
        self.current_expert = expert
        self.current_process = process
        return expert, port

    def close(self) -> None:
        if self.current_process is not None:
            stop_experts([self.current_process])
        self.current_process = None
        self.current_expert = None


def chat(expert_name: str, expert_port: int, messages: list[dict]) -> str:
    """调用专家实例 OpenAI 兼容接口."""
    body = json.dumps(
        {
            "model": expert_name,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{expert_port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", help="JSON: {expert: {model, port, device, gpu_memory_utilization}}"
    )
    ap.add_argument("--probe", help="仅打印该 prompt 的路由结果")
    ap.add_argument("--serve", action="store_true", help="启动全部专家实例")
    ap.add_argument("--chat", action="store_true", help="交互对话(自动路由)")
    ap.add_argument("--webui", action="store_true", help="网页对话(自动路由)")
    ap.add_argument("--web-port", type=int, default=7860, help="WebUI 端口")
    ap.add_argument("--quant", choices=tuple(EXPERT_MODEL_ALIASES), default="bf16")
    ap.add_argument("--devices", help="专家可用 GPU，例如 0,1,2,3；默认自动检测")
    ap.add_argument("--gpu-mem", type=float, help="每个专家的 GPU 显存占比")
    args = ap.parse_args()

    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        local = "127.0.0.1,localhost"
        os.environ[key] = f"{local},{current}" if current else local

    config: dict = {}
    if args.config:
        config = json.loads(_repo_path(args.config).read_text())

    if args.probe:
        expert, port = route_request(args.probe, DEFAULT_PORTS)
        print(f"[router] {args.probe[:50]!r} -> {expert} (port {port})")
        return 0

    ports = {
        name: int(config.get(name, {}).get("port", DEFAULT_PORTS[name]))
        for name in EXPERT_ORDER
    }
    processes: list[subprocess.Popen] = []
    lazy_pool: LazyExpertPool | None = None
    if args.serve:
        devices = _visible_devices(args.devices)
        if not devices:
            print(
                "[router] 警告: 未检测到 GPU 编号，将由 vLLM 使用当前可见设备。",
                file=sys.stderr,
            )
        if len(devices) <= 1 and (args.chat or args.webui):
            lazy_pool = LazyExpertPool(
                args.quant,
                device=devices[0] if devices else None,
                gpu_memory_utilization=args.gpu_mem or 0.85,
            )
            print(
                "[router] 单 GPU 模式: 按 prompt 加载命中专家，切换专家时自动释放旧实例。",
                flush=True,
            )
        else:
            ports, processes = start_all_experts(
                args.quant,
                config,
                devices=devices,
                gpu_memory_utilization=args.gpu_mem,
            )
            detail = " ".join(f"{name}:{ports[name]}" for name in EXPERT_ORDER)
            print(f"[router] 全部专家就绪: {detail}", flush=True)

    if args.webui:
        command = [
            DEFAULT_PYTHON,
            "-m",
            "rsmllm.webui",
            "--router",
            "--ports-json",
            json.dumps(ports),
            "--web-port",
            str(args.web_port),
        ]
        if lazy_pool is not None:
            command += ["--lazy-router", "--quant", args.quant]
            if lazy_pool.device is not None:
                command += ["--device", lazy_pool.device]
            command += ["--gpu-mem", str(lazy_pool.gpu_memory_utilization)]
        try:
            return subprocess.run(command, cwd=str(REPO_ROOT), check=False).returncode
        finally:
            if processes:
                stop_experts(processes)

    if args.chat:
        try:
            while True:
                try:
                    prompt = input("router> ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nbye")
                    return 0
                if not prompt:
                    continue
                expert, port = route_request(prompt, ports)
                print(f"  → {expert} (port {port})")
                try:
                    if lazy_pool is not None:
                        with lazy_pool.lock:
                            expert, port = lazy_pool.client_endpoint(expert)
                            answer = chat(
                                expert, port, [{"role": "user", "content": prompt}]
                            )
                    else:
                        answer = chat(
                            expert, port, [{"role": "user", "content": prompt}]
                        )
                    print(answer)
                except Exception as e:
                    print(f"  ! 调用失败: {e}")
        finally:
            if processes:
                stop_experts(processes)
            if lazy_pool is not None:
                lazy_pool.close()

    if args.serve:
        print("[router] 专家服务保持后台运行；可用 --chat 或 --webui 连接。")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
