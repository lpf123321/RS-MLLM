"""RS-MLLM 交互式控制台: 菜单驱动, 一问一答, 运行时才拉取模型.

启动:  python -m rsmllm.console   (或仓库根 `rsmllm` 脚本)
     rsmllm> 1      # 评测
     ...           # 问答式收集参数 -> 自动解析模型(首次下载) -> 执行
"""
from __future__ import annotations

import subprocess
import sys

from rsmllm.config import REPORT_CONF, SUBSETS, MANIFESTS
from rsmllm.models import get_model, MODEL_REGISTRY

MAIN_MENU = {
    "1": ("评测实验", "按子集(950/770/590/400/full)评估指定模型"),
    "2": ("推理服务", "以报告口径启动 vLLM 推理"),
    "3": ("训练", "多专家 SFT / 蒸馏流程"),
    "4": ("量化", "W8A8 / GPTQ 转换与部署"),
    "5": ("数据预处理", "四个数据集清洗与清单构建"),
    "6": ("Token 剪枝", "L2/Scope/路由剪枝与评测"),
    "7": ("容错探针", "位翻转注入与完整性验证"),
    "8": ("工具", "TTFT 测量 / batch 扫描 / 模型查看"),
    "q": ("退出", ""),
}


def _ask(prompt: str, default: str = "") -> str:
    r = input(f"rsmllm> {prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return r or default


def _ask_model() -> str:
    aliases = ", ".join(f"{k}" for k in MODEL_REGISTRY)
    raw = _ask(f"模型别名({aliases})或 ModelScope id 或本地路径", "mmerestore_bf16")
    print(f"  ↳ 解析模型: {raw}")
    return get_model(raw)  # 首次自动下载(缓存命中则秒回)


def _cmd_eval() -> None:
    model = _ask_model()
    subsets = _ask("评测子集(逗号分隔: " + ",".join(SUBSETS) + ")", "950")
    datasets = _ask("数据集(逗号分隔: vrsbench,mme,xlrs,levircc) 或 all", "all")
    print(f"  → 评测 model={model} subsets={subsets} datasets={datasets}")
    cmd = [sys.executable, "-m", "evaluation.main",
           "--model", model, "--subsets", subsets, "--datasets", datasets,
           "--batch-size", str(REPORT_CONF["batch_size"]),
           "--max-num-seqs", str(REPORT_CONF["max_num_seqs"]),
           "--max-model-len", str(REPORT_CONF["max_model_len"])]
    subprocess.run(cmd, check=False)


def _cmd_serve() -> None:
    model = _ask_model()
    port = _ask("端口", "8001")
    print(f"  → vLLM serve model={model} port={port}")
    cmd = [sys.executable, "-m", "rsmllm.serve",
           "--model", model, "--port", port,
           "--gpu-mem", str(REPORT_CONF["gpu_memory_utilization"])]
    subprocess.run(cmd, check=False)


def _cmd_quantize() -> None:
    method = _ask("方法 (w8a8-int8 / w4a16-gptq)", "w8a8-int8")
    model = _ask_model()
    out = _ask("输出目录", "quantized_models/out")
    cmd = [sys.executable, "rsmllm/quantize.py", "--method", method,
           "--model", model, "--output", out]
    subprocess.run(cmd, check=False)


def _cmd_data() -> None:
    ds = _ask("数据集 (vrsbench/mme/xlrs/levircc)", "vrsbench")
    cmd = [sys.executable, f"scripts/preprocess_{ds}.py"]
    subprocess.run(cmd, check=False)


def _cmd_simple(name: str, script: str) -> None:
    print(f"  → {name}: {script}")
    subprocess.run([sys.executable, script], check=False)


def main() -> int:
    print("=" * 60)
    print("RS-MLLM 交互控制台  (报告功能 ←→ 可执行入口)")
    print("模型首次使用会从 ModelScope 拉取; 所有参数均可直接回车取默认值")
    print("=" * 60)
    while True:
        print()
        for k, (title, desc) in MAIN_MENU.items():
            print(f"  [{k}] {title}  — {desc}" if desc else f"  [{k}] {title}")
        choice = _ask("选择功能").lower()
        if choice in ("q", "quit", "exit"):
            print("bye")
            return 0
        try:
            if choice == "1":
                _cmd_eval()
            elif choice == "2":
                _cmd_serve()
            elif choice == "3":
                _cmd_simple("训练流程", "scripts/train.sh")
            elif choice == "4":
                _cmd_quantize()
            elif choice == "5":
                _cmd_data()
            elif choice == "6":
                _cmd_simple("Token 剪枝", "scripts/run_prune_sweep.py")
            elif choice == "7":
                _cmd_simple("容错探针", "scripts/quant_tol_probe.py")
            elif choice == "8":
                sub = _ask("工具子项: ttft=batch=模型查看").lower()
                if sub == "ttft":
                    _cmd_simple("TTFT 测量", "scripts/ttft_serve_probe.py")
                elif sub == "batch":
                    _cmd_simple("batch 扫描", "scripts/run_batch_scan.py")
                else:
                    print("\n".join(f"  {k} → {v}" for k, v in MODEL_REGISTRY.items()))
            else:
                print("未知功能, 重试")
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
