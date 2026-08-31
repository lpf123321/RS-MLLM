"""RS-MLLM 统一入口: 报告功能一一对应的命令行入口.

用法示例:
  rsmllm data   preprocess --dataset vrsbench
  rsmllm eval   --model modelscope://HITSZ-JBGS/rs-mllm-qwen35-4b --subsets 950,770
  rsmllm serve  --model mmerestore_bf16 --port 8001
  rsmllm train  --stage stage1_clean --slurm
  rsmllm quantize --method w8a8-int8 --model mmerestore_bf16
  rsmllm prune  --method l2 --ratio 0.5
  rsmllm tol-probe
  rsmllm ttft-probe
  rsmllm batch-scan
  rsmllm download --model HITSZ-JBGS/rs-mllm-qwen35-4b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 报告配置(附录) ---
# 推理/评测统一配置, 与《技术报告》附录"评测校验条件"及"模型与版本"保持一致.
REPORT_CONF = {
    "vllm_version": "0.26.0",
    "dtype": "bfloat16",
    "sampling": "greedy",
    "max_model_len": 16384,
    "batch_size": 64,
    "max_num_seqs": 64,
    "min_pixels": 200704,
    "max_pixels": 2097152,
    "decode_strategy": "temperature=0",
}

# ModelScope 模型注册表: 报告使用的全部模型/量化产物
MODEL_REGISTRY = {
    "base":            "HITSZ-JBGS/rs-mllm-qwen35-4b",
    "mmerestore_bf16": "HITSZ-JBGS/rs-mllm-mmerestore-bf16",
    "w8a8":            "HITSZ-JBGS/rs-mllm-mmerestore-w8a8-int8",
    "gptq":            "HITSZ-JBGS/rs-mllm-mmerestore-w4a16-gptq",
    "expert_general":  "HITSZ-JBGS/rs-mllm-expert-general",
    "expert_ground":   "HITSZ-JBGS/rs-mllm-expert-ground",
    "expert_change":   "HITSZ-JBGS/rs-mllm-expert-change",
    "expert_caption":  "HITSZ-JBGS/rs-mllm-expert-caption",
}


def resolve_model(model: str) -> str:
    """模型标识解析: 本地路径 / 注册表别名 / modelscope:// 引用 -> 本地可达路径."""
    if model.startswith("modelscope://"):
        return download_model(model.removeprefix("modelscope://"))
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]
    p = Path(model).expanduser()
    if p.exists() and p.is_dir():
        return str(p)
    raise ValueError(f"无法识别的模型标识: {model!r} (本地路径/注册表别名/modelscope://id)")


def download_model(model_id: str, cache_dir: str | None = None) -> str:
    """从 ModelScope 按需下载模型(本地缓存命中即不重复下载)."""
    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise RuntimeError("需要 modelscope 包: uv add modelscope") from e
    path = snapshot_download(model_id, cache_dir=cache_dir or os.environ.get("MODEL_CACHE", str(REPO_ROOT / ".models")))
    print(f"[rsmllm] model cached at {path}", file=sys.stderr)
    return path


# ---------------------------------------------------------------- 子命令实现 ---------
def _cmd_data(args: argparse.Namespace) -> int:
    """数据预处理: 四个数据集原始 -> 统一格式清单."""
    scripts = {
        "vrsbench": "scripts/preprocess_vrsbench.py",
        "mme": "scripts/preprocess_mme.py",
        "xlrs": "scripts/preprocess_xlrs.py",
        "levircc": "scripts/preprocess_levircc.py",
    }
    s = scripts.get(args.dataset)
    if not s:
        raise SystemExit(f"未知数据集: {args.dataset} (可选: {', '.join(scripts)})")
    os.execv(sys.executable, [sys.executable, str(REPO_ROOT / s)])


def _cmd_eval(args: argparse.Namespace) -> int:
    """统一评测: 指定模型 + 子集(950/770/590/400/full)."""
    model = resolve_model(args.model)
    from evaluation.main import main as eval_main  # noqa: PLC0415
    sys.argv = ["evaluation/main.py", "--model", model, "--subsets", args.subsets,
                "--batch-size", str(REPORT_CONF["batch_size"]),
                "--max-num-seqs", str(REPORT_CONF["max_num_seqs"]),
                "--max-model-len", str(REPORT_CONF["max_model_len"])]
    if args.output:
        sys.argv += ["--output", args.output]
    return eval_main()


def _cmd_serve(args: argparse.Namespace) -> int:
    """vLLM serve: 与报告一致的推理配置."""
    model = resolve_model(args.model)
    from vllm import LLM  # noqa: PLC0415
    llm = LLM(model=model, dtype=REPORT_CONF["dtype"], max_model_len=args.max_model_len or REPORT_CONF["max_model_len"],
              gpu_memory_utilization=args.gpu_mem, enforce_eager=args.eager)
    print(f"[rsmllm] vLLM serve ready: {model}")
    return 0


def _cmd_quantize(args: argparse.Namespace) -> int:
    """量化转换(W8A8-INT8 / W4A16-GPTQ), 调用 quantization/ 工具链."""
    model = resolve_model(args.model)
    script = REPO_ROOT / "quantization" / "convert_ptq.py"
    os.execv(sys.executable, [sys.executable, str(script), "--method", args.method,
                              "--model", model, "--output", args.output, "--calibration", args.calibration or ""])


def _cmd_download(args: argparse.Namespace) -> int:
    path = download_model(args.id)
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rsmllm", description="RS-MLLM 统一入口(与报告功能一一对应)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="数据集预处理")
    d.add_argument("--dataset", required=True, choices=["vrsbench", "mme", "xlrs", "levircc"])
    d.set_defaults(fn=_cmd_data)

    e = sub.add_parser("eval", help="统一评测(950/770/590/400/full)")
    e.add_argument("--model", required=True)
    e.add_argument("--subsets", default="950", help="950|770|590|400|full(逗号分隔)")
    e.add_argument("--output")
    e.set_defaults(fn=_cmd_eval)

    s = sub.add_parser("serve", help="vLLM 推理服务(报告配置)")
    s.add_argument("--model", required=True)
    s.add_argument("--port", type=int, default=8001)
    s.add_argument("--max-model-len", type=int, default=REPORT_CONF["max_model_len"])
    s.add_argument("--gpu-mem", type=float, default=0.85)
    s.add_argument("--eager", action="store_true")
    s.set_defaults(fn=_cmd_serve)

    q = sub.add_parser("quantize", help="量化转换(W8A8/GPTQ)")
    q.add_argument("--method", required=True, choices=["w8a8-int8", "w4a16-gptq"])
    q.add_argument("--model", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--calibration")
    q.set_defaults(fn=_cmd_quantize)

    dl = sub.add_parser("download", help="ModelScope 按需下载")
    dl.add_argument("--id", required=True)
    dl.set_defaults(fn=_cmd_download)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
