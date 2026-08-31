"""RS-MLLM 推理服务(报告口径): vLLM 0.26 + bf16/greedy + 像素 200704-2097152 + maxlen 16384.

注意: 当前 vLLM 0.26.0 官方发布版未内置 Qwen3_5ForConditionalGeneration 注册(见 run_smoke);
若加载报 "failed to be implemented", 需依赖评测链路 transformers 路径或补注册。
用法: python -m rsmllm.serve --model w8a8
"""
from __future__ import annotations

import argparse
import os

from rsmllm.config import REPORT_CONF
from rsmllm.models import get_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="别名/ModelScope id/本地路径")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--max-model-len", type=int, default=REPORT_CONF["max_model_len"])
    ap.add_argument("--gpu-mem", type=float, default=REPORT_CONF["gpu_memory_utilization"])
    ap.add_argument("--eager", action="store_true", default=True)
    args = ap.parse_args()

    model = get_model(args.model)  # 首次自动从 ModelScope 拉取
    print(f"[rsmllm] model resolved: {model}")

    try:
        from vllm import LLM  # noqa: PLC0415
        llm_kwargs = dict(
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem,
            enforce_eager=args.eager,
        )
        # 多模态参数(报告口径)
        if "limit_mm_per_prompt" in LLM.__init__.__code__.co_varnames or True:
            llm_kwargs["limit_mm_per_prompt"] = {"image": 2}
            llm_kwargs["mm_processor_kwargs"] = {
                "min_pixels": REPORT_CONF["min_pixels"], "max_pixels": REPORT_CONF["max_pixels"]}
        llm = LLM(model=model, **llm_kwargs)
        print(f"[rsmllm] vLLM loaded: {model}; awaiting requests on port {args.port}")
        # 简单 serve 循环
        try:
            import json
            import urllib.request
            from vllm import SamplingParams
            sp = SamplingParams(temperature=0, max_tokens=16)
            while True:
                req = input("prompt> ").strip()
                if not req:
                    continue
                out = llm.generate([req], sp)
                print(out[0].outputs[0].text)
        except (KeyboardInterrupt, EOFError):
            print("\n[rsmllm] bye")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[rsmllm] vLLM 加载失败({e}); 请走评测链路(transformers 路径)或检查 Qwen3.5 注册", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
