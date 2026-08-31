#!/usr/bin/env python3
"""RS-MLLM 量化转换入口: 与报告 W8A8/GPTQ 同一条链路(LLM Compressor).

用法: python rsmllm/quantize.py --method w8a8-int8 --model <别名/路径> --output <dir>
"""
from __future__ import annotations

import argparse
import sys

from rsmllm.models import get_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["w8a8-int8", "w4a16-gptq"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default="quantized_models/out")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    model = get_model(args.model)  # 首次自动拉取
    script = "scripts/quantize_qwen35_vlm.py"
    argv = [sys.executable, script, "--method", args.method, "--model", model,
            "--output", args.output]
    if args.calibration:
        argv += ["--calibration", args.calibration]
    argv += ["--workers", str(args.workers)]
    import os
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    raise SystemExit(main())
