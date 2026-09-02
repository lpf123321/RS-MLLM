#!/usr/bin/env python3
"""RS-MLLM 量化转换入口: 与报告 W8A8/GPTQ 同一条链路(LLM Compressor).

用法: python rsmllm/quantize.py --method w8a8-int8 --model <别名/路径> --output <dir>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsmllm.models import get_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["w8a8-int8", "w4a16-gptq"])
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--output",
        default=str(REPO_ROOT / "quantized_models" / "out"),
        help="输出目录(相对路径按仓库根解析)",
    )
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    model = get_model(args.model)  # 首次自动拉取
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = REPO_ROOT / output
    calibration = Path(args.calibration).expanduser() if args.calibration else None
    if calibration is not None and not calibration.is_absolute():
        calibration = REPO_ROOT / calibration
    script = REPO_ROOT / "scripts" / "quantize_qwen35_vlm.py"
    argv = [sys.executable, str(script), "--method", args.method, "--model", model,
            "--output", str(output)]
    if calibration is not None:
        argv += ["--calibration", str(calibration)]
    argv += ["--workers", str(args.workers)]
    os.chdir(REPO_ROOT)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    raise SystemExit(main())
