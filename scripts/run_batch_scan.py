#!/usr/bin/env python3
"""批量并发扫描: 报告 §6.3「批处理参数调优」——固定样本, 6 档并发, 测吞吐/时延/显存."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.main import SYSTEM_PROMPTS, evaluate  # noqa: E402
from evaluation.base.adapter import BaseModelAdapter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default="results/batch_manifest_256.jsonl")
    ap.add_argument("--sizes", default="1,4,16,64,128,192")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--out", default="results/batch_scan.json")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    results = {}
    for bs in sizes:
        t0 = time.time()
        try:
            # 复用评测框架 run batch; 只统计效率(不做评分)
            # 需要模型适配器 —— 见 eval 框架, 用其 LLM 适配器按 batch 档跑
            from evaluation.adapters.qwen3vl import Qwen3VLAdapter
            adapter = Qwen3VLAdapter(model_path=args.model, max_model_len=args.max_model_len,
                                     batch_size=bs, max_num_seqs=args.max_num_seqs)
            # 固定样本清单
            lines = open(args.manifest, encoding="utf-8").read().splitlines()[:256]
            start = time.time()
            n = 0
            for _ in adapter.stream(lines):
                n += 1
            elapsed = time.time() - start
            results[bs] = {"samples": n, "total_seconds": round(elapsed, 2)}
            print(f"[batch-scan] B={bs}: {n} samples in {elapsed:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            results[bs] = {"error": str(e)[:120]}
            print(f"[batch-scan] B={bs}: FAIL {e}", file=sys.stderr)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
