#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from evaluation.adapters.qwen35vl import Qwen35VLAdapter
from evaluation.evalsets import vrsbench, mme, xlrs, levircc
from evaluation.main import evaluate, print_results, save_results, DATASETS, SYSTEM_PROMPTS


def _ckpt_path(output_path):
    return output_path.replace(".json", ".ckpt.json")


def _load_checkpoint(output_path):
    cp = _ckpt_path(output_path)
    if os.path.exists(cp):
        with open(cp, "r") as f:
            return json.load(f)
    return {}


def _save_checkpoint(output_path, all_results):
    cp = _ckpt_path(output_path)
    os.makedirs(os.path.dirname(cp) or ".", exist_ok=True)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="RS-MLLM Evaluation — Qwen3.5VL")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+",
                        choices=list(DATASETS.keys()) + ["all"], default=["all"])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "evaluation" / "results_qwen35vl.json"),
        help="结果文件(相对路径按仓库根解析)",
    )
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--compile_model", action="store_true", default=False)
    parser.add_argument("--disable_thinking", action="store_true", default=True,
                        help="Disable thinking mode for cleaner output (default: True)")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Enable thinking mode (overrides --disable_thinking)")
    parser.add_argument("--start_offset", type=int, default=0,
                        help="Skip first N samples of each dataset")
    parser.add_argument("--no_resume", action="store_true", default=False,
                        help="Ignore checkpoint and start from scratch")
    parser.add_argument("--prune_method", type=str, default=None,
                        choices=["l2", "k2", "divprune", "scope"],
                        help="Token pruning method (default: no pruning)")
    parser.add_argument("--prune_r", type=float, default=0.5,
                        help="Pruning ratio (0.5 = keep 50%% of image tokens)")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds_names = list(DATASETS.keys()) if "all" in args.datasets else args.datasets

    disable_thinking = args.disable_thinking and not args.enable_thinking

    all_results = {} if args.no_resume else _load_checkpoint(args.output)
    done_ds = set(all_results.keys())

    pending = [d for d in ds_names if d not in done_ds]
    if done_ds:
        print(f"[Resume] Already evaluated: {', '.join(sorted(done_ds))}", flush=True)
    if not pending:
        print("[Resume] All datasets already evaluated. Exit.", flush=True)
        print_results(all_results)
        save_results(all_results, args.output)
        return

    print("Loading Qwen3.5VL adapter ...", flush=True)
    adapter = Qwen35VLAdapter(
        args.model_path, device=args.device,
        compile_model=args.compile_model,
        disable_thinking=disable_thinking,
        prune_method=args.prune_method,
        prune_r=args.prune_r,
    )
    if args.prune_method:
        print(f"  Pruning: {args.prune_method} (R={args.prune_r})", flush=True)
    print(f"  Model loaded on {adapter.device}", flush=True)
    print(f"  Compile model: {args.compile_model}", flush=True)
    print(f"  Eval batch size: {args.eval_batch_size}", flush=True)
    print(f"  Disable thinking: {disable_thinking}", flush=True)
    if disable_thinking and adapter.device == "cuda":
        print(f"  GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB allocated",
              flush=True)

    for ds_name in pending:
        module, data_path = DATASETS[ds_name]
        if not os.path.exists(data_path):
            print(f"\n[WARN] Data not found: {data_path}, skipping {ds_name}")
            continue

        adapter.system_prompt = SYSTEM_PROMPTS.get(ds_name, "")
        print(f"\n{'=' * 60}")
        print(f"  Evaluating {ds_name} ({module.NAME})")
        print(f"  Data: {data_path}")
        print(f"  System: {adapter.system_prompt}")
        print(f"{'=' * 60}")

        all_results[ds_name] = evaluate(adapter, module, data_path,
                                        args.max_samples, args.eval_batch_size,
                                        start_offset=args.start_offset)
        _save_checkpoint(args.output, all_results)
        print(f"[Checkpoint] Saved partial results ({', '.join(all_results.keys())})", flush=True)

    _save_checkpoint(args.output, all_results)
    print_results(all_results)
    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
