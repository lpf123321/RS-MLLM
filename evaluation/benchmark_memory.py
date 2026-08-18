#!/usr/bin/env python3
"""Measure CUDA peak memory for a visual-token compression method.

Each invocation benchmarks one method in a fresh Python process.  This is
important because PyTorch's CUDA caching allocator otherwise makes peak memory
from a previous method visible to the next one.
"""

import argparse
import json
import os
import random
import subprocess
import threading
import time
from pathlib import Path

import torch

from evaluation.evalsets import mme, vrsbench, xlrs, levircc
from evaluation.main import SYSTEM_PROMPTS


DATASETS = {
    "mme": mme,
    "vrsbench": vrsbench,
    "xlrs": xlrs,
    "levircc": levircc,
}


class NvidiaSmiPeakMonitor:
    """Poll the current process's GPU memory so non-PyTorch allocations count."""

    def __init__(self, interval_seconds=0.1):
        self.interval_seconds = interval_seconds
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval_seconds * 3 + 1)

    def _poll(self):
        pid = str(os.getpid())
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                for line in output.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) == 2 and fields[0] == pid:
                        self.peak_mib = max(self.peak_mib, int(fields[1]))
            except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)


def build_adapter(args):
    """Build the adapter matching the method name used by the benchmark."""
    common = {
        "model_path": args.model_path,
        "device": args.device,
        "compile_model": args.compile_model,
    }
    if args.method == "baseline":
        from evaluation.adapters.qwen3vl import Qwen3VLAdapter

        return Qwen3VLAdapter(**common)
    if args.method == "mmtok":
        from evaluation.adapters.qwen35_mmtok import Qwen35MMTokAdapter

        return Qwen35MMTokAdapter(**common, keep_ratio=args.keep_ratio)
    if args.method == "divprune":
        from evaluation.adapters.qwen35_divprune import Qwen35DivPruneAdapter

        return Qwen35DivPruneAdapter(
            **common, keep_ratio=args.keep_ratio, pruner_seed=args.pruner_seed
        )
    if args.method == "fourier":
        from evaluation.adapters.qwen35_fourier import Qwen35FourierAdapter

        return Qwen35FourierAdapter(
            **common, keep_ratio=args.keep_ratio, pruner_seed=args.pruner_seed
        )

    from evaluation.adapters.qwen35_pruned import Qwen35PrunedAdapter

    pruner = "scope_l2" if args.method == "scope" else args.method

    return Qwen35PrunedAdapter(
        **common,
        keep_ratio=args.keep_ratio,
        pruner=pruner,
        pruner_seed=args.pruner_seed,
    )


def select_samples(module, data_path, sample_count, sample_seed):
    samples = module.load_data(data_path)
    if not samples:
        raise RuntimeError(f"No samples loaded from {data_path}")
    return random.Random(sample_seed).sample(samples, min(sample_count, len(samples)))


def mib(value):
    return value / (1024 ** 2)


def main():
    parser = argparse.ArgumentParser(description="Visual-token compression GPU memory benchmark")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="mme")
    parser.add_argument("--data_path", required=True)
    parser.add_argument(
        "--method",
        required=True,
        choices=["baseline", "uniform", "random", "mmtok", "l2norm", "scope", "divprune", "fourier"],
    )
    # Kept for compatibility with evaluation/run_memory_smoke.slurm. The
    # method name is now the single source of truth for selecting an adapter.
    parser.add_argument("--pruner", choices=["uniform", "random", "mmtok", "l2norm", "scope_l2"], default=None)
    parser.add_argument("--keep_ratio", type=float, default=1.0)
    parser.add_argument("--pruner_seed", type=int, default=2026)
    parser.add_argument("--sample_count", type=int, default=20)
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an available CUDA GPU.")
    if args.method == "baseline":
        args.keep_ratio = 1.0
    elif not 0 < args.keep_ratio <= 1:
        raise ValueError("--keep_ratio must be in (0, 1].")

    module = DATASETS[args.dataset]
    samples = select_samples(module, args.data_path, args.sample_count, args.sample_seed)
    batch = [(sample["images"], sample["prompt"]) for sample in samples]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    monitor = NvidiaSmiPeakMonitor()
    monitor.start()
    try:
        # Do not reset peak statistics after loading: the reported value covers
        # both model weights and every allocation made while generating.
        adapter = build_adapter(args)
        torch.cuda.synchronize()
        model_allocated_mib = mib(torch.cuda.memory_allocated())
        model_reserved_mib = mib(torch.cuda.memory_reserved())
        start = time.perf_counter()
        adapter.system_prompt = SYSTEM_PROMPTS[args.dataset]
        predictions = adapter.batch_generate(batch, batch_size=1)
        torch.cuda.synchronize()
    finally:
        monitor.stop()
    elapsed_seconds = time.perf_counter() - start

    result = {
        "method": args.method,
        "dataset": args.dataset,
        "data_path": os.path.abspath(args.data_path),
        "model_path": args.model_path,
        "samples": len(samples),
        "sample_seed": args.sample_seed,
        "pruner": (
            "none" if args.method == "baseline" else
            "divprune_l2" if args.method == "divprune" else
            "fourier" if args.method == "fourier" else
            "scope_l2" if args.method == "scope" else args.method
        ),
        "keep_ratio": args.keep_ratio,
        "pruner_seed": args.pruner_seed,
        "eval_batch_size": 1,
        "inference_seconds": elapsed_seconds,
        "seconds_per_sample": elapsed_seconds / len(samples),
        "model_allocated_mib": model_allocated_mib,
        "model_reserved_mib": model_reserved_mib,
        "torch_peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
        "torch_peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
        "nvidia_smi_process_peak_mib": monitor.peak_mib or None,
        "gpu_name": torch.cuda.get_device_name(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Results saved to {output}", flush=True)


if __name__ == "__main__":
    main()
