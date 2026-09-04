#!/usr/bin/env python3
"""Calibrate and export deployable compressed-tensors Qwen3.5 VLM checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

MIN_PIXELS = 200_704
MAX_PIXELS = 2_097_152
MAX_SEQUENCE_LENGTH = 4096
EXPECTED_CALIBRATION_SAMPLES = 512
IGNORE_MODULES = [
    "re:visual.*",
    "re:model.visual.*",
    "re:.*lm_head$",
    "re:.*embed_tokens$",
    "re:.*linear_attn\\.in_proj_a$",
    "re:.*linear_attn\\.in_proj_b$",
]


class CalibrationMessages(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if not row.get("messages"):
                        raise ValueError("calibration row has no messages")
                    self.rows.append(row)
        if len(self.rows) != EXPECTED_CALIBRATION_SAMPLES:
            raise ValueError(
                f"expected {EXPECTED_CALIBRATION_SAMPLES} calibration rows, "
                f"found {len(self.rows)}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def recipe_for(method: str) -> list[Any]:
    if method == "w8a8-int8":
        return [
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(
                targets="Linear",
                scheme="W8A8",
                ignore=IGNORE_MODULES,
            ),
        ]
    if method == "w4a16-gptq":
        return [
            GPTQModifier(
                targets="Linear",
                scheme="W4A16",
                ignore=IGNORE_MODULES,
            ),
        ]
    raise ValueError(f"unsupported method: {method}")


def output_inventory(root: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        inventory.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["w8a8-int8", "w4a16-gptq"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for PTQ conversion")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    calibration_meta = json.loads(args.calibration_meta.read_text(encoding="utf-8"))
    if sha256_file(args.calibration) != calibration_meta["manifest_sha256"]:
        raise ValueError("calibration manifest hash does not match metadata")

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    staging = args.output.with_name(f"{args.output.name}.staging-{job_id}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite staging directory: {staging}")
    staging.mkdir(parents=True)

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    print(
        json.dumps(
            {
                "event": "start",
                "method": args.method,
                "model": str(args.model),
                "output": str(args.output),
                "gpu": torch.cuda.get_device_name(0),
                "torch_cuda": torch.version.cuda,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    dataset = CalibrationMessages(args.calibration)

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(batch) != 1:
            raise ValueError(f"PTQ collator requires batch size 1, got {len(batch)}")
        messages = batch[0]["messages"]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return dict(
            processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors="pt",
            )
        )

    loader_options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.workers,
        "collate_fn": collate,
        "pin_memory": True,
    }
    if args.workers:
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
    calibration_loader = DataLoader(**loader_options)
    recipe = recipe_for(args.method)
    model = oneshot(
        model=model,
        processor=processor,
        dataset=calibration_loader,
        recipe=recipe,
        num_calibration_samples=len(dataset),
        max_seq_length=MAX_SEQUENCE_LENGTH,
        sequential_targets=["Qwen3_5DecoderLayer"],
        sequential_offload_device="cpu",
        shuffle_calibration_samples=False,
        pipeline="independent",
    )
    model.save_pretrained(
        staging,
        save_compressed=True,
        safe_serialization=True,
        max_shard_size="2GB",
        save_original_format=False,
    )
    # The calibration collator deliberately enables max_length truncation.
    # Do not persist that backend state into the deployable tokenizer: a
    # 4096x4096 image expands to 16384 visual placeholders at evaluation time.
    backend_tokenizer = getattr(processor.tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is not None:
        backend_tokenizer.no_truncation()
    processor.save_pretrained(staging)

    elapsed = time.monotonic() - started
    manifest = {
        "method": args.method,
        "model_source": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "calibration_manifest": str(args.calibration.resolve()),
        "calibration_manifest_sha256": sha256_file(args.calibration),
        "calibration_source_sha256": calibration_meta["source_sha256"],
        "calibration_samples": len(dataset),
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "ignored_modules": IGNORE_MODULES,
        "recipe": [repr(modifier) for modifier in recipe],
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(0),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "llmcompressor": package_version("llmcompressor"),
            "compressed-tensors": package_version("compressed-tensors"),
            "qwen-vl-utils": package_version("qwen-vl-utils"),
        },
    }
    manifest["files"] = output_inventory(staging)
    (staging / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    staging.rename(args.output)
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.output),
                "elapsed_seconds": elapsed,
                "peak_cuda_bytes": manifest["peak_cuda_bytes"],
                "total_output_bytes": sum(item["bytes"] for item in manifest["files"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
