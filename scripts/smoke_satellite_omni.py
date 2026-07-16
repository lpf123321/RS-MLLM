#!/usr/bin/env python3
"""Run one real text and segmentation smoke test for Satellite Omni LISA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def load_inference_module(model_dir: Path) -> Any:
    module_path = model_dir / "inference.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location("satellite_omni_inference", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Satellite Omni inference module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    module = load_inference_module(args.model_dir)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    sync_cuda()
    start = time.perf_counter()
    model = module.load_model(model_dir=str(args.model_dir), device=args.device)
    sync_cuda()
    load_s = time.perf_counter() - start

    image = Image.open(args.image).convert("RGB")

    sync_cuda()
    start = time.perf_counter()
    text_response = module.generate_text(
        model,
        image,
        "What color are the large vehicles seen in the image? Answer briefly.",
        max_new_tokens=32,
    )
    sync_cuda()
    text_s = time.perf_counter() - start

    sync_cuda()
    start = time.perf_counter()
    segmentation_text, mask = module.generate_segmentation(
        model,
        image,
        "Segment all vehicles in this satellite image.",
    )
    sync_cuda()
    segmentation_s = time.perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mask_saved = mask is not None
    if mask_saved:
        args.mask_output.parent.mkdir(parents=True, exist_ok=True)
        mask.save(args.mask_output)

    result = {
        "model": "rahuldshetty/satellite-omni-lisa",
        "image": str(args.image),
        "image_size": list(image.size),
        "load_s": round(load_s, 3),
        "text_infer_s": round(text_s, 3),
        "segmentation_infer_s": round(segmentation_s, 3),
        "peak_mem_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1)
        if torch.cuda.is_available()
        else 0.0,
        "text_response": text_response.strip(),
        "segmentation_text": segmentation_text.strip(),
        "mask_saved": mask_saved,
        "mask_output": str(args.mask_output) if mask_saved else None,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
