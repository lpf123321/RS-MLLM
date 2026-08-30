#!/usr/bin/env python3
"""Merge the VOPD LoRA adapter into a CVSearch-compatible SAM3 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml


SAM3_LORA_REPO = Path(
    os.environ.get("SAM3_LORA_ROOT", os.environ.get("SAM3_REPO", "sam3_lora"))
).resolve()
if str(SAM3_LORA_REPO) not in sys.path:
    sys.path.insert(0, str(SAM3_LORA_REPO))

from lora_layers import (  # noqa: E402
    LoRAConfig,
    apply_lora_to_model,
    load_lora_weights,
)
from merge_lora_weights import (  # noqa: E402
    merge_lora_linears,
    restore_multihead_attention,
)
from sam3.model_builder import build_sam3_image_model  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path(os.environ.get("SAM_T0", "sam3.pt")),
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path(os.environ.get("SAM_ADAPTER", "best_lora_weights.pt")),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("SAM_TRAIN_CONFIG", "configs/vopd_bbox_lora.yaml")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("SAM_MERGED_OUTPUT", "sam3_lora.pt")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.base_checkpoint, args.adapter, args.config):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    lora = config["lora"]
    bpe_path = SAM3_LORA_REPO / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"

    print("Building SAM3 from the local base checkpoint...", flush=True)
    model = build_sam3_image_model(
        device="cpu",
        eval_mode=True,
        checkpoint_path=str(args.base_checkpoint),
        load_from_HF=False,
        bpe_path=str(bpe_path),
        enable_segmentation=True,
        compile=False,
    )
    original_model_keys = set(model.state_dict())

    lora_config = LoRAConfig(
        rank=int(lora["rank"]),
        alpha=int(lora["alpha"]),
        dropout=0.0,
        target_modules=list(lora["target_modules"]),
        apply_to_vision_encoder=bool(lora["apply_to_vision_encoder"]),
        apply_to_text_encoder=bool(lora["apply_to_text_encoder"]),
        apply_to_geometry_encoder=bool(lora["apply_to_geometry_encoder"]),
        apply_to_detr_encoder=bool(lora["apply_to_detr_encoder"]),
        apply_to_detr_decoder=bool(lora["apply_to_detr_decoder"]),
        apply_to_mask_decoder=bool(lora["apply_to_mask_decoder"]),
    )
    model = apply_lora_to_model(model, lora_config)
    load_lora_weights(model, str(args.adapter))

    merged_linears = merge_lora_linears(model)
    restored_attention = restore_multihead_attention(model)
    merged_state = model.state_dict()
    if set(merged_state) != original_model_keys:
        missing = sorted(original_model_keys - set(merged_state))[:10]
        extra = sorted(set(merged_state) - original_model_keys)[:10]
        raise RuntimeError(f"Merged architecture mismatch: missing={missing}, extra={extra}")

    print("Rebuilding the original detector.* checkpoint layout...", flush=True)
    base_checkpoint = torch.load(
        args.base_checkpoint, map_location="cpu", weights_only=True
    )
    checkpoint_state = base_checkpoint.get("model", base_checkpoint)
    if not isinstance(checkpoint_state, dict):
        raise TypeError("Unsupported base checkpoint format")

    detector_keys = {key for key in checkpoint_state if key.startswith("detector.")}
    expected_detector_keys = {f"detector.{key}" for key in merged_state}
    missing_from_base = expected_detector_keys - detector_keys
    if missing_from_base:
        raise RuntimeError(
            "Merged model contains detector tensors absent from the base checkpoint: "
            f"{sorted(missing_from_base)[:10]}"
        )
    # Some official checkpoints carry image-model tensors unused by the current
    # builder (for example sam2_convs). Preserve those base-only entries verbatim.
    unused_detector_keys = detector_keys - expected_detector_keys

    output_state = dict(checkpoint_state)
    for key, value in merged_state.items():
        output_state[f"detector.{key}"] = value.detach().cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(output_state, temporary)
    os.replace(temporary, args.output)

    manifest = {
        "format": "sam3_full_checkpoint_with_detector_prefix",
        "base_checkpoint": str(args.base_checkpoint),
        "base_sha256": sha256_file(args.base_checkpoint),
        "adapter": str(args.adapter),
        "adapter_sha256": sha256_file(args.adapter),
        "config": str(args.config),
        "rank": int(lora["rank"]),
        "alpha": int(lora["alpha"]),
        "merged_lora_linear_modules": merged_linears,
        "restored_multihead_attention_modules": restored_attention,
        "detector_tensors": len(detector_keys),
        "merged_model_tensors": len(merged_state),
        "unused_detector_tensors_preserved": len(unused_detector_keys),
        "tracker_tensors_preserved": sum(
            key.startswith("tracker.") for key in output_state
        ),
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256_file(args.output),
    }
    manifest_path = args.output.parent / "merge_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
