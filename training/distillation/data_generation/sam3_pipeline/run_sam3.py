#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .common import DATASET_OBJECT_CLASSES, class_prompts, object_classes, safe_id, sha256, write_jsonl
except ImportError:  # direct script execution
    from common import DATASET_OBJECT_CLASSES, class_prompts, object_classes, safe_id, sha256, write_jsonl


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text-prompted SAM3 only for vision-selected classes.")
    parser.add_argument("--class-predictions", required=True)
    parser.add_argument("--selection", required=True, help="selected_images.jsonl from select_images.py")
    parser.add_argument("--dataset-profile", choices=sorted(DATASET_OBJECT_CLASSES), default="vrsbench")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--sam3-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raw-confidence-threshold", type=float, default=0.05)
    parser.add_argument(
        "--inference-max-pixels", type=int, default=0,
        help=("Resize only the SAM3 inference copy when the source image exceeds this "
              "pixel count; map SAM3 boxes back to original pixels. 0 keeps native size."),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    allowed_classes = object_classes(args.dataset_profile)
    prompts_by_class = class_prompts(args.dataset_profile)

    sys.path.insert(0, str(Path(args.sam3_repo).resolve()))
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if not args.device.startswith("cuda"):
        raise SystemExit("this local SAM3 implementation allocates CUDA tensors internally; use --device cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; do not replace a real SAM3 run with fabricated boxes")
    selected = {row["image"] for row in read_jsonl(Path(args.selection))}
    predictions = read_jsonl(Path(args.class_predictions))
    seen = set()
    for row in predictions:
        image = str(row.get("image", ""))
        classes = row.get("classes")
        if image not in selected or image in seen:
            raise SystemExit(f"class prediction image invalid/duplicate: {image}")
        if not isinstance(classes, list) or len(classes) != len(set(classes)) or any(cls not in allowed_classes for cls in classes):
            raise SystemExit(f"invalid fixed-class prediction for {image}: {classes}")
        seen.add(image)
    if seen != selected:
        raise SystemExit(f"class predictions must cover every selected image; missing={sorted(selected-seen)[:20]}")

    checkpoint = Path(args.checkpoint).resolve()
    out = Path(args.output_dir).resolve(); masks_dir = out / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint), load_from_HF=False, device=args.device,
        eval_mode=True, enable_segmentation=True,
    )
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.raw_confidence_threshold)

    detections = []
    query_count = 0
    for selection in predictions:
        image_name = selection["image"]
        image_path = Path(args.image_root) / image_name
        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size
            inference = image
            if args.inference_max_pixels > 0 and width * height > args.inference_max_pixels:
                scale = math.sqrt(args.inference_max_pixels / float(width * height))
                inference = image.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    Image.Resampling.LANCZOS,
                )
            inference_width, inference_height = inference.size
            scale_x = width / inference_width
            scale_y = height / inference_height
            state = processor.set_image(inference)
            for cls in selection["classes"]:
                for prompt_ordinal, prompt in enumerate(prompts_by_class[cls]):
                    query_count += 1
                    state = processor.set_text_prompt(prompt, state)
                    boxes = state["boxes"].detach().float().cpu().tolist()
                    scores = state["scores"].detach().float().cpu().tolist()
                    masks = state["masks"].detach().cpu().numpy()
                    for ordinal, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
                        detection_id = safe_id(f"sam3_{Path(image_name).stem}_{cls}_p{prompt_ordinal:02d}_{ordinal:03d}")
                        original_box = [
                            float(box[0]) * scale_x, float(box[1]) * scale_y,
                            float(box[2]) * scale_x, float(box[3]) * scale_y,
                        ]
                        mask_array = np.asarray(mask).squeeze().astype(np.uint8) * 255
                        mask_path = masks_dir / f"{detection_id}.png"
                        Image.fromarray(mask_array, mode="L").save(mask_path)
                        detections.append({
                            "detection_id": detection_id, "image": image_name, "class": cls,
                            "prompt": prompt, "prompt_ordinal": prompt_ordinal, "score": float(score),
                            "bbox_xyxy_px": original_box, "image_size": [width, height],
                            "inference_image_size": [inference_width, inference_height],
                            "mask_path": str(mask_path), "source": "sam3",
                        })
    write_jsonl(out / "detections.jsonl", detections)
    manifest = {
        "sam3_repo": str(Path(args.sam3_repo).resolve()), "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "device": args.device,
        "dataset_profile": args.dataset_profile, "allowed_classes": list(allowed_classes),
        "class_prompts": {name: list(values) for name, values in prompts_by_class.items()},
        "raw_confidence_threshold": args.raw_confidence_threshold,
        "inference_max_pixels": args.inference_max_pixels,
        "selected_images": len(selected), "sam3_class_queries": query_count,
        "raw_detections": len(detections), "fixture_only": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
