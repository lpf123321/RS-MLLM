#!/usr/bin/env python3
"""Deterministic filtering and official-box deduplication for XLRS smoke."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from .common import box_iou, clip_box, object_classes, valid_box, write_jsonl
except ImportError:  # direct script execution
    from common import box_iou, clip_box, object_classes, valid_box, write_jsonl


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render(
    image_path: Path, mask_path: Path, box: list[float], output: Path, label: str,
    max_preview_edge: int = 2048,
) -> None:
    with Image.open(image_path) as source, Image.open(mask_path) as mask_source:
        source_width, source_height = source.size
        # JPEG draft decoding avoids materializing a 10000x10000 RGB raster for
        # every candidate overlay.  The training image and frozen SAM3 geometry
        # remain untouched; only the review artifact is downsampled.
        source.draft("RGB", (max_preview_edge, max_preview_edge))
        image = source.convert("RGB")
        image.thumbnail((max_preview_edge, max_preview_edge), Image.Resampling.LANCZOS)
        mask = mask_source.convert("L").resize(image.size, Image.Resampling.NEAREST)
        scale_x = image.width / source_width
        scale_y = image.height / source_height
        preview_box = [
            float(box[0]) * scale_x,
            float(box[1]) * scale_y,
            float(box[2]) * scale_x,
            float(box[3]) * scale_y,
        ]
        tint = Image.new("RGB", image.size, (255, 40, 40))
        alpha = mask.point(lambda value: 90 if value else 0)
        image = Image.composite(tint, image, alpha)
        draw = ImageDraw.Draw(image)
        width = max(3, image.width // 500)
        draw.rectangle(tuple(preview_box), outline=(255, 255, 0), width=width)
        draw.text((max(2, preview_box[0]), max(2, preview_box[1] - 18)), label, fill=(255, 255, 0))
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--official-records", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--official-iou", type=float, default=0.50)
    parser.add_argument("--min-area-ratio", type=float, default=0.0001)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    allowed = object_classes("xlrs")
    official = read_jsonl(Path(args.official_records))
    official_by_image = defaultdict(list)
    for row in official:
        official_by_image[row["image"]].append(row)
    raw = read_jsonl(Path(args.detections))
    grouped, decisions = defaultdict(list), []
    for detection in raw:
        reason = None
        image, cls = str(detection.get("image", "")), str(detection.get("class", ""))
        size, box = detection.get("image_size"), detection.get("bbox_xyxy_px")
        if detection.get("source") != "sam3": reason = "not_sam3"
        elif image not in official_by_image: reason = "not_selected_train_image"
        elif cls not in allowed: reason = "class_not_allowed"
        elif not isinstance(size, list) or len(size) != 2: reason = "invalid_image_size"
        elif not isinstance(box, list) or len(box) != 4: reason = "invalid_box"
        else:
            width, height = map(int, size)
            box = clip_box([float(value) for value in box], width, height)
            detection["bbox_xyxy_px"] = box
            area_ratio = (box[2] - box[0]) * (box[3] - box[1]) / max(1, width * height)
            if float(detection.get("score", -1)) < args.score_threshold: reason = "low_score"
            elif not valid_box(box, width, height): reason = "invalid_box"
            elif area_ratio < args.min_area_ratio: reason = "too_small"
            elif not Path(str(detection.get("mask_path", ""))).is_file(): reason = "missing_mask"
        if reason:
            decisions.append({"detection_id": detection.get("detection_id"), "status": "rejected", "reason": reason})
        else:
            grouped[(image, cls)].append(detection)

    after_nms = []
    for detections in grouped.values():
        kept = []
        for detection in sorted(detections, key=lambda item: float(item["score"]), reverse=True):
            if any(box_iou(detection["bbox_xyxy_px"], other["bbox_xyxy_px"]) >= args.nms_iou for other in kept):
                decisions.append({"detection_id": detection["detection_id"], "status": "rejected", "reason": "same_class_nms"})
            else:
                kept.append(detection)
        after_nms.extend(kept)

    accepted = []
    for detection in after_nms:
        comparable = [row for row in official_by_image[detection["image"]] if row.get("inferred_class") == detection["class"]]
        max_iou = max((box_iou(detection["bbox_xyxy_px"], row["bbox_xyxy_px"]) for row in comparable), default=0.0)
        if max_iou >= args.official_iou:
            decisions.append({"detection_id": detection["detection_id"], "status": "rejected", "reason": "official_same_class_iou", "max_official_iou": max_iou})
            continue
        candidate = {**detection, "candidate_id": detection["detection_id"], "max_same_class_official_iou": max_iou}
        accepted.append(candidate)
        decisions.append({"detection_id": detection["detection_id"], "status": "candidate", "reason": "passed_deterministic_filters", "max_official_iou": max_iou})

    out = Path(args.output_dir)
    for candidate in accepted:
        overlay = out / "overlays" / f"{candidate['candidate_id']}.jpg"
        render(Path(args.image_root) / candidate["image"], Path(candidate["mask_path"]), candidate["bbox_xyxy_px"], overlay, f"{candidate['class']} {candidate['score']:.2f}")
        candidate["overlay_path"] = str(overlay.resolve())
    write_jsonl(out / "candidates.jsonl", accepted)
    write_jsonl(out / "filter_decisions.jsonl", decisions)
    summary = {"dataset_profile": "xlrs", "raw": len(raw), "after_filters": len(accepted), "rejected": len(raw) - len(accepted), "thresholds": {"score": args.score_threshold, "nms_iou": args.nms_iou, "official_iou": args.official_iou, "min_area_ratio": args.min_area_ratio}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
