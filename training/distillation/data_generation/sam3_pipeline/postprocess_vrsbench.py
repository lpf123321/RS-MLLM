#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from .common import DATASET_OBJECT_CLASSES, box_iou, clip_box, object_classes, official_objects, read_rows, valid_box, write_jsonl
except ImportError:  # direct script execution
    from common import DATASET_OBJECT_CLASSES, box_iou, clip_box, object_classes, official_objects, read_rows, valid_box, write_jsonl


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render(image_path: Path, mask_path: Path, box: list[float], output: Path, label: str) -> None:
    with Image.open(image_path).convert("RGB") as image, Image.open(mask_path).convert("L") as mask:
        mask = mask.resize(image.size)
        tint = Image.new("RGB", image.size, (255, 40, 40))
        alpha = mask.point(lambda value: 90 if value else 0)
        image = Image.composite(tint, image, alpha)
        draw = ImageDraw.Draw(image)
        draw.rectangle(tuple(box), outline=(255, 255, 0), width=max(2, image.width // 160))
        draw.text((max(2, box[0]), max(2, box[1] - 12)), label, fill=(255, 255, 0))
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--dataset-profile", choices=sorted(DATASET_OBJECT_CLASSES), default="vrsbench")
    parser.add_argument("--train", required=True)
    parser.add_argument("--annotations-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--official-iou", type=float, default=0.50)
    parser.add_argument("--min-area-ratio", type=float, default=0.0001)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    allowed_classes = object_classes(args.dataset_profile)

    train, _ = read_rows(args.train)
    train_images = {Path(row["image"] if isinstance(row["image"], str) else row["image"][0]).name for row in train}
    raw = read_jsonl(Path(args.detections))
    grouped = defaultdict(list)
    decisions = []
    for det in raw:
        reason = None
        image, cls = str(det.get("image", "")), str(det.get("class", ""))
        size, box = det.get("image_size"), det.get("bbox_xyxy_px")
        if det.get("source") != "sam3": reason = "not_sam3"
        elif image not in train_images: reason = "not_train_image"
        elif cls not in allowed_classes: reason = "class_not_allowed"
        elif not isinstance(size, list) or len(size) != 2: reason = "invalid_image_size"
        else:
            width, height = int(size[0]), int(size[1])
            if not isinstance(box, list) or len(box) != 4: reason = "invalid_box"
            else:
                box = clip_box([float(v) for v in box], width, height)
                det["bbox_xyxy_px"] = box
                area_ratio = (box[2]-box[0]) * (box[3]-box[1]) / max(1, width*height)
                if float(det.get("score", -1)) < args.score_threshold: reason = "low_score"
                elif not valid_box(box, width, height): reason = "invalid_box"
                elif area_ratio < args.min_area_ratio: reason = "too_small"
                elif not Path(str(det.get("mask_path", ""))).is_file(): reason = "missing_mask"
        if reason:
            decisions.append({"detection_id": det.get("detection_id"), "status": "rejected", "reason": reason})
        else:
            grouped[(image, cls)].append(det)

    after_nms = []
    for key, detections in grouped.items():
        kept = []
        for det in sorted(detections, key=lambda item: float(item["score"]), reverse=True):
            if any(box_iou(det["bbox_xyxy_px"], other["bbox_xyxy_px"]) >= args.nms_iou for other in kept):
                decisions.append({"detection_id": det["detection_id"], "status": "rejected", "reason": "same_class_nms"})
            else:
                kept.append(det)
        after_nms.extend(kept)

    accepted = []
    official_cache = {}
    for det in after_nms:
        image, cls = det["image"], det["class"]
        width, height = map(int, det["image_size"])
        if image not in official_cache:
            official_cache[image] = official_objects(args.annotations_root, image, width, height)
        max_iou = max((box_iou(det["bbox_xyxy_px"], obj["bbox_xyxy_px"]) for obj in official_cache[image] if obj["class"] == cls), default=0.0)
        if max_iou >= args.official_iou:
            decisions.append({"detection_id": det["detection_id"], "status": "rejected", "reason": "official_same_class_iou", "max_official_iou": max_iou})
            continue
        candidate = {**det, "candidate_id": det["detection_id"], "max_same_class_official_iou": max_iou}
        accepted.append(candidate)
        decisions.append({"detection_id": det["detection_id"], "status": "candidate", "reason": "passed_deterministic_filters", "max_official_iou": max_iou})

    out = Path(args.output_dir); overlay_root = out / "overlays"
    for candidate in accepted:
        overlay = overlay_root / f"{candidate['candidate_id']}.jpg"
        render(Path(args.image_root) / candidate["image"], Path(candidate["mask_path"]), candidate["bbox_xyxy_px"], overlay, f"{candidate['class']} {candidate['score']:.2f}")
        candidate["overlay_path"] = str(overlay.resolve())
    write_jsonl(out / "candidates.jsonl", accepted)
    write_jsonl(out / "filter_decisions.jsonl", decisions)
    summary = {"dataset_profile": args.dataset_profile, "allowed_classes": list(allowed_classes), "raw": len(raw), "after_filters": len(accepted), "rejected": len(raw)-len(accepted), "thresholds": {"score": args.score_threshold, "nms_iou": args.nms_iou, "official_iou": args.official_iou, "min_area_ratio": args.min_area_ratio}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
