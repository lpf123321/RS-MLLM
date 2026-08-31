#!/usr/bin/env python3
"""Shared utilities for the isolated VOPD -> SAM3 bbox experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision.transforms import v2


SAM3_REPO = Path(os.environ.get("SAM3_REPO", "sam3_lora")).resolve()
SAM3_CHECKPOINT = Path(
    os.environ.get("SAM3_CHECKPOINT", "sam3.pt")
).resolve()
SAM3_BPE = SAM3_REPO / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"

if str(SAM3_REPO) not in sys.path:
    sys.path.insert(0, str(SAM3_REPO))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def clean_question(raw_question: str) -> str:
    """Remove answer choices and VQA-only instructions while retaining the question."""
    text = raw_question.replace("<image>", " ").strip()
    text = re.split(r"\n\s*\n", text, maxsplit=1)[0].strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_prompt(text: str) -> str:
    text = text.strip().strip('`"\'')
    text = re.sub(r"^(prompt|target|phrase)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.rstrip(".?! ")


def prompt_is_valid(prompt: str) -> bool:
    words = re.findall(r"[a-z0-9][a-z0-9'-]*", prompt.lower())
    if not 1 <= len(words) <= 24:
        return False
    bad = (
        "answer with",
        "only focus",
        "option's letter",
        "cannot determine",
        "i cannot",
    )
    return not any(item in prompt.lower() for item in bad)


def question_family(question: str) -> str:
    words = re.findall(r"[a-z0-9]+", question.lower())
    return " ".join(words[:3]) if words else "unknown"


def bbox_xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, x2 - x1, y2 - y1]


def bbox_xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def box_iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = bbox_xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = bbox_xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(
        0.0, (bx2 - bx1) * (by2 - by1)
    ) - inter
    return inter / union if union > 0 else 0.0


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def lexical_tokens(text: str) -> set[str]:
    stop = {
        "a", "an", "the", "of", "in", "on", "at", "to", "from", "with",
        "and", "or", "is", "are", "visible", "shown", "located", "near",
    }
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in stop}


@dataclass
class SplitRecord:
    image: dict[str, Any]
    annotations: list[dict[str, Any]]

    @property
    def annotation(self) -> dict[str, Any]:
        """Backward-compatible access for prompt/sample-level diagnostics."""
        return self.annotations[0]


class VOPDBBoxDataset(Dataset):
    """COCO bbox dataset whose text query lives on each annotation."""

    def __init__(
        self,
        split_dir: str | Path,
        num_negatives: int = 0,
        seed: int = 42,
        limit: int | None = None,
        prompt_overrides: dict[int, str] | None = None,
    ) -> None:
        self.split_dir = Path(split_dir).resolve()
        self.coco = load_json(self.split_dir / "_annotations.coco.json")
        images = sorted(self.coco["images"], key=lambda item: item["id"])
        annotations: dict[int, list[dict[str, Any]]] = {}
        for ann in self.coco["annotations"]:
            annotations.setdefault(int(ann["image_id"]), []).append(ann)
        self.records = [
            SplitRecord(img, annotations[int(img["id"])])
            for img in images if int(img["id"]) in annotations
        ]
        for record in self.records:
            prompts = {str(ann["prompt"]) for ann in record.annotations}
            if len(prompts) != 1:
                raise ValueError(
                    f"Expected one exhaustive prompt per image record {record.image['id']}, got {sorted(prompts)}"
                )
        if limit is not None:
            self.records = self.records[:limit]
        self.num_negatives = int(num_negatives)
        self.seed = int(seed)
        self.epoch = 0
        self.prompt_overrides = prompt_overrides or {}
        self.prompts = [str(record.annotation["prompt"]) for record in self.records]
        self.prompt_tokens = [lexical_tokens(prompt) for prompt in self.prompts]
        self.resolution = 1008
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample_negatives(self, idx: int, positive_prompt: str) -> list[str]:
        if self.num_negatives <= 0 or len(self.prompts) <= 1:
            return []
        rng = random.Random(self.seed + self.epoch * 1_000_003 + idx)
        candidates = list(range(len(self.prompts)))
        rng.shuffle(candidates)
        positive_tokens = lexical_tokens(positive_prompt)
        selected: list[str] = []
        for candidate_idx in candidates:
            if candidate_idx == idx:
                continue
            candidate = self.prompts[candidate_idx]
            if candidate == positive_prompt:
                continue
            if positive_tokens & self.prompt_tokens[candidate_idx]:
                continue
            selected.append(candidate)
            if len(selected) == self.num_negatives:
                break
        return selected

    def __getitem__(self, idx: int):
        from sam3.train.data.sam3_image_dataset import (
            Datapoint,
            FindQueryLoaded,
            Image,
            InferenceMetadata,
            Object,
        )

        record = self.records[idx]
        image_info, anns = record.image, record.annotations
        image_path = self.split_dir / image_info["file_name"]
        raw_image = PILImage.open(image_path).convert("RGB")
        orig_w, orig_h = raw_image.size
        resized = raw_image.resize((self.resolution, self.resolution), PILImage.Resampling.BILINEAR)
        image_tensor = self.transform(resized)

        objects = []
        for object_id, ann in enumerate(anns):
            x, y, w, h = map(float, ann["bbox"])
            box = torch.tensor(
                [(x + w / 2.0) / orig_w, (y + h / 2.0) / orig_h, w / orig_w, h / orig_h],
                dtype=torch.float32,
            )
            objects.append(
                Object(bbox=box, area=float(box[2] * box[3]), object_id=object_id, segment=None)
            )
        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution),
        )
        prompt = self.prompt_overrides.get(int(image_info["id"]), str(anns[0]["prompt"]))
        metadata = InferenceMetadata(
            coco_image_id=int(image_info["id"]),
            original_image_id=int(image_info["id"]),
            original_category_id=1,
            original_size=(orig_h, orig_w),
            object_id=-1,
            frame_index=-1,
        )
        queries = [
            FindQueryLoaded(
                query_text=prompt,
                image_id=0,
                object_ids_output=list(range(len(objects))),
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=metadata,
            )
        ]
        for negative in self._sample_negatives(idx, prompt):
            queries.append(
                FindQueryLoaded(
                    query_text=negative,
                    image_id=0,
                    object_ids_output=[],
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=int(image_info["id"]),
                        original_image_id=int(image_info["id"]),
                        original_category_id=-1,
                        original_size=(orig_h, orig_w),
                        object_id=-1,
                        frame_index=-1,
                    ),
                )
            )
        return Datapoint(find_queries=queries, images=[image_obj], raw_images=[raw_image])


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        for field in value.__dataclass_fields__:
            setattr(value, field, move_to_device(getattr(value, field), device))
        return value
    return value


def make_lora_config(config: dict[str, Any]):
    from lora_layers import LoRAConfig

    lora = config["lora"]
    return LoRAConfig(
        rank=int(lora["rank"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        apply_to_vision_encoder=bool(lora["apply_to_vision_encoder"]),
        apply_to_text_encoder=bool(lora["apply_to_text_encoder"]),
        apply_to_geometry_encoder=bool(lora["apply_to_geometry_encoder"]),
        apply_to_detr_encoder=bool(lora["apply_to_detr_encoder"]),
        apply_to_detr_decoder=bool(lora["apply_to_detr_decoder"]),
        apply_to_mask_decoder=bool(lora["apply_to_mask_decoder"]),
    )


def build_sam3(device: torch.device, eval_mode: bool = True):
    from sam3.model_builder import build_sam3_image_model

    if not SAM3_CHECKPOINT.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {SAM3_CHECKPOINT}")
    return build_sam3_image_model(
        device=device.type,
        eval_mode=eval_mode,
        checkpoint_path=str(SAM3_CHECKPOINT),
        load_from_HF=False,
        bpe_path=str(SAM3_BPE),
        enable_segmentation=True,
        compile=False,
    )


def joint_scores(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    scores = torch.sigmoid(outputs["pred_logits"].float()).squeeze(-1)
    presence = outputs.get("presence_logit_dec")
    if presence is not None:
        presence_prob = torch.sigmoid(presence.float()).reshape(scores.shape[0], -1)[:, 0]
        scores = scores * presence_prob[:, None]
    return scores


def normalized_cxcywh_to_xywh(
    box: Sequence[float], width: int, height: int
) -> list[float]:
    cx, cy, bw, bh = map(float, box)
    x1 = max(0.0, min(float(width), (cx - bw / 2.0) * width))
    y1 = max(0.0, min(float(height), (cy - bh / 2.0) * height))
    x2 = max(0.0, min(float(width), (cx + bw / 2.0) * width))
    y2 = max(0.0, min(float(height), (cy + bh / 2.0) * height))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def bootstrap_ci(
    values: Sequence[float], statistic=np.mean, repeats: int = 10_000, seed: int = 42
) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    stats = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        stats[i] = statistic(arr[rng.integers(0, arr.size, arr.size)])
    low, high = np.quantile(stats, [0.025, 0.975])
    return [float(low), float(high)]


def compute_top1_metrics(
    coco_gt: dict[str, Any], predictions: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gt_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco_gt["annotations"]:
        gt_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    pred_by_image: dict[int, list[dict[str, Any]]] = {}
    for pred in predictions:
        pred_by_image.setdefault(int(pred["image_id"]), []).append(pred)

    rows: list[dict[str, Any]] = []
    for image_id, ground_truths in sorted(gt_by_image.items()):
        candidates = sorted(
            pred_by_image.get(int(image_id), []), key=lambda item: item["score"], reverse=True
        )
        top = candidates[0] if candidates else None
        iou = max(
            (box_iou_xywh(gt["bbox"], top["bbox"]) for gt in ground_truths),
            default=0.0,
        ) if top else 0.0
        center_hit = 0.0
        score = 0.0
        if top:
            px, py, pw, ph = map(float, top["bbox"])
            cx, cy = px + pw / 2.0, py + ph / 2.0
            center_hit = float(any(
                gx <= cx <= gx + gw and gy <= cy <= gy + gh
                for gx, gy, gw, gh in (map(float, gt["bbox"]) for gt in ground_truths)
            ))
            score = float(top["score"])
        rows.append(
            {
                "image_id": int(image_id),
                "iou": float(iou),
                "center_hit": center_hit,
                "top1_score": score,
                "num_predictions": len(candidates),
                "num_ground_truth_instances": len(ground_truths),
            }
        )

    ious = np.asarray([row["iou"] for row in rows], dtype=np.float64)
    hits = np.asarray([row["center_hit"] for row in rows], dtype=np.float64)
    metrics: dict[str, Any] = {
        "num_samples": len(rows),
        "top1_mean_iou": float(ious.mean()) if len(ious) else 0.0,
        "top1_median_iou": float(np.median(ious)) if len(ious) else 0.0,
        "center_hit_rate": float(hits.mean()) if len(hits) else 0.0,
    }
    for threshold in (0.25, 0.50, 0.75):
        key = f"recall_at_1_iou_{threshold:.2f}"
        values = (ious >= threshold).astype(np.float64)
        metrics[key] = float(values.mean()) if len(values) else 0.0
        metrics[f"{key}_ci95"] = bootstrap_ci(values)
    metrics["top1_mean_iou_ci95"] = bootstrap_ci(ious)
    return metrics, rows


def compute_coco_bbox_metrics(
    gt_path: Path, predictions: list[dict[str, Any]], prediction_path: Path
) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    save_json(prediction_path, predictions)
    if not predictions:
        return {"bbox_map": 0.0, "bbox_ap50": 0.0, "bbox_ap75": 0.0, "bbox_ar1": 0.0, "bbox_ar100": 0.0}
    # The exported benchmark deliberately uses zero-based, contiguous image
    # and annotation IDs. COCOeval internally uses annotation ID 0 as its
    # unmatched sentinel, so evaluate an in-memory copy with IDs shifted by
    # one while leaving the public GT file unchanged.
    gt = load_json(gt_path)
    if any(int(annotation["id"]) == 0 for annotation in gt["annotations"]):
        gt = dict(gt)
        gt["annotations"] = [
            {**annotation, "id": int(annotation["id"]) + 1}
            for annotation in gt["annotations"]
        ]
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(str(prediction_path))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.useCats = False
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return {
        "bbox_map": float(stats[0]),
        "bbox_ap50": float(stats[1]),
        "bbox_ap75": float(stats[2]),
        "bbox_ar1": float(stats[6]),
        "bbox_ar100": float(stats[8]),
    }


def relative_delta(before: float, after: float) -> float | None:
    if math.isclose(before, 0.0):
        return None
    return (after - before) / abs(before)
