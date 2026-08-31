#!/usr/bin/env python3
"""Negative-prompt datasets and score diagnostics for isolated VOPD experiments."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision.transforms import v2

from vopd_bbox_common import SplitRecord, lexical_tokens, load_json, read_jsonl


def load_negative_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        for row in read_jsonl(path):
            if row.get("accepted", True) is not True:
                continue
            key = (
                int(row["image_id"]),
                str(row["negative_prompt"]),
                str(row.get("negative_type", "unknown")),
                tuple(row.get("crop_xyxy", [])),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append({**row, "manifest_path": str(path)})
    return rows


def requested_negative_count(ratio: float, seed: int, epoch: int, index: int) -> int:
    """Convert a query ratio such as 0.5/1/2 into a deterministic count."""
    if ratio <= 0:
        return 0
    base = int(math.floor(ratio))
    fractional = float(ratio) - base
    if fractional > 0:
        rng = random.Random(seed + epoch * 1_000_003 + index * 9_973)
        base += int(rng.random() < fractional)
    return base


@dataclass
class NegativeDatasetEntry:
    record_index: int
    crop_row: dict[str, Any] | None = None


class ManifestVOPDDataset(Dataset):
    """Annotation-level VOPD dataset with verified manifest negatives.

    Text negatives are added as empty exhaustive queries on the original image.
    Crop negatives are represented as separate, negative-only datapoints because
    they have their own natural image view.
    """

    def __init__(
        self,
        split_dir: str | Path,
        negative_manifests: Iterable[str | Path] = (),
        negative_ratio: float = 1.0,
        seed: int = 42,
        positive_limit: int | None = None,
        allowed_types: set[str] | None = None,
        include_positive: bool = True,
    ) -> None:
        self.split_dir = Path(split_dir).resolve()
        self.coco = load_json(self.split_dir / "_annotations.coco.json")
        images = sorted(self.coco["images"], key=lambda item: int(item["id"]))
        anns: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in self.coco["annotations"]:
            anns[int(ann["image_id"])].append(ann)
        self.records = [
            SplitRecord(image, anns[int(image["id"])])
            for image in images if int(image["id"]) in anns
        ]
        for record in self.records:
            prompts = {str(ann["prompt"]) for ann in record.annotations}
            if len(prompts) != 1:
                raise ValueError(
                    f"Expected one exhaustive prompt per image record {record.image['id']}, got {sorted(prompts)}"
                )
        if positive_limit is not None:
            self.records = self.records[:positive_limit]
        self.record_index = {int(record.image["id"]): i for i, record in enumerate(self.records)}
        rows = load_negative_rows(negative_manifests)
        if allowed_types:
            rows = [row for row in rows if str(row.get("negative_type")) in allowed_types]
        self.rows_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.crop_rows_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            image_id = int(row["image_id"])
            if image_id not in self.record_index:
                continue
            if row.get("crop_xyxy"):
                self.crop_rows_by_image[image_id].append(row)
            else:
                self.rows_by_image[image_id].append(row)
        for mapping in (self.rows_by_image, self.crop_rows_by_image):
            for image_id in mapping:
                mapping[image_id].sort(
                    key=lambda row: (
                        -float(row.get("hardness_score", row.get("lora_max_score", 0.0))),
                        str(row.get("negative_type", "")),
                        str(row["negative_prompt"]),
                    )
                )
        self.negative_ratio = float(negative_ratio)
        self.seed = int(seed)
        self.epoch = 0
        self.include_positive = bool(include_positive)
        self.resolution = 1008
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        self.entries: list[NegativeDatasetEntry] = []
        self.fixed_crop_count_by_image: dict[int, int] = {}
        if self.include_positive:
            self.entries.extend(NegativeDatasetEntry(i) for i in range(len(self.records)))
        # Crop negatives are separate samples. Keep a fixed count so dataset
        # length is stable across epochs and DistributedSampler remains valid.
        for i, record in enumerate(self.records):
            image_id = int(record.image["id"])
            count = requested_negative_count(self.negative_ratio, self.seed, 0, i)
            crop_rows = self.crop_rows_by_image.get(image_id, [])
            fixed_crop_count = min(len(crop_rows), count)
            self.fixed_crop_count_by_image[image_id] = fixed_crop_count
            for row in crop_rows[:fixed_crop_count]:
                self.entries.append(NegativeDatasetEntry(i, row))

    def __len__(self) -> int:
        return len(self.entries)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def manifest_summary(self) -> dict[str, Any]:
        positive_ids = {int(record.image["id"]) for record in self.records}
        normal_coverage = sum(image_id in self.rows_by_image for image_id in positive_ids)
        crop_coverage = sum(image_id in self.crop_rows_by_image for image_id in positive_ids)
        types: dict[str, int] = defaultdict(int)
        for mapping in (self.rows_by_image, self.crop_rows_by_image):
            for rows in mapping.values():
                for row in rows:
                    types[str(row.get("negative_type", "unknown"))] += 1
        return {
            "positive_samples": len(self.records),
            "dataset_entries": len(self.entries),
            "negative_ratio": self.negative_ratio,
            "normal_prompt_coverage": normal_coverage / max(1, len(positive_ids)),
            "crop_prompt_coverage": crop_coverage / max(1, len(positive_ids)),
            "negative_types": dict(sorted(types.items())),
        }

    def _image_tensor(self, image: PILImage.Image) -> torch.Tensor:
        resized = image.resize((self.resolution, self.resolution), PILImage.Resampling.BILINEAR)
        return self.transform(resized)

    @staticmethod
    def _metadata(image_id: int, original_size: tuple[int, int], category_id: int):
        from sam3.train.data.sam3_image_dataset import InferenceMetadata

        return InferenceMetadata(
            coco_image_id=image_id,
            original_image_id=image_id,
            original_category_id=category_id,
            original_size=original_size,
            object_id=-1,
            frame_index=-1,
        )

    def _negative_rows(self, record_index: int, image_id: int) -> list[dict[str, Any]]:
        candidates = self.rows_by_image.get(image_id, [])
        if not candidates:
            return []
        count = requested_negative_count(
            self.negative_ratio, self.seed, self.epoch, record_index
        )
        # Crop negatives require their own image view and are represented as
        # fixed dataset entries. Count them against the total requested ratio
        # so a mixed crop/text manifest does not silently over-sample negatives.
        count = max(0, count - self.fixed_crop_count_by_image.get(image_id, 0))
        if count <= 0:
            return []
        # Rotate equally hard alternatives between epochs rather than sampling
        # non-deterministically inside DataLoader workers.
        offset = (self.epoch + record_index) % len(candidates)
        ordered = candidates[offset:] + candidates[:offset]
        return ordered[:count]

    def __getitem__(self, index: int):
        from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, Image, Object

        entry = self.entries[index]
        record = self.records[entry.record_index]
        image_info, anns = record.image, record.annotations
        image_id = int(image_info["id"])
        raw = PILImage.open(self.split_dir / image_info["file_name"]).convert("RGB")

        if entry.crop_row is not None:
            x1, y1, x2, y2 = map(int, entry.crop_row["crop_xyxy"])
            cropped = raw.crop((x1, y1, x2, y2))
            crop_id = 10_000_000 + image_id
            image = Image(data=self._image_tensor(cropped), objects=[], size=(self.resolution, self.resolution))
            query = FindQueryLoaded(
                query_text=str(entry.crop_row["negative_prompt"]),
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=self._metadata(crop_id, (cropped.height, cropped.width), -1),
            )
            return Datapoint(find_queries=[query], images=[image], raw_images=[cropped])

        orig_w, orig_h = raw.size
        objects = []
        for object_id, ann in enumerate(anns):
            x, y, w, h = map(float, ann["bbox"])
            box = torch.tensor(
                [(x + w / 2) / orig_w, (y + h / 2) / orig_h, w / orig_w, h / orig_h],
                dtype=torch.float32,
            )
            objects.append(
                Object(bbox=box, area=float(box[2] * box[3]), object_id=object_id, segment=None)
            )
        image = Image(
            data=self._image_tensor(raw), objects=objects, size=(self.resolution, self.resolution)
        )
        queries = [
            FindQueryLoaded(
                query_text=str(anns[0]["prompt"]),
                image_id=0,
                object_ids_output=list(range(len(objects))),
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=self._metadata(image_id, (orig_h, orig_w), 1),
            )
        ]
        for row in self._negative_rows(entry.record_index, image_id):
            queries.append(
                FindQueryLoaded(
                    query_text=str(row["negative_prompt"]),
                    image_id=0,
                    object_ids_output=[],
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=self._metadata(image_id, (orig_h, orig_w), -1),
                )
            )
        return Datapoint(find_queries=queries, images=[image], raw_images=[raw])


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), dim=-1
    )


def pairwise_iou_xyxy(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0))[:, None]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0))[None, :]
    return intersection / (area_a + area_b - intersection).clamp(min=1e-8)


def ranking_loss(
    final_outputs: dict[str, torch.Tensor],
    targets_list: list[dict[str, Any]],
    margin: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Separate IoU>=.5 true queries from IoU<.1 false queries."""
    logits = final_outputs["pred_logits"].float().squeeze(-1)
    scores = logits.sigmoid()
    presence = final_outputs.get("presence_logit_dec")
    if presence is not None:
        scores = scores * presence.float().sigmoid().reshape(scores.shape[0], -1)[:, :1]
    boxes = final_outputs["pred_boxes"].float()
    losses: list[torch.Tensor] = []
    true_scores: list[float] = []
    false_scores: list[float] = []
    prompt_offset = 0
    for target in targets_list:
        num_boxes = target["num_boxes"].tolist()
        packed_offset = 0
        for query_index, count in enumerate(num_boxes):
            pred_scores = scores[prompt_offset + query_index]
            pred_boxes = boxes[prompt_offset + query_index]
            if int(count) > 0:
                gt = target["boxes"][packed_offset : packed_offset + int(count)].float()
                packed_offset += int(count)
                iou = pairwise_iou_xyxy(cxcywh_to_xyxy(pred_boxes), cxcywh_to_xyxy(gt)).amax(dim=1)
                true_mask = iou >= 0.5
                false_mask = iou < 0.1
                true_score = pred_scores[true_mask].max() if true_mask.any() else pred_scores.new_zeros(())
                false_score = pred_scores[false_mask].max() if false_mask.any() else pred_scores.new_zeros(())
                losses.append(torch.relu(pred_scores.new_tensor(margin) + false_score - true_score))
                true_scores.append(float(true_score.detach()))
                false_scores.append(float(false_score.detach()))
        prompt_offset += len(num_boxes)
    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = scores.sum() * 0.0
    return loss, {
        "rank_true_score": sum(true_scores) / max(1, len(true_scores)),
        "rank_false_score": sum(false_scores) / max(1, len(false_scores)),
        "rank_pairs": float(len(losses)),
    }


class TeacherReplayDataset(Dataset):
    """One prompt-image pair per cached original-SAM3 response."""

    def __init__(self, cache_path: str | Path, limit: int | None = None) -> None:
        payload = torch.load(Path(cache_path), map_location="cpu", weights_only=False)
        if payload.get("format") != "vopd-sam3-teacher-v1":
            raise ValueError(f"Unsupported teacher cache format in {cache_path}")
        self.entries = payload["entries"][:limit] if limit is not None else payload["entries"]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, Image, InferenceMetadata

        entry = self.entries[index]
        raw = PILImage.open(entry["image_path"]).convert("RGB")
        if entry.get("crop_xyxy"):
            raw = raw.crop(tuple(map(int, entry["crop_xyxy"])))
        transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.5] * 3, [0.5] * 3)]
        )
        tensor = transform(raw.resize((1008, 1008), PILImage.Resampling.BILINEAR))
        image = Image(data=tensor, objects=[], size=(1008, 1008))
        query = FindQueryLoaded(
            query_text=str(entry["prompt"]),
            image_id=0,
            object_ids_output=[],
            # This batch never enters the standard criterion, but false here
            # documents that it is a soft-label teacher query, not a hard empty target.
            is_exhaustive=False,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=index,
                original_image_id=int(entry["image_id"]),
                original_category_id=-2,
                original_size=(raw.height, raw.width),
                object_id=-1,
                frame_index=-1,
            ),
        )
        return Datapoint(find_queries=[query], images=[image], raw_images=[raw]), entry


def collate_teacher_replay(batch):
    from sam3.train.data.collator import collate_fn_api

    datapoints, entries = zip(*batch)
    collated = collate_fn_api(list(datapoints), dict_key="input", with_seg_masks=False)
    return collated, list(entries)


def teacher_distillation_loss(
    final_outputs: dict[str, torch.Tensor],
    teacher_entries: list[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Directly replay base query logits/boxes; query order is architecture-stable."""
    student_logits = final_outputs["pred_logits"].float()
    student_boxes = final_outputs["pred_boxes"].float()
    teacher_logits = torch.stack([entry["pred_logits"].float() for entry in teacher_entries]).to(student_logits.device)
    teacher_boxes = torch.stack([entry["pred_boxes"].float() for entry in teacher_entries]).to(student_boxes.device)
    if teacher_logits.shape != student_logits.shape or teacher_boxes.shape != student_boxes.shape:
        raise RuntimeError(
            f"Teacher/student shape mismatch: logits {teacher_logits.shape}/{student_logits.shape}, "
            f"boxes {teacher_boxes.shape}/{student_boxes.shape}"
        )
    logit_loss = F.binary_cross_entropy_with_logits(student_logits, teacher_logits.sigmoid())
    student_presence = final_outputs.get("presence_logit_dec")
    teacher_presence_values = [entry.get("presence_logit") for entry in teacher_entries]
    if student_presence is not None and all(value is not None for value in teacher_presence_values):
        teacher_presence = torch.stack([value.float() for value in teacher_presence_values]).to(student_presence.device)
        presence_loss = F.binary_cross_entropy_with_logits(
            student_presence.float(), teacher_presence.sigmoid()
        )
        teacher_joint = teacher_logits.sigmoid().squeeze(-1) * teacher_presence.sigmoid().reshape(len(teacher_entries), -1)[:, :1]
    else:
        presence_loss = student_logits.sum() * 0.0
        teacher_joint = teacher_logits.sigmoid().squeeze(-1)
    # Preserve boxes only for the teacher's most relevant queries. Weighting by
    # the teacher joint score avoids anchoring arbitrary background boxes.
    top_count = min(20, teacher_joint.shape[1])
    top_indices = torch.topk(teacher_joint, k=top_count, dim=1).indices
    gather_index = top_indices[..., None].expand(-1, -1, 4)
    selected_student = torch.gather(student_boxes, 1, gather_index)
    selected_teacher = torch.gather(teacher_boxes, 1, gather_index)
    box_weights = torch.gather(teacher_joint, 1, top_indices).detach()
    box_loss = (
        F.smooth_l1_loss(selected_student, selected_teacher, reduction="none").mean(dim=-1)
        * box_weights
    ).sum() / box_weights.sum().clamp(min=1e-6)
    total = logit_loss + presence_loss + box_loss
    return total, {
        "distill_logit_loss": float(logit_loss.detach()),
        "distill_presence_loss": float(presence_loss.detach()),
        "distill_box_loss": float(box_loss.detach()),
    }
