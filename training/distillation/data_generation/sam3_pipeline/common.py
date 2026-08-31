#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


VRSBENCH_OBJECT_CLASSES = (
    "vehicle", "airplane", "ship", "harbor", "bridge", "tennis-court",
    "storage-tank", "baseball-diamond", "swimming-pool", "ground-track-field",
)

XLRS_OBJECT_CLASSES = (
    "building", "vehicle", "ship", "lake", "lawn", "park", "settlement",
    "farmland", "forest", "sports-field", "road", "soccer-field",
    "swimming-pool", "parking-lot", "tennis-court", "basketball-court",
)

DATASET_OBJECT_CLASSES = {
    "vrsbench": VRSBENCH_OBJECT_CLASSES,
    "xlrs": XLRS_OBJECT_CLASSES,
}

DATASET_CLASS_PROMPTS = {
    "vrsbench": {name: (name.replace("-", " "),) for name in VRSBENCH_OBJECT_CLASSES},
    "xlrs": {
        "building": ("building", "house", "roof"),
        "vehicle": ("vehicle", "car", "truck"),
        "ship": ("ship", "boat", "vessel"),
        "lake": ("lake", "pond"),
        "lawn": ("lawn", "green space"),
        "park": ("park", "plaza", "square"),
        "settlement": ("settlement", "town", "village"),
        "farmland": ("farmland", "field"),
        "forest": ("forest", "woodland"),
        "sports-field": ("sports field",),
        "road": ("road", "intersection", "roundabout"),
        "soccer-field": ("soccer field",),
        "swimming-pool": ("swimming pool",),
        "parking-lot": ("parking lot",),
        "tennis-court": ("tennis court",),
        "basketball-court": ("basketball court",),
    },
}


def object_classes(dataset_profile: str) -> tuple[str, ...]:
    try:
        return DATASET_OBJECT_CLASSES[dataset_profile]
    except KeyError as exc:
        raise ValueError(f"unknown dataset profile: {dataset_profile}") from exc


def class_prompts(dataset_profile: str) -> dict[str, tuple[str, ...]]:
    try:
        return DATASET_CLASS_PROMPTS[dataset_profile]
    except KeyError as exc:
        raise ValueError(f"unknown dataset profile: {dataset_profile}") from exc


# Backward-compatible aliases for old VRSBench scripts and artifacts.
OBJECT_CLASSES = VRSBENCH_OBJECT_CLASSES
CLASS_PROMPTS = {name: prompts[0] for name, prompts in DATASET_CLASS_PROMPTS["vrsbench"].items()}
ANSWER_RE = re.compile(r"^\{<(-?\d+(?:\.\d+)?)><(-?\d+(?:\.\d+)?)><(-?\d+(?:\.\d+)?)><(-?\d+(?:\.\d+)?)>\}$")


def read_rows(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], "jsonl"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON array: {path}")
    return payload, "json"


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_name(row: dict[str, Any]) -> str:
    value = row.get("image")
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not value and isinstance(row.get("messages"), list):
        for message in row["messages"]:
            for content in message.get("content", []) if isinstance(message, dict) else []:
                if isinstance(content, dict) and content.get("type") == "image":
                    value = content.get("image")
                    break
            if value:
                break
    if not isinstance(value, str) or not value:
        raise ValueError("row has no scalar image path")
    return Path(value).name


def question_answer(row: dict[str, Any]) -> tuple[str, str]:
    turns = row.get("conversations")
    if not isinstance(turns, list) or len(turns) < 2:
        raise ValueError("row has no two-turn conversations")
    return str(turns[0].get("value", "")), str(turns[1].get("value", ""))


def parse_native_box(answer: str) -> list[float]:
    match = ANSWER_RE.fullmatch(answer.strip())
    if not match:
        raise ValueError(f"not a native VRSBench box: {answer!r}")
    return [float(value) / 100.0 for value in match.groups()]


def valid_box(box: list[float], width: int, height: int) -> bool:
    return (
        len(box) == 4 and all(math.isfinite(v) for v in box)
        and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
    )


def clip_box(box: list[float], width: int, height: int) -> list[float]:
    return [
        max(0.0, min(float(width), float(box[0]))),
        max(0.0, min(float(height), float(box[1]))),
        max(0.0, min(float(width), float(box[2]))),
        max(0.0, min(float(height), float(box[3]))),
    ]


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def official_objects(annotation_root: str | Path, image: str, width: int, height: int) -> list[dict[str, Any]]:
    path = Path(annotation_root) / f"{Path(image).stem}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for obj in payload.get("objects", []):
        cls = str(obj.get("obj_cls", "")).lower()
        coord = obj.get("obj_coord")
        if cls and isinstance(coord, list) and len(coord) == 4:
            output.append({"class": cls, "bbox_xyxy_px": [coord[0] * width, coord[1] * height, coord[2] * width, coord[3] * height]})
    return output


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
