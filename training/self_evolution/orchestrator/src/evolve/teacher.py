from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps

from .bbox import area, clip_xyxy, union_box, xywh_to_xyxy


def crop_boxes(image: Image.Image, boxes_xyxy: list[Sequence[float]]) -> list[Image.Image]:
    crops = []
    for box in boxes_xyxy:
        clipped = clip_xyxy(box, image.width, image.height)
        crops.append(image.crop(tuple(clipped)))
    return crops


def make_montage(crops: list[Image.Image], tile_size: int = 672) -> Image.Image:
    if not crops:
        raise ValueError("Cannot construct teacher image without crops")
    if len(crops) == 1:
        return crops[0].copy()
    columns = min(2, len(crops))
    rows = math.ceil(len(crops) / columns)
    canvas = Image.new("RGB", (columns * tile_size, rows * tile_size), (127, 127, 127))
    for index, crop in enumerate(crops):
        tile = ImageOps.contain(crop.convert("RGB"), (tile_size, tile_size))
        x = (index % columns) * tile_size + (tile_size - tile.width) // 2
        y = (index // columns) * tile_size + (tile_size - tile.height) // 2
        canvas.paste(tile, (x, y))
    return canvas


def materialize_teacher_image(
    image_path: str | Path,
    boxes_xywh: list[Sequence[float]],
    output_path: str | Path,
) -> tuple[Image.Image, list[list[int]], float]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    boxes = [clip_xyxy(xywh_to_xyxy(box), image.width, image.height) for box in boxes_xywh]
    teacher = make_montage(crop_boxes(image, boxes))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    teacher.save(output_path, format="JPEG", quality=95, subsampling=0)
    covered = area(union_box(boxes)) / float(image.width * image.height)
    return teacher, boxes, covered


def scaled_crop(image: Image.Image, box: Sequence[float], scale_delta: float) -> Image.Image:
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = (x2 - x1) * (1 + scale_delta) / 2
    half_h = (y2 - y1) * (1 + scale_delta) / 2
    clipped = clip_xyxy([cx - half_w, cy - half_h, cx + half_w, cy + half_h], image.width, image.height)
    return image.crop(tuple(clipped))
