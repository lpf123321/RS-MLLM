from __future__ import annotations

from typing import Iterable, Sequence


def xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, x2 - x1, y2 - y1]


def xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def clip_xyxy(box: Sequence[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0, min(width, round(x1)))
    y1 = max(0, min(height, round(y1)))
    x2 = max(0, min(width, round(x2)))
    y2 = max(0, min(height, round(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate bbox after clipping: {box}")
    return [x1, y1, x2, y2]


def area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(left: Sequence[float], right: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, left)
    bx1, by1, bx2, by2 = map(float, right)
    intersection = area([max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)])
    union = area(left) + area(right) - intersection
    return intersection / union if union > 0 else 0.0


def union_box(boxes: Iterable[Sequence[float]]) -> list[float]:
    boxes = [list(map(float, box)) for box in boxes]
    if not boxes:
        raise ValueError("Cannot union an empty box list")
    return [
        min(box[0] for box in boxes), min(box[1] for box in boxes),
        max(box[2] for box in boxes), max(box[3] for box in boxes),
    ]


def jitter_xyxy(box: Sequence[float], width: int, height: int, ratio: float) -> list[int]:
    x1, y1, x2, y2 = map(float, box)
    dx, dy = (x2 - x1) * ratio, (y2 - y1) * ratio
    return clip_xyxy([x1 - dx, y1 - dy, x2 + dx, y2 + dy], width, height)
