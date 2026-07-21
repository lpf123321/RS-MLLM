import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from evaluation.base.metric import BaseMetric

_BBOX_ANGLE_RE = re.compile(
    r"\{<\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*><\s*(\d+(?:\.\d+)?)\s*>\}"
)
_BBOX_COMMA_RE = re.compile(
    r"\{\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*[,;\s]+\s*(\d+(?:\.\d+)?)\s*\}"
)


def _parse_bbox(text: str) -> Optional[Tuple[float, float, float, float]]:
    text = _BBOX_COMMA_RE.sub(r"{<\1><\2><\3><\4>}", text)
    m = _BBOX_ANGLE_RE.search(text)
    return tuple(map(float, m.groups())) if m else None


def _iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    x1 = max(min(b1[0], b1[2]), min(b2[0], b2[2]))
    y1 = max(min(b1[1], b1[3]), min(b2[1], b2[3]))
    x2 = min(max(b1[0], b1[2]), max(b2[0], b2[2]))
    y2 = min(max(b1[1], b1[3]), max(b2[1], b2[3]))
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = abs(b1[2] - b1[0]) * abs(b1[3] - b1[1])
    a2 = abs(b2[2] - b2[0]) * abs(b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class ReferringAcc(BaseMetric):
    name = "referring"

    def __init__(self, thresholds: Optional[List[float]] = None):
        self.thresholds = thresholds or [0.25, 0.5, 0.7]

    def compute(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        ious = []
        for refs, pred in zip(references, predictions):
            ref_box = _parse_bbox(refs[0]) if refs else None
            pred_box = _parse_bbox(pred)
            ious.append(_iou(ref_box, pred_box) if ref_box and pred_box else 0.0)

        if not ious:
            return {"mean_iou": 0.0, **{f"Acc@{t}": 0.0 for t in self.thresholds}}

        total = len(ious)
        result = {"mean_iou": float(np.mean(ious))}
        for t in self.thresholds:
            result[f"Acc@{t}"] = sum(1 for v in ious if v >= t) / total
        return result
