from typing import Dict, List, Optional

import numpy as np

from evaluation.base.metric import BaseMetric
from evaluation.metrics.referring import _iou, _parse_bbox


class GroundingIoU(BaseMetric):
    """IoU for 0-1 normalized grounding bbox.

    The reference is a 0-1 normalized ``[xmin, ymin, xmax, ymax]`` text. The
    prediction may be 0-1 / 0-100 / pixel coordinates (the pixel -> 0-100
    conversion is already done by ``evaluation.main.evaluate``), so this metric
    normalizes everything to 0-1 before computing IoU.
    """

    name = "grounding_iou"

    def __init__(self, thresholds: Optional[List[float]] = None):
        self.thresholds = thresholds or [0.25, 0.5, 0.7]

    @staticmethod
    def _to_01(box):
        if box is None:
            return None
        m = max(abs(v) for v in box)
        if m <= 1.5:
            return tuple(box)
        if m <= 100.5:
            return tuple(v / 100.0 for v in box)
        if m <= 1000.5:
            return tuple(v / 1000.0 for v in box)
        return tuple(box)

    def compute(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        ious = []
        n_parseable = 0
        for refs, pred in zip(references, predictions):
            ref_box = self._to_01(_parse_bbox(refs[0] if refs else ""))
            pred_box = self._to_01(_parse_bbox(pred))
            if ref_box is None or pred_box is None:
                ious.append(0.0)
                continue
            n_parseable += 1
            ious.append(_iou(ref_box, pred_box))

        if not ious:
            return {
                "mean_iou": 0.0,
                **{f"Acc@{t}": 0.0 for t in self.thresholds},
                "parseable": 0.0,
            }

        total = len(ious)
        result = {"mean_iou": float(np.mean(ious)), "parseable": n_parseable / total}
        for t in self.thresholds:
            result[f"Acc@{t}"] = sum(1 for v in ious if v >= t) / total
        return result
