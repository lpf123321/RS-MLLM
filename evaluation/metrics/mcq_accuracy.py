import re
from typing import Dict, List

from evaluation.base.metric import BaseMetric


class MCQAccuracy(BaseMetric):
    name = "mcq_accuracy"

    @staticmethod
    def _extract_letter(text: str) -> str:
        # 优先匹配 "C." / "C)" / "(C)" 等格式
        m = re.search(r'(?<!\w)([A-Ea-e])\s*[.)]', text.strip())
        if m:
            return m.group(1).upper()
        # 回退：裸字母（如 "C" / "A"），用于输出仅为字母的模型
        m = re.search(r'(?<!\w)([A-Ea-e])(?!\w)', text.strip())
        return m.group(1).upper() if m else ""

    def compute(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        if not references:
            return {"MCQ_Accuracy": 0.0}
        correct = 0
        for refs, pred in zip(references, predictions):
            if not refs:
                continue
            pred_letter = self._extract_letter(pred)
            ref_letter = self._extract_letter(refs[0])
            if pred_letter and pred_letter == ref_letter:
                correct += 1
        return {"MCQ_Accuracy": correct / len(references)}
