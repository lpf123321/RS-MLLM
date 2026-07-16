from typing import Dict, List

from evaluation.base.metric import BaseMetric


class Accuracy(BaseMetric):
    name = "accuracy"

    def compute(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        if not references:
            return {"Accuracy": 0.0}
        correct = sum(
            1 for refs, pred in zip(references, predictions)
            if refs and refs[0].strip().lower() == pred.strip().lower()
        )
        return {"Accuracy": correct / len(references)}
