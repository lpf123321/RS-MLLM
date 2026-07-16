from typing import Dict, List

import numpy as np
from rouge_score import rouge_scorer

from evaluation.base.metric import BaseMetric


class ROUGEL(BaseMetric):
    name = "rouge_l"

    def compute(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = []
        for refs, hyp in zip(references, predictions):
            best = max(scorer.score(ref, hyp)["rougeL"].fmeasure for ref in refs)
            scores.append(best)
        return {"ROUGE-L": float(np.mean(scores)) if scores else 0.0}
