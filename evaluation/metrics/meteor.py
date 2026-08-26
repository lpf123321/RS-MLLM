import re
from typing import Dict, List

import numpy as np
from nltk.translate.meteor_score import meteor_score

from evaluation.base.metric import BaseMetric


def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class METEOR(BaseMetric):
    name = "meteor"

    def compute(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        scores = []
        for refs, hyp in zip(references, predictions):
            hyp_tokens = _tokenize(hyp)
            if not hyp_tokens:
                scores.append(0.0)
                continue
            scores.append(meteor_score([_tokenize(ref) for ref in refs], hyp_tokens))
        return {"METEOR": float(np.mean(scores)) if scores else 0.0}
