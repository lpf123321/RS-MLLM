import re
from typing import Dict, List

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from evaluation.base.metric import BaseMetric


def _tokenize(text: str) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return text.split()


class BLEU(BaseMetric):
    name = "bleu"

    def __init__(self, max_n: int = 4):
        self.max_n = max_n

    def compute(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        list_of_refs = [[_tokenize(ref) for ref in ref_group] for ref_group in references]
        list_of_hyps = [_tokenize(hyp) for hyp in predictions]
        smooth = SmoothingFunction().method1
        scores = {}
        for n in range(1, self.max_n + 1):
            weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
            try:
                score = corpus_bleu(list_of_refs, list_of_hyps, weights=weights, smoothing_function=smooth)
                if score > 1.0:
                    score = score / 100.0
                scores[f"BLEU-{n}"] = score
            except Exception:
                scores[f"BLEU-{n}"] = 0.0
        return scores
