import re
from collections import defaultdict
from math import log, sqrt
from typing import Dict, List

import numpy as np

from evaluation.base.metric import BaseMetric


def _tokenize(text: str) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return text.split()


def _precook(tokens: List[str], n: int):
    """Build n-gram counts for all orders 1..n (mirrors pycocoevalcap precook)."""
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(tokens) - k + 1):
            counts[tuple(tokens[i:i + k])] += 1
    return counts


class CIDEr(BaseMetric):
    """CIDEr-D as implemented by pycocoevalcap/cider_scorer.py (tylin/coco-caption).

    Mirrors the reference algorithm:
      - TF-IDF weight = term_freq * (log(N) - log(df)), no count normalization
      - similarity = sum(min(h,r)*r) / (||h|| * ||r||)   (asymmetric, min-clipped)
      - per-sample Gaussian length penalty with sigma=6.0
      - mean over n-grams, divided by #refs, multiplied by 10
    """

    name = "cider"

    def __init__(self, n: int = 4, sigma: float = 6.0):
        self.n = n
        self.sigma = sigma

    def compute(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        ref_tokens = [[_tokenize(ref) for ref in group] for group in references]
        hyp_tokens = [_tokenize(hyp) for hyp in predictions]
        N = len(references)
        if N == 0:
            return {"CIDEr": 0.0}

        doc_freq = defaultdict(int)
        for group in ref_tokens:
            for gram in set(gram for ref in group for gram in _precook(ref, self.n)):
                doc_freq[gram] += 1

        ref_len = np.log(float(N))

        def counts2vec(cnts):
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                df = np.log(max(1.0, doc_freq[ngram]))
                n = len(ngram) - 1
                vec[n][ngram] = float(term_freq) * (ref_len - df)
                norm[n] += pow(vec[n][ngram], 2)
                if n == 1:
                    length += term_freq
            norm = [np.sqrt(x) for x in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            delta = float(length_hyp - length_ref)
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                for (ngram, count) in vec_hyp[n].items():
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n] * norm_ref[n])
                val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            return val

        scores = []
        for group_refs, hyp in zip(ref_tokens, hyp_tokens):
            vec, norm, length = counts2vec(_precook(hyp, self.n))
            score = np.array([0.0 for _ in range(self.n)])
            for ref in group_refs:
                vec_ref, norm_ref, length_ref = counts2vec(_precook(ref, self.n))
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            score_avg /= len(group_refs)
            score_avg *= 10.0
            scores.append(score_avg)

        return {"CIDEr": float(np.mean(scores)) if scores else 0.0}
