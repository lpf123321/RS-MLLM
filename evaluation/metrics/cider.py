import re
from collections import defaultdict
from math import log, sqrt
from typing import Dict, List

import numpy as np

from evaluation.base.metric import BaseMetric


def _tokenize(text: str) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return text.split()


def _ngrams(tokens: List[str], n: int):
    ngs = {}
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i:i + n])
        ngs[g] = ngs.get(g, 0) + 1
    return ngs


class CIDEr(BaseMetric):
    name = "cider"

    def __init__(self, n: int = 4, sigma: float = 6.0):
        self.n = n
        self.sigma = sigma

    def compute(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        all_ref_tokens = [[_tokenize(ref) for ref in group] for group in references]
        all_hyp_tokens = [_tokenize(hyp) for hyp in predictions]
        N = len(references)
        corpus_refs = [t for group in all_ref_tokens for t in group]

        cider_scores = []
        for ngram_n in range(1, self.n + 1):
            df = defaultdict(int)
            for tokens in corpus_refs:
                for gram in set(tuple(tokens[i:i + ngram_n]) for i in range(len(tokens) - ngram_n + 1)):
                    df[gram] += 1

            n_scores = []
            for hyp_tokens, ref_groups in zip(all_hyp_tokens, all_ref_tokens):
                if not hyp_tokens:
                    continue
                hyp_ng = _ngrams(hyp_tokens, ngram_n)
                hyp_max = max(hyp_ng.values()) if hyp_ng else 1
                hyp_vec = {g: (c / hyp_max) * (log((N + 1) / (df.get(g, 0) + 1)) + 1)
                           for g, c in hyp_ng.items()}

                ref_avg = []
                for ref_tokens in ref_groups:
                    if not ref_tokens:
                        continue
                    ref_ng = _ngrams(ref_tokens, ngram_n)
                    ref_max = max(ref_ng.values()) if ref_ng else 1
                    ref_vec = {g: (c / ref_max) * (log((N + 1) / (df.get(g, 0) + 1)) + 1)
                               for g, c in ref_ng.items()}

                    dot = sum(hyp_vec.get(k, 0) * ref_vec.get(k, 0) for k in set(hyp_vec) | set(ref_vec))
                    n1 = sqrt(sum(v ** 2 for v in hyp_vec.values()))
                    n2 = sqrt(sum(v ** 2 for v in ref_vec.values()))
                    ref_avg.append(dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0)

                n_scores.append(np.mean(ref_avg) if ref_avg else 0.0)

            cider_scores.append(float(np.mean(n_scores)) if n_scores else 0.0)

        weights = [1.0 / self.n] * self.n
        cider = sum(w * s for w, s in zip(weights, cider_scores))

        ref_lens = [len(t) for group in all_ref_tokens for t in group]
        hyp_lens = [len(t) for t in all_hyp_tokens]
        avg_ref_len = float(np.mean(ref_lens)) if ref_lens else 1.0
        avg_hyp_len = float(np.mean(hyp_lens)) if hyp_lens else 0.0
        sigma_val = float(np.std(ref_lens)) if len(ref_lens) > 1 else avg_ref_len / self.sigma
        diff = abs(avg_ref_len - avg_hyp_len)
        penalty = np.exp(-(diff ** 2) / (2 * sigma_val ** 2)) if sigma_val > 0 else 1.0

        return {"CIDEr": float(max(cider * penalty * 10.0, 0.0))}
