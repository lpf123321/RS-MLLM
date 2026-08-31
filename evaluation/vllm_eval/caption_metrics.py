"""Dependency-free lexical caption metrics for smoke tests.

These metrics deliberately carry a ``_proxy`` suffix. They verify that the scoring
pipeline is working; they are not claimed to reproduce each dataset's official
COCO-caption package bit for bit.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)
    )


def _closest_reference_length(candidate_len: int, references: list[list[str]]) -> int:
    lengths = [len(reference) for reference in references]
    return min(lengths, key=lambda length: (abs(length - candidate_len), length))


def corpus_bleu_proxy(
    predictions: list[str], references: list[list[str]]
) -> dict[str, float]:
    clipped = [0] * 4
    totals = [0] * 4
    candidate_length = 0
    reference_length = 0
    for prediction, sample_references in zip(predictions, references, strict=True):
        candidate = tokenize(prediction)
        refs = [tokenize(reference) for reference in sample_references]
        candidate_length += len(candidate)
        reference_length += _closest_reference_length(len(candidate), refs)
        for n in range(1, 5):
            candidate_counts = _ngrams(candidate, n)
            maximum_reference: Counter[tuple[str, ...]] = Counter()
            for reference in refs:
                for gram, count in _ngrams(reference, n).items():
                    maximum_reference[gram] = max(maximum_reference[gram], count)
            clipped[n - 1] += sum(
                min(count, maximum_reference[gram])
                for gram, count in candidate_counts.items()
            )
            totals[n - 1] += sum(candidate_counts.values())
    if candidate_length == 0:
        return {f"bleu_{n}_proxy": 0.0 for n in range(1, 5)}
    brevity_penalty = (
        1.0
        if candidate_length > reference_length
        else math.exp(1.0 - reference_length / candidate_length)
    )
    precisions = [
        clipped[index] / totals[index] if totals[index] else 0.0 for index in range(4)
    ]
    result: dict[str, float] = {}
    for order in range(1, 5):
        active = precisions[:order]
        if any(value == 0 for value in active):
            score = 0.0
        else:
            score = brevity_penalty * math.exp(
                sum(math.log(value) for value in active) / order
            )
        result[f"bleu_{order}_proxy"] = score
    return result


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_proxy(prediction: str, references: list[str]) -> float:
    candidate = tokenize(prediction)
    if not candidate:
        return 0.0
    best = 0.0
    for reference_text in references:
        reference = tokenize(reference_text)
        if not reference:
            continue
        lcs = _lcs_length(candidate, reference)
        precision = lcs / len(candidate)
        recall = lcs / len(reference)
        score = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        best = max(best, score)
    return best


def meteor_exact_proxy(prediction: str, references: list[str]) -> float:
    """Exact-token METEOR-style score without stemming or WordNet synonyms."""
    candidate = tokenize(prediction)
    if not candidate:
        return 0.0
    best = 0.0
    for reference_text in references:
        reference = tokenize(reference_text)
        used: set[int] = set()
        matches: list[tuple[int, int]] = []
        for candidate_index, token in enumerate(candidate):
            for reference_index, reference_token in enumerate(reference):
                if reference_index not in used and token == reference_token:
                    used.add(reference_index)
                    matches.append((candidate_index, reference_index))
                    break
        match_count = len(matches)
        if not match_count or not reference:
            continue
        precision = match_count / len(candidate)
        recall = match_count / len(reference)
        harmonic = 10 * precision * recall / (recall + 9 * precision)
        chunks = 1 + sum(
            1
            for previous, current in zip(matches, matches[1:])
            if current[0] != previous[0] + 1 or current[1] != previous[1] + 1
        )
        penalty = 0.5 * (chunks / match_count) ** 3
        best = max(best, harmonic * (1 - penalty))
    return best


def cider_proxy(predictions: list[str], references: list[list[str]]) -> float:
    """Small-corpus CIDEr-style TF-IDF cosine average over 1-4 grams."""
    if not predictions:
        return 0.0
    document_frequency: list[Counter[tuple[str, ...]]] = [Counter() for _ in range(4)]
    for sample_references in references:
        for order in range(1, 5):
            grams = set()
            for reference in sample_references:
                grams.update(_ngrams(tokenize(reference), order))
            document_frequency[order - 1].update(grams)
    document_count = len(references)

    def vector(tokens: list[str], order: int) -> dict[tuple[str, ...], float]:
        counts = _ngrams(tokens, order)
        total = sum(counts.values()) or 1
        return {
            gram: (count / total)
            * math.log((document_count + 1) / (document_frequency[order - 1][gram] + 1))
            for gram, count in counts.items()
        }

    def cosine(left: dict, right: dict) -> float:
        numerator = sum(value * right.get(key, 0.0) for key, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    sample_scores: list[float] = []
    for prediction, sample_references in zip(predictions, references, strict=True):
        order_scores: list[float] = []
        for order in range(1, 5):
            candidate_vector = vector(tokenize(prediction), order)
            similarities = [
                cosine(candidate_vector, vector(tokenize(reference), order))
                for reference in sample_references
            ]
            order_scores.append(
                sum(similarities) / len(similarities) if similarities else 0.0
            )
        sample_scores.append(10.0 * sum(order_scores) / 4)
    return sum(sample_scores) / len(sample_scores)


def summarize_caption_proxies(
    predictions: list[str], references: list[list[str]]
) -> dict[str, float | str]:
    if len(predictions) != len(references):
        raise ValueError("Prediction/reference length mismatch")
    if not predictions:
        return {}
    rouge = [
        rouge_l_proxy(prediction, refs)
        for prediction, refs in zip(predictions, references, strict=True)
    ]
    meteor = [
        meteor_exact_proxy(prediction, refs)
        for prediction, refs in zip(predictions, references, strict=True)
    ]
    return {
        "protocol": "lexical_smoke_proxy_not_official_coco_caption",
        **corpus_bleu_proxy(predictions, references),
        "meteor_exact_proxy": sum(meteor) / len(meteor),
        "rouge_l_proxy": sum(rouge) / len(rouge),
        "cider_tfidf_proxy": cider_proxy(predictions, references),
    }
