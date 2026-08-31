"""Deterministic coverage-stratified sampling helpers."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def _rank(seed: int, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()


def coverage_stratified_sample(
    records: Iterable[T],
    *,
    sample_id: Callable[[T], str],
    group: Callable[[T], str],
    rate: float,
    seed: int,
) -> list[T]:
    """Select ceil(rate * group_size), with a minimum of one sample per non-empty group."""
    if not 0 < rate <= 1:
        raise ValueError(f"rate must be in (0, 1], got {rate}")
    grouped: dict[str, list[T]] = defaultdict(list)
    seen: set[str] = set()
    for record in records:
        identifier = sample_id(record)
        if identifier in seen:
            raise ValueError(f"Duplicate sample id: {identifier}")
        seen.add(identifier)
        grouped[group(record)].append(record)

    selected: list[T] = []
    for group_name in sorted(grouped):
        candidates = grouped[group_name]
        target = max(1, math.ceil(len(candidates) * rate))
        candidates.sort(key=lambda item: _rank(seed, sample_id(item)))
        selected.extend(candidates[:target])
    return sorted(selected, key=lambda item: (group(item), sample_id(item)))
