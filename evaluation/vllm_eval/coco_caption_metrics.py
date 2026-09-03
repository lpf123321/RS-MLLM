"""Official COCO Caption scorers with explicit raw/report scales.

The project has two intentionally different caption metric families:
``evaluation.metrics`` mirrors the report's local implementation, while this
module uses the ``pycocoevalcap==1.2`` PTB-tokenized reference implementation.
The distinction is part of the output schema; callers must not relabel either
family as the other one.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
from typing import Any

CORE_NAMES = ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr")
RAW_SCALES = {
    "Bleu_1..4": "0-1",
    "METEOR": "0-1",
    "ROUGE_L": "0-1",
    "CIDEr": "0-10",
    "SPICE": "0-1",
}


def _unavailable(
    error: str, *, sample_count: int, include_spice: bool
) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "protocol": "pycocoevalcap==1.2",
        "sample_count": sample_count,
        "raw_score_scales": RAW_SCALES,
        "error": error,
        "raw": {},
        "percent": {},
        "spice_requested": include_spice,
    }


def _close_meteor(scorer: Any) -> None:
    """Stop the Java Meteor worker, including when scoring raises."""
    process = getattr(scorer, "meteor_p", None)
    if process is None:
        return
    with contextlib.suppress(Exception):
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
    with contextlib.suppress(Exception):
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def _validate_inputs(predictions: list[str], references: list[list[str]]) -> None:
    if len(predictions) != len(references):
        raise ValueError("Prediction/reference length mismatch")
    if not predictions:
        raise ValueError("Caption metric input is empty")
    for index, (prediction, sample_references) in enumerate(
        zip(predictions, references, strict=True)
    ):
        if not isinstance(prediction, str):
            raise TypeError(f"prediction {index} is not a string")
        if not sample_references or not all(
            isinstance(reference, str) and reference.strip()
            for reference in sample_references
        ):
            raise ValueError(f"caption sample {index} has no non-empty references")


def _coco_inputs(
    predictions: list[str], references: list[list[str]]
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    gts = {
        str(index): [
            {"image_id": str(index), "caption": reference}
            for reference in sample_references
        ]
        for index, sample_references in enumerate(references)
    }
    results = {
        str(index): [{"image_id": str(index), "caption": prediction}]
        for index, prediction in enumerate(predictions)
    }
    return gts, results


def _scale_report_percent(raw: dict[str, float]) -> dict[str, float]:
    # The report prints every caption metric as a percentage, including CIDEr-D
    # (whose native COCO range is 0--10). Keep the unscaled values beside this
    # representation so downstream code cannot confuse units.
    return {name: value * 100.0 for name, value in raw.items()}


def compute_official_coco_caption(
    predictions: list[str],
    references: list[list[str]],
    *,
    include_spice: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Compute official COCO Caption scores for one prediction set.

    ``strict=False`` turns missing optional runtime pieces (for example the
    Meteor Java process or Stanford SPICE model) into explicit status/error
    fields instead of hiding them or aborting an otherwise useful report.
    ``strict=True`` re-raises those errors for CI/acceptance paths.
    """
    _validate_inputs(predictions, references)
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        return _unavailable(
            f"{type(exc).__name__}: {exc}",
            sample_count=len(predictions),
            include_spice=include_spice,
        )

    raw_gts, raw_results = _coco_inputs(predictions, references)
    try:
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(raw_gts)
        results = tokenizer.tokenize(raw_results)
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        return _unavailable(
            f"PTBTokenizer {type(exc).__name__}: {exc}",
            sample_count=len(predictions),
            include_spice=include_spice,
        )

    raw: dict[str, float] = {}
    errors: dict[str, str] = {}

    try:
        bleu, _ = Bleu(4).compute_score(gts, results, verbose=0)
        raw.update(
            {f"Bleu_{index + 1}": float(value) for index, value in enumerate(bleu)}
        )
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        errors["BLEU"] = f"{type(exc).__name__}: {exc}"

    meteor_scorer = None
    try:
        meteor_scorer = Meteor()
        meteor, _ = meteor_scorer.compute_score(gts, results)
        raw["METEOR"] = float(meteor)
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        errors["METEOR"] = f"{type(exc).__name__}: {exc}"
    finally:
        if meteor_scorer is not None:
            _close_meteor(meteor_scorer)

    try:
        raw["ROUGE_L"] = float(Rouge().compute_score(gts, results)[0])
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        errors["ROUGE_L"] = f"{type(exc).__name__}: {exc}"

    try:
        raw["CIDEr"] = float(Cider().compute_score(gts, results)[0])
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        errors["CIDEr"] = f"{type(exc).__name__}: {exc}"

    if include_spice:
        try:
            from pycocoevalcap.spice.spice import Spice

            raw["SPICE"] = float(Spice().compute_score(gts, results)[0])
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            errors["SPICE"] = f"{type(exc).__name__}: {exc}"

    if not raw:
        status = "unavailable"
    elif errors:
        status = "partial"
    else:
        status = "ok"
    try:
        version = importlib.metadata.version("pycocoevalcap")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    result: dict[str, Any] = {
        "status": status,
        "protocol": "pycocoevalcap==1.2",
        "evaluator_version": version,
        "tokenizer": "Stanford PTBTokenizer, -lowerCase",
        "sample_count": len(predictions),
        "raw_score_scales": RAW_SCALES,
        "raw": raw,
        "percent": _scale_report_percent(raw),
        "spice_requested": include_spice,
    }
    if errors:
        result["errors"] = errors
    return result
