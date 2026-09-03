#!/usr/bin/env python3
"""Rescore an existing predictions.jsonl without repeating model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scoring import prediction_letter_distribution, summarize_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    predictions_path = result_dir / "predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"No predictions found in {predictions_path}")
    official, clean = summarize_predictions(rows)
    official["prediction_letter_distribution"] = prediction_letter_distribution(rows)
    clean["prediction_letter_distribution"] = prediction_letter_distribution(rows)
    payload = {
        "source_predictions": str(predictions_path),
        "official": official,
        "clean": clean,
    }
    output = result_dir / "reported_rescore.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
