#!/usr/bin/env python3
"""Check fixed-subset initial -> short -> historical-final training trends."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PATHS = {
    "general": {
        "mme_mcq": ("mme", "vqa", "MCQ_Accuracy"),
        "xlrs_mcq": ("xlrs", "vqa", "MCQ_Accuracy"),
        "vrsbench_vqa": ("vrsbench", "vqa", "Accuracy"),
    },
    "grounding": {
        "vrsbench_acc_0_5": ("vrsbench", "referring", "Acc@0.5"),
        "xlrs_acc_0_5": ("xlrs_grounding", "referring", "Acc@0.5"),
    },
}


def read_metric(document: dict, path: tuple[str, ...]) -> float:
    value: object = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric {'.'.join(path)}: {result}")
    return result


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(PATHS), required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--short", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-drop", type=float, default=0.03)
    args = parser.parse_args()

    documents = {
        "initial": load(args.initial),
        "short": load(args.short),
        "final": load(args.final),
    }
    vectors = {
        endpoint: {
            name: read_metric(document, metric_path)
            for name, metric_path in PATHS[args.kind].items()
        }
        for endpoint, document in documents.items()
    }
    initial_distance = sum(
        abs(vectors["initial"][name] - vectors["final"][name])
        for name in PATHS[args.kind]
    ) / len(PATHS[args.kind])
    short_distance = sum(
        abs(vectors["short"][name] - vectors["final"][name])
        for name in PATHS[args.kind]
    ) / len(PATHS[args.kind])
    drops = {
        name: vectors["initial"][name] - vectors["short"][name]
        for name in PATHS[args.kind]
    }
    failures = []
    if not short_distance < initial_distance:
        failures.append(
            f"short distance {short_distance:.6f} is not below initial "
            f"distance {initial_distance:.6f}"
        )
    for name, drop in drops.items():
        if drop > args.max_drop:
            failures.append(f"{name} dropped {drop:.6f} (> {args.max_drop:.6f})")
    parseable = None
    if args.kind == "grounding":
        parseable = read_metric(
            documents["short"],
            ("xlrs_grounding", "referring", "parseable"),
        )
        if parseable != 1.0:
            failures.append(f"XLRS parseable rate is {parseable:.6f}, expected 1.0")

    report = {
        "kind": args.kind,
        "vectors": vectors,
        "mean_absolute_distance": {
            "initial_to_final": initial_distance,
            "short_to_final": short_distance,
        },
        "drops_from_initial": drops,
        "max_allowed_drop": args.max_drop,
        "xlrs_parseable": parseable,
        "passed": not failures,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
