#!/usr/bin/env python3
"""Score fixed-ID subsets from aligned historical prediction files."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evalsets import mme, vrsbench, xlrs, xlrs_grounding


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = list(path.glob("*predictions*.json"))
    if not files:
        raise FileNotFoundError(f"no prediction JSON found under {path}")

    def key(file: Path) -> tuple[int, str]:
        match = re.search(r"part(\d+)", file.name)
        return (int(match.group(1)) if match else -1, file.name)

    return sorted(files, key=key)


def load_predictions(path: Path) -> tuple[list[dict], list[dict[str, object]]]:
    rows: list[dict] = []
    provenance = []
    for file in prediction_files(path):
        value = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"{file}: expected a JSON list")
        rows.extend(value)
        provenance.append(
            {"path": str(file.resolve()), "records": len(value), "sha256": sha256(file)}
        )
    return rows, provenance


def score_dataset(
    module,
    subset: Path,
    ids_path: Path,
    predictions_path: Path,
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    samples = module.load_data(str(subset))
    identifiers = json.loads(ids_path.read_text(encoding="utf-8"))
    predictions, provenance = load_predictions(predictions_path)
    if len(samples) != len(identifiers):
        raise ValueError(f"{subset}: samples/ids length mismatch")
    selected = []
    for item in identifiers:
        index = int(item["eligible_index"])
        if index < 0 or index >= len(predictions):
            raise IndexError(
                f"{predictions_path}: eligible_index {index} outside {len(predictions)} rows"
            )
        record = predictions[index]
        if not isinstance(record, dict) or "prediction" not in record:
            raise ValueError(f"{predictions_path}: malformed prediction at {index}")
        selected.append(str(record["prediction"]))
    grouped: dict[str, dict[str, list]] = {}
    for sample, prediction in zip(samples, selected, strict=True):
        task = sample["task"]
        entry = grouped.setdefault(task, {"references": [], "predictions": []})
        entry["references"].append(sample["references"])
        entry["predictions"].append(prediction)
    results = {}
    for task, values in grouped.items():
        task_result = {}
        for metric in module.TASK_METRICS.get(task, []):
            task_result.update(
                metric.compute(values["references"], values["predictions"])
            )
        task_result["samples"] = len(values["predictions"])
        results[task] = task_result
    return results, provenance


def required(value: Path | None, flag: str) -> Path:
    if value is None:
        raise ValueError(f"{flag} is required for this kind")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("general", "grounding"), required=True)
    parser.add_argument("--subsets", type=Path, required=True)
    parser.add_argument("--mme-predictions", type=Path)
    parser.add_argument("--xlrs-predictions", type=Path)
    parser.add_argument("--vrsbench-predictions", type=Path, required=True)
    parser.add_argument("--xlrs-grounding-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.kind == "general":
        specs = {
            "mme": (
                mme,
                args.subsets / "mme_1000.jsonl",
                args.subsets / "mme_1000.ids.json",
                required(args.mme_predictions, "--mme-predictions"),
            ),
            "xlrs": (
                xlrs,
                args.subsets / "xlrs_1000.jsonl",
                args.subsets / "xlrs_1000.ids.json",
                required(args.xlrs_predictions, "--xlrs-predictions"),
            ),
            "vrsbench": (
                vrsbench,
                args.subsets / "vrsbench_vqa_1000.jsonl",
                args.subsets / "vrsbench_vqa_1000.ids.json",
                args.vrsbench_predictions,
            ),
        }
    else:
        specs = {
            "vrsbench": (
                vrsbench,
                args.subsets / "vrsbench_referring_1000.jsonl",
                args.subsets / "vrsbench_referring_1000.ids.json",
                args.vrsbench_predictions,
            ),
            "xlrs_grounding": (
                xlrs_grounding,
                args.subsets / "xlrs_grounding_1000.jsonl",
                args.subsets / "xlrs_grounding_1000.ids.json",
                required(
                    args.xlrs_grounding_predictions,
                    "--xlrs-grounding-predictions",
                ),
            ),
        }

    results: dict[str, object] = {}
    sources: dict[str, object] = {}
    for name, (module, subset, ids_path, predictions_path) in specs.items():
        if not subset.is_file() or not ids_path.is_file():
            raise FileNotFoundError(f"missing fixed subset or ids for {name}")
        results[name], sources[name] = score_dataset(
            module, subset, ids_path, predictions_path
        )
    results["_provenance"] = {
        "kind": args.kind,
        "subsets_manifest": sha256(args.subsets / "manifest.json"),
        "prediction_files": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
