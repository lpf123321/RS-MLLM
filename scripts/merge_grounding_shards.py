#!/usr/bin/env python3
"""Merge partial Grounding attempts and parallel shard outputs safely.

Successful rows from the base attempt log are retained.  Each shard may be a
finalized run directory or a JSONL file; only one successful row per sample is
accepted, sample definitions are checked against the full manifest, and the
final output is emitted in manifest order.  The evaluator's normal ``finalize``
path then regenerates summaries and metric reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"row at {path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _id(row: dict[str, Any], *, source: Path) -> str:
    value = row.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: every sample id must be a non-empty string")
    return value


def _fingerprint(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _successful(row: dict[str, Any]) -> bool:
    return not row.get("error") and not row.get("generation_truncated", False)


def _source_rows(path: Path) -> tuple[Path, list[dict[str, Any]]]:
    path = path.expanduser().resolve()
    if path.is_dir():
        predictions = path / "predictions.jsonl"
        attempts = path / "prediction_attempts.jsonl"
        if predictions.is_file():
            path = predictions
        elif attempts.is_file():
            path = attempts
        else:
            raise FileNotFoundError(f"no predictions or attempts file in {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, _read_jsonl(path)


def _select_successful_rows(
    source: Path,
    rows: list[dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_sample = row.get("sample")
        if not isinstance(raw_sample, dict):
            raise ValueError(f"{source}: attempt row has no object sample")
        identifier = _id(raw_sample, source=source)
        expected = manifest_by_id.get(identifier)
        if expected is None:
            raise ValueError(f"{source}: unknown manifest id {identifier}")
        if _fingerprint(raw_sample) != _fingerprint(expected):
            raise ValueError(f"{source}: sample definition differs for {identifier}")
        if _successful(row):
            selected[identifier] = row
    return selected


def _reference_config(reference_run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_run = reference_run.expanduser().resolve()
    config_path = reference_run / "run_config.json"
    manifest_path = reference_run / "run_manifest.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = _read_json(config_path)
    model_info: dict[str, Any] = {}
    if manifest_path.is_file():
        model_info = _read_json(manifest_path).get("model", {})
        if not isinstance(model_info, dict):
            raise ValueError(f"model field is not an object: {manifest_path}")
    return config, model_info


class _StaticAdapter:
    def __init__(self, peak_memory_mb: float = 0.0) -> None:
        self._peak_memory_mb = peak_memory_mb

    def peak_memory_mb(self) -> float:
        return self._peak_memory_mb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--base-attempts",
        type=Path,
        required=True,
        help="partial evaluator prediction_attempts.jsonl (may be empty)",
    )
    parser.add_argument(
        "--shard",
        type=Path,
        action="append",
        required=True,
        help="shard run directory or predictions/attempts JSONL; repeat per shard",
    )
    parser.add_argument(
        "--reference-run",
        type=Path,
        required=True,
        help="one run directory providing run_config.json and model metadata",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest_rows = _read_jsonl(manifest_path)
    manifest_by_id: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        identifier = _id(row, source=manifest_path)
        if identifier in manifest_by_id:
            raise ValueError(f"duplicate manifest id: {identifier}")
        manifest_by_id[identifier] = row

    base_path, base_rows = _source_rows(args.base_attempts)
    selected = _select_successful_rows(base_path, base_rows, manifest_by_id)
    provenance: list[dict[str, Any]] = [
        {
            "source": str(base_path),
            "sha256": _sha256(base_path),
            "successful_samples": len(selected),
        }
    ]
    for shard_arg in args.shard:
        shard_path, shard_rows = _source_rows(shard_arg)
        shard_selected = _select_successful_rows(
            shard_path, shard_rows, manifest_by_id
        )
        overlap = sorted(set(selected) & set(shard_selected))
        if overlap:
            raise ValueError(
                f"successful sample overlap between sources at {shard_path}: "
                + ", ".join(overlap[:10])
            )
        selected.update(shard_selected)
        provenance.append(
            {
                "source": str(shard_path),
                "sha256": _sha256(shard_path),
                "successful_samples": len(shard_selected),
            }
        )

    missing = [identifier for identifier in manifest_by_id if identifier not in selected]
    if missing:
        raise ValueError(
            f"cannot merge incomplete results: {len(missing)} samples missing; "
            f"first ids: {missing[:10]}"
        )

    # Rescore from the embedded sample/prediction rather than trusting a shard's
    # serialized score object.  Importing these modules requires the evaluator
    # environment, which is intentional for a final acceptance operation.
    import sys

    eval_dir = Path(__file__).resolve().parents[1] / "evaluation" / "vllm_eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from schema import Sample
    from scoring import score_prediction
    from vision_opd_eval import finalize

    completed: dict[str, dict[str, Any]] = {}
    for identifier, row in selected.items():
        sample = Sample.from_dict(manifest_by_id[identifier], manifest_dir=manifest_path.parent)
        normalized = dict(row)
        normalized["sample"] = sample.to_dict()
        prediction = normalized.get("prediction")
        if not isinstance(prediction, str):
            raise ValueError(f"prediction is not a string for {identifier}")
        normalized["score"] = score_prediction(sample, prediction)
        completed[identifier] = normalized

    config, model_info = _reference_config(args.reference_run)
    config["manifest_path"] = str(manifest_path)
    config["manifest_sha256"] = _sha256(manifest_path)
    config["merge"] = {
        "method": "base_successes_plus_disjoint_parallel_shards",
        "sources": provenance,
        "sample_count": len(manifest_rows),
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    finalize(
        output_dir,
        [
            Sample.from_dict(row, manifest_dir=manifest_path.parent)
            for row in manifest_rows
        ],
        manifest_rows,
        completed,
        config,
        model_info,
        0.0,
        _StaticAdapter(),
    )
    metadata_path = output_dir / "merge_manifest.json"
    metadata_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "sample_count": len(manifest_rows),
                "successful_samples": len(selected),
                "sources": provenance,
                "group_counts": dict(
                    sorted(
                        Counter(
                            f"{row.get('dataset')}/{row.get('subtask')}"
                            for row in manifest_rows
                        ).items()
                    )
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "merged", "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
