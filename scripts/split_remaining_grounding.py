#!/usr/bin/env python3
"""Split unfinished Grounding samples without rerunning successful attempts.

The evaluator writes one or more attempt rows per sample.  This utility keeps
every sample whose attempt is successful, validates that attempt rows still
match the source manifest, and writes the remaining rows in round-robin
shards.  Shards are written beside the source manifest so its relative image
paths retain exactly the same resolution semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _sample_id(row: dict[str, Any], *, path: Path) -> str:
    identifier = row.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(f"{path}: every row must have a non-empty string id")
    return identifier


def _fingerprint(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _successful_attempt(row: dict[str, Any]) -> bool:
    # This utility is intentionally Grounding-specific: unlike XLRS Caption,
    # a truncated Grounding generation is never an accepted result.
    return not row.get("error") and not row.get("generation_truncated", False)


def split_remaining(
    manifest_path: Path, attempts_path: Path, *, shard_count: int
) -> dict[str, Any]:
    if shard_count < 1:
        raise ValueError("--shards must be >= 1")
    manifest_rows = _read_jsonl(manifest_path)
    manifest_by_id: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        identifier = _sample_id(row, path=manifest_path)
        if identifier in manifest_by_id:
            raise ValueError(f"duplicate manifest id: {identifier}")
        manifest_by_id[identifier] = row

    successful: set[str] = set()
    seen_attempts: set[str] = set()
    for row in _read_jsonl(attempts_path):
        raw_sample = row.get("sample")
        if not isinstance(raw_sample, dict):
            raise ValueError("attempt row has no object 'sample' field")
        identifier = _sample_id(raw_sample, path=attempts_path)
        expected = manifest_by_id.get(identifier)
        if expected is None:
            raise ValueError(f"attempt has unknown manifest id: {identifier}")
        if _fingerprint(raw_sample) != _fingerprint(expected):
            raise ValueError(f"attempt sample differs from manifest: {identifier}")
        seen_attempts.add(identifier)
        if _successful_attempt(row):
            successful.add(identifier)

    remaining = [
        row for row in manifest_rows if _sample_id(row, path=manifest_path) not in successful
    ]
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for index, row in enumerate(remaining):
        shards[index % shard_count].append(row)

    return {
        "manifest_rows": manifest_rows,
        "successful_ids": successful,
        "attempt_ids": seen_attempts,
        "remaining": remaining,
        "shards": shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument(
        "--prefix",
        default=".xlrs_grounding_remaining_shard",
        help="output prefix beside --manifest (default: %(default)s)",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    attempts_path = args.attempts.expanduser().resolve()
    result = split_remaining(manifest_path, attempts_path, shard_count=args.shards)
    output_paths: list[Path] = []
    for index, rows in enumerate(result["shards"]):
        output_path = manifest_path.parent / f"{args.prefix}{index}.jsonl"
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite {output_path}")
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_paths.append(output_path)

    metadata_path = manifest_path.parent / f"{args.prefix}.json"
    if metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite {metadata_path}")
    metadata = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "attempts": str(attempts_path),
        "attempts_sha256": _sha256(attempts_path),
        "total_samples": len(result["manifest_rows"]),
        "attempt_rows_with_known_ids": len(result["attempt_ids"]),
        "successful_samples": len(result["successful_ids"]),
        "remaining_samples": len(result["remaining"]),
        "shards": [
            {"path": str(path), "samples": len(rows)}
            for path, rows in zip(output_paths, result["shards"])
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
