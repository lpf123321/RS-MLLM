#!/usr/bin/env python3
"""Append the exact weighted General Exp7 data to a portable dataset bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = "8711391dec337dd5117602eb122b9bdcf8994831458202979d699662b47104cb"
EXPECTED = 15480
PREFIX = 6816


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vrsbench-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    root = args.dataset_root.expanduser().resolve()
    vrs_root = args.vrsbench_root.expanduser().resolve()
    if sha256(source) != SOURCE_SHA256:
        raise ValueError(f"unexpected Exp7 source SHA-256: {sha256(source)}")
    rows = load_json(source)
    if len(rows) != EXPECTED:
        raise ValueError(f"expected {EXPECTED} records, got {len(rows)}")
    if [row.get("loss_weight") for row in rows[:PREFIX]] != [2.0] * PREFIX:
        raise ValueError("Exp7 prefix must have loss_weight=2")
    if [row.get("loss_weight") for row in rows[PREFIX:]] != [1.0] * (EXPECTED - PREFIX):
        raise ValueError("Exp7 VRSBench suffix must have loss_weight=1")

    exp3 = load_json(root / "general" / "exp3_mme3736_xlrs3080.json")
    if len(exp3) != PREFIX:
        raise ValueError("portable Exp3 prefix has unexpected length")
    portable = json.loads(json.dumps(rows, ensure_ascii=False))
    for index, (source_row, prefix_row) in enumerate(zip(rows, exp3)):
        if source_row.get("id") != prefix_row.get("id") or source_row.get("conversations") != prefix_row.get("conversations"):
            raise ValueError(f"Exp7/Exp3 prefix mismatch at row {index}")
        portable[index]["image"] = prefix_row["image"]

    index_path = root / "IMAGE_INDEX.jsonl"
    index_rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_hash = {row["sha256"]: row for row in index_rows}
    for row in portable[PREFIX:]:
        rewritten = []
        for old in row["image"]:
            candidate = Path(old)
            if not candidate.is_file():
                candidate = vrs_root / Path(old).name
            if not candidate.is_file():
                matches = list(vrs_root.rglob(Path(old).name))
                if len(matches) != 1:
                    raise FileNotFoundError(f"cannot uniquely resolve {old}: {len(matches)} matches")
                candidate = matches[0]
            digest = sha256(candidate)
            suffix = candidate.suffix.lower() or ".bin"
            relative = f"images/{digest[:24]}{suffix}"
            rewritten.append(relative)
            if digest not in by_hash:
                try:
                    upstream = candidate.resolve().relative_to(vrs_root).as_posix()
                except ValueError:
                    upstream = candidate.name
                entry = {"path": relative, "sha256": digest, "bytes": candidate.stat().st_size, "upstream_dataset": "vrsbench", "upstream_path": upstream}
                by_hash[digest] = entry
                index_rows.append(entry)
        row["image"] = rewritten

    destination = root / "general" / "exp7_weighted15480.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(portable, ensure_ascii=False) + "\n", encoding="utf-8")
    index_rows.sort(key=lambda row: row["path"])
    index_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in index_rows), encoding="utf-8")

    manifest_path = root / "ASSET_MANIFEST.json"
    manifest = load_json(manifest_path)
    manifest["datasets"]["general_exp7"] = {
        "path": "general/exp7_weighted15480.json", "schema": "general_mixed",
        "records": EXPECTED, "retained_source_sha256": SOURCE_SHA256,
        "portable_sha256": sha256(destination),
        "loss_weights": {"old_mme_plus_old_xlrs": 2.0, "vrsbench_vqa_train": 1.0},
    }
    manifest["unique_image_blobs"] = len({row["sha256"] for row in index_rows})
    manifest["unique_source_paths"] = len(index_rows)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS" and ".git" not in path.parts)
    (root / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    print(json.dumps(manifest["datasets"]["general_exp7"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
