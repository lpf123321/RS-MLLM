from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .io import read_jsonl, sha256_file, write_json


NEGATIVE_TYPES = ("generic", "random", "semantic", "counterfactual", "crop")


def _validate_coverage(manifest: Path, split_dir: Path, threshold: float) -> dict[str, int]:
    coco = json.loads((split_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    expected = {int(row["id"]) for row in coco["images"]}
    rows = list(read_jsonl(manifest))
    covered = {int(row["image_id"]) for row in rows}
    if covered != expected:
        missing = sorted(expected - covered)
        extra = sorted(covered - expected)
        raise ValueError(f"Negative coverage mismatch: missing={missing[:10]} extra={extra[:10]}")
    for row in rows:
        if row.get("accepted") is not True or float(row.get("absent_probability", 0.0)) < threshold:
            raise ValueError(f"Unverified negative row in {manifest}: image_id={row.get('image_id')}")
    counts: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("negative_type", "unknown"))
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def ensure_verified_negatives(
    config: dict[str, Any], sam_data: dict[str, Any], split_name: str, device: int = 0,
) -> dict[str, Any]:
    """Generate, Qwen3-VL-8B verify, freeze, and audit one negative per SAM sample."""
    manifest = Path(sam_data["negative_manifest"])
    split_dir = Path(sam_data["split_dir"])
    negative_cfg = config["negative_generation"]
    threshold = float(negative_cfg.get("absent_probability_threshold", 0.9))
    if manifest.is_file() and manifest.stat().st_size:
        counts = _validate_coverage(manifest, split_dir, threshold)
        return {**sam_data, "negative_types": counts}

    method_root = Path(config["validated_sam_method"]["root"]).resolve()
    script = method_root / "scripts" / "build_negative_manifests.py"
    output_root = manifest.parent / f"{split_name}_negative_repair"
    output_root.mkdir(parents=True, exist_ok=True)
    seed = int(negative_cfg.get("seed", config["seed"]))
    verifier = Path(negative_cfg["verifier_model"]).resolve()
    methods = ("generic", "random") if split_name == "train" else NEGATIVE_TYPES
    verified_paths = []
    for method in methods:
        candidates = output_root / f"{method}.candidates.jsonl"
        verified = output_root / f"{method}.verified.jsonl"
        if not candidates.is_file():
            subprocess.run([
                sys.executable, str(script), "generate", "--split-dir", str(split_dir),
                "--output", str(candidates), "--method", method, "--per-image", "3",
                "--seed", str(seed),
            ], check=True, cwd=method_root)
        if not verified.is_file():
            subprocess.run([
                sys.executable, str(script), "verify", "--input", str(candidates),
                "--output", str(verified), "--model", str(verifier), "--device", str(device),
                "--threshold", str(threshold), "--batch-size", str(negative_cfg.get("batch_size", 8)),
            ], check=True, cwd=method_root)
        verified_paths.append(verified)

    priority_flag = "--balanced-one-per-image"
    priority = list(methods)
    subprocess.run([
        sys.executable, str(script), "compose", "--input", *map(str, verified_paths),
        "--output", str(manifest), priority_flag, *priority, "--fallback-type", "generic",
    ], check=True, cwd=method_root)
    counts = _validate_coverage(manifest, split_dir, threshold)
    verifier_config = verifier / "config.json"
    metadata = {
        "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
        "split_coco_sha256": sha256_file(split_dir / "_annotations.coco.json"),
        "seed": seed, "threshold": threshold, "negative_types": counts,
        "verifier_model": str(verifier),
        "verifier_config_sha256": sha256_file(verifier_config) if verifier_config.is_file() else None,
        "coverage": "100%", "uses_unverified_negatives": False,
    }
    write_json(manifest.with_suffix(".metadata.json"), metadata)
    return {**sam_data, "negative_types": counts, "negative_metadata": metadata}


def assert_no_negative_manifest_leak(train_manifest: str | Path, valid_manifest: str | Path) -> None:
    train_ids = {str(row.get("source_id")) for row in read_jsonl(train_manifest)}
    valid_ids = {str(row.get("source_id")) for row in read_jsonl(valid_manifest)}
    overlap = sorted(train_ids & valid_ids)
    if overlap:
        raise ValueError(f"Train/valid negative manifests leak {len(overlap)} source IDs: {overlap[:10]}")
