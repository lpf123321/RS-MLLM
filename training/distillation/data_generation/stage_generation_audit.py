#!/usr/bin/env python3
"""Stage portable, image-free generation evidence for ModelScope.

The historical runs contain useful decisions and audit fields, but also local
absolute paths and review images derived from upstream datasets.  This tool
exports the structured evidence only, rewrites every local path, and verifies
that no home/data path leaked into the resulting bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


VRS_REVISION = "6cee2968fd752a6d51c6cb2d18dded2bc0baa218"
ABSOLUTE_PATH = re.compile(r"/(?:home|data\d*|users|root|mnt|tmp)/[^\s\"']+")
FORBIDDEN = ("/home/", "/data/", "/data2/", "/users/", "/root/", "/mnt/", "/tmp/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    marker = "Images_train/"
    if marker in normalized:
        member = normalized.rsplit(marker, 1)[1].split("#", 1)[0]
        return (
            "official://xiang709/VRSBench@"
            f"{VRS_REVISION}/Images_train.zip#Images_train/{member}"
        )
    name = Path(normalized.split("#", 1)[0]).name
    return f"omitted-local-artifact://{name or 'artifact'}"


def sanitize_text(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            pass
        else:
            return json.dumps(sanitize(decoded), ensure_ascii=False, sort_keys=True)
    if value.startswith("/"):
        return portable_path(value)
    return ABSOLUTE_PATH.sub(lambda match: portable_path(match.group(0)), value)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def read_json_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected object")
                yield value
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        yield value
    elif isinstance(value, list):
        for index, row in enumerate(value):
            if not isinstance(row, dict):
                raise ValueError(f"{path}[{index}]: expected object")
            yield row
    else:
        raise ValueError(f"{path}: expected object or object array")


def write_jsonl(rows: Iterable[dict[str, Any]], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(sanitize(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def stage_vrs_sft(parquet_path: Path, destination: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install pyarrow from requirements.txt") from exc
    table = pq.read_table(parquet_path)
    rows_by_family: dict[str, list[dict[str, Any]]] = {"vqa": [], "referring": [], "caption": []}
    for row in table.to_pylist():
        extra = row.get("extra_info") or {}
        sample_id = str(extra.get("sample_id") or "")
        family = next(
            (candidate for candidate in rows_by_family if f"_{candidate}_" in sample_id),
            "",
        )
        if not family:
            ability = str(row.get("ability") or "")
            family = "referring" if ability == "spatial_reasoning" else ability
        if family not in rows_by_family:
            raise ValueError(f"unexpected VRSBench family: {family!r}")
        source_id = str(extra.get("source_id") or "")
        if not source_id:
            raise ValueError("VRSBench record lacks source_id")
        portable = {
            "data_source": row.get("data_source"),
            "messages": row.get("messages"),
            "images": [{
                "image": (
                    "official://xiang709/VRSBench@"
                    f"{VRS_REVISION}/Images_train.zip#Images_train/{source_id}.png"
                )
            }],
            "ability": row.get("ability"),
            "extra_info": extra,
        }
        rows_by_family[family].append(portable)
    counts = {}
    for family, rows in rows_by_family.items():
        counts[family] = write_jsonl(rows, destination / f"{family}.jsonl")
    if counts != {"vqa": 1241, "referring": 526, "caption": 296}:
        raise ValueError(f"unexpected VRSBench SFT counts: {counts}")
    return {"records": sum(counts.values()), "families": counts, "source_sha256": sha256(parquet_path)}


def stage_files(source_root: Path, names: list[str], destination: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        source = source_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target_name = Path(name).name
        target = destination / target_name
        if source.suffix in {".json", ".jsonl"}:
            records = list(read_json_records(source))
            if source.suffix == ".jsonl":
                count = write_jsonl(records, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                payload: Any = sanitize(records[0]) if len(records) == 1 else sanitize(records)
                target.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                count = len(records)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(sanitize_text(source.read_text(encoding="utf-8")), encoding="utf-8")
            count = None
        result[target_name] = {"records": count, "source_sha256": sha256(source), "sha256": sha256(target)}
    return result


def verify_portable(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        leaks = [prefix for prefix in FORBIDDEN if prefix in text]
        if leaks:
            raise ValueError(f"absolute path leaked into {path}: {leaks}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vrsbench-sft-parquet", type=Path, required=True)
    parser.add_argument("--xlrs-general-root", type=Path, required=True)
    parser.add_argument("--vrsbench-grounding-root", type=Path, required=True)
    parser.add_argument("--xlrs-grounding-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    generation = output / "generation"
    if generation.exists():
        raise FileExistsError(f"refusing to replace existing audit bundle: {generation}")
    generation.mkdir(parents=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "contains_images": False,
        "contains_absolute_paths": False,
        "split": "train",
        "datasets": {},
    }
    manifest["datasets"]["vrsbench_sft_display"] = stage_vrs_sft(
        args.vrsbench_sft_parquet.expanduser().resolve(), generation / "vrsbench_sft"
    )
    manifest["datasets"]["xlrs_new"] = stage_files(
        args.xlrs_general_root.expanduser().resolve(),
        [
            "source_manifest.jsonl", "question_manifest.jsonl", "style_profile.json",
            "direct_annotations.jsonl", "accepted.jsonl", "coverage.jsonl",
            "run_config.json", "validation.json", "report.md",
        ],
        generation / "xlrs_new",
    )
    vrs_root = args.vrsbench_grounding_root.expanduser().resolve()
    manifest["datasets"]["vrsbench_new_grounding"] = stage_files(
        vrs_root,
        [
            "class_predictions.summary.json", "candidates.jsonl", "geometry_audit.json",
            "question_decisions_final.jsonl", "raw_review_audit.json",
            "source_annotation_gate_audit.json", "strict_semantic_audit.json",
            "small-grounding-test/summary.json", "small-grounding-test/synthetic_audit.jsonl",
            "small-grounding-test/synthetic_grounding.jsonl",
            "small-grounding-test/rejected_question_candidates.jsonl",
        ],
        generation / "vrsbench_new_grounding",
    )
    xlrs_root = args.xlrs_grounding_root.expanduser().resolve()
    manifest["datasets"]["xlrs_new_grounding"] = stage_files(
        xlrs_root,
        [
            "class_predictions.jsonl", "question_decisions.jsonl",
            "question_decisions_strict.jsonl", "final_clean/summary.json",
            "final_clean/synthetic_audit.jsonl", "final_clean/synthetic_grounding.jsonl",
            "final_clean/rejected.jsonl",
        ],
        generation / "xlrs_new_grounding",
    )

    readme = generation / "README.md"
    readme.write_text(
        "# Data-generation audit/display bundle\n\n"
        "This directory contains train-only structured evidence and no upstream images. "
        "Download the official datasets with the repository scripts, then use the "
        "`sam3_pipeline` finalize commands to regenerate visual galleries. MME-new is not "
        "included here because the historical 3,736-example run no longer has its original "
        "source/question/style/audit intermediates; its frozen training JSON remains under "
        "`general/`. The OPSD audit data is published in RS-MLLM-Self-Evolution-Data.\n",
        encoding="utf-8",
    )
    manifest_path = generation / "AUDIT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_portable(generation)

    asset_manifest = output / "ASSET_MANIFEST.json"
    if asset_manifest.is_file():
        value = json.loads(asset_manifest.read_text(encoding="utf-8"))
        value["generation_audit"] = {
            "path": "generation/AUDIT_MANIFEST.json",
            "sha256": sha256(manifest_path),
            "contains_images": False,
        }
        asset_manifest.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    sums = output / "SHA256SUMS"
    files = sorted(path for path in output.rglob("*") if path.is_file() and path != sums)
    with sums.open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n")
    counts = Counter()
    for path in generation.rglob("*"):
        if path.is_file():
            counts[path.suffix or "no_suffix"] += 1
    print(json.dumps({"output": str(generation), "files": sum(counts.values()), "by_suffix": counts}, default=dict))


if __name__ == "__main__":
    main()
