"""Prepare, merge, and verify the fixed Vision-OPD four-benchmark evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from sampling import coverage_stratified_sample
from schema import Sample
from scoring import prediction_letter_distribution, summarize_predictions

EXPECTED_TOTAL = 71_665
EXPECTED_SUBTASKS = 33
SEED = 20260727
OFFICIAL_SPLITS = {
    "vrsbench": "eval",
    "mme_realworld_rs": "test",
    "xlrs_bench_lite": "test",
    "levir_cc": "test",
}
SHARDS = (
    "vrs_caption",
    "vrs_vqa",
    "vrs_refer",
    "mme",
    "xlrs",
    "levir",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def group_name(sample: Sample) -> str:
    return f"{sample.dataset}/{sample.subtask}"


def shard_name(sample: Sample) -> str:
    if sample.dataset == "vrsbench":
        return {
            "caption": "vrs_caption",
            "open_vqa": "vrs_vqa",
            "bbox": "vrs_refer",
        }[sample.task_type]
    return {
        "mme_realworld_rs": "mme",
        "xlrs_bench_lite": "xlrs",
        "levir_cc": "levir",
    }[sample.dataset]


def read_official_samples(source_manifest: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[str] = set()
    for raw in jsonl_rows(source_manifest):
        dataset = raw.get("dataset")
        split = raw.get("split")
        if OFFICIAL_SPLITS.get(dataset) != split:
            continue
        sample = Sample.from_dict(raw, manifest_dir=source_manifest.parent)
        sample.split = split
        sample.source = dict(raw.get("source", {}))
        sample.validate(check_images=True)
        if sample.id in seen:
            raise ValueError(f"Duplicate official sample ID: {sample.id}")
        seen.add(sample.id)
        samples.append(sample)
    if len(samples) != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL} official samples, found {len(samples)}"
        )
    groups = {group_name(sample) for sample in samples}
    if len(groups) != EXPECTED_SUBTASKS:
        raise ValueError(
            f"Expected {EXPECTED_SUBTASKS} official subtasks, found {len(groups)}"
        )
    for sample in samples:
        if OFFICIAL_SPLITS[sample.dataset] != sample.split:
            raise ValueError(f"Training/validation leakage: {sample.id} split={sample.split}")
    return samples


def sample_counts(samples: list[Sample]) -> dict[str, int]:
    return dict(sorted(Counter(group_name(sample) for sample in samples).items()))


def status_counts(samples: list[Sample]) -> dict[str, int]:
    return dict(sorted(Counter(sample.clean_status for sample in samples).items()))


def write_testset(
    directory: Path,
    samples: list[Sample],
    *,
    source_counts: dict[str, int],
    source_manifest: Path,
    selection: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    manifest = directory / "testset.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    summary = {
        "selection": selection,
        "seed": SEED,
        "sample_count": len(samples),
        "covered_subtasks": len(sample_counts(samples)),
        "source_counts": source_counts,
        "selected_counts": sample_counts(samples),
        "status_counts": status_counts(samples),
        "official_split_policy": OFFICIAL_SPLITS,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "manifest_sha256": sha256_file(manifest),
    }
    (directory / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in directory.iterdir():
        path.chmod(0o444)
    directory.chmod(0o555)


def exact_stratified_sample(
    samples: list[Sample], target: int, *, seed: int
) -> list[Sample]:
    if target < EXPECTED_SUBTASKS or target > len(samples):
        raise ValueError(f"Invalid exact sample target: {target}")
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[group_name(sample)].append(sample)
    selected: list[Sample] = []
    remaining: list[tuple[bytes, Sample]] = []
    for name in sorted(grouped):
        ranked = sorted(
            grouped[name],
            key=lambda sample: hashlib.sha256(
                f"{seed}:{sample.id}".encode("utf-8")
            ).digest(),
        )
        selected.append(ranked[0])
        remaining.extend(
            (
                hashlib.sha256(f"{seed}:remaining:{sample.id}".encode("utf-8")).digest(),
                sample,
            )
            for sample in ranked[1:]
        )
    selected.extend(sample for _, sample in sorted(remaining)[: target - len(selected)])
    selected_ids = {sample.id for sample in selected}
    return [sample for sample in samples if sample.id in selected_ids]


def prepare(source_manifest: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    samples = read_official_samples(source_manifest)
    counts = sample_counts(samples)
    output_root.mkdir(parents=True)
    write_testset(
        output_root / "full",
        samples,
        source_counts=counts,
        source_manifest=source_manifest,
        selection="all official eval/test splits",
    )
    by_shard: dict[str, list[Sample]] = {name: [] for name in SHARDS}
    for sample in samples:
        by_shard[shard_name(sample)].append(sample)
    for name in SHARDS:
        shard_samples = by_shard[name]
        shard_counts = sample_counts(shard_samples)
        write_testset(
            output_root / "shards" / name,
            shard_samples,
            source_counts=shard_counts,
            source_manifest=source_manifest,
            selection=f"full shard: {name}",
        )
    smoke = coverage_stratified_sample(
        samples,
        sample_id=lambda sample: sample.id,
        group=group_name,
        rate=0.001,
        seed=SEED,
    )
    if len(smoke) != 92:
        raise ValueError(f"Expected 92-sample smoke, found {len(smoke)}")
    write_testset(
        output_root / "smoke_92",
        smoke,
        source_counts=counts,
        source_manifest=source_manifest,
        selection="coverage-stratified 0.1% smoke",
    )
    pilot = exact_stratified_sample(samples, 500, seed=SEED)
    write_testset(
        output_root / "pilot_500",
        pilot,
        source_counts=counts,
        source_manifest=source_manifest,
        selection="exact 500-sample coverage-stratified pilot",
    )
    report = {
        "status": "pass",
        "full_samples": len(samples),
        "covered_subtasks": len(counts),
        "split_counts": dict(
            sorted(Counter(f"{sample.dataset}/{sample.split}" for sample in samples).items())
        ),
        "status_counts": status_counts(samples),
        "shard_counts": {name: len(by_shard[name]) for name in SHARDS},
        "smoke_samples": len(smoke),
        "pilot_samples": len(pilot),
        "source_manifest_sha256": sha256_file(source_manifest),
        "full_manifest_sha256": sha256_file(output_root / "full" / "testset.jsonl"),
    }
    report_path = output_root / "prepare_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.chmod(0o444)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def merge(testset_dir: Path, shard_dirs: list[Path], output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    expected_rows = list(jsonl_rows(testset_dir / "testset.jsonl"))
    expected_ids = [row["id"] for row in expected_rows]
    expected_by_id = {row["id"]: row for row in expected_rows}
    predictions_by_id: dict[str, dict[str, Any]] = {}
    manifests: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        manifest = json.loads((shard_dir / "run_manifest.json").read_text(encoding="utf-8"))
        manifests.append(manifest)
        for row in jsonl_rows(shard_dir / "predictions.jsonl"):
            identifier = row["sample"]["id"]
            if identifier in predictions_by_id:
                raise ValueError(f"Duplicate prediction across shards: {identifier}")
            if identifier not in expected_by_id:
                raise ValueError(f"Unexpected prediction ID: {identifier}")
            if row["sample"] != expected_by_id[identifier]:
                raise ValueError(f"Prediction sample differs from canonical manifest: {identifier}")
            predictions_by_id[identifier] = row
    if set(predictions_by_id) != set(expected_ids):
        missing = set(expected_ids) - set(predictions_by_id)
        extra = set(predictions_by_id) - set(expected_ids)
        raise ValueError(f"Prediction coverage mismatch: missing={len(missing)} extra={len(extra)}")
    policies = {json.dumps(m["generation_policy"], sort_keys=True) for m in manifests}
    model_hashes = {m["model_artifact_sha256"] for m in manifests}
    code_hashes = {m["code_sha256"] for m in manifests}
    if len(policies) != 1 or len(model_hashes) != 1 or len(code_hashes) != 1:
        raise ValueError("Shard protocol, model hash, or code hash mismatch")
    rows = [predictions_by_id[identifier] for identifier in expected_ids]
    if any(row.get("error") for row in rows):
        raise ValueError("Cannot merge rows with generation errors")
    if any(row.get("generation_truncated") for row in rows):
        raise ValueError("Cannot merge rows with generation truncations")
    output_dir.mkdir(parents=True)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    official, clean = summarize_predictions(rows)
    official["prediction_letter_distribution"] = prediction_letter_distribution(rows)
    clean["prediction_letter_distribution"] = prediction_letter_distribution(rows)
    (output_dir / "official_summary.json").write_text(
        json.dumps(official, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "clean_summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "unit_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            sample = row["sample"]
            unit = {
                "sample_id": sample["id"],
                "dataset": sample["dataset"],
                "subtask": sample["subtask"],
                "task_type": sample["task_type"],
                "clean_status": sample["clean_status"],
                "score": row["score"],
            }
            handle.write(json.dumps(unit, ensure_ascii=False) + "\n")
    with (output_dir / "issues.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "dataset", "subtask", "clean_status", "issue"])
        for row in rows:
            sample = row["sample"]
            for issue in sample.get("issues", []):
                writer.writerow(
                    [sample["id"], sample["dataset"], sample["subtask"], sample["clean_status"], issue]
                )
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    efficiency = {
        "samples": len(rows),
        "mean_inference_seconds": statistics.fmean(elapsed),
        "p50_inference_seconds": statistics.median(elapsed),
        "p95_inference_seconds": percentile(elapsed, 0.95),
        "total_inference_seconds": sum(elapsed),
        "peak_cuda_memory_mb": max(float(m["peak_cuda_memory_mb"]) for m in manifests),
        "source_jobs": [m["environment"]["slurm_job_id"] for m in manifests],
    }
    (output_dir / "efficiency.json").write_text(
        json.dumps(efficiency, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    first = manifests[0]
    run_manifest = {
        "model_path": first["model_path"],
        "model": first["model"],
        "model_profile": first["model_profile"],
        "model_artifact_sha256": first["model_artifact_sha256"],
        "manifest_path": str((testset_dir / "testset.jsonl").resolve()),
        "manifest_sha256": sha256_file(testset_dir / "testset.jsonl"),
        "sample_count": len(rows),
        "group_counts": dict(
            sorted(Counter(f"{row['sample']['dataset']}/{row['sample']['subtask']}" for row in rows).items())
        ),
        "generation_errors": 0,
        "generation_truncations": 0,
        "generation_policy": first["generation_policy"],
        "processor_pixel_policy": first["processor_pixel_policy"],
        "code_sha256": first["code_sha256"],
        "source_shard_manifests": [str(path / "run_manifest.json") for path in shard_dirs],
        "environment": {
            "slurm_job_id": None,
            "source_slurm_job_ids": efficiency["source_jobs"],
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2), flush=True)


def verify(testset_dir: Path, run_dir: Path) -> None:
    manifest_rows = list(jsonl_rows(testset_dir / "testset.jsonl"))
    prediction_rows = list(jsonl_rows(run_dir / "predictions.jsonl"))
    unit_rows = list(jsonl_rows(run_dir / "unit_scores.jsonl"))
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    official = json.loads((run_dir / "official_summary.json").read_text(encoding="utf-8"))
    clean = json.loads((run_dir / "clean_summary.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    ids = [row["id"] for row in manifest_rows]
    prediction_ids = [row["sample"]["id"] for row in prediction_rows]
    if len(manifest_rows) != EXPECTED_TOTAL:
        failures.append(f"manifest rows {len(manifest_rows)} != {EXPECTED_TOTAL}")
    if len(ids) != len(set(ids)):
        failures.append("manifest IDs are not unique")
    if prediction_ids != ids:
        failures.append("prediction IDs/order differ from canonical manifest")
    if len(unit_rows) != EXPECTED_TOTAL:
        failures.append("unit score count mismatch")
    for row in manifest_rows:
        if OFFICIAL_SPLITS.get(row["dataset"]) != row.get("split"):
            failures.append(f"non-official split: {row['id']} {row.get('split')}")
            break
    if any(row.get("error") for row in prediction_rows):
        failures.append("generation errors remain")
    if any(row.get("generation_truncated") for row in prediction_rows):
        failures.append("generation truncations remain")
    if run_manifest.get("manifest_sha256") != sha256_file(testset_dir / "testset.jsonl"):
        failures.append("manifest hash mismatch")
    if run_manifest.get("sample_count") != EXPECTED_TOTAL:
        failures.append("run manifest count mismatch")
    if official.get("prediction_rows") != EXPECTED_TOTAL or clean.get("prediction_rows") != EXPECTED_TOTAL:
        failures.append("summary count mismatch")
    report = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "manifest_samples": len(manifest_rows),
        "unique_ids": len(set(ids)),
        "prediction_rows": len(prediction_rows),
        "unit_score_rows": len(unit_rows),
        "covered_subtasks": len({f"{row['dataset']}/{row['subtask']}" for row in manifest_rows}),
        "split_counts": dict(sorted(Counter(f"{row['dataset']}/{row.get('split')}" for row in manifest_rows).items())),
        "generation_errors": sum(bool(row.get("error")) for row in prediction_rows),
        "generation_truncations": sum(bool(row.get("generation_truncated")) for row in prediction_rows),
    }
    report_path = run_dir / "final_validation_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-manifest", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--testset-dir", type=Path, required=True)
    merge_parser.add_argument("--output-dir", type=Path, required=True)
    merge_parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--testset-dir", type=Path, required=True)
    verify_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source_manifest.resolve(), args.output_root.resolve())
    elif args.command == "merge":
        merge(
            args.testset_dir.resolve(),
            [path.resolve() for path in args.shard_dir],
            args.output_dir.resolve(),
        )
    else:
        verify(args.testset_dir.resolve(), args.run_dir.resolve())


if __name__ == "__main__":
    main()
