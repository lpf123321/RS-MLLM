"""Validate end-to-end build and inference outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from schema import Sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    testset_dir = args.testset_dir.resolve()
    run_dir = args.run_dir.resolve()
    report_path = run_dir / "validation_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")

    dataset_summary = _read_json(testset_dir / "dataset_summary.json")
    manifest_rows = _read_jsonl(testset_dir / "testset.jsonl")
    samples = [Sample.from_dict(row) for row in manifest_rows]
    prediction_rows = _read_jsonl(run_dir / "predictions.jsonl")
    run_manifest = _read_json(run_dir / "run_manifest.json")
    official = _read_json(run_dir / "official_summary.json")
    clean = _read_json(run_dir / "clean_summary.json")

    failures: list[str] = []
    expected_ids = [sample.id for sample in samples]
    actual_ids = [row["sample"]["id"] for row in prediction_rows]
    if actual_ids != expected_ids:
        failures.append(
            "prediction IDs or ordering do not exactly match the test-set manifest"
        )
    if [row["sample"] for row in prediction_rows] != manifest_rows:
        failures.append("embedded prediction samples differ from the test-set manifest")
    if run_manifest["manifest_sha256"] != _sha256(testset_dir / "testset.jsonl"):
        failures.append("run_manifest manifest_sha256 mismatch")
    if len(set(expected_ids)) != len(expected_ids):
        failures.append("test-set manifest contains duplicate IDs")
    if dataset_summary["sample_count"] != len(samples):
        failures.append("dataset_summary sample_count mismatch")
    expected_groups = Counter(
        f"{sample.dataset}/{sample.subtask}" for sample in samples
    )
    if dict(sorted(expected_groups.items())) != dataset_summary["selected_counts"]:
        failures.append("dataset_summary selected_counts mismatch")
    if set(dataset_summary["source_counts"]) != set(dataset_summary["selected_counts"]):
        failures.append("0.1% sample does not cover every source subtask")
    if run_manifest["group_counts"] != dict(sorted(expected_groups.items())):
        failures.append("run_manifest group_counts mismatch")
    if run_manifest["sample_count"] != len(samples):
        failures.append("run_manifest sample_count mismatch")
    if run_manifest["generation_errors"]:
        failures.append(f"generation errors: {run_manifest['generation_errors']}")
    if run_manifest["generation_truncations"]:
        failures.append(
            f"generation truncations: {run_manifest['generation_truncations']}"
        )
    if official["prediction_rows"] != len(samples) or clean["prediction_rows"] != len(
        samples
    ):
        failures.append("summary prediction row count mismatch")

    for sample in samples:
        if sample.dataset == "levir_cc":
            if len(sample.images) != 2 or len(sample.references) != 5:
                failures.append(f"LEVIR protocol violation: {sample.id}")
        if (
            sample.dataset == "xlrs_bench_lite"
            and sample.subtask == "counting_with_changing_detection"
        ):
            if len(sample.images) != 2 or [image.role for image in sample.images] != [
                "before",
                "after",
            ]:
                failures.append(f"XLRS temporal input violation: {sample.id}")
        if (
            sample.task_type == "multi_choice"
            and sample.subtask != "overall_land_use_classification"
        ):
            failures.append(f"Unexpected multi-choice task: {sample.id}")

    manual_review = [
        {
            "sample_id": sample.id,
            "dataset": sample.dataset,
            "subtask": sample.subtask,
            "issues": sample.issues,
            "image_paths": [image.path for image in sample.images],
        }
        for sample in samples
        if sample.clean_status == "manual_review"
    ]
    report = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "manifest_samples": len(samples),
        "covered_subtasks": len(expected_groups),
        "covered_subtask_names": sorted(expected_groups),
        "model_type": run_manifest["model"]["model_type"],
        "generation_errors": run_manifest["generation_errors"],
        "generation_truncations": run_manifest["generation_truncations"],
        "manual_visual_review_count": len(manual_review),
        "manual_visual_review": manual_review,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
