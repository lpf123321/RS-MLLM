"""Run one Qwen model over a corrected manifest and emit traceable outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_cli_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in (REPO_ROOT, Path(__file__).resolve().parent):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (REPO_ROOT / path).resolve()

from model import QwenAdapter
from schema import Sample
from scoring import (
    prediction_letter_distribution,
    score_prediction,
    summarize_predictions,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _max_new_tokens(sample: Sample) -> int:
    return {
        "caption": 384,
        "change_caption": 128,
        "open_vqa": 128,   # thinking 模型预留思考空间, 防正文截断
        "bbox": 128,
        "single_choice": 128,
        "multi_choice": 64,
    }[sample.task_type]


def _load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    manifest_dir = path.parent
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sample = Sample.from_dict(json.loads(line), manifest_dir=manifest_dir)
                sample.validate(check_images=True)
            except Exception as exc:
                raise ValueError(f"Invalid manifest row {line_number}: {exc}") from exc
            samples.append(sample)
    if not samples:
        raise ValueError("Manifest contains no samples")
    if len({sample.id for sample in samples}) != len(samples):
        raise ValueError("Manifest contains duplicate sample IDs")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=16_777_216)
    parser.add_argument("--min-pixels", type=int, default=200_704)
    parser.add_argument(
        "--quantization", choices=("bf16", "int8", "nf4"), default="bf16"
    )
    parser.add_argument("--prune-ratio", type=float, default=0.0)
    args = parser.parse_args()

    manifest_path = _resolve_cli_path(args.manifest)
    model_path = _resolve_cli_path(args.model)
    output_dir = _resolve_cli_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    samples = _load_samples(manifest_path)

    adapter = QwenAdapter(
        model_path,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        quantization=args.quantization,
        prune_ratio=args.prune_ratio,
    )
    load_started = time.perf_counter()
    model_info = adapter.load()
    model_load_seconds = time.perf_counter() - load_started
    rows: list[dict] = []
    predictions_path = output_dir / "predictions.jsonl"

    with predictions_path.open("w", encoding="utf-8") as output:
        for index, sample in enumerate(samples, start=1):
            print(f"[{index}/{len(samples)}] {sample.id}", flush=True)
            error = None
            prediction = ""
            elapsed = 0.0
            generated_tokens = 0
            truncated = False
            try:
                result = adapter.generate(
                    [image.path for image in sample.images],
                    sample.prompt,
                    max_new_tokens=_max_new_tokens(sample),
                )
                prediction = result.text
                elapsed = result.elapsed_seconds
                generated_tokens = result.generated_tokens
                truncated = result.truncated
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"ERROR {sample.id}: {error}", file=sys.stderr, flush=True)
            score = score_prediction(sample, prediction)
            row = {
                "sample": sample.to_dict(),
                "prediction": prediction,
                "score": score,
                "elapsed_seconds": elapsed,
                "generated_tokens": generated_tokens,
                "generation_truncated": truncated,
                "error": error,
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()

    official, clean = summarize_predictions(rows)
    official["prediction_letter_distribution"] = prediction_letter_distribution(rows)
    clean["prediction_letter_distribution"] = prediction_letter_distribution(rows)
    (output_dir / "official_summary.json").write_text(
        json.dumps(official, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "clean_summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "issues.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "dataset", "subtask", "clean_status", "issue"])
        for sample in samples:
            for issue in sample.issues:
                writer.writerow(
                    [
                        sample.id,
                        sample.dataset,
                        sample.subtask,
                        sample.clean_status,
                        issue,
                    ]
                )

    elapsed_values = [row["elapsed_seconds"] for row in rows if not row["error"]]
    versions = {}
    for package in (
        "torch",
        "transformers",
        "bitsandbytes",
        "qwen-vl-utils",
        "pillow",
        "pyarrow",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    run_manifest = {
        "model_path": str(model_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "model": model_info,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "sample_count": len(samples),
        "group_counts": dict(
            sorted(
                Counter(
                    f"{sample.dataset}/{sample.subtask}" for sample in samples
                ).items()
            )
        ),
        "model_load_seconds": model_load_seconds,
        "mean_inference_seconds": sum(elapsed_values) / len(elapsed_values)
        if elapsed_values
        else None,
        "peak_cuda_memory_mb": adapter.peak_memory_mb(),
        "generation_errors": sum(bool(row["error"]) for row in rows),
        "generation_truncations": sum(row["generation_truncated"] for row in rows),
        "processor_pixel_policy": {
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "source_images_are_not_pre_resized": True,
            "quantization": args.quantization,
            "prune_ratio": args.prune_ratio,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "packages": versions,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2), flush=True)
    if run_manifest["generation_errors"] or run_manifest["generation_truncations"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
