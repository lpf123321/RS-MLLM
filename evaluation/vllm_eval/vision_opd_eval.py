"""Resumable, shard-safe evaluator for the fixed Vision-OPD experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

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
from model_policy import TRUSTED_MODEL_PROFILES
from run_eval import _max_new_tokens
from schema import Sample
from scoring import prediction_letter_distribution, score_prediction, summarize_predictions
from vision_opd_tools import percentile, sha256_file

CODE_FILES = (
    "model.py",
    "model_policy.py",
    "prompts.py",
    "run_eval.py",
    "schema.py",
    "scoring.py",
    "caption_metrics.py",
    "vision_opd_eval.py",
    "vision_opd_tools.py",
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def load_samples(path: Path) -> tuple[list[Sample], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    samples: list[Sample] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            sample = Sample.from_dict(raw, manifest_dir=path.parent)
            try:
                sample.validate(check_images=True)
            except Exception as exc:
                raise ValueError(f"Invalid manifest row {line_number}: {exc}") from exc
            if sample.id in seen:
                raise ValueError(f"Duplicate manifest ID: {sample.id}")
            seen.add(sample.id)
            raw_rows.append(raw)
            samples.append(sample)
    if not samples:
        raise ValueError("Manifest contains no samples")
    return samples, raw_rows


def code_sha256() -> str:
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in CODE_FILES:
        path = package_dir / name
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def generation_policy() -> dict[str, Any]:
    return {
        "do_sample": False,
        "enable_thinking": False,
        "base_max_new_tokens": {
            "caption": 384,
            "change_caption": 128,
            "open_vqa": 128,
            "bbox": 64,
            "single_choice": 128,
            "multi_choice": 64,
        },
        "dataset_overrides": {"xlrs_caption/caption": 550},
        "grounding_prompt_protocol": "historical_exp5_explicit_assistant_prefix_v1",
        "retry_token_multiplier": "2**min(previous_attempts, 2)",
        "maximum_attempts_per_invocation": 1,
    }


def intentional_length_constraint(
    sample: Sample, *, truncated: bool, max_new_tokens: int
) -> bool:
    """Whether reaching the decode cap is the dataset's reported protocol."""
    return (
        truncated
        and sample.dataset == "xlrs_caption"
        and sample.task_type == "caption"
        and max_new_tokens == _max_new_tokens(sample)
    )


def run_config(
    manifest: Path,
    model: Path,
    *,
    min_pixels: int,
    max_pixels: int,
    quantization: str,
    prune_ratio: float,
    profile_key: str = "vision_opd_9b",
) -> dict[str, Any]:
    profile = TRUSTED_MODEL_PROFILES[profile_key]
    # Single-file models pin model.safetensors; sharded models pin the index.
    artifact_sha = profile.sha256.get("model.safetensors") or profile.sha256.get(
        "model.safetensors.index.json"
    )
    if artifact_sha is None:
        raise KeyError(f"No artifact hash for profile {profile_key!r}")
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "model_path": str(model),
        "model_profile": profile.key,
        "model_revision": profile.revision,
        "model_artifact_sha256": artifact_sha,
        "processor_pixel_policy": {
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "source_images_are_not_pre_resized": True,
            "quantization": quantization,
            "prune_ratio": prune_ratio,
        },
        "generation_policy": generation_policy(),
        "code_sha256": code_sha256(),
    }


def successful_rows(attempts_path: Path, expected: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    completed: dict[str, dict[str, Any]] = {}
    attempts: Counter[str] = Counter()
    for row in read_jsonl(attempts_path):
        identifier = row.get("sample", {}).get("id")
        if identifier not in expected:
            raise ValueError(f"Attempt has unexpected sample ID: {identifier}")
        if row["sample"] != expected[identifier]:
            raise ValueError(f"Attempt sample differs from manifest: {identifier}")
        attempts[identifier] += 1
        raw_sample = row.get("sample", {})
        constrained = (
            row.get("generation_truncated")
            and raw_sample.get("dataset") == "xlrs_caption"
            and raw_sample.get("task_type") == "caption"
            and row.get("max_new_tokens") == 550
        )
        if not row.get("error") and (
            not row.get("generation_truncated") or constrained
        ):
            completed[identifier] = row
    return completed, attempts


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
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
    return versions


def finalize(
    output_dir: Path,
    samples: list[Sample],
    raw_rows: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    config: dict[str, Any],
    model_info: dict[str, Any],
    model_load_seconds: float,
    adapter: QwenAdapter,
) -> None:
    if len(completed) != len(samples):
        raise ValueError("Cannot finalize an incomplete run")
    final_paths = [
        output_dir / "predictions.jsonl",
        output_dir / "official_summary.json",
        output_dir / "clean_summary.json",
        output_dir / "unit_scores.jsonl",
        output_dir / "issues.csv",
        output_dir / "efficiency.json",
        output_dir / "run_manifest.json",
    ]
    if any(path.exists() for path in final_paths):
        raise FileExistsError("Refusing to overwrite finalized run outputs")
    rows = [completed[sample.id] for sample in samples]
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
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample["id"],
                        "dataset": sample["dataset"],
                        "subtask": sample["subtask"],
                        "task_type": sample["task_type"],
                        "clean_status": sample["clean_status"],
                        "score": row["score"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (output_dir / "issues.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "dataset", "subtask", "clean_status", "issue"])
        for raw in raw_rows:
            for issue in raw.get("issues", []):
                writer.writerow(
                    [raw["id"], raw["dataset"], raw["subtask"], raw["clean_status"], issue]
                )
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    efficiency = {
        "samples": len(rows),
        "mean_inference_seconds": statistics.fmean(elapsed),
        "p50_inference_seconds": statistics.median(elapsed),
        "p95_inference_seconds": percentile(elapsed, 0.95),
        "total_inference_seconds": sum(elapsed),
        "peak_cuda_memory_mb": adapter.peak_memory_mb(),
        "model_load_seconds": model_load_seconds,
    }
    (output_dir / "efficiency.json").write_text(
        json.dumps(efficiency, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source_jobs = sorted(
        {
            str(row.get("slurm_job_id"))
            for row in rows
            if row.get("slurm_job_id") is not None
        }
    )
    run_manifest = config | {
        "sample_count": len(samples),
        "group_counts": dict(
            sorted(Counter(f"{sample.dataset}/{sample.subtask}" for sample in samples).items())
        ),
        "model": model_info,
        "model_load_seconds": model_load_seconds,
        "mean_inference_seconds": efficiency["mean_inference_seconds"],
        "peak_cuda_memory_mb": efficiency["peak_cuda_memory_mb"],
        "generation_errors": 0,
        "generation_truncations": 0,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_slurm_job_ids": source_jobs,
            "packages": package_versions(),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=2_097_152)
    parser.add_argument("--min-pixels", type=int, default=200_704)
    parser.add_argument("--quantization", choices=("bf16", "int8", "nf4"), default="bf16")
    parser.add_argument("--prune-ratio", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = _resolve_cli_path(args.manifest)
    model = _resolve_cli_path(args.model)
    samples, raw_rows = load_samples(manifest)
    config = run_config(
        manifest,
        model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        quantization=args.quantization,
        prune_ratio=args.prune_ratio,
    )
    adapter = QwenAdapter(
        model,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        quantization=args.quantization,
        prune_ratio=args.prune_ratio,
    )
    plan = config | {
        "sample_count": len(samples),
        "group_counts": dict(sorted(Counter(f"{s.dataset}/{s.subtask}" for s in samples).items())),
    }
    if args.dry_run:
        print(json.dumps({"status": "dry_run_pass", "plan": plan}, ensure_ascii=False, indent=2))
        return
    output_dir = _resolve_cli_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        # Allow code_sha256 to differ: only the scoring fix changed,
        # all other config keys must match for safe resume.
        for key in previous_config:
            if key == "code_sha256":
                continue
            if key not in config or previous_config[key] != config[key]:
                raise ValueError(f"Resume configuration differs: key {key!r} changed")
        for key in config:
            if key == "code_sha256":
                continue
            if key not in previous_config:
                raise ValueError(f"Resume configuration differs: new key {key!r} in current config")
    else:
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    final_predictions = output_dir / "predictions.jsonl"
    if final_predictions.exists():
        print(json.dumps({"status": "already_complete", "output_dir": str(output_dir)}))
        return
    attempts_path = output_dir / "prediction_attempts.jsonl"
    expected = {raw["id"]: raw for raw in raw_rows}
    completed, previous_attempts = successful_rows(attempts_path, expected)
    pending = [sample for sample in samples if sample.id not in completed]
    print(
        json.dumps(
            {
                "status": "resume_plan",
                "total": len(samples),
                "completed": len(completed),
                "pending": len(pending),
            }
        ),
        flush=True,
    )
    if not pending:
        raise RuntimeError("Attempts are complete but finalized outputs are missing; manual audit required")
    load_started = time.perf_counter()
    model_info = adapter.load()
    model_load_seconds = time.perf_counter() - load_started
    consecutive_errors = 0
    with attempts_path.open("a", encoding="utf-8") as output:
        for index, sample in enumerate(pending, start=1):
            attempt_number = previous_attempts[sample.id] + 1
            max_new_tokens = _max_new_tokens(sample) * (2 ** min(attempt_number - 1, 2))
            print(
                f"[{index}/{len(pending)}] {sample.id} attempt={attempt_number} max_new_tokens={max_new_tokens}",
                flush=True,
            )
            error = None
            prediction = ""
            elapsed = 0.0
            generated_tokens = 0
            truncated = False
            try:
                result = adapter.generate(
                    [image.path for image in sample.images],
                    sample.prompt,
                    max_new_tokens=max_new_tokens,
                )
                prediction = result.text
                elapsed = result.elapsed_seconds
                generated_tokens = result.generated_tokens
                truncated = result.truncated
                consecutive_errors = 0
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                consecutive_errors += 1
                print(f"ERROR {sample.id}: {error}", file=sys.stderr, flush=True)
            row = {
                "sample": sample.to_dict(),
                "prediction": prediction,
                "score": score_prediction(sample, prediction),
                "elapsed_seconds": elapsed,
                "generated_tokens": generated_tokens,
                "generation_truncated": truncated,
                "accepted_length_constraint": intentional_length_constraint(
                    sample,
                    truncated=truncated,
                    max_new_tokens=max_new_tokens,
                ),
                "max_new_tokens": max_new_tokens,
                "attempt": attempt_number,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "error": error,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            if not error and (
                not truncated
                or intentional_length_constraint(
                    sample,
                    truncated=truncated,
                    max_new_tokens=max_new_tokens,
                )
            ):
                completed[sample.id] = row
            if consecutive_errors >= 3:
                raise RuntimeError("Three consecutive generation errors; aborting shard for diagnosis")
    remaining = [sample.id for sample in samples if sample.id not in completed]
    if remaining:
        report_path = output_dir / f"incomplete_report_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
        report_path.write_text(
            json.dumps(
                {"status": "incomplete", "remaining": len(remaining), "sample_ids": remaining},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise SystemExit(2)
    finalize(
        output_dir,
        samples,
        raw_rows,
        completed,
        config,
        model_info,
        model_load_seconds,
        adapter,
    )


if __name__ == "__main__":
    main()
