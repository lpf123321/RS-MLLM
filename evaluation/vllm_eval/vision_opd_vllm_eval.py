"""Batched vLLM evaluator for the fixed Vision-OPD experiment.

Performance path: same manifest, pixel policy, prompts, decode policy and
scoring as ``vision_opd_eval`` / ``vision_opd_eval_v6``, but inference runs on
a vLLM offline engine with continuous batching instead of per-sample
Transformers ``generate``. Outputs use the identical JSONL schema so the
existing finalize/validate tooling works unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def setup_runtime_env() -> None:
    """自动设置 vLLM 运行时所需环境变量(用户无需手动 export)。

    - LD_LIBRARY_PATH: 从本 venv site-packages 自动推导 av.libs / nvidia/*/lib
    - VLLM_USE_FLASHINFER_SAMPLER=0: flashinfer 0.6.14 与 nvcc12.4 不兼容
    - VLLM_WORKER_MULTIPROC_METHOD=spawn: vllm 多进程必须 spawn
    """
    import glob

    site_pkgs = Path(__file__).resolve().parent.parent / ".venv" / "lib" / "python3*" / "site-packages"
    site_pkgs = next(iter(glob.glob(str(site_pkgs))), None)
    if site_pkgs:
        lib_dirs = [d for d in glob.glob(f"{site_pkgs}/av.libs") + glob.glob(f"{site_pkgs}/nvidia/*/lib")]
        if lib_dirs:
            os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


setup_runtime_env()

import vision_opd_eval as base
from model import GenerationResult
from model_policy import (
    assert_model_allowed,
    load_derived_profile,
    register_derived_profile,
    verify_derived_model_content,
    verify_trusted_model_content,
)
from run_eval import _max_new_tokens
from scoring import score_prediction
from vision_opd_profile import register_vision_opd_profile
from vision_opd_tools import sha256_file

register_vision_opd_profile()

CODE_FILES = base.CODE_FILES + (
    "vision_opd_profile.py",
    "vision_opd_vllm_eval.py",
)

MAX_PASSES = 3  # token multiplier 1x, 2x, 4x -- same policy as the serial evaluator


def code_sha256() -> str:
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in CODE_FILES:
        path = package_dir / name
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _check_resume_config(
    previous_config: dict[str, Any], current_config: dict[str, Any]
) -> None:
    """Fail-closed resume gate: the run configuration -- including
    ``code_sha256`` -- is immutable for an output directory.

    A resumed run must be produced by the exact same evaluator code, so
    attempts written under a previous evaluator version are never accepted:
    any key drift (changed or new) refuses the resume and requires a fresh
    output directory.
    """
    for key in previous_config:
        if key not in current_config or previous_config[key] != current_config[key]:
            raise ValueError(f"Resume configuration differs: key {key!r} changed")
    for key in current_config:
        if key not in previous_config:
            raise ValueError(f"Resume configuration differs: new key {key!r} in current config")


class VLLMBatchAdapter:
    """vLLM offline engine bound to the fixed Vision-OPD pixel/decode policy."""

    def __init__(
        self,
        model_path: Path,
        *,
        profile_key: str,
        max_pixels: int,
        min_pixels: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        enforce_eager: bool,
        cudagraph_mm_encoder: bool = False,
        mm_encoder_attn_backend: str | None = None,
        skip_content_verification: bool = False,
    ) -> None:
        if enforce_eager and cudagraph_mm_encoder:
            raise ValueError("Encoder CUDA Graph requires enforce_eager=False")
        assert_model_allowed(model_path)
        if not skip_content_verification:
            verify_trusted_model_content(model_path, profile_key=profile_key)
        self.profile_key = profile_key
        self.model_path = model_path.resolve()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.cudagraph_mm_encoder = cudagraph_mm_encoder
        self.mm_encoder_attn_backend = mm_encoder_attn_backend
        self.family = ""
        self.llm: Any = None
        self.processor: Any = None
        self.torch: Any = None

    def load(self) -> dict[str, Any]:
        import torch
        from transformers import AutoConfig, AutoProcessor
        from vllm import LLM

        # NOTE: do NOT touch torch.cuda before LLM(...) -- vLLM forks its
        # EngineCore and any parent-side CUDA context aborts the child.
        config = AutoConfig.from_pretrained(str(self.model_path), trust_remote_code=True)
        self.family = str(config.model_type)
        if self.family != "qwen3_5":
            raise ValueError(f"Unexpected Vision-OPD model_type: {self.family}")
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        started = time.perf_counter()
        llm_kwargs: dict[str, Any] = {
            "model": str(self.model_path),
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "max_model_len": self.max_model_len,
            "limit_mm_per_prompt": {"image": 2},
            "mm_processor_kwargs": {
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            },
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "enforce_eager": self.enforce_eager,
        }
        if self.cudagraph_mm_encoder:
            llm_kwargs["compilation_config"] = {"cudagraph_mm_encoder": True}
        if self.mm_encoder_attn_backend is not None:
            llm_kwargs["mm_encoder_attn_backend"] = self.mm_encoder_attn_backend
        self.llm = LLM(**llm_kwargs)
        load_seconds = time.perf_counter() - started
        self.torch = torch
        return {
            "model_type": self.family,
            "model_class": "Qwen3_5ForConditionalGeneration",
            "processor_class": type(self.processor).__name__,
            "loader": "vllm.LLM",
            "loader_vllm_version": __import__("vllm").__version__,
            "loader_transformers_version": __import__("transformers").__version__,
            "auto_load_seconds": load_seconds,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "cudagraph_mm_encoder": self.cudagraph_mm_encoder,
            "mm_encoder_attn_backend": self.mm_encoder_attn_backend,
            "quantization": "bf16",
            "pruning": {"ratio": 0.0, "modules": 0, "parameters": 0, "zeros": 0},
            "cuda_device": torch.cuda.get_device_name(),
            "cuda_capability": list(torch.cuda.get_device_capability()),
        }

    def _prompt_text(self, image_paths: list[str], prompt: str) -> str:
        content = [
            {
                "type": "image",
                "image": path,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            }
            for path in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def generate_batch(
        self, items: list[tuple[list[str], str, int]]
    ) -> list[GenerationResult]:
        """Run one chunk; items are (image_paths, prompt, max_new_tokens)."""
        if self.llm is None or self.processor is None:
            raise RuntimeError("Engine must be loaded before generation")
        from PIL import Image
        from vllm import SamplingParams

        requests: list[dict[str, Any]] = []
        params: list[Any] = []
        for image_paths, prompt, max_new_tokens in items:
            images = [Image.open(path).convert("RGB") for path in image_paths]
            requests.append(
                {
                    "prompt": self._prompt_text(image_paths, prompt),
                    "multi_modal_data": {
                        "image": images[0] if len(images) == 1 else images
                    },
                }
            )
            params.append(
                SamplingParams(
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                    skip_special_tokens=True,
                )
            )
        started = time.perf_counter()
        outputs = self.llm.generate(requests, params)
        wall = time.perf_counter() - started
        results: list[GenerationResult] = []
        for (_, _, max_new_tokens), output in zip(items, outputs):
            completion = output.outputs[0]
            generated_tokens = len(completion.token_ids)
            metrics = getattr(output, "metrics", None)
            arrival = getattr(metrics, "arrival_time", None) if metrics else None
            last_token = getattr(metrics, "last_token_time", None) if metrics else None
            first_token = getattr(metrics, "first_token_time", None) if metrics else None
            if arrival and last_token and last_token > arrival:
                elapsed = last_token - arrival
            else:
                elapsed = wall / max(len(items), 1)
            ttft = (first_token - arrival) if (arrival and first_token and first_token > arrival) else 0.0
            results.append(
                GenerationResult(
                    text=completion.text.strip(),
                    elapsed_seconds=elapsed,
                    first_token_seconds=ttft,
                    generated_tokens=generated_tokens,
                    truncated=(
                        completion.finish_reason == "length"
                        or generated_tokens >= max_new_tokens
                    ),
                )
            )
        return results

    def peak_memory_mb(self) -> float:
        if self.torch is None or not self.torch.cuda.is_available():
            return 0.0
        return self.torch.cuda.max_memory_allocated() / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录(默认: results/<manifest文件名>_<profile>)",
    )
    parser.add_argument("--max-pixels", type=int, default=2_097_152)
    parser.add_argument("--min-pixels", type=int, default=200_704)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=16_384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--cudagraph-mm-encoder", action="store_true")
    parser.add_argument(
        "--mm-encoder-attn-backend",
        choices=("FLASH_ATTN", "FLASHINFER"),
        default=None,
    )
    parser.add_argument(
        "--model-profile",
        default=None,
        help="trusted base profile key (required unless --derived-profile is used)",
    )
    parser.add_argument(
        "--derived-profile",
        type=Path,
        default=None,
        help=(
            "external derived candidate profile/provenance JSON: exact model path, "
            "base profile/revision/license, training/fused evidence, and every "
            "loader-selected weight and runtime file with size+SHA (required "
            "unless --model-profile is used)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")
    # 默认输出目录: results/<manifest文件名>_<profile>
    if args.output_dir is None:
        manifest_name = Path(args.manifest).stem
        key = args.model_profile or (Path(args.derived_profile).stem if args.derived_profile else "model")
        args.output_dir = Path("results") / f"{manifest_name}_{key}"
    if args.enforce_eager and args.cudagraph_mm_encoder:
        raise ValueError("--cudagraph-mm-encoder cannot be used with --enforce-eager")
    if (args.model_profile is None) == (args.derived_profile is None):
        parser.error("exactly one of --model-profile / --derived-profile is required")
    manifest = args.manifest.resolve()
    model = args.model.resolve()
    profile_key = args.model_profile
    derived_manifest: Path | None = None
    if args.derived_profile is not None:
        derived = load_derived_profile(args.derived_profile)
        if derived.model_path.resolve() != model:
            raise ValueError(
                f"--model {model} does not match derived profile model_path "
                f"{derived.model_path}"
            )
        verify_derived_model_content(derived)
        register_derived_profile(derived)
        profile_key = derived.key
        derived_manifest = args.derived_profile.resolve()
    samples, raw_rows = base.load_samples(manifest)
    config = base.run_config(
        manifest,
        model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        quantization="bf16",
        prune_ratio=0.0,
        profile_key=profile_key,
    )
    config["code_sha256"] = code_sha256()
    engine: dict[str, Any] = {
        "name": "vllm",
        "model_profile": profile_key,
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.enforce_eager,
        "cudagraph_mm_encoder": args.cudagraph_mm_encoder,
        "mm_encoder_attn_backend": args.mm_encoder_attn_backend,
        "max_passes": MAX_PASSES,
    }
    if derived_manifest is not None:
        engine["derived_profile_manifest"] = str(derived_manifest)
        engine["derived_profile_manifest_sha256"] = sha256_file(derived_manifest)
    config["engine"] = engine
    plan = config | {
        "sample_count": len(samples),
        "group_counts": dict(
            sorted(base.Counter(f"{s.dataset}/{s.subtask}" for s in samples).items())
        ),
    }
    if args.dry_run:
        print(json.dumps({"status": "dry_run_pass", "plan": plan}, ensure_ascii=False, indent=2))
        return
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        _check_resume_config(previous_config, config)
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
    completed, previous_attempts = base.successful_rows(attempts_path, expected)
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
    if not pending and completed:
        base.finalize(
            output_dir, samples, raw_rows, completed, config,
            {"note": "finalized from existing attempts"}, 0.0, _NullAdapter(),
        )
        return
    load_started = time.perf_counter()
    adapter = VLLMBatchAdapter(
        model,
        profile_key=profile_key,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        cudagraph_mm_encoder=args.cudagraph_mm_encoder,
        mm_encoder_attn_backend=args.mm_encoder_attn_backend,
        # The derived profile was fully hash-verified in-process above on the
        # same resolved path; the adapter re-verification would re-read every
        # weight file (multi-GB) without adding a new trust check.
        skip_content_verification=derived_manifest is not None,
    )
    model_info = adapter.load()
    model_load_seconds = time.perf_counter() - load_started

    with attempts_path.open("a", encoding="utf-8") as output:
        for pass_index in range(1, MAX_PASSES + 1):
            if not pending:
                break
            chunks = [
                pending[offset : offset + args.batch_size]
                for offset in range(0, len(pending), args.batch_size)
            ]
            print(
                json.dumps(
                    {
                        "status": "pass_start",
                        "pass": pass_index,
                        "pending": len(pending),
                        "chunks": len(chunks),
                    }
                ),
                flush=True,
            )
            consecutive_chunk_errors = 0
            for chunk_number, chunk in enumerate(chunks, start=1):
                attempt_numbers = {
                    sample.id: previous_attempts[sample.id] + 1 for sample in chunk
                }
                items = [
                    (
                        [image.path for image in sample.images],
                        sample.prompt,
                        _max_new_tokens(sample)
                        * (2 ** min(attempt_numbers[sample.id] - 1, 2)),
                    )
                    for sample in chunk
                ]
                try:
                    results = adapter.generate_batch(items)
                    consecutive_chunk_errors = 0
                except Exception as exc:
                    consecutive_chunk_errors += 1
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"ERROR chunk {chunk_number}: {error}", file=sys.stderr, flush=True)
                    for sample, (_, _, max_new_tokens) in zip(chunk, items):
                        row = {
                            "sample": sample.to_dict(),
                            "prediction": "",
                            "score": score_prediction(sample, ""),
                            "elapsed_seconds": 0.0,
                            "first_token_seconds": 0.0,
                            "generated_tokens": 0,
                            "generation_truncated": False,
                            "max_new_tokens": max_new_tokens,
                            "attempt": attempt_numbers[sample.id],
                            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                            "error": error,
                        }
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        previous_attempts[sample.id] += 1
                    output.flush()
                    if consecutive_chunk_errors >= 2:
                        raise RuntimeError(
                            "Two consecutive chunk errors; aborting shard for diagnosis"
                        )
                    continue
                for sample, (_, _, max_new_tokens), result in zip(chunk, items, results):
                    row = {
                        "sample": sample.to_dict(),
                        "prediction": result.text,
                        "score": score_prediction(sample, result.text),
                        "elapsed_seconds": result.elapsed_seconds,
                        "first_token_seconds": result.first_token_seconds,
                        "generated_tokens": result.generated_tokens,
                        "generation_truncated": result.truncated,
                        "max_new_tokens": max_new_tokens,
                        "attempt": attempt_numbers[sample.id],
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "error": None,
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    previous_attempts[sample.id] += 1
                    if not result.truncated:
                        completed[sample.id] = row
                output.flush()
                if chunk_number % 10 == 0 or chunk_number == len(chunks):
                    done = len(samples) - len(
                        [s for s in samples if s.id not in completed]
                    )
                    print(
                        f"[pass {pass_index} chunk {chunk_number}/{len(chunks)}] "
                        f"completed {done}/{len(samples)}",
                        flush=True,
                    )
            pending = [sample for sample in samples if sample.id not in completed]

    if pending:
        report_path = output_dir / f"incomplete_report_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
        report_path.write_text(
            json.dumps(
                {
                    "status": "incomplete",
                    "remaining": len(pending),
                    "sample_ids": [sample.id for sample in pending],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise SystemExit(2)
    base.finalize(
        output_dir,
        samples,
        raw_rows,
        completed,
        config,
        model_info,
        model_load_seconds,
        adapter,
    )


class _NullAdapter:
    def peak_memory_mb(self) -> float:
        return 0.0


if __name__ == "__main__":
    main()
