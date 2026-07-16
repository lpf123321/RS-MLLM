#!/usr/bin/env python3
"""Candidate registry for RS-MLLM model evaluation.

All candidate model IDs, sources, loading routes, and cache policy live here so
that download, fit tests, and summaries share one source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

CacheStatus = Literal["baseline", "cached", "download-first"]
Loader = Literal[
    "baseline_qwen3vl",
    "qwen3vl_dense",
    "minicpm",
    "qwen35_dense",
    "qwen35_moe",
    "internvlu",
    "uhr_bat",
    "auto_image_text",
]

ROOT = Path.home() / "RS-MLLM"
MS_CACHE = ROOT / ".ms_cache"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
RESULTS_DIR = ROOT / "results"
CACHE_MANIFEST = RESULTS_DIR / "candidate_cache_manifest.json"
SOURCE_REPORT = RESULTS_DIR / "candidate_sources.json"


@dataclass(frozen=True)
class Candidate:
    key: str
    model_id: str
    source_url: str
    license: str
    support_status: str
    cache_status: CacheStatus
    role: str
    loader: Loader
    expected_route: str
    enabled_stage: str
    group: str
    params: str
    quantization: str
    release_verified: bool
    modelscope_id: str | None = None
    hf_id: str | None = None
    cache_dir_name: str | None = None
    requires_external_code: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_baseline(self) -> bool:
        return self.cache_status == "baseline"

    @property
    def default_modelscope_id(self) -> str | None:
        return self.modelscope_id or self.model_id

    @property
    def default_hf_id(self) -> str:
        return self.hf_id or self.model_id

    @property
    def expected_modelscope_path(self) -> Path | None:
        if self.cache_status == "baseline":
            return ROOT / self.model_id
        mid = self.default_modelscope_id
        if not mid:
            return None
        dirname = self.cache_dir_name or mid.replace("/", "--")
        return MS_CACHE / "models" / dirname


CANDIDATES: dict[str, Candidate] = {
    "qwen3vl_2b_baseline": Candidate(
        key="qwen3vl_2b_baseline",
        model_id="models/Qwen3-VL-2B-Instruct",
        source_url="/users/u2024311136/shared/shared_models/Qwen3-VL-2B-Instruct",
        license="unverified",
        support_status="Existing project baseline; only permitted non-2026 reference.",
        cache_status="baseline",
        role="baseline",
        loader="baseline_qwen3vl",
        expected_route="local shared model via eval_baseline.load_model",
        enabled_stage="all",
        group="baseline",
        params="2B",
        quantization="bf16",
        release_verified=False,
    ),
    "minicpm_v46": Candidate(
        key="minicpm_v46",
        model_id="OpenBMB/MiniCPM-V-4.6",
        hf_id="openbmb/MiniCPM-V-4.6",
        source_url="https://huggingface.co/openbmb/MiniCPM-V-4.6",
        license="Apache-2.0",
        support_status="HF task image-text-to-text; Transformers AutoModelForImageTextToText + AutoProcessor; vLLM/SGLang/llama.cpp/Ollama and quantized variants are provided.",
        cache_status="cached",
        role="primary-lightweight",
        loader="minicpm",
        expected_route="transformers",
        enabled_stage="all",
        group="cached",
        params="1.3B",
        quantization="bf16",
        release_verified=True,
    ),
    "qwen3remote_4b": Candidate(
        key="qwen3remote_4b",
        model_id="kaiz0603/qwen3remote",
        source_url="https://huggingface.co/kaiz0603/qwen3remote",
        license="Apache-2.0",
        support_status="HF repository created 2026-06-22 with complete Qwen3-VL-4B BF16 weights. Repository naming indicates remote-sensing specialization, but the generic model card does not disclose training data or benchmark results.",
        cache_status="download-first",
        role="primary-remote-specialized",
        loader="qwen3vl_dense",
        expected_route="transformers",
        enabled_stage="all",
        group="download-first",
        params="4B",
        quantization="bf16",
        release_verified=True,
    ),
    "uhr_bat": Candidate(
        key="uhr_bat",
        model_id="RL-MIND/UHR-BAT",
        source_url="https://huggingface.co/RL-MIND/UHR-BAT",
        license="Apache-2.0",
        support_status="ICML 2026 remote-sensing VLM with complete HF weights and remote-code inference; reports XLRS-Bench and MMERealworld-RS results. Single-image tasks only in the unified adapter.",
        cache_status="download-first",
        role="primary-remote-specialized",
        loader="uhr_bat",
        expected_route="transformers-remote-code",
        enabled_stage="all-single-image",
        group="download-first",
        params="7B",
        quantization="bf16",
        release_verified=True,
    ),
    "miril_drone_2b1": Candidate(
        key="miril_drone_2b1",
        model_id="MirilAI/Miril-Drone-2B-1",
        source_url="https://huggingface.co/MirilAI/Miril-Drone-2B-1",
        license="Apache-2.0",
        support_status="HF repository released 2026-07-07 with complete Gemma 4 E2B BF16 weights, official AutoModelForImageTextToText quickstart, and aerial-image caption/VQA training on a WALDO subset.",
        cache_status="download-first",
        role="primary-aerial-lightweight",
        loader="auto_image_text",
        expected_route="transformers",
        enabled_stage="all",
        group="download-first",
        params="2B-class",
        quantization="bf16",
        release_verified=True,
    ),
    "qwen35_4b": Candidate(
        key="qwen35_4b",
        model_id="Qwen/Qwen3.5-4B",
        source_url="https://huggingface.co/Qwen/Qwen3.5-4B",
        license="Apache-2.0",
        support_status="HF task image-text-to-text; use Qwen3_5ForConditionalGeneration + AutoProcessor for image-conditioned inference.",
        cache_status="cached",
        role="primary-mid-size",
        loader="qwen35_dense",
        expected_route="transformers",
        enabled_stage="all",
        group="cached",
        params="4B",
        quantization="bf16",
        release_verified=True,
    ),
    "qwen36_27b_fp8": Candidate(
        key="qwen36_27b_fp8",
        model_id="Qwen/Qwen3.6-27B-FP8",
        source_url="https://huggingface.co/Qwen/Qwen3.6-27B-FP8",
        license="Apache-2.0",
        support_status="HF task image-text-to-text; FP8 quantized; Transformers/vLLM/SGLang/KTransformers compatible, but A100 FP8 route requires fit validation.",
        cache_status="cached",
        role="large-fp8-fit-only",
        loader="qwen35_dense",
        expected_route="transformers-fit-then-serving-fallback",
        enabled_stage="fit",
        group="cached",
        params="27B",
        quantization="FP8",
        release_verified=True,
    ),
    "qwen36_35b_a3b_fp8": Candidate(
        key="qwen36_35b_a3b_fp8",
        model_id="Qwen/Qwen3.6-35B-A3B-FP8",
        source_url="https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8",
        license="Apache-2.0",
        support_status="HF task image-text-to-text; FP8 quantized MoE; Transformers/vLLM/SGLang/KTransformers compatible, but A100 FP8 route requires fit validation.",
        cache_status="cached",
        role="large-fp8-fit-only",
        loader="qwen35_moe",
        expected_route="transformers-fit-then-serving-fallback",
        enabled_stage="fit",
        group="cached",
        params="35B/3B active",
        quantization="FP8",
        release_verified=True,
    ),
    "internvlu_4b": Candidate(
        key="internvlu_4b",
        model_id="InternVL-U/InternVL-U",
        source_url="https://github.com/OpenGVLab/InternVL-U",
        license="MIT",
        support_status="Official repo released inference code and checkpoint on 2026/03/06; custom internvlu.InternVLUPipeline.",
        cache_status="download-first",
        role="primary-new-4b",
        loader="internvlu",
        expected_route="custom-pipeline",
        enabled_stage="all",
        group="download-first",
        params="4B",
        quantization="bf16",
        release_verified=True,
        requires_external_code="external/InternVL-U",
    ),
    "qwen35_9b": Candidate(
        key="qwen35_9b",
        model_id="Qwen/Qwen3.5-9B",
        source_url="https://huggingface.co/Qwen/Qwen3.5-9B",
        license="Apache-2.0",
        support_status="HF task image-text-to-text; use Qwen3_5ForConditionalGeneration + AutoProcessor.",
        cache_status="download-first",
        role="primary-9b",
        loader="qwen35_dense",
        expected_route="transformers",
        enabled_stage="all",
        group="download-first",
        params="9B",
        quantization="bf16",
        release_verified=True,
    ),
    "qwen35_35b_a3b_gptq_int4": Candidate(
        key="qwen35_35b_a3b_gptq_int4",
        model_id="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
        source_url="https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
        license="Apache-2.0",
        support_status="HF task image-text-to-text; GPTQ 4-bit; use Qwen3_5MoeForConditionalGeneration + AutoProcessor without load_in_4bit.",
        cache_status="download-first",
        role="large-int4",
        loader="qwen35_moe",
        expected_route="transformers-gptq",
        enabled_stage="all",
        group="download-first",
        params="35B/3B active",
        quantization="GPTQ-Int4",
        release_verified=True,
    ),
    "minicpm_v46_thinking": Candidate(
        key="minicpm_v46_thinking",
        model_id="OpenBMB/MiniCPM-V-4.6-Thinking",
        hf_id="openbmb/MiniCPM-V-4.6-Thinking",
        source_url="https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking",
        license="Apache-2.0",
        support_status="HF task image-text-to-text; Transformers AutoModelForImageTextToText; long-CoT reasoning variant.",
        cache_status="download-first",
        role="secondary-reasoning",
        loader="minicpm",
        expected_route="transformers",
        enabled_stage="mini-after-minicpm",
        group="download-first",
        params="1.3B",
        quantization="bf16",
        release_verified=True,
    ),
    "minicpm_v46_awq": Candidate(
        key="minicpm_v46_awq",
        model_id="OpenBMB/MiniCPM-V-4.6-AWQ",
        hf_id="openbmb/MiniCPM-V-4.6-AWQ",
        source_url="https://huggingface.co/openbmb/MiniCPM-V-4.6-AWQ",
        license="Apache-2.0",
        support_status="HF task image-text-to-text; AWQ W4A16 quantized deployment variant of MiniCPM-V-4.6.",
        cache_status="download-first",
        role="deployment-variant",
        loader="minicpm",
        expected_route="transformers-awq",
        enabled_stage="variant-after-minicpm",
        group="download-first",
        params="1.3B",
        quantization="AWQ-W4A16",
        release_verified=True,
    ),
}


def get_candidate(key: str) -> Candidate:
    try:
        return CANDIDATES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown candidate {key!r}. Available: {', '.join(CANDIDATES)}") from exc


def iter_candidates(include_baseline: bool = True):
    for candidate in CANDIDATES.values():
        if include_baseline or not candidate.is_baseline:
            yield candidate
