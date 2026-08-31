"""Unified Qwen3.5 and Qwen3-VL generation adapter."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_policy import assert_model_allowed, verify_trusted_model_content


@dataclass(frozen=True)
class GenerationResult:
    text: str
    elapsed_seconds: float
    generated_tokens: int
    truncated: bool
    first_token_seconds: float = 0.0


def model_load_kwargs(torch: Any, quantization: str) -> dict[str, Any]:
    """Build explicit, auditable loading arguments for one precision mode."""
    if quantization not in {"bf16", "int8", "nf4"}:
        raise ValueError(f"Unsupported quantization mode: {quantization}")
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if quantization == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
        return kwargs
    from transformers import BitsAndBytesConfig

    if quantization == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        return kwargs
    if quantization == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return kwargs


def generation_kwargs(
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
    """Validate generation controls and omit inactive sampling arguments."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
    return kwargs


def prune_language_layers(model: Any, torch: Any, ratio: float) -> dict[str, Any]:
    """Apply per-matrix unstructured L1 pruning to transformer language layers."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError("prune_ratio must be in [0, 1)")
    if ratio == 0.0:
        return {"ratio": 0.0, "modules": 0, "parameters": 0, "zeros": 0}

    from torch.nn.utils import prune

    modules = 0
    parameters = 0
    zeros = 0
    for name, module in model.named_modules():
        normalized = name.lower()
        is_language_layer = ".layers." in normalized and (
            ".mlp." in normalized or ".self_attn." in normalized
        )
        if not is_language_layer or not isinstance(module, torch.nn.Linear):
            continue
        count = module.weight.numel()
        prune.l1_unstructured(module, name="weight", amount=ratio)
        prune.remove(module, "weight")
        modules += 1
        parameters += count
        zeros += int(torch.count_nonzero(module.weight == 0).item())
    if modules == 0:
        raise RuntimeError("No language-layer linear modules matched the pruning policy")
    return {
        "ratio": ratio,
        "modules": modules,
        "parameters": parameters,
        "zeros": zeros,
    }


class QwenAdapter:
    def __init__(
        self,
        model_path: Path,
        *,
        max_pixels: int,
        min_pixels: int,
        quantization: str = "bf16",
        prune_ratio: float = 0.0,
    ) -> None:
        assert_model_allowed(model_path)
        verify_trusted_model_content(model_path)
        self.model_path = model_path.resolve()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        if quantization not in {"bf16", "int8", "nf4"}:
            raise ValueError(f"Unsupported quantization mode: {quantization}")
        self.quantization = quantization
        if not 0.0 <= prune_ratio < 1.0:
            raise ValueError("prune_ratio must be in [0, 1)")
        if prune_ratio and quantization != "bf16":
            raise ValueError("Pruning requires bf16; do not combine it with quantization")
        self.prune_ratio = prune_ratio
        self.family = ""
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None

    def load(self) -> dict[str, Any]:
        import torch
        from transformers import AutoConfig, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Qwen evaluation")
        config = AutoConfig.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )
        self.family = str(config.model_type)
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            if hasattr(image_processor, "min_pixels"):
                image_processor.min_pixels = self.min_pixels
            if hasattr(image_processor, "max_pixels"):
                image_processor.max_pixels = self.max_pixels

        if self.family == "qwen3_5":
            from transformers import Qwen3_5ForConditionalGeneration

            model_class = Qwen3_5ForConditionalGeneration
        elif self.family == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration

            model_class = Qwen3VLForConditionalGeneration
        else:
            raise ValueError(
                f"Unsupported model_type {self.family!r}; expected qwen3_5 or qwen3_vl"
            )
        self.model = model_class.from_pretrained(
            str(self.model_path),
            **model_load_kwargs(torch, self.quantization),
        ).eval()
        pruning = prune_language_layers(self.model, torch, self.prune_ratio)
        self.torch = torch
        return {
            "model_type": self.family,
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "quantization": self.quantization,
            "pruning": pruning,
            "cuda_device": torch.cuda.get_device_name(),
            "cuda_capability": list(torch.cuda.get_device_capability()),
        }

    @staticmethod
    def _messages(
        image_paths: list[str], prompt: str, *, max_pixels: int, min_pixels: int
    ):
        content = [
            {
                "type": "image",
                "image": path,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
            for path in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def generate(
        self,
        image_paths: list[str],
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        seed: int = 0,
    ) -> GenerationResult:
        if self.model is None or self.processor is None:
            raise RuntimeError("Adapter must be loaded before generation")
        messages = self._messages(
            image_paths,
            prompt,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )
        started = time.perf_counter()
        if self.family == "qwen3_5":
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            from qwen_vl_utils import process_vision_info

            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[rendered],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        controls = generation_kwargs(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **controls)
        input_length = inputs["input_ids"].shape[-1]
        generated = generated_ids[:, input_length:]
        generated_tokens = int(generated.shape[-1])
        decoded = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        elapsed = time.perf_counter() - started
        return GenerationResult(
            text=decoded,
            elapsed_seconds=elapsed,
            generated_tokens=generated_tokens,
            truncated=generated_tokens >= max_new_tokens,
        )

    def peak_memory_mb(self) -> float:
        if self.torch is None or not self.torch.cuda.is_available():
            return 0.0
        return self.torch.cuda.max_memory_allocated() / 1024**2
