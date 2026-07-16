#!/usr/bin/env python3
"""Evaluate RS-MLLM candidate VLMs through shared baseline dataset loops."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

import eval_baseline
from candidate_registry import CANDIDATES, CACHE_MANIFEST, MS_CACHE, RESULTS_DIR, ROOT, Candidate, get_candidate


class CandidateRuntimeError(RuntimeError):
    pass


class BaseAdapter:
    def __init__(self, candidate: Candidate, local_path: Path):
        self.candidate = candidate
        self.local_path = local_path
        self.model: Any = None
        self.processor: Any = None
        self.infer_times: list[float] = []
        self.unsupported: set[str] = set()

    def load(self) -> None:
        raise NotImplementedError

    def generate_messages(self, messages, max_new_tokens: int = 256) -> str:
        raise NotImplementedError

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    @classmethod
    def _trim_decode(cls, processor, inputs, generated_ids) -> str:
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return cls._clean_text(text)

    @staticmethod
    def _extract_images_and_text(messages) -> tuple[list[Image.Image], str]:
        images: list[Image.Image] = []
        texts: list[str] = []
        for msg in messages:
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    image = item.get("image")
                    if isinstance(image, Image.Image):
                        images.append(image)
                elif item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
        return images, "\n".join(t for t in texts if t)

    def timed(self, fn):
        start = time.time()
        try:
            return fn()
        finally:
            self.infer_times.append(time.time() - start)


class BaselineQwen3VLAdapter(BaseAdapter):
    def load(self) -> None:
        self.model, self.processor = eval_baseline.load_model("cuda")

    def generate_messages(self, messages, max_new_tokens: int = 256) -> str:
        def run():
            # Call the original baseline path by bypassing adapter dispatch.
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
            return self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        return self.timed(run)


class Qwen3VLDenseAdapter(BaseAdapter):
    def load(self) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(str(self.local_path), trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.local_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    def generate_messages(self, messages, max_new_tokens: int = 256) -> str:
        def run():
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self._trim_decode(self.processor, inputs, generated_ids)
        return self.timed(run)


class MiniCPMAdapter(BaseAdapter):
    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(str(self.local_path), trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.local_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

    def generate_messages(self, messages, max_new_tokens: int = 256) -> str:
        def run():
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                downsample_mode="16x",
                max_slice_nums=36,
            ).to(self.model.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, downsample_mode="16x", max_new_tokens=max_new_tokens
                )
            return self._trim_decode(self.processor, inputs, generated_ids)
        return self.timed(run)


class Qwen35DenseAdapter(BaseAdapter):
    def load(self) -> None:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(str(self.local_path), trust_remote_code=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(self.local_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    def generate_messages(self, messages, max_new_tokens: int = 256) -> str:
        def run():
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self._trim_decode(self.processor, inputs, generated_ids)
        return self.timed(run)


class Qwen35MoeAdapter(BaseAdapter):
    def load(self) -> None:
        from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(str(self.local_path), trust_remote_code=True)
        self.model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            str(self.local_path),
            device_map="auto",
            trust_remote_code=True,
        )

    def generate_messages(self, messages, max_new_tokens: int = 256) -> str:
        def run():
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self._trim_decode(self.processor, inputs, generated_ids)
        return self.timed(run)


class AutoImageTextAdapter(BaseAdapter):
    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            str(self.local_path), trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.local_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    @staticmethod
    def _unwrap_structured_text(text: str) -> str:
        cleaned = BaseAdapter._clean_text(text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return cleaned
        if isinstance(payload, dict):
            for key in ("answer", "caption"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value.strip()
        return cleaned

    def generate_messages(self, messages, max_new_tokens: int = 256) -> str:
        def run():
            images, _ = self._extract_images_and_text(messages)
            if not images:
                raise CandidateRuntimeError(
                    "AutoImageText adapter received no PIL images"
                )
            rendered = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[rendered],
                images=[images],
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            generated = output[:, inputs["input_ids"].shape[-1]:]
            decoded = self.processor.batch_decode(
                generated, skip_special_tokens=True
            )[0]
            return self._unwrap_structured_text(decoded)

        return self.timed(run)


class UHRBATAdapter(BaseAdapter):
    target_sizes = [672, 1344, 2688, 4032]
    multiscale_topk = [80, 320, 600, 2000]

    def __init__(self, candidate: Candidate, local_path: Path):
        super().__init__(candidate, local_path)
        self.tokenizer: Any = None
        self.uhrbat: Any = None

    def load(self) -> None:
        import importlib
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.local_path), trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.processor = AutoImageProcessor.from_pretrained(str(self.local_path))
        config = AutoConfig.from_pretrained(str(self.local_path), trust_remote_code=True)
        config.mm_vision_tower = str(resolve_uhr_vision_tower())
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.local_path),
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.uhrbat = importlib.import_module(self.model.__class__.__module__)

    @staticmethod
    def _chatml_prompt(question: str) -> str:
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<image>\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate_messages(self, messages, max_new_tokens: int = 256) -> str:
        def run():
            images, question = self._extract_images_and_text(messages)
            if len(images) != 1:
                if len(images) > 1:
                    self.unsupported.add("levir_cc")
                raise CandidateRuntimeError(
                    f"UHR-BAT adapter requires exactly one image, received {len(images)}"
                )

            image = images[0].convert("RGB")
            prompt = self._chatml_prompt(question)
            image_token_id = getattr(self.model.config, "image_token_index", -200)
            input_ids = self.uhrbat.tokenizer_image_token(
                prompt,
                self.tokenizer,
                image_token_id,
                return_tensors="pt",
            ).unsqueeze(0).to(self.model.device)
            attention_mask = torch.ones_like(input_ids)
            multiscale_pixels = [
                self.uhrbat.split_image_to_multiscale_tiles(
                    image,
                    self.processor,
                    target_sizes=self.target_sizes,
                    tile_size=336,
                )
            ]

            with torch.inference_mode():
                output = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    image_sizes=[image.size],
                    modalities=["image"],
                    multiscale_pixels=multiscale_pixels,
                    multiscale_masks=[{}],
                    multiscale_topk=self.multiscale_topk,
                    multiscale_target_sizes=self.target_sizes,
                    do_sample=False,
                    repetition_penalty=1.0,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            prompt_len = output.sequences.shape[1] - len(output.scores)
            answer_ids = output.sequences[:, prompt_len:].clone()
            replacement_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            answer_ids[answer_ids < 0] = replacement_id
            return self._clean_text(
                self.tokenizer.decode(answer_ids[0], skip_special_tokens=True)
            )

        return self.timed(run)


class InternVLUAdapter(BaseAdapter):
    def load(self) -> None:
        external = ROOT / "external" / "InternVL-U"
        if str(external) not in sys.path:
            sys.path.insert(0, str(external))
        from internvlu import InternVLUPipeline
        self.model = InternVLUPipeline.from_pretrained(
            str(self.local_path), torch_dtype=torch.bfloat16
        )
        self.model.to("cuda")
        self.processor = self.model.processor

    def generate_messages(self, messages, max_new_tokens: int = 256) -> str:
        def run():
            images, prompt = self._extract_images_and_text(messages)
            if not images:
                raise CandidateRuntimeError("InternVL-U adapter received no PIL images")
            image_arg: Any = images[0]
            if len(images) > 1:
                image_arg = [images]
            try:
                output = self.model(
                    prompt=prompt,
                    image=image_arg,
                    max_new_tokens=max_new_tokens,
                    generation_mode="text",
                )
            except Exception as exc:
                if len(images) > 1:
                    self.unsupported.add("levir_cc")
                raise exc
            return self._clean_text(self.processor.tokenizer.decode(output.generate_output[0], skip_special_tokens=True))
        return self.timed(run)


ADAPTERS = {
    "baseline_qwen3vl": BaselineQwen3VLAdapter,
    "qwen3vl_dense": Qwen3VLDenseAdapter,
    "minicpm": MiniCPMAdapter,
    "qwen35_dense": Qwen35DenseAdapter,
    "qwen35_moe": Qwen35MoeAdapter,
    "internvlu": InternVLUAdapter,
    "uhr_bat": UHRBATAdapter,
    "auto_image_text": AutoImageTextAdapter,
}


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def first_snapshot(path: Path) -> Path | None:
    if not path.exists():
        return None
    snapshots = path / "snapshots"
    if snapshots.exists():
        entries = sorted(snapshots.iterdir())
        if entries:
            return entries[-1]
    return path

def resolve_uhr_vision_tower(cache_root: Path = MS_CACHE) -> Path:
    cache_path = cache_root / "models" / "openai--clip-vit-large-patch14-336"
    snapshot = first_snapshot(cache_path)
    if snapshot and (snapshot / "preprocessor_config.json").exists():
        if (snapshot / "pytorch_model.bin").exists() or (snapshot / "model.safetensors").exists():
            return snapshot
    raise FileNotFoundError(
        "UHR-BAT requires a local openai/clip-vit-large-patch14-336 snapshot; "
        f"expected it under {cache_path}"
    )


def resolve_local_path(candidate: Candidate) -> Path:
    if candidate.is_baseline:
        return ROOT / candidate.model_id

    manifest = read_json(CACHE_MANIFEST, {})
    entry = manifest.get(candidate.key, {}) if isinstance(manifest, dict) else {}
    if entry.get("cache_path"):
        path = Path(entry["cache_path"])
        if path.exists():
            return path

    ms_path = candidate.expected_modelscope_path
    if ms_path:
        snapshot = first_snapshot(ms_path)
        if snapshot and snapshot.exists():
            return snapshot

    raise FileNotFoundError(f"No cache found for {candidate.key}; run download_models.py first")


def load_adapter(candidate: Candidate) -> tuple[BaseAdapter, float]:
    local_path = resolve_local_path(candidate)
    adapter_cls = ADAPTERS[candidate.loader]
    adapter = adapter_cls(candidate, local_path)
    start = time.time()
    adapter.load()
    return adapter, time.time() - start


def set_jsonl_path(candidate: Candidate, dataset: str, stage: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))
    path = Path(os.environ.get(
        "CANDIDATE_JSONL",
        str(RESULTS_DIR / f"candidate_{candidate.key}_{dataset}_{stage}_{job_id}.jsonl"),
    ))
    eval_baseline.JSONL_PATH = path
    if path.exists():
        path.unlink()
    return path


def run_dataset(adapter: BaseAdapter, dataset: str, task: str, max_samples: int | None) -> list[dict]:
    if dataset == "vrsbench":
        return eval_baseline.eval_vrsbench(adapter, None, task=task, max_samples=max_samples)
    if dataset == "xlrs_lite":
        return eval_baseline.eval_xlrs_lite(adapter, None, max_samples=max_samples)
    if dataset == "mme":
        return eval_baseline.eval_mme_realworld(adapter, None, max_samples=max_samples)
    if dataset == "levir_cc":
        return eval_baseline.eval_levir_cc(adapter, None, max_samples=max_samples)
    if dataset == "all":
        results: list[dict] = []
        results.extend(eval_baseline.eval_vrsbench(adapter, None, task=task, max_samples=max_samples))
        results.extend(eval_baseline.eval_xlrs_lite(adapter, None, max_samples=max_samples))
        results.extend(eval_baseline.eval_mme_realworld(adapter, None, max_samples=max_samples))
        results.extend(eval_baseline.eval_levir_cc(adapter, None, max_samples=max_samples))
        return results
    raise ValueError(f"Unsupported dataset: {dataset}")


def accuracy_by_task(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for row in rows:
        key = f"{row.get('dataset')}/{row.get('task')}"
        stat = summary.setdefault(key, {"correct": 0, "total": 0, "acc": 0.0})
        stat["total"] = int(stat["total"]) + 1
        stat["correct"] = int(stat["correct"]) + int(bool(row.get("correct")))
    for stat in summary.values():
        total = int(stat["total"])
        stat["acc"] = float(stat["correct"]) / total if total else 0.0
    return summary


def count_infer_errors(rows: list[dict]) -> int:
    return sum(1 for row in rows if str(row.get("prediction", "")).startswith("[INFER ERROR]"))


def resolve_dataset_sample(dataset: str, task: str) -> dict[str, Any]:
    """Resolve one dataset image without loading a model; used by CLI dry checks."""
    if dataset == "vrsbench":
        task_name = "vqa" if task == "all" else task
        task_files = {
            "caption": eval_baseline.VRSBENCH_DIR / "VRSBench_EVAL_Cap.json",
            "vqa": eval_baseline.VRSBENCH_DIR / "VRSBench_EVAL_vqa.json",
            "refer": eval_baseline.VRSBENCH_DIR / "VRSBench_EVAL_referring.json",
        }
        sample = json.loads(task_files[task_name].read_text())[0]
        image_path = eval_baseline.VRSBENCH_DIR / "images" / "val" / sample["image_id"]
        if not image_path.exists():
            raise FileNotFoundError(f"Sample image is missing: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        return {
            "dataset": dataset,
            "task": task_name,
            "json": str(task_files[task_name]),
            "image_path": str(image_path),
            "image_size": [width, height],
            "image_id": sample["image_id"],
        }
    raise ValueError("--resolve-only currently supports --dataset vrsbench for a non-extracting smoke check")


def sanity_errors(rows: list[dict], stage: str, dataset: str) -> list[str]:
    errors: list[str] = []
    if stage == "fit" and dataset == "vrsbench":
        if len(rows) < 3:
            errors.append(f"fit gate expected at least 3 predictions, got {len(rows)}")
        if count_infer_errors(rows):
            errors.append("fit gate has inference errors")
        yellow_rows = [
            row for row in rows
            if row.get("dataset") == "vrsbench"
            and row.get("task") == "vqa"
            and str(row.get("ground_truth", "")).strip().lower() == "yellow"
        ]
        if not yellow_rows:
            errors.append("fit gate did not find the known Yellow VQA sanity row")
        else:
            pred = str(yellow_rows[0].get("prediction", "")).lower()
            if "yellow" not in pred:
                errors.append("fit gate Yellow VQA sanity answer did not contain yellow")
    return errors


def peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)


def runtime_warnings_for(adapter: BaseAdapter | None) -> list[str]:
    """Warnings that gate deployment suitability without erasing completed runs."""
    if adapter is None or adapter.model is None:
        return []
    warnings: list[str] = []
    device_map = getattr(adapter.model, "hf_device_map", None)
    if isinstance(device_map, dict):
        offloaded = sorted({str(device) for device in device_map.values() if str(device) in {"cpu", "disk"}})
        if offloaded:
            warnings.append(
                "runtime_gated_offload: model completed inference but uses "
                f"{','.join(offloaded)} offload, unsuitable for single-A100 efficient deployment"
            )
    if adapter.candidate.quantization.upper() == "FP8" and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability < (8, 9):
            warnings.append(
                "runtime_gated_fp8_fallback: A100 compute capability "
                f"{capability[0]}.{capability[1]} cannot run this Transformers FP8 path natively"
            )
    return warnings


def write_summary(candidate: Candidate, dataset: str, stage: str, load_ok: bool, load_s: float,
                  adapter: BaseAdapter | None, rows: list[dict], errors: list[str],
                  runtime_warnings: list[str] | None = None) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))
    avg_infer = None
    if adapter and adapter.infer_times:
        avg_infer = round(sum(adapter.infer_times) / len(adapter.infer_times), 3)
    summary = {
        "candidate": candidate.key,
        "model_id": candidate.model_id,
        "stage": stage,
        "dataset": dataset,
        "load_ok": load_ok,
        "load_s": round(load_s, 3),
        "peak_mem_mb": peak_memory_mb(),
        "avg_infer_s": avg_infer,
        "num_samples": len(rows),
        "accuracy_by_task": accuracy_by_task(rows),
        "runtime_warnings": runtime_warnings or [],
        "errors": errors,
    }
    path = RESULTS_DIR / f"candidate_{candidate.key}_{dataset}_{stage}_{job_id}_summary.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved summary: {path}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--dataset", choices=["vrsbench", "xlrs_lite", "mme", "levir_cc", "all"], default="vrsbench")
    parser.add_argument("--task", choices=["caption", "vqa", "refer", "all"], default="all")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--stage", choices=["fit", "mini", "full"], default="fit")
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Resolve candidate cache and one dataset image, then exit without loading the model.",
    )
    args = parser.parse_args()

    eval_baseline.RESULTS_DIR.mkdir(exist_ok=True)
    eval_baseline.EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    candidate = get_candidate(args.candidate)
    if args.resolve_only:
        local_path = resolve_local_path(candidate)
        sample = resolve_dataset_sample(args.dataset, args.task)
        print(json.dumps({
            "candidate": candidate.key,
            "model_id": candidate.model_id,
            "local_path": str(local_path),
            "sample": sample,
        }, ensure_ascii=False, indent=2), flush=True)
        return
    set_jsonl_path(candidate, args.dataset, args.stage)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    adapter: BaseAdapter | None = None
    load_ok = False
    rows: list[dict] = []
    errors: list[str] = []
    load_s = 0.0
    try:
        adapter, load_s = load_adapter(candidate)
        load_ok = True
        rows = run_dataset(adapter, args.dataset, args.task, args.max_samples)
        errors.extend(sanity_errors(rows, args.stage, args.dataset))
        if adapter.unsupported:
            errors.append("unsupported datasets: " + ",".join(sorted(adapter.unsupported)))
        infer_errors = count_infer_errors(rows)
        if rows and infer_errors / len(rows) > 0.05:
            errors.append(f"inference error rate >5%: {infer_errors}/{len(rows)}")
        if peak_memory_mb() > 39000:
            errors.append(f"peak memory exceeds A100 gate: {peak_memory_mb()} MB")
    except Exception as exc:
        errors.append(f"load_or_eval: {type(exc).__name__}: {exc}")
        print(f"ERROR: {errors[-1]}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        write_summary(
            candidate, args.dataset, args.stage, load_ok, load_s, adapter, rows, errors,
            runtime_warnings=runtime_warnings_for(adapter),
        )

    if errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
