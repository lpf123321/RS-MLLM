from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

import torch


def add_cvsearch_import_path(config: dict[str, Any]) -> Path:
    repository = Path(config["repositories"]["cvsearch"])
    path = repository / "cvsearch"
    for candidate in (repository, path):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    return path


def build_cvsearch_mllm(config: dict[str, Any], model_path: str, device: str = "cuda:0"):
    add_cvsearch_import_path(config)
    from models.modeling_qwenvl import ModelQwenVL

    model_config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    architectures = set(model_config.get("architectures", []))

    if "Qwen3VLForConditionalGeneration" in architectures:
        class ModelQwen3VL(ModelQwenVL):
            def get_prompt_from_qs(self, qs, response=None, show_prompt=False):
                messages = [{"role": "user", "content": []}]
                while "<image>\n" in qs:
                    index = qs.find("<image>\n")
                    if index == 0:
                        messages[0]["content"].append({"type": "image"})
                        qs = qs[len("<image>\n"):]
                    else:
                        messages[0]["content"].append({"type": "text", "text": qs[:index]})
                        qs = qs[index:]
                if qs:
                    messages[0]["content"].append({"type": "text", "text": qs})
                if response is not None:
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
                return self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=response is None, enable_thinking=False,
                )

        cv_cfg = config["cvsearch"]
        return ModelQwen3VL(
            model_path, device=device, torch_dtype=torch.bfloat16,
            max_pixels=int(cv_cfg["max_pixels"]), min_pixels=int(cv_cfg["min_pixels"]),
            patch_scale=float(cv_cfg["patch_scale"]),
        )

    if "Qwen3_5ForConditionalGeneration" not in architectures:
        raise ValueError(f"Unsupported MLLM architectures: {sorted(architectures)}")

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    class ModelQwen35(ModelQwenVL):
        def __init__(self) -> None:
            self.device = device
            self.dtype = torch.bfloat16
            self.use_flash_attn = True
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            self.tokenizer = self.processor.tokenizer
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_path,
                dtype=self.dtype,
                attn_implementation="flash_attention_2",
                device_map=device,
                local_files_only=True,
            )
            cv_cfg = config["cvsearch"]
            self.processor.image_processor.max_pixels = int(cv_cfg["max_pixels"])
            self.processor.image_processor.min_pixels = int(cv_cfg["min_pixels"])
            self.bias_value = 0.6
            self.input_size = (448, 448)
            self.view_size = 336
            self.scale_size = 672
            self.background_color = tuple(int(x * 255) for x in self.processor.image_processor.image_mean)
            self.patch_scale = float(cv_cfg["patch_scale"])
            self.init_prompts()
            yes_ids = self.tokenizer("Yes", add_special_tokens=False).input_ids
            no_ids = self.tokenizer("No", add_special_tokens=False).input_ids
            if not yes_ids or not no_ids:
                raise ValueError("Tokenizer produced no tokens for Yes/No")
            self.index_yes = yes_ids[-1]
            self.index_no = no_ids[-1]

        def get_prompt_from_qs(self, qs, response=None, show_prompt=False):
            messages = [{"role": "user", "content": []}]
            while "<image>\n" in qs:
                index = qs.find("<image>\n")
                if index == 0:
                    messages[0]["content"].append({"type": "image"})
                    qs = qs[len("<image>\n"):]
                else:
                    messages[0]["content"].append({"type": "text", "text": qs[:index]})
                    qs = qs[index:]
            if qs:
                messages[0]["content"].append({"type": "text", "text": qs})
            if response is not None:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=response is None, enable_thinking=False,
            )

    return ModelQwen35()


# Backward-compatible name for already-created run metadata/imports.
build_qwen35_cvsearch_model = build_cvsearch_mllm


def build_tracing_sam(config: dict[str, Any], model_path: str):
    add_cvsearch_import_path(config)
    from models.modeling_sam3 import sam3_inference

    class TracingSAM3(sam3_inference):
        def __init__(self) -> None:
            from sam3.eval.postprocessors import PostProcessImage
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
            from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI

            bpe = Path(config["repositories"]["cvsearch"]) / "assets/bpe_simple_vocab_16e6.txt.gz"
            self.model = build_sam3_image_model(
                checkpoint_path=model_path, load_from_HF=False, bpe_path=str(bpe),
                device="cuda", eval_mode=True, enable_segmentation=True, compile=False,
            )
            self.model.eval().to("cuda")
            self.processor = Sam3Processor(self.model)
            self.transform = ComposeAPI(transforms=[
                RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                ToTensorAPI(), NormalizeAPI(mean=[0.5] * 3, std=[0.5] * 3),
            ])
            self.postprocessor = PostProcessImage(
                max_dets_per_img=-1, iou_type="segm", use_original_sizes_box=True,
                use_original_sizes_mask=True, convert_mask_to_rle=False,
                detection_threshold=0.5, to_cpu=False,
            )
            self.GLOBAL_COUNTER = 1
            self.calls: list[dict[str, Any]] = []
            self._trace_context: dict[str, Any] | None = None

        def reset_trace(self) -> None:
            self.calls.clear()
            self._trace_context = None

        def set_trace_context(self, *, stage: str, crop_xyxy, target_indices) -> None:
            self._trace_context = {
                "stage": str(stage),
                "crop_xyxy": [int(value) for value in crop_xyxy],
                "target_indices": [int(value) for value in target_indices],
            }

        def batch_inference(self, image, text_prompts):
            context = self._trace_context or {
                "stage": "unlabeled",
                "crop_xyxy": [0, 0, image.width, image.height],
                "target_indices": list(range(len(text_prompts))),
            }
            self._trace_context = None
            backbone, results, target_ids = super().batch_inference(image, text_prompts)
            predictions = []
            for prompt_position, (prompt, target_id) in enumerate(zip(text_prompts, target_ids)):
                result = results[target_id]
                boxes = result["boxes"].detach().float().cpu()
                scores = result["scores"].detach().float().cpu()
                masks = result.get("masks")
                if masks is None:
                    mask_tensor = torch.empty((0, image.height, image.width), dtype=torch.uint8)
                else:
                    mask_tensor = masks.detach().cpu()
                    if mask_tensor.ndim == 4 and mask_tensor.shape[1] == 1:
                        mask_tensor = mask_tensor[:, 0]
                    elif mask_tensor.ndim == 2:
                        mask_tensor = mask_tensor.unsqueeze(0)
                    mask_tensor = (mask_tensor > 0).to(torch.uint8)
                if int(mask_tensor.shape[0]) != int(boxes.shape[0]):
                    raise RuntimeError(
                        f"SAM trace box/mask count mismatch: {boxes.shape[0]} boxes, "
                        f"{mask_tensor.shape[0]} masks"
                    )
                if context["stage"] == "initial":
                    retained_indices = list(range(len(scores)))
                else:
                    retained_indices = [
                        index for index, score in enumerate(scores.tolist()) if float(score) > 0.6
                    ]
                    if not retained_indices and len(scores):
                        retained_indices = [int(torch.argmax(scores).item())]
                target_indices = context.get("target_indices", [])
                predictions.append({
                    "prompt": str(prompt),
                    "target_index": int(target_indices[prompt_position])
                    if prompt_position < len(target_indices) else prompt_position,
                    "boxes_xyxy": boxes.tolist(),
                    "joint_scores": scores.tolist(),
                    "retained_prediction_indices": retained_indices,
                    "_mask_tensors": mask_tensor,
                })
            self.calls.append({
                "call_index": len(self.calls),
                "image_size": list(image.size),
                "stage": context["stage"],
                "crop_xyxy": context["crop_xyxy"],
                "predictions": predictions,
            })
            return backbone, results, target_ids

    return TracingSAM3()


@torch.inference_mode()
def answer_image(model, image, prompt: str, max_new_tokens: int = 32) -> str:
    formatted = model.get_prompt_from_qs(prompt)
    inputs = model.processor(text=[formatted], images=[image], return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    generated = model.model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=max_new_tokens)
    continuation = generated[:, inputs.input_ids.shape[1]:]
    return model.processor.batch_decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


@torch.inference_mode()
def answer_logprob(model, image, prompt: str, answer: str) -> float:
    prompt_text = model.get_prompt_from_qs(prompt)
    prompt_inputs = model.processor(text=[prompt_text], images=[image], return_tensors="pt", padding=True).to(model.device)
    # Teacher-force directly after the exact generation prompt. Some merged
    # processors serialize a completed assistant message differently from an
    # open generation prompt, so the two chat-template variants are not always
    # token-prefix compatible.
    full_text = prompt_text + answer
    full_inputs = model.processor(text=[full_text], images=[image], return_tensors="pt", padding=True).to(model.device)
    outputs = model.model(**full_inputs)
    prompt_ids = prompt_inputs.input_ids[0]
    full_ids = full_inputs.input_ids[0]
    common = min(prompt_ids.numel(), full_ids.numel())
    start = 0
    while start < common and int(prompt_ids[start]) == int(full_ids[start]):
        start += 1
    labels = full_inputs.input_ids[:, start:]
    if labels.numel() == 0:
        # Defend against a tokenizer boundary merge at the prompt/answer join.
        full_text = prompt_text + " " + answer
        full_inputs = model.processor(text=[full_text], images=[image], return_tensors="pt", padding=True).to(model.device)
        outputs = model.model(**full_inputs)
        full_ids = full_inputs.input_ids[0]
        common = min(prompt_ids.numel(), full_ids.numel())
        start = 0
        while start < common and int(prompt_ids[start]) == int(full_ids[start]):
            start += 1
        labels = full_inputs.input_ids[:, start:]
    if labels.numel() == 0 or start == 0:
        raise ValueError("Could not align answer tokens with the generation prompt")
    logits = outputs.logits[:, start - 1:full_inputs.input_ids.shape[1] - 1]
    token_logprobs = torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return float(token_logprobs.mean().item())
