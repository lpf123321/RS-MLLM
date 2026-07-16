import json
import os
from typing import List

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None

from evaluation.base.adapter import BaseModelAdapter


def _get_model_class(model_path: str):
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        if any("Qwen3_5" in a for a in archs):
            if Qwen3_5ForConditionalGeneration is not None:
                return Qwen3_5ForConditionalGeneration
            raise ImportError(
                "Qwen3_5ForConditionalGeneration is not available in this transformers version. "
                "Please upgrade transformers."
            )
    return Qwen3VLForConditionalGeneration


class Qwen3VLAdapter(BaseModelAdapter):
    def __init__(self, model_path: str, device: str = "cuda", max_new_tokens: int = 256):
        self.max_new_tokens = max_new_tokens
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model_class = _get_model_class(model_path)
        self.model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, images: List[str], prompt: str) -> str:
        messages = [{
            "role": "user",
            "content": (
                [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": prompt}]
            ),
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip()
