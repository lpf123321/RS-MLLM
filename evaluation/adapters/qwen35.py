from typing import List, Tuple

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)

from evaluation.base.adapter import BaseModelAdapter


class Qwen35Adapter(BaseModelAdapter):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 256,
        compile_model: bool = False,
        system_prompt: str = "",
    ):
        self.max_new_tokens = max_new_tokens
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.system_prompt = system_prompt

        if compile_model and self.device == "cuda":
            print("  Compiling model with torch.compile ...")
            self.model = torch.compile(self.model, mode="default")

    def _build_messages(self, images: List[str], prompt: str):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({
            "role": "user",
            "content": (
                [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": prompt}]
            ),
        })
        return messages

    def generate(self, images: List[str], prompt: str) -> str:
        return self._run_batch([(images, prompt)])[0]

    def batch_generate(
        self,
        batch: List[Tuple[List[str], str]],
        batch_size: int = 4,
    ) -> List[str]:
        results = []
        for i in tqdm(range(0, len(batch), batch_size), desc="Inference", unit="batch"):
            results.extend(self._run_batch(batch[i:i + batch_size]))
        return results

    def _run_batch(self, batch: List[Tuple[List[str], str]]) -> List[str]:
        texts = []
        all_images = []
        all_videos = []

        for images, prompt in batch:
            messages = self._build_messages(images, prompt)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            texts.append(text)
            all_images.append(image_inputs)
            all_videos.append(video_inputs)

        video_arg = all_videos if any(v is not None for v in all_videos) else None
        inputs = self.processor(
            text=texts,
            images=all_images,
            videos=video_arg,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [text.strip() for text in output_texts]
