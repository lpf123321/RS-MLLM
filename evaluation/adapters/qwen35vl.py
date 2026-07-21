import json
import os
import re
from typing import List, Tuple

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from evaluation.base.adapter import BaseModelAdapter


THINKING_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class Qwen35VLAdapter(BaseModelAdapter):
    def __init__(self, model_path: str, device: str = "cuda",
                 max_new_tokens: int = 256, compile_model: bool = False,
                 system_prompt: str = "", disable_thinking: bool = True):
        self.max_new_tokens = max_new_tokens
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.disable_thinking = disable_thinking

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.processor.tokenizer.padding_side = "left"

        dt = torch.bfloat16
        if self.device == "cuda":
            cap = torch.cuda.get_device_capability()[0]
            if cap < 8 and not torch.cuda.is_bf16_supported():
                dt = torch.float16
                print(f"  GPU CC {cap} → using float16 (bf16 not supported)", flush=True)

        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dt,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.system_prompt = system_prompt

        if compile_model and self.device == "cuda":
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 8:
                print("  Compiling model with torch.compile ...", flush=True)
                self.model = torch.compile(self.model, mode="default")
            else:
                print(f"  GPU CC {cap} → torch.compile disabled (needs SM80+)", flush=True)

    def _build_messages(self, images: List[str], prompt: str) -> List[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({
            "role": "user",
            "content": (
                [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": prompt}]
            ),
        })
        return messages

    def _clean_output(self, text: str) -> str:
        text = THINKING_PATTERN.sub("", text)
        return text.strip()

    def generate(self, images: List[str], prompt: str) -> str:
        messages = self._build_messages(images, prompt)
        chat_kwargs = {"add_generation_prompt": True}
        if self.disable_thinking:
            chat_kwargs["enable_thinking"] = False
        text = self.processor.apply_chat_template(
            messages, tokenize=False, **chat_kwargs,
        )
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
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return self._clean_output(output_text)

    def batch_generate(self, batch: List[Tuple[List[str], str]],
                       batch_size: int = 4) -> List[str]:
        results = []
        for i in tqdm(range(0, len(batch), batch_size), desc="Inference", unit="batch"):
            sub_batch = batch[i:i + batch_size]
            results.extend(self._run_batch(sub_batch))
        return results

    def _run_batch(self, batch: List[Tuple[List[str], str]]) -> List[str]:
        texts = []
        all_images = []
        all_videos = []
        for images, prompt in batch:
            messages = self._build_messages(images, prompt)
            chat_kwargs = {"add_generation_prompt": True}
            if self.disable_thinking:
                chat_kwargs["enable_thinking"] = False
            text = self.processor.apply_chat_template(
                messages, tokenize=False, **chat_kwargs,
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
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [self._clean_output(t) for t in output_texts]
