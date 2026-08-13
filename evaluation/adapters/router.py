"""Task Router：把 prompt 路由到对应 Expert LoRA。

单 Qwen3.5-4B 基座 + 3 个 LoRA Adapter，动态切换（AdapterManager）。
"""
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from evaluation.base.adapter import BaseModelAdapter
from evaluation.router import rules


class RouterAdapter(BaseModelAdapter):
    """单基座 + 多 LoRA，按 prompt 路由并动态切换 Adapter。"""

    ADAPTER_PATHS = {
        rules.GENERAL: None,
        rules.GROUNDING: None,
        rules.CHANGE: None,
    }

    def __init__(self, model_path: str, general_lora: str, grounding_lora: str,
                 change_lora: str, device: str = "cuda",
                 max_new_tokens: int = 256, system_prompt: str = "",
                 image_min_pixels: int = None, image_max_pixels: int = None):
        self.max_new_tokens = max_new_tokens
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if image_min_pixels is not None:
            self.processor.image_min_pixels = image_min_pixels
        if image_max_pixels is not None:
            self.processor.image_max_pixels = image_max_pixels
        self.processor.tokenizer.padding_side = "left"

        torch.cuda.reset_peak_memory_stats()
        print("  Loading base model ...", flush=True)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, general_lora, adapter_name="general")
        self.model.load_adapter(grounding_lora, adapter_name="grounding")
        self.model.load_adapter(change_lora, adapter_name="change")
        self.model = self.model.to(self.device)
        self.model.eval()

        self._adapter_map = {
            rules.GENERAL: "general",
            rules.GROUNDING: "grounding",
            rules.CHANGE: "change",
        }
        self.active_adapter = None
        self.peak_memory = torch.cuda.max_memory_allocated(self.device)
        print(f"  Peak memory after load: {self.peak_memory / 1e9:.2f} GB", flush=True)

    def _switch(self, expert: str):
        name = self._adapter_map[expert]
        if self.active_adapter != name:
            self.model.set_adapter(name)
            self.active_adapter = name

    def _extract_answer(self, text: str) -> str:
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        return text.strip()

    def _run_batch(self, batch, max_new_tokens=None):
        import time
        from qwen_vl_utils import process_vision_info

        tok = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        results = [None] * len(batch)
        groups = {}
        for i, (images, prompt) in enumerate(batch):
            expert = rules.route(prompt)
            groups.setdefault(expert, []).append(i)

        for expert, indices in groups.items():
            self._switch(expert)
            sub = [batch[i] for i in indices]
            texts, all_images, all_videos = [], [], []
            for images, prompt in sub:
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
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False)
                text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                im, vid = process_vision_info(messages)
                texts.append(text)
                all_images.append(im)
                all_videos.append(vid)

            video_arg = all_videos if any(v is not None for v in all_videos) else None
            inputs = self.processor(
                text=texts, images=all_images, videos=video_arg,
                padding=True, return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs, max_new_tokens=tok,
                    do_sample=False, temperature=None, top_p=None,
                )
            trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated)
            ]
            outputs = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for j, idx in enumerate(indices):
                results[idx] = self._extract_answer(outputs[j])

        return results

    def batch_generate(self, batch, batch_size=4, max_new_tokens=None):
        import time
        start = time.time()
        results = []
        for i in range(0, len(batch), batch_size):
            results.extend(self._run_batch(batch[i:i + batch_size], max_new_tokens=max_new_tokens))
        self.inference_time = time.time() - start
        self.inference_samples_per_sec = len(batch) / self.inference_time if self.inference_time > 0 else 0
        print(f"  Inference: {len(batch)} samples in {self.inference_time:.1f}s "
              f"({self.inference_samples_per_sec:.2f} samples/s)", flush=True)
        return results

    def generate(self, images, prompt):
        return self._run_batch([(images, prompt)])[0]
