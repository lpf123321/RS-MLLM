"""Task Router：把 prompt 路由到对应 Expert LoRA。

单 Qwen3.5-4B 基座 + 3 个 LoRA Adapter，动态切换（AdapterManager）。
"""
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from evaluation.base.adapter import BaseModelAdapter
from evaluation.router import rules


class RouterAdapter(BaseModelAdapter):
    """单基座 + 多 expert LoRA，按 prompt 路由并动态切换 Adapter。

    expert 注册表由 ``expert_lora`` dict 驱动（任意数量的 expert），同时保留
    ``general_lora`` / ``grounding_lora`` / ``change_lora`` 作为便捷参数向后兼容。
    task -> expert 的映射见 ``evaluation.router.rules.TASK_TO_EXPERT``。
    """

    def __init__(self, model_path: str, general_lora: str = None,
                 grounding_lora: str = None, change_lora: str = None,
                 expert_lora: dict = None, device: str = "cuda",
                 max_new_tokens: int = 256, system_prompt: str = "",
                 image_min_pixels: int = None, image_max_pixels: int = None,
                 force_think: bool = True):
        self.max_new_tokens = max_new_tokens
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt
        self.force_think = force_think

        # 构建 expert 注册表：{expert_name: lora_path}
        expert_lora = dict(expert_lora or {})
        for name, path in (
            (rules.GENERAL, general_lora),
            (rules.GROUNDING, grounding_lora),
            (rules.CHANGE, change_lora),
        ):
            if path is not None:
                expert_lora[name] = path
        if not expert_lora:
            raise ValueError(
                "No expert LoRA provided. Pass --expert_lora dict or "
                "--general_lora/--grounding_lora/--change_lora."
            )
        self.expert_lora = expert_lora
        self._task_to_expert = dict(rules.TASK_TO_EXPERT)

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
        names = list(self.expert_lora)
        self.model = PeftModel.from_pretrained(
            base, self.expert_lora[names[0]], adapter_name=names[0]
        )
        for name in names[1:]:
            self.model.load_adapter(self.expert_lora[name], adapter_name=name)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._adapter_map = {name: name for name in names}
        self.active_adapter = None
        self.peak_memory = torch.cuda.max_memory_allocated(self.device)
        print(f"  Peak memory after load: {self.peak_memory / 1e9:.2f} GB", flush=True)

    def _switch(self, expert: str):
        name = self._adapter_map.get(expert, expert)
        if self.active_adapter != name:
            self.model.set_adapter(name)
            self.active_adapter = name

    def _extract_answer(self, text: str) -> str:
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        return text.strip()

    def _group_key(self, prompt: str) -> str:
        """分组键（默认按 expert 分组）。子类可覆盖为按 task 分组。"""
        return rules.route(prompt)

    def _prepare_group(self, key: str):
        """每组生成前的准备（默认切到对应 expert）。子类可覆盖。"""
        self._switch(key)

    def _run_batch(self, batch, max_new_tokens=None):
        import time
        from qwen_vl_utils import process_vision_info

        tok = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        results = [None] * len(batch)
        groups = {}
        for i, (images, prompt) in enumerate(batch):
            key = self._group_key(prompt)
            groups.setdefault(key, []).append(i)

        for key, indices in groups.items():
            self._prepare_group(key)
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
                if self.force_think:
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False)
                    text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                else:
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False)
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
