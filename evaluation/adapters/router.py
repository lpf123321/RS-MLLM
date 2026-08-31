"""Task Router：把 prompt 路由到对应 Expert LoRA。

单 Qwen3.5-4B 基座 + 多 expert LoRA，按 prompt 路由并动态切换 Adapter。
"""
import re

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from evaluation.base.adapter import BaseModelAdapter
from evaluation.router import rules

# 与 qwen35vl/qwen3vl 一致的 think 剥离；额外兼容 <thinking>（模型可能把 <think> 特殊 token 后接 "ing" 文本）
THINKING_PATTERN = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>\s*", re.DOTALL)
_STRAY_THINK_TAG = re.compile(r"</?think(?:ing)?>")


class RouterAdapter(BaseModelAdapter):
    """单基座 + 多 expert LoRA，按 prompt 路由并动态切换 Adapter。

    expert 注册表由 ``expert_lora`` dict 驱动（任意数量的 expert），同时保留
    ``general_lora`` / ``grounding_lora`` / ``change_lora`` 作为便捷参数向后兼容。
    task -> expert 的映射见 ``evaluation.router.rules.TASK_TO_EXPERT``。
    """

    def __init__(self, model_path: str, general_lora: str = None,
                 grounding_lora: str = None, change_lora: str = None,
                 caption_lora: str = None,
                 expert_lora: dict = None, device: str = "cuda",
                 max_new_tokens: int = 256, system_prompt: str = "",
                 image_min_pixels: int = None, image_max_pixels: int = None,
                 force_think: bool = False):
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
            (rules.CAPTION_EXPERT, caption_lora),
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
        self._delta_mode = all(str(path).endswith(".pt") for path in expert_lora.values())

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if image_min_pixels is not None or image_max_pixels is not None:
            size = self.processor.image_processor.size
            if image_min_pixels is not None:
                size["shortest_edge"] = image_min_pixels
            if image_max_pixels is not None:
                size["longest_edge"] = image_max_pixels
        self.processor.tokenizer.padding_side = "left"

        torch.cuda.reset_peak_memory_stats()
        print("  Loading base model ...", flush=True)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        names = list(self.expert_lora)
        if self._delta_mode:
            # Delta files contain full-rank W_expert - W0 state dictionaries.
            self.model = base.to(self.device)
            self._delta_cache = {}
            self._active_delta = None
        else:
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
            if self._delta_mode:
                if self._active_delta is not None:
                    self._apply_delta(self._load_delta(self._active_delta), sign=-1)
                self._apply_delta(self._load_delta(name), sign=1)
                self._active_delta = name
            else:
                self.model.set_adapter(name)
            self.active_adapter = name

    def _load_delta(self, expert: str):
        if expert not in self._delta_cache:
            self._delta_cache[expert] = torch.load(
                self.expert_lora[expert], map_location="cpu", weights_only=True
            )
        return self._delta_cache[expert]

    def _apply_delta(self, delta: dict, sign: int = 1):
        state = self.model.state_dict()
        with torch.no_grad():
            for name, value in delta.items():
                if name in state:
                    state[name].add_((sign * value).to(device=state[name].device, dtype=state[name].dtype))

    def _extract_answer(self, text: str) -> str:
        if not text:
            return text
        if self._delta_mode and " response" in text:
            text = text.split(" response", 1)[1]
        # 与 qwen35vl/qwen3vl 一致：去掉 <think>/<thinking> 块
        text = THINKING_PATTERN.sub("", text)
        # 去掉残留的孤立 think 标签（如模型仅输出闭合标签 </thinking>）
        text = _STRAY_THINK_TAG.sub("", text)
        # 截断角色标记重复（answer\nuser\nuser... 或 <|im_start|>...）
        text = re.split(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>", text)[0]
        text = re.split(r"\n(?:user|assistant|system)\b", text)[0]
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
                if self._delta_mode and self.force_think:
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    # Match the shared Delta inference script's empty think block.
                    text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                elif self.force_think:
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
