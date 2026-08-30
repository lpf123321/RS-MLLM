"""VRSBench/Qwen 多模态数据集与动态 batch collator."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class JsonlOffsetDataset(Dataset):
    """用字节偏移随机读取 JSONL，避免将 VRSBench 全量载入内存."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.offsets: list[int] = []
        with open(self.path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with open(self.path, "rb") as f:
            f.seek(self.offsets[index])
            return json.loads(f.readline())


def load_multimodal_dataset(path: str, max_samples: int | None = None) -> Dataset:
    if not path.endswith(".jsonl"):
        raise ValueError(
            "多模态版本当前要求 JSONL 数据。VRSBench 请使用 "
            "vrsbench_train.jsonl 或 vrsbench_eval.jsonl。"
        )
    dataset = JsonlOffsetDataset(path)
    if max_samples is not None and max_samples > 0:
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    )


def _resolve_image(path: str, image_root: str) -> str:
    if os.path.isfile(path):
        return path
    candidates = [
        os.path.join(image_root, os.path.basename(path)),
        os.path.join(image_root, "Images_train", os.path.basename(path)),
        os.path.join(image_root, "Images_val", os.path.basename(path)),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"找不到 VRSBench 图片 {path!r}；已在 image_root={image_root!r} 下查找"
    )


def normalize_multimodal_example(example: dict[str, Any], image_root: str):
    """统一 VRSBench messages/conversations，并解析本地图片路径."""
    messages = example.get("messages")
    if messages is None and "conversations" in example:
        role_map = {"human": "user", "gpt": "assistant"}
        messages = [
            {"role": role_map.get(m["from"], m["from"]), "content": m["value"]}
            for m in example["conversations"]
        ]
    if not messages:
        raise ValueError("样本缺少 messages/conversations")

    normalized, image_paths = [], []
    fallback_image = example.get("image")
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            parts = []
            if "<image>" in content:
                if not fallback_image:
                    raise ValueError("<image> 样本缺少 image 字段")
                parts.append({"type": "image"})
                image_paths.append(_resolve_image(fallback_image, image_root))
                content = content.replace("<image>", "").lstrip()
            parts.append({"type": "text", "text": content})
        else:
            parts = []
            for item in content:
                if item.get("type") == "image" or "image" in item:
                    raw_path = item.get("image") or item.get("image_url") or fallback_image
                    parts.append({"type": "image"})
                    image_paths.append(_resolve_image(raw_path, image_root))
                elif item.get("type") == "text" or "text" in item:
                    parts.append({"type": "text", "text": item.get("text", "")})
        normalized.append({"role": message["role"], "content": parts})
    return normalized, image_paths


class MultimodalDataCollator:
    """加载图片并用 AutoProcessor 构造 Qwen3.5 CE 与 rollout 输入."""

    def __init__(
        self,
        processor,
        image_root: str,
        max_length: int,
        enable_thinking: bool = False,
    ):
        self.processor = processor
        self.image_root = image_root
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_texts, prompt_texts, image_batches = [], [], []
        for example in examples:
            try:
                messages, paths = normalize_multimodal_example(example, self.image_root)
                if not paths:
                    raise ValueError("多模态样本中没有图片")
                # 尝试打开所有图片，任何一张损坏则跳过该样本
                images = []
                for path in paths:
                    images.append(Image.open(path).convert("RGB"))
                prompt_messages = list(messages)
                if prompt_messages[-1]["role"] == "assistant":
                    prompt_messages = prompt_messages[:-1]
                full_texts.append(
                    self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=self.enable_thinking,
                    )
                )
                prompt_texts.append(
                    self.processor.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.enable_thinking,
                    )
                )
                image_batches.append(images)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"[MultimodalCollator] 跳过损坏/缺失图片的样本: {e}")
                continue

        if not full_texts:
            raise RuntimeError(
                "MultimodalCollator: batch 中所有样本均因损坏/缺失图片被跳过，"
                "请检查 VRSBench 图片目录完整性"
            )

        self.processor.tokenizer.padding_side = "right"
        full = self.processor(
            text=full_texts,
            images=image_batches,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Decoder-only 批量生成必须左 padding，保证最后一列是实际 prompt。
        self.processor.tokenizer.padding_side = "left"
        prompt = self.processor(
            text=prompt_texts,
            images=image_batches,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = full["input_ids"].clone()
        labels[full["attention_mask"] == 0] = IGNORE_INDEX
        for i in range(labels.shape[0]):
            prompt_len = int(prompt["attention_mask"][i].sum())
            labels[i, : min(prompt_len, labels.shape[1])] = IGNORE_INDEX
        full["labels"] = labels
        full["prompt_input_ids"] = prompt["input_ids"]
        full["prompt_attention_mask"] = prompt["attention_mask"]
        if "mm_token_type_ids" in prompt:
            full["prompt_mm_token_type_ids"] = prompt["mm_token_type_ids"]
        return dict(full)
