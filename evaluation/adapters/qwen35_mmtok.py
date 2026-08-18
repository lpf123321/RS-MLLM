import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from evaluation.adapters.qwen3vl import Qwen3VLAdapter, _THINKING_PATTERN


class Qwen35MMTokAdapter(Qwen3VLAdapter):
    """Qwen3.5 inference adapter with MMTok-style greedy coverage selection."""

    def __init__(self, *args, keep_ratio: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.model.config, "model_type", None) != "qwen3_5":
            raise ValueError("qwen35_mmtok only supports Qwen3.5 checkpoints.")
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("--mmtok_keep_ratio must be in (0, 1].")
        self.keep_ratio = keep_ratio

    @staticmethod
    def _coverage_indices(features: torch.Tensor, keep_count: int) -> torch.Tensor:
        """Greedily maximize visual-token similarity coverage."""
        count = features.shape[0]
        if keep_count >= count:
            return torch.arange(count, device=features.device)

        normalized = F.normalize(features.float(), dim=-1)
        similarity = normalized @ normalized.T

        selected = []
        available = torch.ones(count, dtype=torch.bool, device=features.device)

        first = similarity.sum(dim=0).argmax()
        selected.append(first)
        available[first] = False
        coverage = similarity[:, first]

        for _ in range(1, keep_count):
            gains = torch.maximum(coverage[:, None], similarity).sum(dim=0)
            gains -= coverage.sum()
            gains.masked_fill_(~available, -torch.inf)
            chosen = gains.argmax()
            selected.append(chosen)
            available[chosen] = False
            coverage = torch.maximum(coverage, similarity[:, chosen])

        # Retain original spatial/order sequence after selecting representative tokens.
        return torch.sort(torch.stack(selected)).values

    def _pruned_inputs(self, inputs):
        input_ids = inputs["input_ids"]
        if input_ids.shape[0] != 1:
            raise ValueError("MMTok pruning supports eval_batch_size=1 only.")

        image_token_id = self.model.config.image_token_id
        image_mask = input_ids.eq(image_token_id)
        image_positions = image_mask[0].nonzero(as_tuple=False).flatten()
        if image_positions.numel() == 0:
            return None

        vision = self.model.model.get_image_features(
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            return_dict=True,
        )
        pooled = vision.pooler_output
        pooled_features = list(pooled) if isinstance(pooled, tuple) else [pooled]

        if sum(feature.shape[0] for feature in pooled_features) != image_positions.numel():
            raise RuntimeError("Image placeholders do not match visual feature count.")

        selected_positions = []
        selected_features = []
        offset = 0

        for feature in pooled_features:
            count = feature.shape[0]
            keep_count = max(1, math.ceil(count * self.keep_ratio))
            local = self._coverage_indices(feature, keep_count)
            positions = image_positions[offset:offset + count]

            selected_positions.append(positions.index_select(0, local))
            selected_features.append(feature.index_select(0, local))
            offset += count

        selected_positions = torch.cat(selected_positions)
        selected_features = torch.cat(selected_features)

        full_position_ids, _ = self.model.model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=inputs["mm_token_type_ids"],
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"],
        )

        inputs_embeds = self.model.get_input_embeddings()(input_ids).clone()
        inputs_embeds[0, selected_positions] = selected_features

        keep_mask = ~image_mask
        keep_mask[0, selected_positions] = True
        keep_mask = keep_mask[0]

        pruned = {
            "input_ids": input_ids[:, keep_mask],
            "inputs_embeds": inputs_embeds[:, keep_mask],
            "attention_mask": inputs["attention_mask"][:, keep_mask],
            "mm_token_type_ids": inputs["mm_token_type_ids"][:, keep_mask],
            "position_ids": full_position_ids[..., keep_mask],
        }

        max_position = pruned["position_ids"].amax(dim=0).amax(dim=-1, keepdim=True)
        self.model.model.rope_deltas = (
            max_position + 1 - pruned["attention_mask"].sum(dim=-1, keepdim=True)
        )
        return pruned

    def _generate_pruned(self, images: List[str], prompt: str) -> str:
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

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        pruned = self._pruned_inputs(inputs)
        with torch.no_grad():
            if pruned is None:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
                prompt_length = inputs.input_ids.shape[1]
            else:
                generated_ids = self.model.generate(
                    **pruned,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
                prompt_length = pruned["input_ids"].shape[1]

        output = self.processor.batch_decode(
            generated_ids[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return _THINKING_PATTERN.sub("", output).strip()

    def batch_generate(
        self, batch: List[Tuple[List[str], str]], batch_size: int = 1
    ) -> List[str]:
        # keep_ratio=1.0 must use the already verified baseline implementation.
        if self.keep_ratio == 1.0:
            return super().batch_generate(batch, batch_size=batch_size)

        if batch_size != 1:
            raise ValueError("Use --eval_batch_size 1 with qwen35_mmtok.")

        return [
            self._generate_pruned(images, prompt)
            for images, prompt in tqdm(batch, desc="Inference", unit="sample")
        ]