import math
from typing import List, Tuple

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from evaluation.adapters.qwen3vl import Qwen3VLAdapter, _THINKING_PATTERN
from evaluation.pruners import PRUNER_REGISTRY, build_pruner


class Qwen35PrunedAdapter(Qwen3VLAdapter):
    """Generic Qwen3.5 visual-token pruning adapter."""

    def __init__(
        self,
        *args,
        keep_ratio: float = 0.5,
        pruner: str = "mmtok",
        pruner_seed: int = 2026,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if getattr(self.model.config, "model_type", None) != "qwen3_5":
            raise ValueError("qwen35_pruned only supports Qwen3.5 checkpoints.")
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("--keep_ratio must be in (0, 1].")
        if pruner not in PRUNER_REGISTRY:
            choices = ", ".join(sorted(PRUNER_REGISTRY))
            raise ValueError(f"Unknown pruner '{pruner}'. Available: {choices}")

        self.keep_ratio = keep_ratio
        self.pruner_name = pruner
        self.pruner = build_pruner(pruner)
        self.pruner_seed = pruner_seed
        self._selection_counter = 0

    def _prune_inputs(self, inputs):
        input_ids = inputs["input_ids"]
        if input_ids.shape[0] != 1:
            raise ValueError("Pruned inference requires eval_batch_size=1.")

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
        pooled_features = [pooled] if isinstance(pooled, torch.Tensor) else list(pooled)

        expected = sum(feature.shape[0] for feature in pooled_features)
        if expected != image_positions.numel():
            raise RuntimeError(
                f"Visual feature count {expected} != image placeholder count "
                f"{image_positions.numel()}."
            )

        selected_positions = []
        selected_features = []
        offset = 0

        for feature in pooled_features:
            count = feature.shape[0]
            keep_count = max(1, math.ceil(count * self.keep_ratio))
            seed = self.pruner_seed + self._selection_counter
            local_indices = self.pruner.select(feature, keep_count, seed)
            self._selection_counter += 1

            positions = image_positions[offset:offset + count]
            selected_positions.append(positions.index_select(0, local_indices))
            selected_features.append(feature.index_select(0, local_indices))
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

        with torch.no_grad():
            pruned = self._prune_inputs(inputs)

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
        self,
        batch: List[Tuple[List[str], str]],
        batch_size: int = 1,
    ) -> List[str]:
        if self.keep_ratio == 1.0:
            return super().batch_generate(batch, batch_size=batch_size)
        if batch_size != 1:
            raise ValueError("Use --eval_batch_size 1 with qwen35_pruned.")

        return [
            self._generate_pruned(images, prompt)
            for images, prompt in tqdm(batch, desc="Inference", unit="sample")
        ]