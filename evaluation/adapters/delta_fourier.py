"""FourierCompressor adapter for the current Delta expert models."""

from typing import List, Tuple

import torch
from qwen_vl_utils import process_vision_info

from evaluation.adapters.router import RouterAdapter
from token_compression import compress_fourier_tokens


class DeltaFourierAdapter(RouterAdapter):
    """Delta Router + low-frequency 2D DCT visual-token compression.

    Fourier compression changes the spatial grid, so it is implemented
    explicitly rather than through the index-selector pre-LLM hook.
    """

    def __init__(self, *args, keep_ratio: float = 0.5, **kwargs):
        super().__init__(*args, force_think=False, **kwargs)
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.keep_ratio = keep_ratio

    def _build_inputs(self, images: List[str], prompt: str):
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
        if self._delta_mode:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            text += "<|im_start|>assistant\n thinking\n\n response\n\n"
        else:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        image_inputs, video_inputs = process_vision_info(messages)
        video_arg = video_inputs if video_inputs is not None else None
        return self.processor(
            text=[text], images=image_inputs, videos=video_arg,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

    def _compress_inputs(self, inputs):
        input_ids = inputs["input_ids"]
        image_token_id = self.model.config.image_token_id
        image_mask = input_ids.eq(image_token_id)
        image_positions = image_mask[0].nonzero(as_tuple=False).flatten()
        if image_positions.numel() == 0:
            return inputs

        vision = self.model.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True
        )
        pooled = vision.pooler_output
        features_per_image = [pooled] if isinstance(pooled, torch.Tensor) else list(pooled)
        if sum(feature.shape[0] for feature in features_per_image) != image_positions.numel():
            raise RuntimeError("Visual feature and placeholder counts do not match.")

        new_grid = inputs["image_grid_thw"].clone()
        selected_positions = []
        compressed_features = []
        offset = 0
        merge_size = self.model.model.visual.spatial_merge_size
        for image_index, features in enumerate(features_per_image):
            compressed, grid = compress_fourier_tokens(
                features,
                tuple(int(x) for x in new_grid[image_index].tolist()),
                merge_size,
                self.keep_ratio,
            )
            reserve_count = compressed.shape[0]
            positions = image_positions[offset:offset + features.shape[0]]
            selected_positions.append(positions[:reserve_count])
            compressed_features.append(compressed)
            new_grid[image_index] = torch.tensor(
                grid, dtype=new_grid.dtype, device=new_grid.device
            )
            offset += features.shape[0]

        selected_positions = torch.cat(selected_positions)
        compressed_features = torch.cat(compressed_features)
        inputs_embeds = self.model.get_input_embeddings()(input_ids).clone()
        inputs_embeds[0, selected_positions] = compressed_features

        keep_mask = ~image_mask
        keep_mask[0, selected_positions] = True
        keep_mask = keep_mask[0]
        pruned = {
            "input_ids": input_ids[:, keep_mask],
            "inputs_embeds": inputs_embeds[:, keep_mask],
            "attention_mask": inputs["attention_mask"][:, keep_mask],
            "mm_token_type_ids": inputs["mm_token_type_ids"][:, keep_mask],
        }
        position_ids, _ = self.model.model.get_rope_index(
            input_ids=pruned["input_ids"],
            mm_token_type_ids=pruned["mm_token_type_ids"],
            image_grid_thw=new_grid,
            attention_mask=pruned["attention_mask"],
        )
        pruned["position_ids"] = position_ids
        self.model.model.rope_deltas = (
            position_ids.amax(dim=0).amax(dim=-1, keepdim=True)
            + 1 - pruned["attention_mask"].sum(dim=-1, keepdim=True)
        )
        return pruned

    def _generate_one(self, images: List[str], prompt: str) -> str:
        self._switch(self._group_key(prompt))
        inputs = self._compress_inputs(self._build_inputs(images, prompt))
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        output = self.processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return self._extract_answer(output)

    def batch_generate(
        self, batch: List[Tuple[List[str], str]], batch_size: int = 1, max_new_tokens=None
    ):
        if batch_size != 1:
            raise ValueError("DeltaFourierAdapter currently requires eval_batch_size=1.")
        old_tokens = self.max_new_tokens
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens
        try:
            return [self._generate_one(images, prompt) for images, prompt in batch]
        finally:
            self.max_new_tokens = old_tokens
