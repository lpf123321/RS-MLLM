import math

import torch

from evaluation.adapters.qwen35_pruned import Qwen35PrunedAdapter


class Qwen35FourierAdapter(Qwen35PrunedAdapter):
    """Qwen3.5 Fourier-Compressor adapter using low-frequency 2D DCT blocks."""

    def __init__(self, *args, keep_ratio=0.5, pruner_seed=2026, **kwargs):
        super().__init__(
            *args,
            keep_ratio=keep_ratio,
            pruner="uniform",
            pruner_seed=pruner_seed,
            **kwargs,
        )
        self.pruner_name = "fourier"

    @staticmethod
    def _dct(x, norm="ortho"):
        shape = x.shape
        length = shape[-1]
        x = x.contiguous().view(-1, length)

        value = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        value_fft = torch.fft.fft(value, dim=1)

        k = -torch.arange(length, dtype=x.dtype, device=x.device)[None, :]
        k = k * math.pi / (2 * length)
        output = value_fft.real * torch.cos(k) - value_fft.imag * torch.sin(k)

        if norm == "ortho":
            output[:, 0] /= math.sqrt(length) * 2
            output[:, 1:] /= math.sqrt(length / 2) * 2

        return (2 * output).view(*shape)

    @staticmethod
    def _idct(x, norm="ortho"):
        shape = x.shape
        length = shape[-1]
        value = x.contiguous().view(-1, length) / 2

        if norm == "ortho":
            value[:, 0] *= math.sqrt(length) * 2
            value[:, 1:] *= math.sqrt(length / 2) * 2

        k = torch.arange(length, dtype=x.dtype, device=x.device)[None, :]
        k = k * math.pi / (2 * length)

        real = value * torch.cos(k)
        imag = value * torch.sin(k)
        imag = imag + torch.cat(
            [value[:, :1] * 0, -value.flip([1])[:, :-1]], dim=1
        ) * torch.cos(k)
        real = real - torch.cat(
            [value[:, :1] * 0, -value.flip([1])[:, :-1]], dim=1
        ) * torch.sin(k)

        value = torch.view_as_complex(
            torch.cat([real.unsqueeze(2), imag.unsqueeze(2)], dim=2)
        )
        value = torch.fft.ifft(value, dim=1).real

        output = value.new_zeros(value.shape)
        output[:, ::2] += value[:, : length - (length // 2)]
        output[:, 1::2] += value.flip([1])[:, : length // 2]
        return output.view(*shape)

    @classmethod
    def _dct2(cls, x):
        return cls._dct(cls._dct(x).transpose(-1, -2)).transpose(-1, -2)

    @classmethod
    def _idct2(cls, x):
        return cls._idct(cls._idct(x.transpose(-1, -2)).transpose(-1, -2))

    def _compress_one_image(self, features, grid_thw):
        merge_size = self.model.model.visual.spatial_merge_size
        time_steps = int(grid_thw[0].item())
        height = int(grid_thw[1].item()) // merge_size
        width = int(grid_thw[2].item()) // merge_size

        expected = time_steps * height * width
        if features.shape[0] != expected:
            raise RuntimeError(
                f"Visual feature count {features.shape[0]} != expected {expected}."
            )

        # keep_ratio is a token ratio; DCT retains sqrt(keep_ratio) per axis.
        side_ratio = math.sqrt(self.keep_ratio)
        reserve_h = max(1, min(height, int(height * side_ratio)))
        reserve_w = max(1, min(width, int(width * side_ratio)))

        channels = features.shape[-1]
        x = features.float().transpose(0, 1).reshape(
            time_steps, channels, height, width
        )
        coefficients = self._dct2(x)
        low_frequency = coefficients[..., :reserve_h, :reserve_w]
        compressed = self._idct2(low_frequency)

        compressed = compressed.reshape(
            time_steps, channels, reserve_h * reserve_w
        ).transpose(1, 2).reshape(-1, channels)

        return compressed.to(features.dtype), reserve_h, reserve_w

    def _prune_inputs(self, inputs):
        input_ids = inputs["input_ids"]
        if input_ids.shape[0] != 1:
            raise ValueError("Fourier inference requires --eval_batch_size 1.")

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
        features_per_image = (
            [pooled] if isinstance(pooled, torch.Tensor) else list(pooled)
        )

        if sum(item.shape[0] for item in features_per_image) != image_positions.numel():
            raise RuntimeError("Visual feature and placeholder counts do not match.")

        selected_positions = []
        compressed_features = []
        new_grid_thw = inputs["image_grid_thw"].clone()
        offset = 0
        merge_size = self.model.model.visual.spatial_merge_size

        for image_index, features in enumerate(features_per_image):
            compressed, reserve_h, reserve_w = self._compress_one_image(
                features,
                new_grid_thw[image_index],
            )
            count = compressed.shape[0]
            positions = image_positions[offset:offset + features.shape[0]]

            selected_positions.append(positions[:count])
            compressed_features.append(compressed)
            new_grid_thw[image_index, 1] = reserve_h * merge_size
            new_grid_thw[image_index, 2] = reserve_w * merge_size
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
            image_grid_thw=new_grid_thw,
            attention_mask=pruned["attention_mask"],
        )
        pruned["position_ids"] = position_ids

        max_position = position_ids.amax(dim=0).amax(dim=-1, keepdim=True)
        self.model.model.rope_deltas = (
            max_position + 1 - pruned["attention_mask"].sum(dim=-1, keepdim=True)
        )
        return pruned