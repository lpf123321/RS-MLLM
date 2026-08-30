"""Low-frequency 2D DCT visual-token compression."""

import math
from typing import Sequence, Tuple

import torch


def _dct(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
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


def _idct(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
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
    flipped = torch.cat([value[:, :1] * 0, -value.flip([1])[:, :-1]], dim=1)
    imag = imag + flipped * torch.cos(k)
    real = real - flipped * torch.sin(k)
    value = torch.view_as_complex(torch.stack([real, imag], dim=2))
    value = torch.fft.ifft(value, dim=1).real

    output = value.new_zeros(value.shape)
    output[:, ::2] += value[:, : length - (length // 2)]
    output[:, 1::2] += value.flip([1])[:, : length // 2]
    return output.view(*shape)


def _dct2(x: torch.Tensor) -> torch.Tensor:
    return _dct(_dct(x).transpose(-1, -2)).transpose(-1, -2)


def _idct2(x: torch.Tensor) -> torch.Tensor:
    return _idct(_idct(x.transpose(-1, -2)).transpose(-1, -2))


def compress_fourier_tokens(
    features: torch.Tensor,
    grid_thw: Sequence[int],
    spatial_merge_size: int,
    keep_ratio: float,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Keep low-frequency DCT coefficients and reconstruct a smaller feature grid.

    Args:
        features: Visual features shaped ``[num_tokens, hidden_size]``.
        grid_thw: Original Qwen visual grid as ``(time, height, width)``.
        spatial_merge_size: Qwen visual encoder's spatial merge factor.
        keep_ratio: Fraction of tokens to retain, in ``(0, 1]``.

    Returns:
        The compressed features and their new Qwen grid ``(time, height, width)``.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape [num_tokens, hidden_size].")
    if len(grid_thw) != 3:
        raise ValueError("grid_thw must contain exactly (time, height, width).")
    if spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be positive.")
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1].")

    time_steps, raw_height, raw_width = (int(value) for value in grid_thw)
    if raw_height % spatial_merge_size or raw_width % spatial_merge_size:
        raise ValueError("grid dimensions must be divisible by spatial_merge_size.")
    height = raw_height // spatial_merge_size
    width = raw_width // spatial_merge_size
    expected = time_steps * height * width
    if features.shape[0] != expected:
        raise ValueError(
            f"Visual feature count {features.shape[0]} does not match grid size {expected}."
        )

    side_ratio = math.sqrt(keep_ratio)
    reserve_height = max(1, min(height, int(height * side_ratio)))
    reserve_width = max(1, min(width, int(width * side_ratio)))
    channels = features.shape[-1]
    image = features.float().transpose(0, 1).reshape(
        time_steps, channels, height, width
    )
    coefficients = _dct2(image)
    low_frequency = coefficients[..., :reserve_height, :reserve_width]
    compressed = _idct2(low_frequency).reshape(
        time_steps, channels, reserve_height * reserve_width
    )
    compressed = compressed.transpose(1, 2).reshape(-1, channels).to(features.dtype)
    return compressed, (
        time_steps,
        reserve_height * spatial_merge_size,
        reserve_width * spatial_merge_size,
    )


class FourierTokenCompressor:
    """Stateful wrapper for low-frequency Fourier token compression."""

    name = "fourier"

    def __init__(self, keep_ratio: float, spatial_merge_size: int):
        self.keep_ratio = keep_ratio
        self.spatial_merge_size = spatial_merge_size

    def compress(
        self, features: torch.Tensor, grid_thw: Sequence[int]
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        return compress_fourier_tokens(
            features, grid_thw, self.spatial_merge_size, self.keep_ratio
        )
