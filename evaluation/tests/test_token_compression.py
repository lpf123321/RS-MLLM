import torch

from token_compression import (
    DivPruneTokenPruner,
    FourierTokenCompressor,
    MMTokTokenPruner,
    RandomTokenPruner,
    ScopeL2TokenPruner,
    UniformTokenPruner,
)


def test_selectors_return_sorted_unique_indices():
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    for selector in (
        UniformTokenPruner(),
        RandomTokenPruner(),
        MMTokTokenPruner(),
        DivPruneTokenPruner(),
        ScopeL2TokenPruner(),
    ):
        indices = selector.select(features, keep_count=3, seed=42)
        assert indices.shape == (3,)
        assert torch.equal(indices, torch.unique(indices, sorted=True))


def test_random_selector_is_seeded():
    features = torch.randn(16, 4)
    selector = RandomTokenPruner()
    assert torch.equal(
        selector.select(features, keep_count=5, seed=42),
        selector.select(features, keep_count=5, seed=42),
    )


def test_fourier_compressor_reduces_grid():
    features = torch.randn(16, 8)
    compressor = FourierTokenCompressor(keep_ratio=0.25, spatial_merge_size=2)
    compressed, grid_thw = compressor.compress(features, (1, 8, 8))
    assert compressed.shape == (4, 8)
    assert grid_thw == (1, 4, 4)
