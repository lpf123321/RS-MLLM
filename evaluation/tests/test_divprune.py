import torch

from evaluation.pruners.divprune import DivPrunePruner


def test_divprune_matches_greedy_max_min_cosine_selection():
    features = torch.tensor(
        [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )

    selected = DivPrunePruner().select(features, keep_count=2, seed=2026)

    assert selected.tolist() == [0, 3]


def test_divprune_keeps_all_tokens_when_requested():
    features = torch.randn(4, 8)

    selected = DivPrunePruner().select(features, keep_count=4, seed=2026)

    assert selected.tolist() == [0, 1, 2, 3]
