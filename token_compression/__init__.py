"""Standalone visual-token compression methods."""

from .divprune import DivPruneTokenPruner
from .fourier import FourierTokenCompressor, compress_fourier_tokens
from .l2_norm import L2NormTokenPruner, select_l2_tokens
from .mmtok import MMTokTokenPruner
from .random_pruner import RandomTokenPruner
from .scope_l2 import ScopeL2TokenPruner
from .uniform import UniformTokenPruner

__all__ = [
    "DivPruneTokenPruner",
    "FourierTokenCompressor",
    "L2NormTokenPruner",
    "MMTokTokenPruner",
    "RandomTokenPruner",
    "ScopeL2TokenPruner",
    "UniformTokenPruner",
    "TOKEN_PRUNER_REGISTRY",
    "build_token_pruner",
    "compress_fourier_tokens",
    "select_l2_tokens",
]


TOKEN_PRUNER_REGISTRY = {
    "uniform": UniformTokenPruner,
    "random": RandomTokenPruner,
    "mmtok": MMTokTokenPruner,
    "l2norm": L2NormTokenPruner,
    "divprune": DivPruneTokenPruner,
    "scope_l2": ScopeL2TokenPruner,
}


def build_token_pruner(name: str):
    try:
        return TOKEN_PRUNER_REGISTRY[name]()
    except KeyError as error:
        choices = ", ".join(sorted(TOKEN_PRUNER_REGISTRY))
        raise ValueError(f"Unknown token pruner '{name}'. Available: {choices}") from error
