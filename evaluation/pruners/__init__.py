from .l2_norm import L2NormPruner
from .mmtok import MMTokPruner
from .random_pruner import RandomPruner
from .scope_l2 import ScopeL2Pruner
from .uniform import UniformPruner

PRUNER_REGISTRY = {
    "uniform": UniformPruner,
    "random": RandomPruner,
    "mmtok": MMTokPruner,
    "l2norm": L2NormPruner,
    "scope_l2": ScopeL2Pruner,
}


def build_pruner(name: str):
    try:
        return PRUNER_REGISTRY[name]()
    except KeyError as error:
        choices = ", ".join(sorted(PRUNER_REGISTRY))
        raise ValueError(f"Unknown pruner '{name}'. Available: {choices}") from error