# Visual Token Compression

This directory contains standalone visual-token compression methods. Its only
runtime dependency is PyTorch.

## Methods

| Method | Export | Behavior |
| --- | --- | --- |
| L2 norm | `L2NormTokenPruner` | Retains the largest feature norms. |
| Uniform | `UniformTokenPruner` | Retains evenly spaced tokens. |
| Random | `RandomTokenPruner` | Retains a seeded random subset. |
| MMTok | `MMTokTokenPruner` | Greedily maximizes cosine-similarity coverage. |
| DivPrune | `DivPruneTokenPruner` | Greedily maximizes subset diversity. |
| SCOPE-L2 | `ScopeL2TokenPruner` | Combines L2 saliency and coverage. |
| Fourier | `FourierTokenCompressor` | Reconstructs a smaller grid from low-frequency 2D DCT coefficients. |

For token features `features` of shape `[num_tokens, hidden_size]`, the method
computes `||features[i]||_2` for each token and retains the tokens with the
largest norms. Returned indices are sorted into their original token order,
which makes them suitable for `index_select`.

## Use

```python
import math
import torch

from token_compression import select_l2_tokens

features = torch.randn(196, 1536, device="cuda", dtype=torch.bfloat16)
keep_ratio = 0.5
keep_count = max(1, math.ceil(features.shape[0] * keep_ratio))
indices = select_l2_tokens(features, keep_count)
compressed_features = features.index_select(0, indices)
```

All selector classes implement `pruner.select(features, keep_count, seed=None)`
and return sorted indices. The seed affects only `RandomTokenPruner`.

Fourier needs the original Qwen visual grid and merge factor because it produces
new token features rather than selecting existing positions:

```python
from token_compression import FourierTokenCompressor

compressor = FourierTokenCompressor(keep_ratio=0.5, spatial_merge_size=2)
compressed_features, new_grid_thw = compressor.compress(features, (1, 28, 28))
```

## Original evaluation integration

The full Qwen3.5 inference integration remains in
`evaluation/adapters/qwen35_pruned.py`; it obtains visual features, calls the
pruner, and removes the unselected image-token positions from the model input.
The original evaluation integrations remain in `evaluation/pruners/` and
`evaluation/adapters/`.
