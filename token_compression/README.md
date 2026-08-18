# L2 Token Compression

This directory contains the standalone L2-norm visual-token compression method
used by the evaluation pipeline. Its only runtime dependency is PyTorch.

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

`L2NormTokenPruner` is also available when the caller expects a
`pruner.select(features, keep_count, seed)` interface. The `seed` has no effect
because L2-norm ranking is deterministic.

## Original evaluation integration

The full Qwen3.5 inference integration remains in
`evaluation/adapters/qwen35_pruned.py`; it obtains visual features, calls the
pruner, and removes the unselected image-token positions from the model input.
The original evaluation implementation is in `evaluation/pruners/l2_norm.py`.
