# SAM3 LoRA training runtime

This directory is the SAM3 training runtime used by the self-evolution
experiments. It is intentionally separate from `../cvsearch/sam3/`:

- `sam3_lora/sam3/` keeps the original training forward contract and returns a
  single `SAM3Output`.
- `cvsearch/sam3/` is the search-time fork and may additionally return visual
  backbone features for CVSearch.
- `lora_layers.py` injects LoRA into the selected SAM3 attention projections.
- `merge_lora_weights.py` provides the generic adapter merge implementation.
- `../sam_bridge/scripts/` contains the VOPD-specific dataset, training,
  selectivity-evaluation and checkpoint-merge entry points.

The source snapshot comes from `Sompote/sam3_lora` at the commit recorded in
`UPSTREAM_REVISION`. Large demo media, model weights, datasets, checkpoints and
experiment outputs are not included. The required tokenizer vocabulary is
kept at `sam3/assets/bpe_simple_vocab_16e6.txt.gz`.

Install dependencies in the SAM environment:

```bash
conda create -n Sam3_lora python=3.12 -y
conda run -n Sam3_lora python -m pip install -r \
  training/self_evolution/sam3_lora/requirements.txt
```

The lightweight GPU reproduction was verified with Python 3.12.13, PyTorch
2.11.0+cu128 and torchvision 0.26.0+cu128. Install the PyTorch build matching
the host CUDA driver before installing the remaining requirements.

The unified self-evolution runner sets `SAM3_REPO` to this directory and writes
the round-specific training YAML automatically. A standalone self-evolution
run therefore does not depend on `/home/cpy/sam3_lora` or
`/home/cpy/sam3_lora_test`.
