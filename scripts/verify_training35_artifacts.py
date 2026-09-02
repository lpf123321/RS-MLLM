#!/usr/bin/env python3
"""Validate that a training35 LoRA and merged model can be opened safely."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def tensor_files(model: Path) -> list[Path]:
    index=model/'model.safetensors.index.json'
    if index.is_file():
        mapping=json.loads(index.read_text(encoding='utf-8'))['weight_map']
        return [model/name for name in sorted(set(mapping.values()))]
    return sorted(model.glob('*.safetensors'))


def safetensor_keys(path: Path) -> list[str]:
    with path.open('rb') as handle:
        size=int.from_bytes(handle.read(8),'little'); header=json.loads(handle.read(size))
    return [key for key in header if key!='__metadata__']


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--lora',type=Path,required=True); parser.add_argument('--merged',type=Path,required=True); args=parser.parse_args()
    for path in (args.lora/'adapter_config.json',args.merged/'config.json'):
        json.loads(path.read_text(encoding='utf-8'))
    adapter_keys=safetensor_keys(args.lora/'adapter_model.safetensors')
    shards=tensor_files(args.merged)
    if not shards or any(not path.is_file() for path in shards): raise FileNotFoundError('merged safetensor shards are incomplete')
    tensors=sum(len(safetensor_keys(shard)) for shard in shards)
    if not adapter_keys or not tensors: raise ValueError('empty adapter or merged model')
    print(json.dumps({'lora':str(args.lora),'adapter_tensors':len(adapter_keys),'merged':str(args.merged),'shards':len(shards),'model_tensors':tensors,'passed':True},indent=2))
if __name__=='__main__': main()
