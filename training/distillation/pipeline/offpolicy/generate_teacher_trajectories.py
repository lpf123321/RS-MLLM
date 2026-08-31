#!/usr/bin/env python3
"""Generate sharded multimodal trajectories with the frozen SFT teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qwen_vl_utils import process_vision_info

from prompts import SYSTEM_PROMPTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(SYSTEM_PROMPTS))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path, dataset: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for row_index, line in enumerate(stream):
            record = json.loads(line)
            metadata = record["sample_metadata"]
            source_index = int(metadata["source_index"])
            content = record["messages"][0]["content"]
            images = [part["image"] for part in content if part["type"] == "image"]
            prompt = next(part["text"] for part in content if part["type"] == "text")
            rows.append(
                {
                    "trajectory_id": f"{dataset}:{source_index}",
                    "dataset": dataset,
                    "source_index": source_index,
                    "input_row_index": row_index,
                    "images": images,
                    "prompt": prompt,
                }
            )
    return rows


def shard_bounds(total: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    return total * shard_index // num_shards, total * (shard_index + 1) // num_shards


def load_adapter(model_path: Path, max_new_tokens: int):
    from offpolicy.qwen35_teacher_adapter import Qwen35VLAdapter

    return Qwen35VLAdapter(
        str(model_path),
        device="cuda",
        max_new_tokens=max_new_tokens,
        disable_thinking=True,
    )


def trim_generated_ids(ids: list[int], *, eos_ids: set[int], pad_id: int | None) -> list[int]:
    for position, token_id in enumerate(ids):
        if token_id in eos_ids:
            return ids[: position + 1]
    if pad_id is not None:
        while ids and ids[-1] == pad_id:
            ids.pop()
    return ids


def generate_batch(adapter, rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    texts = []
    all_images = []
    all_videos = []
    for row in rows:
        messages = adapter._build_messages(row["images"], row["prompt"])
        text = adapter.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        texts.append(text)
        all_images.append(image_inputs)
        all_videos.append(video_inputs)

    video_arg = all_videos if any(item is not None for item in all_videos) else None
    inputs = adapter.processor(
        text=texts,
        images=all_images,
        videos=video_arg,
        padding=True,
        return_tensors="pt",
    ).to(adapter.model.device)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": not args.greedy,
    }
    if args.greedy:
        generation_kwargs.update({"temperature": None, "top_p": None})
    else:
        generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

    with torch.inference_mode():
        generated = adapter.model.generate(**inputs, **generation_kwargs)

    prompt_width = inputs.input_ids.shape[1]
    tokenizer = adapter.processor.tokenizer
    eos = tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(item) for item in (eos or [])}
    pad_id = tokenizer.pad_token_id
    outputs = []
    for row, generated_row in zip(rows, generated, strict=True):
        response_ids = trim_generated_ids(
            generated_row[prompt_width:].tolist(),
            eos_ids=eos_ids,
            pad_id=pad_id,
        )
        if not response_ids:
            raise RuntimeError(f"Teacher produced an empty token trajectory for {row['trajectory_id']}")
        response_text = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        outputs.append(
            {
                **row,
                "teacher_response": response_text,
                "teacher_response_ids": response_ids,
                "response_tokens": len(response_ids),
            }
        )
    return outputs


def existing_output_is_valid(path: Path, expected: int, args: argparse.Namespace) -> bool:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        manifest.get("records") == expected
        and manifest.get("dataset") == args.dataset
        and manifest.get("shard_index") == args.shard_index
        and manifest.get("num_shards") == args.num_shards
        and manifest.get("model_path") == str(args.model_path.resolve())
        and manifest.get("max_new_tokens") == args.max_new_tokens
    )


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch-size and max-new-tokens must be positive")

    all_rows = load_rows(args.input, args.dataset)
    start, end = shard_bounds(len(all_rows), args.shard_index, args.num_shards)
    rows = all_rows[start:end]
    if args.resume and existing_output_is_valid(args.output, len(rows), args):
        print(f"Validated existing trajectory shard, skipping: {args.output}", flush=True)
        return

    process_seed = args.seed + args.shard_index + 1000 * sorted(SYSTEM_PROMPTS).index(args.dataset)
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)

    adapter = load_adapter(args.model_path, args.max_new_tokens)
    adapter.system_prompt = SYSTEM_PROMPTS[args.dataset]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    token_counts = []
    with temporary.open("w", encoding="utf-8") as stream:
        for offset in range(0, len(rows), args.batch_size):
            generated_rows = generate_batch(adapter, rows[offset : offset + args.batch_size], args)
            for record in generated_rows:
                token_counts.append(record["response_tokens"])
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"{args.dataset} shard {args.shard_index}/{args.num_shards}: "
                f"{min(offset + args.batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )
    temporary.replace(args.output)

    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "model_path": str(args.model_path.resolve()),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "range": [start, end],
        "records": len(rows),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": not args.greedy,
        "temperature": None if args.greedy else args.temperature,
        "top_p": None if args.greedy else args.top_p,
        "seed": process_seed,
        "min_response_tokens": min(token_counts) if token_counts else 0,
        "max_response_tokens": max(token_counts) if token_counts else 0,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
