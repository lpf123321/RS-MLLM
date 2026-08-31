"""Native Qwen3.5-VL LoRA SFT used by the retained Expert experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

from .data import load_records, resolve_image_path, sha256


def main() -> None:
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from qwen_vl_utils import process_vision_info
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from transformers import (
        AutoProcessor,
        Qwen3_5ForConditionalGeneration,
        get_cosine_schedule_with_warmup,
    )

    class SFTDataset(Dataset):
        def __init__(
            self,
            path: Path,
            image_root: Path,
            processor,
            max_pixels: int,
            expected_records: int,
        ):
            self.records = load_records(path)
            if len(self.records) != expected_records:
                raise ValueError(
                    f"expected {expected_records} records, found {len(self.records)}"
                )
            self.image_root = image_root
            self.processor = processor
            self.max_pixels = max_pixels
            weights = [float(record.get("loss_weight", 1.0)) for record in self.records]
            if any(weight <= 0 or not math.isfinite(weight) for weight in weights):
                raise ValueError("loss_weight must be finite and strictly positive")
            self.loss_weight_mean = sum(weights) / len(weights)

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            record = self.records[index]
            prompt = record["conversations"][0]["value"]
            answer = record["conversations"][1]["value"].strip()
            images = [
                str(resolve_image_path(image, self.image_root, allow_absolute=False))
                for image in record["image"]
            ]
            messages = [
                {
                    "role": "user",
                    "content": [
                        *(
                            {
                                "type": "image",
                                "image": image,
                                **(
                                    {"max_pixels": self.max_pixels}
                                    if self.max_pixels
                                    else {}
                                ),
                            }
                            for image in images
                        ),
                        {
                            "type": "text",
                            "text": prompt.replace("<image>\n", "", 1),
                        },
                    ],
                }
            ]
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            suffix = f"{answer}<|im_end|>\n"
            image_inputs, video_inputs = process_vision_info(messages)
            values = self.processor(
                text=[rendered + suffix],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors="pt",
            )
            suffix_ids = self.processor.tokenizer(
                suffix,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
            labels = torch.full_like(values["input_ids"], -100)
            labels[:, -suffix_ids.shape[1] :] = values["input_ids"][
                :, -suffix_ids.shape[1] :
            ]
            values["labels"] = labels
            values["sample_index"] = torch.tensor(index)
            values["sample_weight"] = torch.tensor(
                float(record.get("loss_weight", 1.0)) / self.loss_weight_mean,
                dtype=torch.float32,
            )
            return values

    def single_item_collate(items):
        if len(items) != 1:
            raise ValueError("the retained trainer requires micro-batch size 1")
        return items[0]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-records", required=True, type=int)
    parser.add_argument("--init-lora", type=Path)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-pixels", type=int, default=3211264)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the retained training configuration")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.init_lora:
        model = PeftModel.from_pretrained(
            model, args.init_lora, is_trainable=True
        )
        target_modules = sorted(
            {
                name.rsplit(".lora_", 1)[0]
                for name, _ in model.named_parameters()
                if ".lora_" in name
            }
        )
    else:
        target_modules = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and "visual" not in name
            and "lm_head" not in name
        ]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.to(device)

    if rank == 0:
        trainable, total = model.get_nb_trainable_parameters()
        print(
            json.dumps(
                {
                    "model": str(args.model),
                    "target_modules": len(target_modules),
                    "trainable_parameters": trainable,
                    "total_parameters": total,
                    "trainable_fraction": trainable / total,
                },
                indent=2,
            ),
            flush=True,
        )

    dataset = SFTDataset(
        args.data,
        args.image_root,
        processor,
        args.max_pixels,
        args.expected_records,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=single_item_collate,
        persistent_workers=args.workers > 0,
    )

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=0.0)
    updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    epoch_updates = updates_per_epoch * args.epochs
    total_updates = (
        min(epoch_updates, args.max_updates) if args.max_updates else epoch_updates
    )
    if total_updates <= 0:
        raise ValueError("training must contain at least one optimizer update")
    warmup_steps = max(1, round(total_updates * 0.03))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_updates
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    update = 0
    losses: list[float] = []
    weighted_losses: list[float] = []
    start = time.time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for micro_step, batch in enumerate(loader):
            sample_index = int(batch.pop("sample_index"))
            sample_weight = float(batch.pop("sample_weight").item())
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            should_step = (
                (micro_step + 1) % args.grad_accum == 0
                or micro_step + 1 == len(loader)
            )
            sync_context = nullcontext()
            if distributed and not should_step:
                sync_context = model.no_sync()
            with sync_context:
                # Only the answer suffix contributes labels.  Requesting suffix
                # logits avoids allocating full image/prompt sequence logits.
                answer_tokens = int((batch["labels"] != -100).sum().item())
                logits_to_keep = answer_tokens + 1
                batch["labels"] = batch["labels"][:, -logits_to_keep:]
                output = model(**batch, logits_to_keep=logits_to_keep)
                raw_loss = output.loss
                weighted_loss = raw_loss * sample_weight
                (weighted_loss / args.grad_accum).backward()
            losses.append(float(raw_loss.detach()))
            weighted_losses.append(float(weighted_loss.detach()))
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if rank == 0:
                    recent = losses[-args.grad_accum :]
                    print(
                        json.dumps(
                            {
                                "update": update,
                                "total_updates": total_updates,
                                "loss": sum(recent) / len(recent),
                                "weighted_loss": sum(
                                    weighted_losses[-args.grad_accum :]
                                )
                                / len(weighted_losses[-args.grad_accum :]),
                                "sample_weight": sample_weight,
                                "grad_norm": float(grad_norm),
                                "lr": scheduler.get_last_lr()[0],
                                "sample_index": sample_index,
                                "elapsed_seconds": time.time() - start,
                            }
                        ),
                        flush=True,
                    )
                if args.max_updates and update >= args.max_updates:
                    break
        if args.max_updates and update >= args.max_updates:
            break

    if distributed:
        dist.barrier()
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if distributed else model
        unwrapped.save_pretrained(args.output, safe_serialization=True)
        processor.save_pretrained(args.output)
        metrics = {
            "records": len(dataset),
            "data_sha256": sha256(args.data),
            "epochs": args.epochs,
            "world_size": world_size,
            "effective_batch_size": world_size * args.grad_accum,
            "gradient_accumulation": args.grad_accum,
            "runtime_max_pixels": args.max_pixels,
            "optimizer_updates": update,
            "max_updates": args.max_updates,
            "init_lora": args.init_lora.name if args.init_lora else None,
            "learning_rate": args.lr,
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "seed": args.seed,
            "mean_micro_loss": sum(losses) / len(losses),
            "mean_weighted_micro_loss": sum(weighted_losses) / len(weighted_losses),
            "loss_weight_mean": dataset.loss_weight_mean,
            "loss_weight_min": min(
                float(record.get("loss_weight", 1.0)) for record in dataset.records
            ),
            "loss_weight_max": max(
                float(record.get("loss_weight", 1.0)) for record in dataset.records
            ),
            "elapsed_seconds": time.time() - start,
        }
        (args.output / "training_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2), flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
