#!/usr/bin/env python3
"""BBox-only LoRA fine-tuning for the converted VOPD grounding dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from evaluate_vopd_bbox import collate_bbox, evaluate_loaded_model
from vopd_bbox_common import VOPDBBoxDataset, build_sam3, make_lora_config, move_to_device, save_json
from vopd_negative_common import (
    ManifestVOPDDataset,
    TeacherReplayDataset,
    collate_teacher_replay,
    ranking_loss,
    teacher_distillation_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("SAM_TRAIN_CONFIG", "configs/vopd_bbox_lora.yaml")))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--min-epochs", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int)
    parser.add_argument("--num-negatives", type=int)
    parser.add_argument("--negative-manifest", type=Path, nargs="*", default=[])
    parser.add_argument("--valid-negative-manifest", type=Path)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--negative-types", nargs="*")
    parser.add_argument("--presence-weight", type=float, default=20.0)
    parser.add_argument("--rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--rank-margin", type=float, default=0.2)
    parser.add_argument("--distill-cache", type=Path)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--master-port", type=int, default=29610)
    parser.add_argument("--launched", action="store_true")
    parser.add_argument(
        "--ddp-equivalence-mode",
        action="store_true",
        help="Use the same RNG seed on every rank for the fixed-sample DDP acceptance test.",
    )
    return parser.parse_args()


def launch_distributed(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={len(args.devices)}",
        f"--master_port={args.master_port}",
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--devices",
        *[str(device) for device in args.devices],
        "--launched",
    ]
    for flag, value in (
        ("--output-dir", args.output_dir),
        ("--data-root", args.data_root),
        ("--max-epochs", args.max_epochs),
        ("--min-epochs", args.min_epochs),
        ("--train-limit", args.train_limit),
        ("--valid-limit", args.valid_limit),
        ("--num-negatives", args.num_negatives),
        ("--valid-negative-manifest", args.valid_negative_manifest),
        ("--negative-ratio", args.negative_ratio),
        ("--presence-weight", args.presence_weight),
        ("--rank-loss-weight", args.rank_loss_weight),
        ("--rank-margin", args.rank_margin),
        ("--distill-cache", args.distill_cache),
        ("--distill-weight", args.distill_weight),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    if args.negative_manifest:
        command.append("--negative-manifest")
        command.extend(str(path) for path in args.negative_manifest)
    if args.negative_types:
        command.append("--negative-types")
        command.extend(args.negative_types)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in args.devices)
    if args.ddp_equivalence_mode:
        command.append("--ddp-equivalence-mode")
    raise SystemExit(subprocess.call(command, env=env))


def setup_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def prepare_targets(model, input_batch, outputs_list, matcher, device):
    from sam3.model.model_misc import SAM3Output

    targets_list = [model.back_convert(target) for target in input_batch.find_targets]
    for targets in targets_list:
        for key, value in targets.items():
            if isinstance(value, torch.Tensor):
                targets[key] = value.to(device)
    final_output = None
    with SAM3Output.iteration_mode(
        outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
    ) as outputs_iter:
        for stage_outputs, targets in zip(outputs_iter, targets_list):
            for output in stage_outputs:
                final_output = output
                output["indices"] = matcher(output, targets)
                for auxiliary in output.get("aux_outputs", []):
                    auxiliary["indices"] = matcher(auxiliary, targets)
    if final_output is None:
        raise RuntimeError("SAM3 produced no training output")
    return targets_list, final_output


def extract_final_output(outputs_list):
    from sam3.model.model_misc import SAM3Output

    with SAM3Output.iteration_mode(
        outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
    ) as outputs_iter:
        stages = list(outputs_iter)
    if not stages or not stages[-1]:
        raise RuntimeError("SAM3 produced no output")
    return stages[-1][-1]


def build_loss(presence_weight: float = 20.0):
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher

    matcher = BinaryHungarianMatcherV2(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True)
    loss_functions = [
        Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
        IABCEMdetr(
            pos_weight=10.0,
            weight_dict={"loss_ce": 20.0, "presence_loss": float(presence_weight)},
            pos_focal=False,
            alpha=0.25,
            gamma=2,
            use_presence=True,
            pad_n_queries=200,
        ),
    ]
    wrapper = Sam3LossWrapper(
        loss_fns_find=loss_functions,
        matcher=matcher,
        o2m_matcher=BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4),
        o2m_weight=2.0,
        use_o2m_matcher_on_o2m_aux=False,
        normalization="local",
        normalize_by_valid_object_num=False,
    )
    return matcher, wrapper


def main() -> None:
    args = parse_args()
    if len(args.devices) > 1 and not args.launched and "LOCAL_RANK" not in os.environ:
        launch_distributed(args)

    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = rank = 0
        world_size = 1
        torch.cuda.set_device(args.devices[0])
        device = torch.device(f"cuda:{args.devices[0]}")

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    training = config["training"]
    seed = int(training["seed"])
    setup_seed(seed, 0 if args.ddp_equivalence_mode else rank)
    output_dir = (args.output_dir or Path(config["output"]["output_dir"])).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root or training["data_root"])
    num_negatives = int(training["num_negatives"] if args.num_negatives is None else args.num_negatives)
    if args.negative_manifest:
        train_dataset = ManifestVOPDDataset(
            data_root / "train",
            negative_manifests=args.negative_manifest,
            negative_ratio=args.negative_ratio,
            seed=seed,
            positive_limit=args.train_limit,
            allowed_types=set(args.negative_types) if args.negative_types else None,
        )
    else:
        train_dataset = VOPDBBoxDataset(
            data_root / "train", num_negatives=num_negatives, seed=seed, limit=args.train_limit
        )
    valid_dataset = VOPDBBoxDataset(
        data_root / "valid", num_negatives=0, seed=seed, limit=args.valid_limit
    )
    sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        collate_fn=collate_bbox,
    )
    teacher_dataset = None
    teacher_sampler = None
    teacher_loader = None
    if args.distill_cache and args.distill_weight > 0:
        teacher_dataset = TeacherReplayDataset(args.distill_cache, limit=args.train_limit)
        teacher_sampler = DistributedSampler(teacher_dataset, shuffle=True, seed=seed + 17) if distributed else None
        teacher_loader = DataLoader(
            teacher_dataset,
            batch_size=int(training["batch_size"]),
            shuffle=teacher_sampler is None,
            sampler=teacher_sampler,
            num_workers=int(training["num_workers"]),
            pin_memory=True,
            collate_fn=collate_teacher_replay,
        )

    model = build_sam3(device, eval_mode=False)
    from lora_layers import apply_lora_to_model, count_parameters, save_lora_weights

    model = apply_lora_to_model(model, make_lora_config(config))
    model.to(device)
    stats = count_parameters(model)
    if rank == 0:
        print(json.dumps(stats, indent=2))
    wrapped = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if distributed else model
    optimizer = AdamW(
        [parameter for parameter in wrapped.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    matcher, loss_wrapper = build_loss(args.presence_weight)
    from sam3.train.loss.loss_fns import CORE_LOSS_KEY

    max_epochs = int(args.max_epochs or training["max_epochs"])
    min_epochs = int(args.min_epochs or training["min_epochs"])
    patience = int(training["early_stopping_patience"])
    if rank == 0:
        resolved_config = {
            **config,
            "run": {
                "data_root": str(data_root.resolve()),
                "output_dir": str(output_dir),
                "devices": args.devices,
                "world_size": world_size,
                "max_epochs": max_epochs,
                "min_epochs": min_epochs,
                "train_limit": args.train_limit,
                "valid_limit": args.valid_limit,
                "num_negatives": num_negatives,
                "negative_manifests": [str(path.resolve()) for path in args.negative_manifest],
                "valid_negative_manifest": str(args.valid_negative_manifest.resolve()) if args.valid_negative_manifest else None,
                "negative_ratio": args.negative_ratio,
                "negative_types": args.negative_types,
                "presence_weight": args.presence_weight,
                "rank_loss_weight": args.rank_loss_weight,
                "rank_margin": args.rank_margin,
                "distill_cache": str(args.distill_cache.resolve()) if args.distill_cache else None,
                "distill_weight": args.distill_weight,
                "ddp_equivalence_mode": args.ddp_equivalence_mode,
            },
        }
        if hasattr(train_dataset, "manifest_summary"):
            resolved_config["negative_manifest_summary"] = train_dataset.manifest_summary()
        save_json(output_dir / "resolved_config.json", resolved_config)
    best_map = -1.0
    best_selection_score = -float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()

    for epoch in range(max_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)
        wrapped.train()
        total_loss_sum = 0.0
        total_core_loss_sum = 0.0
        total_rank_loss_sum = 0.0
        total_distill_loss_sum = 0.0
        steps = 0
        epoch_started = time.perf_counter()
        if teacher_sampler is not None:
            teacher_sampler.set_epoch(epoch)
        teacher_iterator = iter(teacher_loader) if teacher_loader is not None else None
        for step, batch in enumerate(train_loader):
            input_batch = move_to_device(batch["input"], device)
            amp_dtype = str(training.get("amp_dtype", "bfloat16")).lower()
            if amp_dtype != "bfloat16":
                raise ValueError(f"Only bfloat16 AMP is supported, got {amp_dtype!r}")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = wrapped(input_batch)
            targets, final_output = prepare_targets(model, input_batch, outputs, matcher, device)
            loss_dict = loss_wrapper(outputs, targets)
            core_loss = loss_dict[CORE_LOSS_KEY]
            rank_term, rank_diagnostics = ranking_loss(
                final_output, targets, margin=args.rank_margin
            )
            distill_term = core_loss.new_zeros(())
            if teacher_iterator is not None:
                try:
                    teacher_batch, teacher_entries = next(teacher_iterator)
                except StopIteration:
                    teacher_iterator = iter(teacher_loader)
                    teacher_batch, teacher_entries = next(teacher_iterator)
                teacher_input = move_to_device(teacher_batch["input"], device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    replay_outputs = wrapped(teacher_input)
                replay_final = extract_final_output(replay_outputs)
                distill_term, distill_diagnostics = teacher_distillation_loss(
                    replay_final, teacher_entries
                )
            loss = (
                core_loss
                + float(args.rank_loss_weight) * rank_term
                + float(args.distill_weight) * distill_term
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, step={step}: {loss}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss_sum += float(loss.detach())
            total_core_loss_sum += float(core_loss.detach())
            total_rank_loss_sum += float(rank_term.detach())
            total_distill_loss_sum += float(distill_term.detach())
            steps += 1
            if rank == 0 and (step + 1) % 10 == 0:
                print(
                    f"epoch={epoch + 1}/{max_epochs} step={step + 1}/{len(train_loader)} "
                    f"loss={float(loss.detach()):.6f} core={float(core_loss.detach()):.6f} "
                    f"rank={float(rank_term.detach()):.6f} distill={float(distill_term.detach()):.6f}",
                    flush=True,
                )

        loss_tensor = torch.tensor(
            [total_loss_sum, total_core_loss_sum, total_rank_loss_sum, total_distill_loss_sum, steps],
            device=device, dtype=torch.float64,
        )
        if distributed:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        average_loss = float(loss_tensor[0] / max(1.0, loss_tensor[4]))
        average_core_loss = float(loss_tensor[1] / max(1.0, loss_tensor[4]))
        average_rank_loss = float(loss_tensor[2] / max(1.0, loss_tensor[4]))
        average_distill_loss = float(loss_tensor[3] / max(1.0, loss_tensor[4]))
        if distributed:
            dist.barrier()

        stop = False
        if rank == 0:
            epoch_path = output_dir / f"epoch_{epoch + 1:02d}_lora_weights.pt"
            save_lora_weights(model, str(epoch_path))
            val_dir = output_dir / f"validation_epoch_{epoch + 1:02d}"
            if args.valid_negative_manifest:
                from evaluate_vopd_selectivity import evaluate_selectivity_loaded_model

                selectivity = evaluate_selectivity_loaded_model(
                    model,
                    split_dir=data_root / "valid",
                    negative_manifest=args.valid_negative_manifest,
                    output_dir=val_dir,
                    device=device,
                    positive_limit=args.valid_limit,
                    negative_limit=args.valid_limit,
                    batch_size=int(config["evaluation"].get("batch_size", 1)),
                    num_workers=int(training["num_workers"]),
                )
                metrics = selectivity["positive"]
                positive_diag = selectivity["positive_diagnostics"]
                negative_metrics = selectivity["negative"]
                negative_fpr = float(negative_metrics["negative_fpr_at_0_5"])
                score_gap = float(positive_diag["positive_true_minus_false_gap"])
            else:
                metrics = evaluate_loaded_model(
                    model,
                    split_dir=data_root / "valid",
                    output_dir=val_dir,
                    device=device,
                    nms_iou=float(config["evaluation"]["nms_iou"]),
                    max_detections=int(config["evaluation"]["max_detections"]),
                    limit=args.valid_limit,
                    shuffled_count=min(100, len(valid_dataset)),
                    seed=seed,
                    num_workers=int(training["num_workers"]),
                    batch_size=int(config["evaluation"].get("batch_size", 1)),
                )
                positive_diag = {}
                negative_metrics = {}
                negative_fpr = 1.0 - float(metrics.get("diagnostics", {}).get("matched_minus_shuffled_score", 0.0))
                score_gap = float(metrics.get("diagnostics", {}).get("matched_minus_shuffled_score", 0.0))
            current_map = float(metrics["bbox_map"])
            # The balanced score is only an epoch selector. Final model choice
            # applies the predeclared hard constraints and CVSearch accuracy.
            selection_score = current_map + 0.20 * (1.0 - negative_fpr) + 0.05 * score_gap
            improved = selection_score > best_selection_score + 1e-12
            if improved:
                best_map = current_map
                best_selection_score = selection_score
                epochs_without_improvement = 0
                save_lora_weights(model, str(output_dir / "best_lora_weights.pt"))
            else:
                epochs_without_improvement += 1
            save_lora_weights(model, str(output_dir / "last_lora_weights.pt"))
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": average_loss,
                    "train_core_loss": average_core_loss,
                    "train_rank_loss": average_rank_loss,
                    "train_distill_loss": average_distill_loss,
                    "valid_bbox_map": current_map,
                    "valid_bbox_ap50": metrics["bbox_ap50"],
                    "valid_matched_vs_shuffled_gap": metrics.get("diagnostics", {}).get(
                        "matched_vs_shuffled_relative_gap"
                    ),
                    "valid_negative_fpr_at_0_5": negative_metrics.get("negative_fpr_at_0_5"),
                    "valid_negative_max_score_mean": negative_metrics.get("negative_max_score_mean"),
                    "valid_positive_true_score_mean": positive_diag.get("positive_true_score_mean"),
                    "valid_positive_false_score_mean": positive_diag.get("positive_false_score_mean"),
                    "valid_positive_true_minus_false_gap": positive_diag.get("positive_true_minus_false_gap"),
                    "selection_score": selection_score,
                    "improved": improved,
                    "epoch_seconds": time.perf_counter() - epoch_started,
                }
            )
            save_json(output_dir / "training_history.json", history)
            print(json.dumps(history[-1], indent=2), flush=True)
            stop = (epoch + 1) >= min_epochs and epochs_without_improvement >= patience

        if distributed:
            stop_tensor = torch.tensor([int(stop)], device=device)
            dist.broadcast(stop_tensor, src=0)
            stop = bool(stop_tensor.item())
            dist.barrier()
        if stop:
            if rank == 0:
                print(f"Early stopping after epoch {epoch + 1}")
            break

    if rank == 0:
        summary = {
            "best_valid_bbox_map": best_map,
            "best_selection_score": best_selection_score,
            "epochs_completed": len(history),
            "num_negatives": num_negatives,
            "negative_manifests": [str(path.resolve()) for path in args.negative_manifest],
            "valid_negative_manifest": str(args.valid_negative_manifest.resolve()) if args.valid_negative_manifest else None,
            "negative_ratio": args.negative_ratio,
            "presence_weight": args.presence_weight,
            "rank_loss_weight": args.rank_loss_weight,
            "distill_cache": str(args.distill_cache.resolve()) if args.distill_cache else None,
            "distill_weight": args.distill_weight,
            "train_samples": len(train_dataset),
            "valid_samples": len(valid_dataset),
            "world_size": world_size,
            "parameter_stats": stats,
            "training_seconds": time.perf_counter() - training_started,
            "amp_dtype": training.get("amp_dtype", "bfloat16"),
            "checkpoint": str(config["model"]["checkpoint"]),
        }
        save_json(output_dir / "training_summary.json", summary)
        print(json.dumps(summary, indent=2))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
