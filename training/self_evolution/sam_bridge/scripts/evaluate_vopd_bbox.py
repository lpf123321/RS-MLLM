#!/usr/bin/env python3
"""Evaluate base SAM3 or a SAM3-LoRA adapter on VOPD bbox grounding."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from vopd_bbox_common import (
    VOPDBBoxDataset,
    build_sam3,
    compute_coco_bbox_metrics,
    compute_top1_metrics,
    load_json,
    make_lora_config,
    move_to_device,
    normalized_cxcywh_to_xywh,
    save_json,
    write_jsonl,
)


def collate_bbox(batch):
    from sam3.train.data.collator import collate_fn_api

    return collate_fn_api(batch, dict_key="input", with_seg_masks=False)


def postprocess_one(
    final_outputs: dict[str, torch.Tensor],
    prompt_index: int,
    width: int,
    height: int,
    nms_iou: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    from torchvision.ops import nms

    logits = final_outputs["pred_logits"][prompt_index].detach().float()
    boxes = final_outputs["pred_boxes"][prompt_index].detach().float()
    probabilities = torch.sigmoid(logits).squeeze(-1)
    presence = final_outputs.get("presence_logit_dec")
    if presence is not None:
        presence_prob = torch.sigmoid(presence[prompt_index].detach().float()).reshape(-1)[0]
        probabilities = probabilities * presence_prob
    valid = probabilities > 1e-6
    scores, filtered_boxes = probabilities[valid], boxes[valid]
    if not len(scores):
        return []
    cx, cy, bw, bh = filtered_boxes.unbind(-1)
    xyxy = torch.stack(
        (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0), dim=-1
    ).clamp(0.0, 1.0)
    keep = nms(xyxy, scores, nms_iou)
    if max_detections > 0:
        keep = keep[:max_detections]
    scores = scores[keep].cpu()
    filtered_boxes = filtered_boxes[keep].cpu()
    results: list[dict[str, Any]] = []
    for index in range(len(scores)):
        results.append(
            {
                "bbox": normalized_cxcywh_to_xywh(filtered_boxes[index].tolist(), width, height),
                "score": float(scores[index]),
            }
        )
    return results


def run_inference(
    model,
    dataset: VOPDBBoxDataset,
    device: torch.device,
    nms_iou: float,
    max_detections: int,
    num_workers: int = 2,
    batch_size: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    from sam3.model.model_misc import SAM3Output

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_bbox,
    )
    model.eval()
    coco_predictions: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    prediction_id = 0
    started = time.perf_counter()
    with torch.inference_mode():
        processed_count = 0
        for _, batch in enumerate(loader):
            input_batch = move_to_device(batch["input"], device)
            # SAM3's own validation path uses CUDA autocast. BF16 is selected
            # here to avoid FP16 gradient/range concerns and is supported by
            # the RTX 5090 GPUs used for this experiment.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs_list = model(input_batch)
            with SAM3Output.iteration_mode(
                outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
            ) as outputs_iter:
                final_outputs = list(outputs_iter)[-1][-1]
            current_size = min(batch_size, len(dataset) - processed_count)
            if int(final_outputs["pred_logits"].shape[0]) != current_size:
                raise RuntimeError(
                    "Expected exactly one annotation-level prompt per evaluation image; "
                    f"got {final_outputs['pred_logits'].shape[0]} prompts for {current_size} images"
                )
            for local_index in range(current_size):
                record = dataset.records[processed_count + local_index]
                image_info, ann = record.image, record.annotation
                processed = postprocess_one(
                    final_outputs,
                    prompt_index=local_index,
                    width=int(image_info["width"]),
                    height=int(image_info["height"]),
                    nms_iou=nms_iou,
                    max_detections=max_detections,
                )
                for pred in processed:
                    coco_predictions.append(
                        {
                            "id": prediction_id,
                            "image_id": int(image_info["id"]),
                            "category_id": 1,
                            "bbox": pred["bbox"],
                            "score": pred["score"],
                        }
                    )
                    prediction_id += 1
                raw_rows.append(
                    {
                        "image_id": int(image_info["id"]),
                        "file_name": image_info["file_name"],
                        "source_id": ann.get("source_id"),
                        "prompt": ann["prompt"],
                        "gt_bbox": ann["bbox"],
                        "gt_bboxes": [item["bbox"] for item in record.annotations],
                        "predictions": processed,
                    }
                )
            processed_count += current_size
            if processed_count % 25 < current_size or processed_count == len(dataset):
                print(f"inference: {processed_count}/{len(dataset)}", flush=True)
    elapsed = time.perf_counter() - started
    return coco_predictions, raw_rows, elapsed


def evaluate_loaded_model(
    model,
    split_dir: Path,
    output_dir: Path,
    device: torch.device,
    nms_iou: float = 0.7,
    max_detections: int = 100,
    limit: int | None = None,
    shuffled_count: int = 0,
    seed: int = 42,
    num_workers: int = 2,
    batch_size: int = 1,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = VOPDBBoxDataset(split_dir, num_negatives=0, seed=seed, limit=limit)
    predictions, raw_rows, elapsed = run_inference(
        model,
        dataset,
        device,
        nms_iou,
        max_detections,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    gt = load_json(split_dir / "_annotations.coco.json")
    if limit is not None:
        image_ids = {record.image["id"] for record in dataset.records}
        gt = dict(gt)
        gt["images"] = [item for item in gt["images"] if item["id"] in image_ids]
        gt["annotations"] = [item for item in gt["annotations"] if item["image_id"] in image_ids]
    gt_path = output_dir / "coco_gt_bbox.json"
    save_json(gt_path, gt)
    prediction_path = output_dir / "coco_predictions_bbox.json"
    coco_metrics = compute_coco_bbox_metrics(gt_path, predictions, prediction_path)
    top1_metrics, per_sample = compute_top1_metrics(gt, predictions)
    write_jsonl(output_dir / "raw_predictions.jsonl", raw_rows)
    write_jsonl(output_dir / "per_sample_metrics.jsonl", per_sample)

    diagnostics: dict[str, Any] = {}
    if shuffled_count > 0 and len(dataset) > 1:
        count = min(shuffled_count, len(dataset))
        selected_ids = list(range(count))
        shuffled_prompts = [dataset.prompts[index] for index in selected_ids]
        random.Random(seed).shuffle(shuffled_prompts)
        if any(shuffled_prompts[i] == dataset.prompts[i] for i in selected_ids):
            shuffled_prompts = shuffled_prompts[1:] + shuffled_prompts[:1]
        overrides = {
            int(dataset.records[index].image["id"]): shuffled_prompts[index]
            for index in selected_ids
        }
        shuffled_dataset = VOPDBBoxDataset(
            split_dir,
            num_negatives=0,
            seed=seed,
            limit=count,
            prompt_overrides=overrides,
        )
        _, shuffled_raw, shuffled_elapsed = run_inference(
            model,
            shuffled_dataset,
            device,
            nms_iou,
            max_detections,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        matched_scores = [
            float(raw_rows[index]["predictions"][0]["score"])
            if raw_rows[index]["predictions"]
            else 0.0
            for index in selected_ids
        ]
        shuffled_scores = [
            float(row["predictions"][0]["score"]) if row["predictions"] else 0.0
            for row in shuffled_raw
        ]
        matched_mean = float(np.mean(matched_scores))
        shuffled_mean = float(np.mean(shuffled_scores))
        diagnostics = {
            "shuffled_count": count,
            "matched_top1_score_mean": matched_mean,
            "shuffled_top1_score_mean": shuffled_mean,
            "matched_minus_shuffled_score": matched_mean - shuffled_mean,
            "matched_vs_shuffled_relative_gap": (
                (matched_mean - shuffled_mean) / abs(shuffled_mean)
                if shuffled_mean != 0
                else None
            ),
            "shuffled_inference_seconds": shuffled_elapsed,
            "warning": "Shuffled prompts are weak diagnostic negatives, not guaranteed absent concepts.",
        }
        write_jsonl(output_dir / "shuffled_raw_predictions.jsonl", shuffled_raw)

    metrics: dict[str, Any] = {
        **coco_metrics,
        **top1_metrics,
        "diagnostics": diagnostics,
        "inference_seconds": elapsed,
        "seconds_per_sample": elapsed / len(dataset) if len(dataset) else 0.0,
        "nms_iou": nms_iou,
        "nms_type": "bbox",
        "max_detections": max_detections,
        "presence_scoring": True,
        "bbox_only": True,
    }
    save_json(output_dir / "metrics.json", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("SAM_TRAIN_CONFIG", "configs/vopd_bbox_lora.yaml")))
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("base", "lora"), required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffled-count", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    model = build_sam3(device, eval_mode=True)
    if args.mode == "lora":
        if args.weights is None:
            raise ValueError("--weights is required in lora mode")
        from lora_layers import apply_lora_to_model, load_lora_weights

        model = apply_lora_to_model(model, make_lora_config(config))
        load_lora_weights(model, str(args.weights))
        model.to(device)
    metrics = evaluate_loaded_model(
        model,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        device=device,
        nms_iou=float(config["evaluation"]["nms_iou"]),
        max_detections=int(config["evaluation"]["max_detections"]),
        limit=args.limit,
        shuffled_count=args.shuffled_count,
        seed=int(config["training"]["seed"]),
        num_workers=args.num_workers,
        batch_size=int(args.batch_size or config["evaluation"].get("batch_size", 1)),
    )
    metrics["mode"] = args.mode
    metrics["weights"] = str(args.weights) if args.weights else None
    metrics["device"] = str(device)
    save_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
