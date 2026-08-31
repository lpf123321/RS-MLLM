#!/usr/bin/env python3
"""Evaluate VOPD positive localization and verified-negative rejection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from build_negative_manifests import PromptPairDataset
from evaluate_vopd_bbox import collate_bbox, evaluate_loaded_model, postprocess_one
from vopd_bbox_common import (
    box_iou_xywh,
    build_sam3,
    make_lora_config,
    move_to_device,
    read_jsonl,
    save_json,
    write_jsonl,
)


def positive_score_diagnostics(raw_path: Path) -> dict[str, Any]:
    rows = read_jsonl(raw_path)
    true_scores: list[float] = []
    false_scores: list[float] = []
    false_counts: list[int] = []
    gaps: list[float] = []
    for row in rows:
        true = [
            float(pred["score"])
            for pred in row["predictions"]
            if box_iou_xywh(row["gt_bbox"], pred["bbox"]) >= 0.5
        ]
        false = [
            float(pred["score"])
            for pred in row["predictions"]
            if box_iou_xywh(row["gt_bbox"], pred["bbox"]) < 0.1
        ]
        true_score = max(true, default=0.0)
        false_score = max(false, default=0.0)
        true_scores.append(true_score)
        false_scores.append(false_score)
        false_counts.append(sum(score >= 0.5 for score in false))
        gaps.append(true_score - false_score)
    return {
        "positive_true_score_mean": float(np.mean(true_scores)) if true_scores else 0.0,
        "positive_false_score_mean": float(np.mean(false_scores)) if false_scores else 0.0,
        "positive_true_minus_false_gap": float(np.mean(gaps)) if gaps else 0.0,
        "positive_false_boxes_at_0_5_mean": float(np.mean(false_counts)) if false_counts else 0.0,
        "positive_miss_rate_at_0_5": float(np.mean([score < 0.5 for score in true_scores])) if true_scores else 1.0,
    }


def evaluate_negative_rows(
    model,
    manifest: Path,
    output_dir: Path,
    device: torch.device,
    limit: int | None = None,
    batch_size: int = 2,
    num_workers: int = 2,
) -> dict[str, Any]:
    rows = [row for row in read_jsonl(manifest) if row.get("accepted", True)]
    if limit is not None:
        rows = rows[:limit]
    dataset = PromptPairDataset(rows)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=collate_bbox,
    )
    from sam3.model.model_misc import SAM3Output

    model.eval()
    result_rows: list[dict[str, Any]] = []
    processed = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = move_to_device(batch["input"], device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(inputs)
            with SAM3Output.iteration_mode(outputs, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as iterator:
                final = list(iterator)[-1][-1]
            current = int(final["pred_logits"].shape[0])
            presence = final.get("presence_logit_dec")
            for local in range(current):
                row = rows[processed + local]
                image = Image.open(row["image_path"])
                if row.get("crop_xyxy"):
                    x1, y1, x2, y2 = map(int, row["crop_xyxy"])
                    width, height = x2 - x1, y2 - y1
                else:
                    width, height = image.size
                predictions = postprocess_one(final, local, width, height, 0.7, 100)
                max_score = float(predictions[0]["score"]) if predictions else 0.0
                presence_probability = (
                    float(torch.sigmoid(presence[local].float()).reshape(-1)[0].cpu())
                    if presence is not None else None
                )
                result_rows.append(
                    {
                        **row,
                        "max_score": max_score,
                        "boxes_at_0_5": sum(float(item["score"]) >= 0.5 for item in predictions),
                        "presence_probability": presence_probability,
                        "predictions": predictions,
                    }
                )
            processed += current
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "negative_predictions.jsonl", result_rows)
    max_scores = [float(row["max_score"]) for row in result_rows]
    box_counts = [int(row["boxes_at_0_5"]) for row in result_rows]
    presence_values = [
        float(row["presence_probability"])
        for row in result_rows if row["presence_probability"] is not None
    ]
    by_type: dict[str, dict[str, Any]] = {}
    for negative_type in sorted({str(row.get("negative_type", "unknown")) for row in result_rows}):
        subset = [row for row in result_rows if str(row.get("negative_type", "unknown")) == negative_type]
        by_type[negative_type] = {
            "count": len(subset),
            "fpr_at_0_5": float(np.mean([row["max_score"] >= 0.5 for row in subset])),
            "max_score_mean": float(np.mean([row["max_score"] for row in subset])),
            "false_boxes_at_0_5_mean": float(np.mean([row["boxes_at_0_5"] for row in subset])),
        }
    metrics = {
        "negative_count": len(result_rows),
        "negative_fpr_at_0_5": float(np.mean([score >= 0.5 for score in max_scores])) if max_scores else 0.0,
        "negative_max_score_mean": float(np.mean(max_scores)) if max_scores else 0.0,
        "negative_max_score_p95": float(np.quantile(max_scores, 0.95)) if max_scores else 0.0,
        "negative_false_boxes_at_0_5_mean": float(np.mean(box_counts)) if box_counts else 0.0,
        "negative_presence_mean": float(np.mean(presence_values)) if presence_values else None,
        "by_type": by_type,
    }
    save_json(output_dir / "negative_metrics.json", metrics)
    return metrics


def evaluate_selectivity_loaded_model(
    model,
    split_dir: Path,
    negative_manifest: Path,
    output_dir: Path,
    device: torch.device,
    positive_limit: int | None = None,
    negative_limit: int | None = None,
    batch_size: int = 2,
    num_workers: int = 2,
) -> dict[str, Any]:
    positive_dir = output_dir / "positive"
    positive = evaluate_loaded_model(
        model, split_dir=split_dir, output_dir=positive_dir, device=device,
        nms_iou=0.7, max_detections=100, limit=positive_limit,
        shuffled_count=0, seed=42, num_workers=num_workers, batch_size=batch_size,
    )
    positive_diag = positive_score_diagnostics(positive_dir / "raw_predictions.jsonl")
    negative = evaluate_negative_rows(
        model, manifest=negative_manifest, output_dir=output_dir / "negative",
        device=device, limit=negative_limit, batch_size=batch_size, num_workers=num_workers,
    )
    combined = {"positive": positive, "positive_diagnostics": positive_diag, "negative": negative}
    save_json(output_dir / "selectivity_metrics.json", combined)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("SAM_TRAIN_CONFIG", "configs/vopd_bbox_lora.yaml")))
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--negative-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("base", "lora"), required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--positive-limit", type=int)
    parser.add_argument("--negative-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--positive-source",
        type=Path,
        help="Reuse positive metrics from an existing selectivity_metrics.json and only rerun negatives.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    model = build_sam3(device, eval_mode=True)
    if args.mode == "lora":
        if args.weights is None:
            raise ValueError("--weights required in lora mode")
        from lora_layers import apply_lora_to_model, load_lora_weights
        model = apply_lora_to_model(model, make_lora_config(config))
        load_lora_weights(model, str(args.weights))
        model.to(device)
    if args.positive_source:
        with args.positive_source.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        negative = evaluate_negative_rows(
            model,
            manifest=args.negative_manifest,
            output_dir=args.output_dir / "negative",
            device=device,
            limit=args.negative_limit,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        metrics = {
            "positive": source["positive"],
            "positive_diagnostics": source["positive_diagnostics"],
            "negative": negative,
            "positive_source": str(args.positive_source.resolve()),
        }
    else:
        metrics = evaluate_selectivity_loaded_model(
            model, args.split_dir, args.negative_manifest, args.output_dir, device,
            positive_limit=args.positive_limit, negative_limit=args.negative_limit,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
    metrics["mode"] = args.mode
    metrics["weights"] = str(args.weights) if args.weights else None
    save_json(args.output_dir / "selectivity_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
