#!/usr/bin/env python3
"""Cache original SAM3 responses for prompt-replay distillation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from build_negative_manifests import PromptPairDataset
from evaluate_vopd_bbox import collate_bbox
from vopd_bbox_common import build_sam3, move_to_device, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument(
        "--one-per-image",
        action="store_true",
        help="Keep the most verifier-ambiguous candidate for each image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.manifest:
        rows.extend(read_jsonl(path))
    if args.accepted_only:
        rows = [row for row in rows if row.get("accepted", True)]
    if args.one_per_image:
        grouped = {}
        for row in rows:
            image_id = int(row["image_id"])
            # Probability near 0.5 is maximally ambiguous. Prefer rejected
            # candidates when equally close so they are never mislabeled as
            # hard negatives and are supervised only by Base-SAM replay.
            key = (
                abs(float(row.get("absent_probability", 0.5)) - 0.5),
                bool(row.get("accepted", True)),
                str(row["negative_prompt"]),
            )
            current = grouped.get(image_id)
            if current is None or key < current[0]:
                grouped[image_id] = (key, row)
        rows = [grouped[image_id][1] for image_id in sorted(grouped)]
    if args.limit is not None:
        rows = rows[: args.limit]
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(args.device)
    model = build_sam3(device, eval_mode=True)
    dataset = PromptPairDataset(rows)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
        pin_memory=True, collate_fn=collate_bbox,
    )
    from sam3.model.model_misc import SAM3Output

    entries = []
    processed = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = move_to_device(batch["input"], device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(inputs)
            with SAM3Output.iteration_mode(outputs, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as iterator:
                final = list(iterator)[-1][-1]
            current = int(final["pred_logits"].shape[0])
            for local in range(current):
                row = rows[processed + local]
                entries.append(
                    {
                        "image_id": int(row["image_id"]),
                        "image_path": str(row["image_path"]),
                        "prompt": str(row["negative_prompt"]),
                        "crop_xyxy": row.get("crop_xyxy"),
                        "pred_logits": final["pred_logits"][local].detach().cpu().to(torch.float16),
                        "pred_boxes": final["pred_boxes"][local].detach().cpu().to(torch.float16),
                        "presence_logit": (
                            final["presence_logit_dec"][local].detach().cpu().to(torch.float16)
                            if final.get("presence_logit_dec") is not None else None
                        ),
                    }
                )
            processed += current
            if processed % 50 < current or processed == len(rows):
                print(f"teacher cache {processed}/{len(rows)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "vopd-sam3-teacher-v1",
            "base_checkpoint": os.environ.get("SAM3_CHECKPOINT", "sam3.pt"),
            "source_manifests": [str(path.resolve()) for path in args.manifest],
            "one_per_image": args.one_per_image,
            "entries": entries,
        },
        args.output,
    )
    print(json.dumps({"entries": len(entries), "output": str(args.output), "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
