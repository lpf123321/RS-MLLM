#!/usr/bin/env python3
"""Generate, visually verify, score, and audit VOPD negative prompts."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from evaluate_vopd_bbox import collate_bbox, postprocess_one
from vopd_bbox_common import (
    build_sam3,
    lexical_tokens,
    load_json,
    make_lora_config,
    move_to_device,
    read_jsonl,
    write_jsonl,
)


GENERIC_POOL = [
    "microscopic cell under a microscope",
    "astronaut floating in outer space",
    "submarine under deep ocean water",
    "dinosaur fossil skeleton in a museum",
    "MRI scanner in a hospital room",
    "volcanic lava flowing from a crater",
    "polar bear on an ice floe",
    "space satellite orbiting earth",
    "deep sea jellyfish underwater",
    "snowmobile on a snowy trail",
    "laboratory centrifuge machine",
    "hot air balloon flying in the sky",
]

FLIPS = [
    (r"\bleftmost\b", "rightmost"),
    (r"\brightmost\b", "leftmost"),
    (r"\bleft side\b", "right side"),
    (r"\bright side\b", "left side"),
    (r"\bupper-left\b", "lower-right"),
    (r"\blower-right\b", "upper-left"),
    (r"\bupper-right\b", "lower-left"),
    (r"\blower-left\b", "upper-right"),
    (r"\bupper\b", "lower"),
    (r"\blower\b", "upper"),
    (r"\btop\b", "bottom"),
    (r"\bbottom\b", "top"),
    (r"\bfront\b", "rear"),
    (r"\brear\b", "front"),
    (r"\binside\b", "outside"),
    (r"\boutside\b", "inside"),
    (r"\babove\b", "below"),
    (r"\bbelow\b", "above"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate")
    generate.add_argument("--split-dir", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--method", choices=("generic", "random", "semantic", "counterfactual", "crop"), required=True
    )
    generate.add_argument("--limit", type=int)
    generate.add_argument("--per-image", type=int, default=3)
    generate.add_argument("--seed", type=int, default=42)

    verify = sub.add_parser("verify")
    verify.add_argument("--input", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--model", type=Path, default=Path(os.environ.get("NEGATIVE_VERIFIER_MODEL", "Qwen3-VL-8B-Instruct")))
    verify.add_argument("--device", type=int, default=0)
    verify.add_argument("--threshold", type=float, default=0.9)
    verify.add_argument("--limit", type=int)
    verify.add_argument("--max-pixels", type=int, default=448 * 448)
    verify.add_argument("--batch-size", type=int, default=8)

    rewrite = sub.add_parser("rewrite-counterfactual")
    rewrite.add_argument("--split-dir", type=Path, required=True)
    rewrite.add_argument("--output", type=Path, required=True)
    rewrite.add_argument("--model", type=Path, default=Path(os.environ.get("NEGATIVE_VERIFIER_MODEL", "Qwen3-VL-8B-Instruct")))
    rewrite.add_argument("--device", type=int, default=0)
    rewrite.add_argument("--limit", type=int)
    rewrite.add_argument("--batch-size", type=int, default=8)

    score = sub.add_parser("score")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--config", type=Path, default=Path(os.environ.get("SAM_TRAIN_CONFIG", "configs/vopd_bbox_lora.yaml")))
    score.add_argument("--mode", choices=("base", "lora"), required=True)
    score.add_argument("--weights", type=Path)
    score.add_argument("--device", type=int, default=0)
    score.add_argument("--batch-size", type=int, default=2)
    score.add_argument("--limit", type=int)

    active = sub.add_parser("active")
    active.add_argument("--input", type=Path, nargs="+", required=True)
    active.add_argument("--output", type=Path, required=True)
    active.add_argument("--threshold", type=float, default=0.5)
    active.add_argument("--per-image", type=int, default=2)

    compose = sub.add_parser("compose")
    compose.add_argument("--input", type=Path, nargs="+", required=True)
    compose.add_argument("--output", type=Path, required=True)
    compose.add_argument(
        "--quota", nargs="+",
        help="Per-image quotas such as generic=2 random=1; earlier entries have priority.",
    )
    compose.add_argument(
        "--one-per-image-priority", nargs="+",
        help="Choose one row per image using this ordered type fallback list.",
    )
    compose.add_argument(
        "--balanced-one-per-image", nargs="+",
        help="Choose one row per image while rotating the preferred type by image id.",
    )
    compose.add_argument(
        "--fallback-type",
        help="Fallback type used only when no priority/balanced candidate exists for an image.",
    )

    audit = sub.add_parser("audit")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--count", type=int, default=50)
    audit.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_records(split_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    coco = load_json(split_dir / "_annotations.coco.json")
    anns = {int(ann["image_id"]): ann for ann in coco["annotations"]}
    rows = []
    for image in sorted(coco["images"], key=lambda item: int(item["id"])):
        ann = anns[int(image["id"])]
        rows.append({"image": image, "annotation": ann})
    return rows[:limit] if limit is not None else rows


def base_row(split_dir: Path, record: dict[str, Any], method: str, prompt: str) -> dict[str, Any]:
    image, ann = record["image"], record["annotation"]
    return {
        "image_id": int(image["id"]),
        "source_id": ann.get("source_id"),
        "image_path": str((split_dir / image["file_name"]).resolve()),
        "positive_prompt": str(ann["prompt"]),
        "negative_prompt": str(prompt).lower().strip(),
        "negative_type": method,
        "accepted": None,
    }


def generate_candidates(args: argparse.Namespace) -> None:
    split_dir = args.split_dir.resolve()
    records = load_records(split_dir, args.limit)
    rng = random.Random(args.seed)
    prompts = [str(row["annotation"]["prompt"]).lower() for row in records]
    token_sets = [lexical_tokens(prompt) for prompt in prompts]
    inverted: dict[str, list[int]] = defaultdict(list)
    for index, tokens in enumerate(token_sets):
        for token in tokens:
            inverted[token].append(index)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        positive = prompts[index]
        positive_tokens = token_sets[index]
        candidates: list[tuple[str, dict[str, Any]]] = []
        if args.method == "generic":
            offset = (index * args.per_image) % len(GENERIC_POOL)
            for j in range(args.per_image):
                candidates.append((GENERIC_POOL[(offset + j) % len(GENERIC_POOL)], {}))
        elif args.method == "random":
            order = list(range(len(records)))
            rng.shuffle(order)
            for other in order:
                if other == index or token_sets[other] & positive_tokens:
                    continue
                candidates.append((prompts[other], {"candidate_source_image_id": int(records[other]["image"]["id"])}))
                if len(candidates) >= args.per_image:
                    break
        elif args.method == "semantic":
            pool = set()
            for token in positive_tokens:
                pool.update(inverted[token])
            scored = []
            for other in pool:
                if other == index or prompts[other] == positive:
                    continue
                union = positive_tokens | token_sets[other]
                overlap = len(positive_tokens & token_sets[other]) / max(1, len(union))
                if overlap <= 0 or overlap >= 0.9:
                    continue
                scored.append((overlap, other))
            scored.sort(key=lambda item: (-item[0], item[1]))
            for overlap, other in scored[: args.per_image]:
                candidates.append(
                    (
                        prompts[other],
                        {"candidate_source_image_id": int(records[other]["image"]["id"]), "lexical_jaccard": overlap},
                    )
                )
        elif args.method == "counterfactual":
            for pattern, replacement in FLIPS:
                rewritten, count = re.subn(pattern, replacement, positive, count=1, flags=re.I)
                if count and rewritten != positive:
                    candidates.append((rewritten, {"counterfactual_rule": f"{pattern}->{replacement}"}))
                if len(candidates) >= args.per_image:
                    break
        elif args.method == "crop":
            image = record["image"]
            x, y, width, height = map(float, record["annotation"]["bbox"])
            image_w, image_h = int(image["width"]), int(image["height"])
            margin = max(4, int(0.02 * min(image_w, image_h)))
            rectangles = [
                (0, 0, max(0, int(x) - margin), image_h),
                (min(image_w, int(x + width) + margin), 0, image_w, image_h),
                (0, 0, image_w, max(0, int(y) - margin)),
                (0, min(image_h, int(y + height) + margin), image_w, image_h),
            ]
            rectangles = [
                rect for rect in rectangles
                if rect[2] - rect[0] >= 128 and rect[3] - rect[1] >= 128
                and (rect[2] - rect[0]) * (rect[3] - rect[1]) >= 0.1 * image_w * image_h
            ]
            rectangles.sort(key=lambda rect: -((rect[2] - rect[0]) * (rect[3] - rect[1])))
            for rectangle in rectangles[: args.per_image]:
                candidates.append((positive, {"crop_xyxy": list(rectangle)}))
        for prompt, extra in candidates:
            rows.append({**base_row(split_dir, record, args.method, prompt), **extra})
    write_jsonl(args.output, rows)
    print(json.dumps({"method": args.method, "images": len(records), "candidates": len(rows), "output": str(args.output)}, indent=2))


def token_id(tokenizer, word: str) -> int:
    ids = tokenizer(word).input_ids
    return int(ids[-1])


def verify_candidates(args: argparse.Namespace) -> None:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    device = torch.device(f"cuda:{args.device}")
    processor = AutoProcessor.from_pretrained(str(args.model))
    processor.image_processor.max_pixels = args.max_pixels
    processor.image_processor.min_pixels = 56 * 56
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model), dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=str(device)
    ).eval()
    yes_id, no_id = token_id(processor.tokenizer, "Yes"), token_id(processor.tokenizer, "No")
    verified: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        images: list[Image.Image] = []
        texts: list[str] = []
        for row in batch:
            image = Image.open(row["image_path"]).convert("RGB")
            if row.get("crop_xyxy"):
                image = image.crop(tuple(map(int, row["crop_xyxy"])))
            question = (
                f"Target phrase: '{row['negative_prompt']}'. Does this exact target or region visibly exist "
                "in the image? Consider every noun, attribute, part, and spatial relation in the phrase. "
                "Answer Yes or No."
            )
            message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
            texts.append(processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
            images.append(image)
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :][:, [yes_id, no_id]].float()
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        for row, probs in zip(batch, probabilities, strict=True):
            absent = float(probs[1])
            verified.append(
                {
                    **row,
                    "verifier_model": str(args.model),
                    "present_probability": float(probs[0]),
                    "absent_probability": absent,
                    "verification_threshold": args.threshold,
                    "accepted": absent >= args.threshold,
                }
            )
        completed = min(start + len(batch), len(rows))
        if completed % 25 < args.batch_size or completed == len(rows):
            accepted = sum(item["accepted"] for item in verified)
            print(f"verified {completed}/{len(rows)} accepted={accepted}", flush=True)
    write_jsonl(args.output, verified)
    print(json.dumps({"total": len(verified), "accepted": sum(row["accepted"] for row in verified), "output": str(args.output)}, indent=2))


def rewrite_counterfactuals(args: argparse.Namespace) -> None:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    records = load_records(args.split_dir.resolve(), args.limit)
    device = torch.device(f"cuda:{args.device}")
    processor = AutoProcessor.from_pretrained(str(args.model))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model), dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=str(device)
    ).eval()
    generated_rows: list[dict[str, Any]] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        texts = []
        for record in batch:
            ann = record["annotation"]
            instruction = (
                "Rewrite the visual target phrase below by changing exactly one meaningful spatial relation, "
                "object part, or distinguishing attribute, so it refers to a plausible but different target. "
                "Keep the same concise noun-phrase style, use 2 to 20 words, do not add 'not', and output only "
                f"the rewritten phrase. Original phrase: {ann['prompt']}"
            )
            message = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
            texts.append(processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
        inputs = processor(text=texts, images=None, return_tensors="pt", padding=True, padding_side="left").to(device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, outputs)]
        generations = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for record, generation in zip(batch, generations):
            phrase = generation.strip().strip("`\"'").rstrip(".?! ").lower()
            positive = str(record["annotation"]["prompt"]).lower().strip()
            words = re.findall(r"[a-z0-9][a-z0-9'-]*", phrase)
            row = base_row(args.split_dir.resolve(), record, "counterfactual", phrase)
            row.update(
                {
                    "generator_model": str(args.model),
                    "generator_instruction": "minimal-counterfactual-v1",
                    "generation_valid": bool(2 <= len(words) <= 20 and phrase != positive and "not " not in phrase),
                }
            )
            generated_rows.append(row)
        print(f"rewritten {min(start + len(batch), len(records))}/{len(records)}", flush=True)
    write_jsonl(args.output, [row for row in generated_rows if row["generation_valid"]])
    print(json.dumps({"input": len(records), "valid": sum(row["generation_valid"] for row in generated_rows), "output": str(args.output)}, indent=2))


class PromptPairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, Image as SAMImage, InferenceMetadata
        from torchvision.transforms import v2

        row = self.rows[index]
        raw = Image.open(row["image_path"]).convert("RGB")
        if row.get("crop_xyxy"):
            raw = raw.crop(tuple(map(int, row["crop_xyxy"])))
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.5] * 3, [0.5] * 3)])
        tensor = transform(raw.resize((1008, 1008), Image.Resampling.BILINEAR))
        image = SAMImage(data=tensor, objects=[], size=(1008, 1008))
        metadata = InferenceMetadata(
            coco_image_id=index,
            original_image_id=int(row["image_id"]),
            original_category_id=-1,
            original_size=(raw.height, raw.width),
            object_id=-1,
            frame_index=-1,
        )
        query = FindQueryLoaded(
            query_text=str(row["negative_prompt"]), image_id=0, object_ids_output=[], is_exhaustive=True,
            query_processing_order=0, inference_metadata=metadata,
        )
        return Datapoint(find_queries=[query], images=[image], raw_images=[raw])


def score_candidates(args: argparse.Namespace) -> None:
    import yaml
    from sam3.model.model_misc import SAM3Output

    rows = [row for row in read_jsonl(args.input) if row.get("accepted", True)]
    if args.limit is not None:
        rows = rows[: args.limit]
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(args.device)
    model = build_sam3(device, eval_mode=True)
    if args.mode == "lora":
        if args.weights is None:
            raise ValueError("--weights required for lora mode")
        from lora_layers import apply_lora_to_model, load_lora_weights
        model = apply_lora_to_model(model, make_lora_config(config))
        load_lora_weights(model, str(args.weights))
        model.to(device)
    loader = DataLoader(PromptPairDataset(rows), batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_bbox)
    output_rows: list[dict[str, Any]] = []
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
                image = Image.open(row["image_path"])
                if row.get("crop_xyxy"):
                    x1, y1, x2, y2 = map(int, row["crop_xyxy"])
                    width, height = x2 - x1, y2 - y1
                else:
                    width, height = image.size
                predictions = postprocess_one(final, local, width, height, 0.7, 100)
                max_score = float(predictions[0]["score"]) if predictions else 0.0
                count_over = sum(float(item["score"]) >= 0.5 for item in predictions)
                output_rows.append({**row, f"{args.mode}_max_score": max_score, f"{args.mode}_boxes_at_0_5": count_over})
            processed += current
            if processed % 50 < current or processed == len(rows):
                print(f"scored {processed}/{len(rows)}", flush=True)
    write_jsonl(args.output, output_rows)


def build_active(args: argparse.Namespace) -> None:
    rows = []
    for path in args.input:
        rows.extend(read_jsonl(path))
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        score = float(row.get("lora_max_score", 0.0))
        if row.get("accepted", True) and score >= args.threshold:
            grouped[int(row["image_id"])].append(
                {**row, "negative_type": "active", "hardness_score": score}
            )
    selected = []
    for image_id in sorted(grouped):
        choices = sorted(grouped[image_id], key=lambda row: -float(row["hardness_score"]))
        selected.extend(choices[: args.per_image])
    write_jsonl(args.output, selected)
    print(json.dumps({"images": len(grouped), "rows": len(selected), "output": str(args.output)}, indent=2))


def compose_manifest(args: argparse.Namespace) -> None:
    quotas: list[tuple[str, int]] = []
    for item in args.quota or []:
        negative_type, raw_count = item.split("=", 1)
        quotas.append((negative_type, int(raw_count)))
    if not quotas and not args.one_per_image_priority and not args.balanced_one_per_image:
        raise ValueError("Provide --quota, --one-per-image-priority, or --balanced-one-per-image")
    by_type_image: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for path in args.input:
        for row in read_jsonl(path):
            if not row.get("accepted", True):
                continue
            by_type_image[str(row.get("negative_type", "unknown"))][int(row["image_id"])].append(row)
    image_ids = sorted({image_id for mapping in by_type_image.values() for image_id in mapping})
    output_rows: list[dict[str, Any]] = []
    coverage: dict[str, int] = defaultdict(int)
    complete = 0
    for image_id in image_ids:
        chosen: list[dict[str, Any]] = []
        is_complete = True
        priority = args.one_per_image_priority
        if args.balanced_one_per_image:
            types = list(args.balanced_one_per_image)
            offset = image_id % len(types)
            rotated = types[offset:] + types[:offset]
            rotation_rank = {negative_type: index for index, negative_type in enumerate(rotated)}
            # Greedily use the currently least represented available type.
            # The cyclic rank is only a deterministic tie breaker. This keeps
            # a fixed validation benchmark close to globally balanced even
            # when some counterfactual/crop candidates do not exist.
            priority = sorted(
                types,
                key=lambda negative_type: (
                    coverage[negative_type],
                    rotation_rank[negative_type],
                ),
            )
        if priority:
            is_complete = False
            for negative_type in priority:
                candidates = sorted(
                    by_type_image.get(negative_type, {}).get(image_id, []),
                    key=lambda row: (
                        -float(row.get("hardness_score", row.get("lora_max_score", 0.0))),
                        -float(row.get("absent_probability", 0.0)),
                        str(row["negative_prompt"]),
                    ),
                )
                if candidates:
                    chosen.append(candidates[0])
                    coverage[negative_type] += 1
                    is_complete = True
                    break
            if not is_complete and args.fallback_type:
                candidates = sorted(
                    by_type_image.get(args.fallback_type, {}).get(image_id, []),
                    key=lambda row: (
                        -float(row.get("hardness_score", row.get("lora_max_score", 0.0))),
                        -float(row.get("absent_probability", 0.0)),
                        str(row["negative_prompt"]),
                    ),
                )
                if candidates:
                    chosen.append(candidates[0])
                    coverage[args.fallback_type] += 1
                    is_complete = True
        for negative_type, count in quotas:
            candidates = sorted(
                by_type_image.get(negative_type, {}).get(image_id, []),
                key=lambda row: (
                    -float(row.get("hardness_score", row.get("lora_max_score", 0.0))),
                    -float(row.get("absent_probability", 0.0)),
                    str(row["negative_prompt"]),
                ),
            )
            selected = candidates[:count]
            chosen.extend(selected)
            coverage[negative_type] += int(len(selected) == count)
            is_complete &= len(selected) == count
        if is_complete:
            complete += 1
        output_rows.extend(chosen)
    write_jsonl(args.output, output_rows)
    print(
        json.dumps(
            {
                "images_seen": len(image_ids),
                "images_complete": complete,
                "rows": len(output_rows),
                "quota_coverage": dict(coverage),
                "output": str(args.output),
            },
            indent=2,
        )
    )


def audit_rows(args: argparse.Namespace) -> None:
    rows = [row for row in read_jsonl(args.input) if row.get("accepted", True)]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.count]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=18)
    for sheet_index in range(0, len(rows), 10):
        sheet_rows = rows[sheet_index : sheet_index + 10]
        canvas = Image.new("RGB", (1800, 5 * 420), "white")
        draw = ImageDraw.Draw(canvas)
        for local, row in enumerate(sheet_rows):
            col, line = local % 2, local // 2
            x0, y0 = col * 900, line * 420
            image = Image.open(row["image_path"]).convert("RGB")
            if row.get("crop_xyxy"):
                image = image.crop(tuple(map(int, row["crop_xyxy"])))
            image.thumbnail((860, 330))
            canvas.paste(image, (x0 + 20, y0 + 20))
            label = f"id={row['image_id']} type={row.get('negative_type')} absent={row.get('absent_probability', 'n/a')}\n{row['negative_prompt']}"
            draw.multiline_text((x0 + 20, y0 + 355), label[:240], fill="black", font=font, spacing=4)
        canvas.save(args.output_dir / f"audit_{sheet_index // 10:02d}.jpg", quality=92)
    print(json.dumps({"audited": len(rows), "sheets": (len(rows) + 9) // 10, "output_dir": str(args.output_dir)}, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        generate_candidates(args)
    elif args.command == "rewrite-counterfactual":
        rewrite_counterfactuals(args)
    elif args.command == "verify":
        verify_candidates(args)
    elif args.command == "score":
        score_candidates(args)
    elif args.command == "active":
        build_active(args)
    elif args.command == "compose":
        compose_manifest(args)
    elif args.command == "audit":
        audit_rows(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
