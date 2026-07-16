#!/usr/bin/env python3
"""Baseline evaluation: Qwen3-VL-2B zero-shot on remote sensing datasets.

This is intentionally a simple baseline runner: it validates the environment,
runs zero-shot inference, and saves raw predictions plus lightweight accuracy
estimates for short-answer / multi-choice tasks.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image

from evaluate_results import bbox_iou, parse_bbox

Image.MAX_IMAGE_PIXELS = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "Qwen3-VL-2B-Instruct"
DATA_BASE = BASE_DIR / "datasets" / "shared_datasets"
RESULTS_DIR = BASE_DIR / "results"
EXTRACT_DIR = BASE_DIR / "datasets" / "extracted"

JSONL_PATH = Path(os.environ["BASELINE_JSONL"]) if os.environ.get("BASELINE_JSONL") else None
BASELINE_SKIP = int(os.environ.get("BASELINE_SKIP", "0"))


def record_result(results: list, row: dict) -> None:
    results.append(row)
    if JSONL_PATH is not None:
        JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with JSONL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

VRSBENCH_DIR = DATA_BASE / "VRSBench"
LEVIR_ZIP = DATA_BASE / "LEVIR-CC" / "Levir-CC-dataset.zip"
XLRS_DIR = DATA_BASE / "XLRS-Bench-lite"
MME_DIR = DATA_BASE / "MME-RealWorld-RS"


def load_model(device: str = "cuda"):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    print(f"Loading model from {MODEL_PATH}...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    params_b = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Model: {params_b:.2f}B params", flush=True)
    return model, processor


def generate(model, processor, messages, max_new_tokens: int = 256) -> str:
    if hasattr(model, "generate_messages"):
        return model.generate_messages(messages, max_new_tokens=max_new_tokens)
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def resize_for_vlm(image: Image.Image, max_side: int = 2048) -> Image.Image:
    """Downscale very large remote-sensing images before VLM inference."""
    image = image.convert("RGB")
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    image = image.copy()
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def load_cached_resized_image(image_path: Path, cache_dir: Path, max_side: int = 1024) -> Image.Image:
    """Load a huge PNG once, cache a resized JPEG, and reuse it later."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{image_path.stem}_{max_side}.jpg"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")
    image = resize_for_vlm(Image.open(image_path), max_side=max_side)
    image.save(cache_path, quality=92)
    return image


def eval_vrsbench(model, processor, task: str = "all", max_samples: int | None = None):
    results = []
    img_dir = VRSBENCH_DIR / "images" / "val"
    task_files = {
        "caption": VRSBENCH_DIR / "VRSBench_EVAL_Cap.json",
        "vqa": VRSBENCH_DIR / "VRSBench_EVAL_vqa.json",
        "refer": VRSBENCH_DIR / "VRSBench_EVAL_referring.json",
    }
    tasks = list(task_files) if task == "all" else [task]

    for t in tasks:
        print(f"\n{'=' * 60}\n  VRSBench / {t}\n{'=' * 60}", flush=True)
        samples = json.loads(task_files[t].read_text())
        if max_samples:
            samples = samples[:max_samples]
        correct = 0

        for i, sample in enumerate(samples):
            img_path = img_dir / sample["image_id"]
            if not img_path.exists():
                print(f"  [{i}/{len(samples)}] missing image: {img_path}", flush=True)
                continue
            image = Image.open(img_path).convert("RGB")

            if t == "caption":
                prompt = "Describe this remote sensing image in detail."
                max_tokens = 128
            elif t == "vqa":
                prompt = sample["question"] + "\nAnswer with only the answer, no explanation."
                max_tokens = 32
            else:
                prompt = (
                    f"Locate: {sample['question']}. "
                    "Output only one bounding box as [x1,y1,x2,y2], normalized 0-100."
                )
                max_tokens = 64

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt}
            ]}]
            try:
                pred = generate(model, processor, messages, max_new_tokens=max_tokens)
            except Exception as exc:
                print(f"  [{i}] error: {exc}", flush=True)
                pred = ""

            gt = sample["ground_truth"]
            ok = judge_answer(pred, gt, t)
            correct += int(ok)
            record_result(results, {
                "dataset": "vrsbench",
                "task": t,
                "image_id": sample["image_id"],
                "question_id": sample.get("question_id"),
                "question_type": sample.get("type"),
                "question": sample.get("question", prompt),
                "ground_truth": gt,
                "prediction": pred,
                "correct": ok,
            })
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(samples)}] acc={correct / (i + 1):.3f}", flush=True)

        print(f"  FINAL: {correct / len(samples) if samples else 0:.4f} ({correct}/{len(samples)})", flush=True)
    return results


def eval_levir_cc(model, processor, max_samples: int | None = None):
    print(f"\n{'=' * 60}\n  LEVIR-CC\n{'=' * 60}", flush=True)
    levir_extract = EXTRACT_DIR / "levir_cc"
    if not levir_extract.exists():
        print(f"  Extracting {LEVIR_ZIP}...", flush=True)
        levir_extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(LEVIR_ZIP) as zf:
            zf.extractall(levir_extract)

    json_path = next(levir_extract.rglob("LevirCCcaptions.json"), None)
    if json_path is None:
        print("  ERROR: LevirCCcaptions.json not found", flush=True)
        return []

    data = json.loads(json_path.read_text())
    img_base = json_path.parent / "images"
    results = []
    count = 0
    if BASELINE_SKIP:
        print(f"  Skipping first {BASELINE_SKIP} samples", flush=True)


    for entry in data["images"]:
        sentences = entry["sentences"]
        if BASELINE_SKIP and count + len(sentences) <= BASELINE_SKIP:
            count += len(sentences)
            continue

        split = entry.get("split", entry.get("filepath", "train"))
        img_a = img_base / split / "A" / entry["filename"]
        img_b = img_base / split / "B" / entry["filename"]
        if not img_a.exists() or not img_b.exists():
            count += len(sentences)
            continue
        image_a = Image.open(img_a).convert("RGB")
        image_b = Image.open(img_b).convert("RGB")

        for sentence in sentences:
            if BASELINE_SKIP and count < BASELINE_SKIP:
                count += 1
                continue
            prompt = "Describe the changes between these two temporal remote sensing images."
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image_a},
                {"type": "image", "image": image_b},
                {"type": "text", "text": prompt},
            ]}]
            try:
                pred = generate(model, processor, messages, max_new_tokens=128)
            except Exception as exc:
                print(f"  [{count}] error: {exc}", flush=True)
                pred = ""
            record_result(results, {
                "dataset": "levir_cc",
                "task": "change_caption",
                "image_id": entry["filename"],
                "ground_truth": sentence["raw"].strip(),
                "prediction": pred,
                "changeflag": entry.get("changeflag", 1),
            })
            count += 1
            if count % 20 == 0:
                print(f"  [{count}] processed...", flush=True)
            if max_samples and count >= max_samples:
                break
        if max_samples and count >= max_samples:
            break

    print(f"  Total: {count} samples", flush=True)
    return results


def eval_xlrs_lite(model, processor, max_samples: int | None = None):
    print(f"\n{'=' * 60}\n  XLRS-Bench-lite\n{'=' * 60}", flush=True)
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError:
        print("  ERROR: pyarrow not installed", flush=True)
        return []

    results = []
    correct = 0
    count = 0
    for arrow_file in sorted((XLRS_DIR / "train").glob("*.arrow")):
        with pa.memory_map(str(arrow_file)) as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                for row_idx in range(batch.num_rows):
                    row = batch.slice(row_idx, 1).to_pydict()
                    question = row["question"][0]
                    options = row["multi-choice options"][0]
                    answer = str(row["answer"][0]).strip().upper()
                    image_bytes = row["image"][0][0]["bytes"]
                    image = resize_for_vlm(Image.open(io.BytesIO(image_bytes)), max_side=1024)
                    prompt = (
                        f"{question}\n" + "\n".join(options)
                        + "\nAnswer with only the correct option letter."
                    )
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": image}, {"type": "text", "text": prompt}
                    ]}]
                    try:
                        pred = generate(model, processor, messages, max_new_tokens=16)
                    except Exception as exc:
                        print(f"  [{count}] error: {exc}", flush=True)
                        pred = ""
                    ok = extract_choice(pred) == answer
                    correct += int(ok)
                    record_result(results, {
                        "dataset": "xlrs_lite", "task": "vqa", "question": question,
                        "ground_truth": answer, "prediction": pred, "correct": ok,
                    })
                    count += 1
                    if count % 50 == 0:
                        print(f"  [{count}] acc={correct / count:.3f}", flush=True)
                    if max_samples and count >= max_samples:
                        break
                if max_samples and count >= max_samples:
                    break
            if max_samples and count >= max_samples:
                break

    print(f"  FINAL: {correct / count if count else 0:.4f} ({correct}/{count})", flush=True)
    return results


def eval_mme_realworld(model, processor, max_samples: int | None = None):
    print(f"\n{'=' * 60}\n  MME-RealWorld-RS\n{'=' * 60}", flush=True)
    combined = ensure_mme_tar()
    data = json.loads((MME_DIR / "MME_RealWorld.json").read_text())
    samples = [
        s for s in data
        if "remote_sensing" in s.get("Question_id", "").lower()
        or s.get("Subtask") == "Remote Sensing"
    ]
    if max_samples:
        samples = samples[:max_samples]

    results = []
    correct = 0
    for i, sample in enumerate(samples):
        image_path = ensure_mme_image(combined, sample["Image"])
        if image_path is None or not image_path.exists():
            print(f"  [{i}/{len(samples)}] missing image: {sample['Image']}", flush=True)
            continue
        image = load_cached_resized_image(image_path, EXTRACT_DIR / "mme_rs" / "resized_1024", max_side=1024)
        prompt = (
            f"{sample['Text']}\n" + "\n".join(sample["Answer choices"])
            + "\nAnswer with only the correct option letter."
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}
        ]}]
        try:
            pred = generate(model, processor, messages, max_new_tokens=16)
        except Exception as exc:
            print(f"  [{i}] error: {exc}", flush=True)
            pred = ""
        gt = str(sample["Ground truth"]).strip().upper()
        ok = extract_choice(pred) == gt
        correct += int(ok)
        record_result(results, {
            "dataset": "mme_realworld", "task": "vqa",
            "question_id": sample["Question_id"], "question": sample["Text"],
            "ground_truth": gt, "prediction": pred, "correct": ok,
        })
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(samples)}] acc={correct / (i + 1):.3f}", flush=True)

    print(f"  FINAL: {correct / len(results) if results else 0:.4f} ({correct}/{len(results)})", flush=True)
    return results


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.+-]+", " ", text.lower())).strip()


def extract_choice(text: str) -> str | None:
    match = re.search(r"\b([ABCDE])\b", text.strip().upper())
    return match.group(1) if match else None


def judge_answer(pred: str, gt: str, task: str) -> bool:
    if task == "caption":
        return False  # store raw predictions; caption metrics need BLEU/ROUGE/CIDEr later
    if task == "refer":
        pred_bbox = parse_bbox(pred)
        gt_bbox = parse_bbox(gt)
        return bool(pred_bbox and gt_bbox and bbox_iou(pred_bbox, gt_bbox) >= 0.5)
    norm_pred = normalize_text(pred)
    norm_gt = normalize_text(gt)
    if not norm_gt:
        return False
    return norm_pred == norm_gt or norm_gt in norm_pred


def ensure_mme_tar() -> Path:
    mme_extract = EXTRACT_DIR / "mme_rs"
    mme_extract.mkdir(parents=True, exist_ok=True)
    combined = mme_extract / "remote_sensing.tar.gz"
    parts = sorted(MME_DIR.glob("remote_sensing.tar.gz.part_*"))
    if not parts:
        raise FileNotFoundError("MME tar parts not found")

    expected_size = sum(part.stat().st_size for part in parts)
    if combined.exists() and combined.stat().st_size == expected_size:
        return combined
    if combined.exists():
        print(
            f"  Rebuilding incomplete MME tar: {combined.stat().st_size} != {expected_size}",
            flush=True,
        )
        combined.unlink()

    print(f"  Concatenating {len(parts)} MME tar parts...", flush=True)
    with open(combined, "wb") as out:
        for part in parts:
            with open(part, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024 * 16)
                    if not chunk:
                        break
                    out.write(chunk)
    return combined


def ensure_mme_image(combined: Path, image_rel: str) -> Path | None:
    mme_extract = EXTRACT_DIR / "mme_rs"
    candidates = [
        mme_extract / image_rel,
        mme_extract / "remote_sensing" / Path(image_rel).name,
        mme_extract / "images" / image_rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Tar members are known to look like remote_sensing/<file>. Extract only
    # the requested image instead of reading the full 50GB+ archive index.
    members = [image_rel, f"remote_sensing/{Path(image_rel).name}"]
    for member in members:
        proc = subprocess.run(
            ["tar", "xzf", str(combined), "-C", str(mme_extract), member],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode == 0:
            out = mme_extract / member
            if out.exists():
                return out
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["vrsbench", "levir_cc", "xlrs_lite", "mme", "all"], default="vrsbench")
    parser.add_argument("--task", choices=["caption", "vqa", "refer", "all"], default="all")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.device)
    all_results = []
    if args.dataset in ("vrsbench", "all"):
        all_results.extend(eval_vrsbench(model, processor, args.task, args.max_samples))
    if args.dataset in ("levir_cc", "all"):
        all_results.extend(eval_levir_cc(model, processor, args.max_samples))
    if args.dataset in ("xlrs_lite", "all"):
        all_results.extend(eval_xlrs_lite(model, processor, args.max_samples))
    if args.dataset in ("mme", "all"):
        all_results.extend(eval_mme_realworld(model, processor, args.max_samples))

    out = RESULTS_DIR / f"baseline_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))

    print(f"\n{'=' * 60}\n  SUMMARY\n{'=' * 60}", flush=True)
    summary = {}
    for row in all_results:
        key = f"{row['dataset']}/{row['task']}"
        summary.setdefault(key, {"correct": 0, "total": 0})
        summary[key]["total"] += 1
        summary[key]["correct"] += int(bool(row.get("correct")))
    for key, stat in summary.items():
        total = stat["total"]
        acc = stat["correct"] / total if total else 0
        print(f"  {key}: {acc:.4f} ({stat['correct']}/{total})", flush=True)
    print(f"\n  Saved to: {out}", flush=True)


if __name__ == "__main__":
    main()
