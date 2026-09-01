#!/usr/bin/env python3
"""将 datasets_data/ 的 messages 格式清单转换为评测器 Sample schema v1.0.

用法:
  python scripts/build_sample_manifest.py --all                          # 全部数据集
  python scripts/build_sample_manifest.py --dataset vrsbench --subtask vqa   # 单子任务

任务前缀 -> task_type 映射(与评测器/报告一致):
  [VQA] -> open_vqa, [CAP] -> caption, [REF] -> bbox, [CD] -> change_caption,
  [MCQ] -> single_choice(选项 A-D)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None  # 无 PIL 时跳过尺寸探测(width/height=0, 评测器不查尺寸可跑)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "evaluation" / "vllm_eval" / "manifests"

MCQ_RE = re.compile(r"\(([A-D])\)\s")
PREFIX_RE = re.compile(r"^\s*\[(VQA|CAP|REF|CD|MCQ)\]")

# 数据集 -> 源文件名(datasets_data/ 下)
DATASETS = {
    "vrsbench": "vrsbench_eval",  # VRSBench: [VQA]/[CAP]/[REF]
    "mme": "mme_rs",          # MME-RealWorld-RS: [MCQ]
    "xlrs": "xlrs",           # XLRS-Bench-lite: [MCQ]/[CAP]
    "levircc": "levircc_test",# LEVIR-CC: [CD]
    "xlrs_caption": "xlrs_caption_en",   # 平铺格式: image + references + prompt
    "xlrs_grounding": "xlrs_grounding_test",  # 平铺格式: image + question + bbox
}


def parse_prefix(text: str) -> tuple[str, str]:
    """返回 (task_type, 剩余文本)."""
    m = PREFIX_RE.match(text)
    if not m:
        return "open_vqa", text
    tag = m.group(1)
    rest = text[m.end():].strip()
    return {
        "VQA": "open_vqa",
        "CAP": "caption",
        "REF": "bbox",
        "CD": "change_caption",
        "MCQ": "single_choice",
    }[tag], rest


def parse_mcq(text: str) -> tuple[dict[str, str], list[str]]:
    """解析 (A) xxx (B) xxx ... -> (choices, answer_labels)."""
    opts = MCQ_RE.findall(text)
    if not opts:
        return {}, []
    parts = MCQ_RE.split(text)
    choices: dict[str, str] = {}
    for letter, body in zip(opts, parts[1:]):
        body = body.strip()
        if body:
            choices[letter] = body
    return choices, list(choices)


def image_size(path: str) -> tuple[int, int]:
    if Image is None:
        return 0, 0
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return 0, 0


def convert_messages(raw: dict, dataset: str, index: int) -> dict:
    """一条 messages 记录 -> Sample dict (v1.0)."""
    msgs = raw.get("messages", [])
    user = next((m for m in msgs if m.get("role") == "user"), None)
    if not user:
        raise ValueError(f"{dataset} row {index}: no user message")
    content = user.get("content", [])
    images, text_parts = [], []
    for item in content:
        if item.get("type") == "image":
            images.append(item.get("image", ""))
        elif item.get("type") == "text":
            text_parts.append(item.get("text", ""))
    text = "\n".join(text_parts).strip()
    if not text or not images:
        raise ValueError(f"{dataset} row {index}: empty text/images")

    task_type, prompt = parse_prefix(text)
    mcq_choices: dict[str, str] = {}
    if task_type == "single_choice":
        mcq_choices, _ = parse_mcq(prompt)  # 先解析选项(题干+选项完整文本)
        if mcq_choices:
            prompt = prompt.split("\n(")[0]  # 再去掉选项行, 只留题干
    assistant = next((m for m in msgs if m.get("role") == "assistant"), None)
    answer = ""
    if assistant:
        for item in assistant.get("content", []):
            if item.get("type") == "text":
                answer = item.get("text", "").strip()
                break
        else:
            answer = str(assistant.get("content", "")).strip()

    references = raw.get("references") or ([answer] if answer else [])
    subtask = dataset  # 细粒度子任务由 --subtask 过滤时用提示词关键词细分
    image_refs = []
    for i, p in enumerate(images):
        role = "before" if (len(images) == 2 and i == 0 and task_type == "change_caption") else ("after" if len(images) == 2 and i == 1 and task_type == "change_caption" else "none")
        w, h = image_size(p)
        image_refs.append({"path": p, "role": role, "width": w, "height": h, "transform": "none"})

    return {
        "id": f"{dataset}_{index}",
        "dataset": dataset,
        "subtask": subtask,
        "task_type": task_type,
        "prompt": prompt,
        "images": image_refs,
        "references": references,
        "choices": mcq_choices,
        "answer_labels": list(mcq_choices) if task_type == "single_choice" else [],
        "accepted_labels": list(mcq_choices) if task_type == "single_choice" else [],
        "clean_status": "keep",
        "issues": [],
        "metadata": {"source": f"{dataset}_messages"},
        "split": "test",
        "source": {"format": "messages", "index": index},
        "schema_version": "1.0",
    }


def convert_flat_caption(raw: dict, dataset: str, index: int) -> dict:
    """平铺 caption 记录 -> Sample dict (v1.0)."""
    path = raw.get("image", "")
    w, h = image_size(path)
    return {
        "id": str(raw.get("id", f"{dataset}_{index}")),
        "dataset": dataset,
        "subtask": "caption",
        "task_type": "caption",
        "prompt": raw.get("prompt", "Describe the image in detail."),
        "images": [{"path": path, "role": "none", "width": w, "height": h, "transform": "none"}],
        "references": raw.get("references", []) or [],
        "choices": {},
        "answer_labels": [],
        "accepted_labels": [],
        "clean_status": "keep",
        "issues": [],
        "metadata": {"source": f"{dataset}_flat"},
        "split": "test",
        "source": {"format": "flat_caption", "index": index},
        "schema_version": "1.0",
    }


def convert_flat_grounding(raw: dict, dataset: str, index: int) -> dict:
    """平铺 grounding 记录 -> Sample dict (v1.0, bbox)."""
    path = raw.get("image", "")
    w = int(raw.get("image_width", 0) or 0)
    h = int(raw.get("image_height", 0) or 0)
    bbox = raw.get("bbox") or []
    # bbox 写入 references(计分层 _bbox_score 读 references[0]); 格式与 parse_bbox 兼容
    refs = []
    if len(bbox) >= 4:
        refs = ["[{:.6f},{:.6f},{:.6f},{:.6f}]".format(*[float(v) for v in bbox[:4]])]
    return {
        "id": str(raw.get("id", f"{dataset}_{index}")),
        "dataset": dataset,
        "subtask": "grounding",
        "task_type": "bbox",
        "prompt": raw.get("question", ""),
        "images": [{"path": path, "role": "none", "width": w, "height": h, "transform": "none"}],
        "references": refs,
        "choices": {},
        "answer_labels": [],
        "accepted_labels": [],
        "clean_status": "keep",
        "issues": [],
        "metadata": {"source": f"{dataset}_flat", "bbox": bbox, "category": raw.get("category", "")},
        "split": "test",
        "source": {"format": "flat_grounding", "index": index},
        "schema_version": "1.0",
    }


def build(dataset: str, subtask_filter: str | None, limit: int | None,
          images_root: str | None = None) -> Path:
    name = DATASETS[dataset]
    src = REPO_ROOT / "datasets_data" / f"{name}.jsonl"
    if not src.exists():
        raise SystemExit(f"缺失源清单: {src}")
    convert = {
        "xlrs_caption": convert_flat_caption,
        "xlrs_grounding": convert_flat_grounding,
    }.get(dataset, convert_messages)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{name}{'_' + subtask_filter if subtask_filter else ''}.jsonl"
    written = 0
    with src.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            sample = convert(raw, dataset, i)
            if images_root:
                # 路径前缀替换: /users/.../shared_datasets/<X> -> <images_root>/<X>
                marker = "/shared_datasets/"
                for img in sample["images"]:
                    p = img["path"]
                    if marker in p:
                        img["path"] = str(Path(images_root) / p.split(marker, 1)[1])
                    # 路径已本地化, 重新探测真实尺寸(转换时用的是原始路径, 探测不到)
                    w, h = image_size(img["path"])
                    img["width"], img["height"] = w, h
            if subtask_filter:
                # 子任务过滤: vqa/caption/ref/cd/mcq 前缀匹配
                tag_map = {"vqa": "open_vqa", "caption": "caption", "referring": "bbox",
                           "change": "change_caption", "mcq": "single_choice"}
                want = tag_map.get(subtask_filter)
                if want is None or sample["task_type"] != want:
                    continue
            g.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
            if limit and written >= limit:
                break
    print(f"OK: {out} ({written} samples)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="构建全部数据集")
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--subtask", help="vqa|caption|referring|change|mcq")
    ap.add_argument("--limit", type=int, help="每个输出最多样本数(调试用)")
    ap.add_argument("--images-root", help="图片根(替换 /users/.../shared_datasets/ 前缀, 如 assets)")
    args = ap.parse_args()
    if args.all:
        for ds in DATASETS:
            build(ds, None, args.limit, args.images_root)
        return 0
    if not args.dataset:
        ap.error("需 --dataset 或 --all")
    build(args.dataset, args.subtask, args.limit, args.images_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
