#!/usr/bin/env python3
"""将 datasets_data/ 的 messages 格式清单转换为评测器 Sample schema v1.0.

构建是 fail-closed 的: Pillow 缺失、路径解析失败或图片尺寸无效都会
终止构建，并且不会覆盖已有清单。评测器不得接收 0x0 图片元数据。

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
import os
import re
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:
    Image = None
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "evaluation" / "vllm_eval" / "manifests"
XLRS_CAPTION_PROMPT_PATH = REPO_ROOT / "evaluation" / "prompts" / "xlrs_caption_en.txt"

PREFIX_RE = re.compile(r"^\s*\[(VQA|CAP|REF|CD|MCQ)\]")


class ManifestBuildError(RuntimeError):
    """Raised when a manifest cannot be made safe for the evaluator."""


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
    """解析 (A) xxx (B) xxx ... -> (choices, answer_labels).

    逐行显式解析: 每行以 (X) 开头则视为选项, 避免 split/捕获组歧义.
    """
    choices: dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if len(s) >= 4 and s[0] == "(" and s[2] == ")" and s[1] in "ABCD":
            choices[s[1]] = s[4:].strip()
    return choices, list(choices)


def _require_pillow() -> None:
    if Image is None:
        raise ManifestBuildError(
            "Pillow is required to build an evaluation manifest with valid image "
            f"dimensions (Python: {sys.executable}). Run the builder with the "
            "evaluation venv, or install Pillow in the interpreter used to launch "
            "the console."
        ) from PIL_IMPORT_ERROR


def image_size(path: str | Path) -> tuple[int, int]:
    """Read a real image size; never turn a probe failure into ``(0, 0)``."""
    _require_pillow()
    image_path = Path(path)
    if not image_path.is_file():
        raise ManifestBuildError(f"image file does not exist: {image_path}")
    if Image is None:
        raise AssertionError("Pillow availability was not checked")
    try:
        with Image.open(image_path) as im:
            width, height = im.size
    except Exception as exc:
        raise ManifestBuildError(
            f"cannot read image {image_path}: {type(exc).__name__}: {exc}"
        ) from exc
    if width <= 0 or height <= 0:
        raise ManifestBuildError(
            f"image has invalid dimensions {width}x{height}: {image_path}"
        )
    return width, height


def _resolve_image_path(
    raw_path: str,
    *,
    source_dir: Path,
    images_root: str | None,
) -> tuple[str, Path]:
    """Return the stored manifest path and the path used for probing."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ManifestBuildError("image path is empty")

    marker = "/shared_datasets/"
    if images_root and marker in raw_path:
        suffix = raw_path.split(marker, 1)[1]
        stored = Path(images_root) / suffix
        probe = stored if stored.is_absolute() else (OUTPUT_DIR / stored).resolve()
    else:
        raw = Path(raw_path)
        normalized = raw_path.replace("\\", "/")
        dataset_marker = "datasets/shared_datasets/"
        if not raw.is_absolute() and dataset_marker in normalized:
            suffix = normalized.split(dataset_marker, 1)[1]
            probe = (REPO_ROOT / "datasets" / "shared_datasets" / suffix).resolve()
        else:
            probe = raw if raw.is_absolute() else (source_dir / raw).resolve()
        try:
            stored = Path(os.path.relpath(probe, OUTPUT_DIR))
        except ValueError:
            stored = probe

    if not probe.is_file():
        raise ManifestBuildError(
            f"image file does not exist after path resolution: {probe} "
            f"(source={raw_path!r})"
        )
    return stored.as_posix(), probe


def _image_ref(
    raw_path: str,
    *,
    role: str,
    source_dir: Path,
    images_root: str | None,
) -> dict[str, object]:
    stored_path, probe_path = _resolve_image_path(
        raw_path, source_dir=source_dir, images_root=images_root
    )
    width, height = image_size(probe_path)
    return {
        "path": stored_path,
        "role": role,
        "width": width,
        "height": height,
        "transform": "none",
    }


def convert_messages(
    raw: dict,
    dataset: str,
    index: int,
    *,
    source_dir: Path,
    images_root: str | None,
) -> dict:
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

    task_type, _ = parse_prefix(text)   # 判定任务类型
    prompt = text.strip()               # 保留 [VQA] 等前缀(与报告 system prompt 口径一致)
    mcq_choices: dict[str, str] = {}
    if task_type == "single_choice":
        mcq_choices, _ = parse_mcq(prompt)  # 先解析选项(题干+选项完整文本)
        # 保留选项行: 模型需要看到 A/B/C/D 选项才能作答(评测器不加回选项)
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
    # 正确答案字母: 从参考答案提取(格式 "D. (D) xxx" 或 "D" 或 "(D) xxx")
    answer_labels: list[str] = []
    if task_type == "single_choice" and mcq_choices:
        for ref in references:
            m = re.search(r"\(([A-D])\)", ref)
            if m and m.group(1) in mcq_choices:
                answer_labels = [m.group(1)]
                break
        else:
            # 兜底: 答案本身是单个字母
            for ref in references:
                s = ref.strip().upper()
                if s in mcq_choices:
                    answer_labels = [s]
                    break
        if not answer_labels:
            # 回捞: 参考答案文本匹配 choices 文本(如答案 = 选项全文)
            norm = lambda t: re.sub(r"[^a-z0-9]+", "", t.lower())
            for ref in references:
                rn = norm(ref)
                for label, text in mcq_choices.items():
                    if rn and rn == norm(text):
                        answer_labels = [label]
                        break
                if answer_labels:
                    break
    subtask = dataset  # 细粒度子任务由 --subtask 过滤时用提示词关键词细分
    image_refs = []
    for i, p in enumerate(images):
        role = "before" if (len(images) == 2 and i == 0 and task_type == "change_caption") else ("after" if len(images) == 2 and i == 1 and task_type == "change_caption" else "none")
        image_refs.append(
            _image_ref(
                p,
                role=role,
                source_dir=source_dir,
                images_root=images_root,
            )
        )

    return {
        "id": f"{dataset}_{index}",
        "dataset": dataset,
        "subtask": subtask,
        "task_type": task_type,
        "prompt": prompt,
        "images": image_refs,
        "references": references,
        "choices": mcq_choices,
        "answer_labels": answer_labels,
        "accepted_labels": answer_labels,
        "clean_status": "keep",
        "issues": [],
        "metadata": {"source": f"{dataset}_messages"},
        "split": "test",
        "source": {"format": "messages", "index": index},
        "schema_version": "1.0",
    }


def convert_flat_caption(
    raw: dict,
    dataset: str,
    index: int,
    *,
    source_dir: Path,
    images_root: str | None,
) -> dict:
    """平铺 caption 记录 -> Sample dict (v1.0)."""
    path = raw.get("image", "")
    image = _image_ref(
        path,
        role="none",
        source_dir=source_dir,
        images_root=images_root,
    )
    # XLRS caption expert was trained with this long nine-grid instruction in
    # the user turn.  The source eval JSONL only contains a generic prompt, so
    # forwarding it silently changes the task and causes extremely short
    # generations (and a catastrophic BLEU brevity penalty).
    if dataset == "xlrs_caption":
        prompt = XLRS_CAPTION_PROMPT_PATH.read_text(encoding="utf-8").strip()
        prompt_protocol = "xlrs_caption_nine_grid_user_v1"
    else:
        prompt = raw.get("prompt", "Describe the image in detail.")
        prompt_protocol = "source_prompt"
    return {
        "id": str(raw.get("id", f"{dataset}_{index}")),
        "dataset": dataset,
        "subtask": "caption",
        "task_type": "caption",
        "prompt": prompt,
        "images": [image],
        "references": raw.get("references", []) or [],
        "choices": {},
        "answer_labels": [],
        "accepted_labels": [],
        "clean_status": "keep",
        "issues": [],
        "metadata": {
            "source": f"{dataset}_flat",
            "prompt_protocol": prompt_protocol,
        },
        "split": "test",
        "source": {"format": "flat_caption", "index": index},
        "schema_version": "1.0",
    }


def convert_flat_grounding(
    raw: dict,
    dataset: str,
    index: int,
    *,
    source_dir: Path,
    images_root: str | None,
    image_index_by_source: dict[str, int] | None = None,
) -> dict:
    """平铺 grounding 记录 -> Sample dict (v1.0, bbox)."""
    path = raw.get("image", "")
    # The reported Exp4/Exp5 XLRS grounding protocol feeds the 4096px export
    # into Qwen's processor.  Its attempted image_max_pixels assignment did
    # not reach the inner image processor, so the actual run retained the full
    # 4096 input.  The 1024px JPEG cache is a different protocol.
    if dataset == "xlrs_grounding":
        path = path.replace(
            "/images_exported_test/", "/images_exported_test_4096/"
        )
        # The 1024 export retained global/sparse image numbers, whereas the
        # 4096 export was numbered densely by each original image's first
        # occurrence in the test Arrow dataset.  The JSONL preserves that row
        # order and original ``path``, so reconstruct the dense name without
        # importing datasets/pyarrow into the evaluation environment.
        if image_index_by_source is not None:
            source_key = str(raw.get("path", "")).strip()
            if not source_key:
                raise ManifestBuildError(
                    f"{dataset} row {index}: missing original image path"
                )
            if source_key not in image_index_by_source:
                image_index_by_source[source_key] = len(image_index_by_source)
            image_number = image_index_by_source[source_key]
            path = str(Path(path).with_name(f"xlrs_vg_{image_number:05d}.jpg"))
    image = _image_ref(
        path,
        role="none",
        source_dir=source_dir,
        images_root=images_root,
    )
    # The annotations use the declared source coordinate system (normally
    # 4096x4096). Probe the file first, then preserve those coordinates.
    source_width = int(raw.get("image_width", 0) or 0)
    source_height = int(raw.get("image_height", 0) or 0)
    if source_width > 0 and source_height > 0:
        image["width"] = source_width
        image["height"] = source_height
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
        "images": [image],
        "references": refs,
        "choices": {},
        "answer_labels": [],
        "accepted_labels": [],
        "clean_status": "keep",
        "issues": [],
        "metadata": {
            "source": f"{dataset}_flat",
            "bbox": bbox,
            "category": raw.get("category", ""),
            "image_protocol": "xlrs_grounding_4096_export_v1",
        },
        "split": "test",
        "source": {"format": "flat_grounding", "index": index},
        "schema_version": "1.0",
    }


def build(dataset: str, subtask_filter: str | None, limit: int | None,
          images_root: str | None = None) -> Path:
    _require_pillow()
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
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    written = 0
    dropped = 0
    grounding_image_numbers: dict[str, int] = {}
    try:
        with src.open(encoding="utf-8") as f, tmp.open("w", encoding="utf-8") as g:
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                convert_kwargs = {
                    "source_dir": src.parent,
                    "images_root": images_root,
                }
                if dataset == "xlrs_grounding":
                    convert_kwargs["image_index_by_source"] = grounding_image_numbers
                sample = convert(raw, dataset, i, **convert_kwargs)
                # 无正确选项答案的 choice 样本直接丢弃(否则评测器校验抛错杀全量)
                if sample["task_type"] in {"single_choice", "multi_choice"} and not sample["answer_labels"]:
                    dropped += 1
                    continue
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
        if written == 0:
            raise ManifestBuildError(
                f"no valid samples were written for dataset={dataset!r}, "
                f"subtask={subtask_filter!r}"
            )
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"OK: {out} ({written} samples, 丢弃 {dropped})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="构建全部数据集")
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--subtask", help="vqa|caption|referring|change|mcq")
    ap.add_argument("--limit", type=int, help="每个输出最多样本数(调试用)")
    ap.add_argument("--images-root", help="图片根(替换 /users/.../shared_datasets/ 前缀, 如 assets)")
    args = ap.parse_args()
    try:
        if args.all:
            for ds in DATASETS:
                build(ds, None, args.limit, args.images_root)
            return 0
        if not args.dataset:
            ap.error("需 --dataset 或 --all")
        build(args.dataset, args.subtask, args.limit, args.images_root)
        return 0
    except (ManifestBuildError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: manifest build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
