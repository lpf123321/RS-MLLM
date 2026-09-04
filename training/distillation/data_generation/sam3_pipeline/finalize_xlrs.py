#!/usr/bin/env python3
"""Finalize XLRS grounding after fixed-box language review and template alignment."""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from .common import clip_box, object_classes, safe_id, write_jsonl
except ImportError:  # direct script execution
    from common import clip_box, object_classes, safe_id, write_jsonl


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def template(question: str) -> str:
    marker = "Description:"
    index = question.find(marker)
    if index < 0:
        raise ValueError("official XLRS question lacks Description: marker")
    return question[: index + len(marker)]


def validate_description(description: str, candidate_id: str) -> None:
    words = re.findall(r"[A-Za-z0-9]+", description)
    if not 35 <= len(words) <= 100:
        raise SystemExit(
            f"XLRS inner description must contain 35-100 words: {candidate_id} ({len(words)})"
        )
    sentences = [part for part in re.split(r"[.!?]+", description) if part.strip()]
    if len(sentences) < 2:
        raise SystemExit(f"XLRS inner description needs at least two sentences: {candidate_id}")
    relation_groups = {
        "vertical": r"\b(?:above|below|upper|lower|top|bottom)\b",
        "horizontal": r"\b(?:left|right|west|east)\b",
        "adjacency": r"\b(?:beside|adjacent|near|next to|between|surrounded|immediately)\b",
        "containment": r"\b(?:inside|within|along|through|edge|corner|center|centre)\b",
    }
    matched = sum(bool(re.search(pattern, description, flags=re.IGNORECASE)) for pattern in relation_groups.values())
    if matched < 2:
        raise SystemExit(f"XLRS description lacks two independent relation types: {candidate_id}")


GALLERY_MAX_EDGE = 1280


def open_preview(image_path: Path) -> tuple[Image.Image, tuple[int, int]]:
    """Decode a bounded preview without retaining a full 10k XLRS raster."""
    with Image.open(image_path) as source:
        original_size = source.size
        source.draft("RGB", (GALLERY_MAX_EDGE, GALLERY_MAX_EDGE))
        image = source.convert("RGB")
        image.thumbnail((GALLERY_MAX_EDGE, GALLERY_MAX_EDGE), Image.Resampling.LANCZOS)
        return image.copy(), original_size


def save_preview(image_path: Path, output: Path) -> None:
    image, _ = open_preview(image_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=82, optimize=True)
    image.close()


def render_official(image_path: Path, box: list[float], output: Path, label: str) -> None:
    image, (original_width, original_height) = open_preview(image_path)
    scale_x = image.width / original_width
    scale_y = image.height / original_height
    x1, y1, x2, y2 = clip_box(box, original_width, original_height)
    preview_box = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
    draw = ImageDraw.Draw(image)
    draw.rectangle(tuple(preview_box), outline=(40, 255, 80), width=max(3, image.width // 500))
    draw.text((max(2, preview_box[0]), max(2, preview_box[1] - 18)), label, fill=(40, 255, 80))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=82, optimize=True)
    image.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--official-records", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--question-decisions", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--gallery-only-accepted",
        action="store_true",
        help="Render Gallery cards only for images with accepted synthetic rows.",
    )
    parser.add_argument(
        "--gallery-page-size",
        type=int,
        default=0,
        help="Split the Gallery into pages of this many images; 0 writes one page.",
    )
    args = parser.parse_args()

    selection = read_jsonl(Path(args.selection))
    official = read_jsonl(Path(args.official_records))
    candidates = read_jsonl(Path(args.candidates))
    decisions = read_jsonl(Path(args.question_decisions))
    official_by_image = defaultdict(list)
    for row in official: official_by_image[row["image"]].append(row)
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    decision_by_id = {row["candidate_id"]: row for row in decisions}
    if len(candidate_by_id) != len(candidates) or set(candidate_by_id) != set(decision_by_id):
        raise SystemExit("question decisions must cover every candidate exactly once")

    out = Path(args.output_dir)
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    accepted, rejected, audits = [], [], []
    seen = defaultdict(set)
    allowed = object_classes("xlrs")
    for candidate in candidates:
        decision = decision_by_id[candidate["candidate_id"]]
        if decision.get("status") == "rejected":
            rejected.append({"candidate_id": candidate["candidate_id"], "reason": decision.get("reason", "ambiguous")})
            continue
        if decision.get("status") != "accepted" or candidate["class"] not in allowed:
            raise SystemExit(f"invalid decision/class: {candidate['candidate_id']}")
        description = str(decision.get("description", "")).strip()
        lowered_description = description.lower()
        forbidden_meta = ("description:", "bbox", "mask", "highlighted", "overlay", "sam3", "bounding box")
        if not description or any(token in lowered_description for token in forbidden_meta):
            rejected.append({
                "candidate_id": candidate["candidate_id"],
                "reason": "language_visualization_leak",
            })
            continue
        if re.search(r"\bred (?:area|surface|field|region)\b", lowered_description):
            rejected.append({
                "candidate_id": candidate["candidate_id"],
                "reason": "possible_red_mask_color_leak",
            })
            continue
        validate_description(description, candidate["candidate_id"])
        official_template = template(official_by_image[candidate["image"]][0]["question"])
        final_question = f"{official_template} {description}"
        if not final_question.startswith(official_template) or final_question[len(official_template):].strip() != description:
            raise SystemExit("inner description changed during template alignment")
        normalized = re.sub(r"\s+", " ", final_question.strip().lower())
        if normalized in seen[candidate["image"]]:
            rejected.append({
                "candidate_id": candidate["candidate_id"],
                "reason": "same_image_duplicate_question",
            })
            continue
        seen[candidate["image"]].add(normalized)
        width, height = map(int, candidate["image_size"])
        x1, y1, x2, y2 = clip_box(candidate["bbox_xyxy_px"], width, height)
        answer = f"[{round(x1)}, {round(y1)}, {round(x2)}, {round(y2)}]"
        row = {
            "id": f"sam3_xlrs_aug_{safe_id(candidate['candidate_id'])}",
            "image": str((Path(args.image_root) / candidate["image"]).resolve()),
            "messages": [{"role": "user", "content": f"<image>\n{final_question}"}, {"role": "assistant", "content": answer}],
            "class": candidate["class"],
            "candidate_id": candidate["candidate_id"],
        }
        accepted.append(row)
        audits.append({
            "candidate_id": candidate["candidate_id"], "image": candidate["image"], "class": candidate["class"],
            "description_before_alignment": description, "description_after_alignment": final_question[len(official_template):].strip(),
            "description_unchanged": True, "bbox_xyxy_px": candidate["bbox_xyxy_px"], "answer": answer,
            "official_template": official_template, "template_from_same_train_image": True,
            "mask_path": candidate["mask_path"], "overlay_path": candidate["overlay_path"],
            "review_reason": decision.get("reason", ""),
        })
    write_jsonl(out / "synthetic_grounding.jsonl", accepted)
    write_jsonl(out / "synthetic_audit.jsonl", audits)
    write_jsonl(out / "rejected.jsonl", rejected)

    image_names = [row["image"] for row in selection]
    gallery_image_names = image_names
    if args.gallery_only_accepted:
        accepted_images = {row["image"] for row in audits}
        gallery_image_names = [image for image in image_names if image in accepted_images]
    for image in gallery_image_names:
        save_preview(Path(args.image_root) / image, assets / image)
    audit_by_image = defaultdict(list)
    for audit in audits: audit_by_image[audit["image"]].append(audit)
    cards = []
    for image in gallery_image_names:
        official_html = []
        for ordinal, row in enumerate(official_by_image[image], 1):
            name = f"{safe_id(Path(image).stem)}_official_{ordinal:02d}.jpg"
            render_official(Path(args.image_root) / image, row["bbox_xyxy_px"], assets / name, f"official {ordinal}")
            official_html.append(f"<div><img src='assets/{html.escape(name)}'><p>{html.escape(row['question'])}</p><code>{html.escape(str(row['bbox_xyxy_px']))}</code></div>")
        synthetic_html = []
        for audit in audit_by_image[image]:
            name = f"{safe_id(audit['candidate_id'])}.jpg"
            save_preview(Path(audit["overlay_path"]), assets / name)
            final = next(row for row in accepted if row["candidate_id"] == audit["candidate_id"])
            final_question = final["messages"][0]["content"].replace("<image>\n", "")
            synthetic_html.append(f"<div><img src='assets/{html.escape(name)}'><p><b>内部描述：</b>{html.escape(audit['description_before_alignment'])}</p><p><b>同图官方模板：</b>{html.escape(audit['official_template'])}</p><p><b>最终问题：</b>{html.escape(final_question)}</p><code>{html.escape(audit['answer'])}</code></div>")
        cards.append(f"<section><h2>{html.escape(image)}</h2><h3>原图</h3><img class='raw' src='assets/{html.escape(image)}'><h3>官方问题与框</h3><div class='grid'>{''.join(official_html)}</div><h3>新 SAM3 实例与模板对齐问题</h3><div class='grid'>{''.join(synthetic_html) or '<p>无通过候选</p>'}</div></section>")
    header = """<!doctype html><meta charset='utf-8'><title>XLRS SAM3 smoke</title><style>body{font-family:system-ui;background:#151515;color:#eee;margin:24px}section{border:1px solid #555;padding:16px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.grid img,.raw{max-width:100%;max-height:700px;object-fit:contain;background:#222}p{line-height:1.45}code{color:#9fe}a{color:#8cf}</style><h1>XLRS train-only SAM3：外层模板对齐、内部描述不变</h1>"""
    page_size = max(0, args.gallery_page_size)
    gallery_pages = 1
    if page_size and len(cards) > page_size:
        chunks = [cards[index : index + page_size] for index in range(0, len(cards), page_size)]
        gallery_pages = len(chunks)
        links = []
        for ordinal, chunk in enumerate(chunks, 1):
            page_name = f"gallery_{ordinal:03d}.html"
            navigation = "<p>" + " | ".join(
                f"<a href='gallery_{number:03d}.html'>第 {number} 页</a>"
                for number in range(1, len(chunks) + 1)
            ) + "</p>"
            (out / page_name).write_text(header + navigation + "".join(chunk), encoding="utf-8")
            links.append(f"<li><a href='{page_name}'>第 {ordinal} 页（{len(chunk)} 张）</a></li>")
        (out / "gallery.html").write_text(
            header + f"<p>仅展示有新通过样本的图像，共 {len(cards)} 张。</p><ul>{''.join(links)}</ul>",
            encoding="utf-8",
        )
    else:
        (out / "gallery.html").write_text(header + "".join(cards), encoding="utf-8")
    summary = {"dataset_profile": "xlrs", "selected_images": len(image_names), "gallery_images": len(gallery_image_names), "gallery_pages": gallery_pages, "candidates": len(candidates), "accepted": len(accepted), "rejected": len(rejected), "descriptions_changed_by_alignment": 0, "boxes_changed_by_alignment": 0, "test_used_for_generation": False}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
