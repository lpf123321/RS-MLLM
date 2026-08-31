#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from .common import DATASET_OBJECT_CLASSES, clip_box, image_name, object_classes, parse_native_box, question_answer, read_rows, safe_id, sha256, write_jsonl
except ImportError:  # direct script execution
    from common import DATASET_OBJECT_CLASSES, clip_box, image_name, object_classes, parse_native_box, question_answer, read_rows, safe_id, sha256, write_jsonl


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def native_answer(box: list[float], width: int, height: int) -> str:
    values = [round(box[0] / width * 100), round(box[1] / height * 100), round(box[2] / width * 100), round(box[3] / height * 100)]
    x1, y1, x2, y2 = [max(0, min(100, int(value))) for value in values]
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"SAM3 box degenerates in native 0-100 coordinates: {box}")
    return f"{{<{x1}><{y1}><{x2}><{y2}>}}"


def render_official_overlay(image_path: Path, answer: str, output_path: Path, label: str) -> None:
    normalized = parse_native_box(answer)
    with Image.open(image_path).convert("RGB") as image:
        width, height = image.size
        box = clip_box(
            [normalized[0] * width, normalized[1] * height, normalized[2] * width, normalized[3] * height],
            width,
            height,
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle(tuple(box), outline=(40, 255, 80), width=max(2, width // 160))
        draw.text((max(2, box[0]), max(2, box[1] - 12)), label, fill=(40, 255, 80))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dataset-profile", choices=sorted(DATASET_OBJECT_CLASSES), default="vrsbench")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--question-decisions", required=True)
    parser.add_argument("--sam3-manifest", required=True)
    parser.add_argument("--selection", required=True, help="selected_images.jsonl used for this run")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    allowed_classes = object_classes(args.dataset_profile)

    train, _ = read_rows(args.train)
    candidates = read_jsonl(Path(args.candidates)); decisions = read_jsonl(Path(args.question_decisions))
    selected_rows = read_jsonl(Path(args.selection))
    selected_images = [str(row["image"]) for row in selected_rows]
    if len(selected_images) != len(set(selected_images)):
        raise SystemExit("selection contains duplicate images")
    sam3_manifest = json.loads(Path(args.sam3_manifest).read_text(encoding="utf-8"))
    manifest_profile = sam3_manifest.get("dataset_profile", "vrsbench")
    if manifest_profile != args.dataset_profile:
        raise SystemExit(f"SAM3 manifest profile mismatch: {manifest_profile} != {args.dataset_profile}")
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    decision_by_id = {row["candidate_id"]: row for row in decisions}
    if len(candidate_by_id) != len(candidates) or len(decision_by_id) != len(decisions) or set(candidate_by_id) != set(decision_by_id):
        raise SystemExit("question decisions must cover every candidate exactly once")
    train_images = {image_name(row) for row in train}
    if not set(selected_images) <= train_images:
        raise SystemExit("selection is not a train-only image subset")
    if any(str(row.get("image", "")) not in set(selected_images) for row in candidates):
        raise SystemExit("candidate image is outside the selected dry-run subset")
    official_ids = {str(row.get("id", "")) for row in train}
    official_questions = defaultdict(set); official_rows = defaultdict(list)
    for row in train:
        image = image_name(row); q, _ = question_answer(row)
        official_questions[image].add(q.replace("<image>\n", "", 1).strip()); official_rows[image].append(row)

    out = Path(args.output_dir); image_out = out / "images"; overlay_out = out / "overlays"
    official_overlay_out = out / "official_overlays"
    out.mkdir(parents=True, exist_ok=True)
    native_rows, audits, rejected = [], [], []
    seen_questions = defaultdict(set)
    top_keys = set(train[0])
    for candidate_id, candidate in candidate_by_id.items():
        decision = decision_by_id[candidate_id]
        if decision.get("status") == "rejected":
            rejected.append({"candidate_id": candidate_id, "reason": decision.get("reason", "ambiguous_fixed_instance")})
            continue
        if decision.get("status") != "accepted":
            raise SystemExit(f"invalid question decision status: {candidate_id}")
        image, cls = candidate["image"], candidate["class"]
        question = str(decision.get("question", ""))
        if image not in train_images or cls not in allowed_classes:
            raise SystemExit(f"invalid candidate provenance: {candidate_id}")
        if "[refer]" not in question.lower() or "<p>" not in question or "</p>" not in question:
            raise SystemExit(f"non-native question: {candidate_id}")
        clean_question = question.replace("<image>\n", "", 1).strip()
        if clean_question in official_questions[image] or clean_question in seen_questions[image]:
            raise SystemExit(f"duplicate question: {candidate_id}")
        seen_questions[image].add(clean_question)
        width, height = map(int, candidate["image_size"])
        answer = native_answer(candidate["bbox_xyxy_px"], width, height)
        native_id = f"sam3_aug_{safe_id(Path(image).stem)}_{cls}_{safe_id(candidate_id)[-12:]}"
        if native_id in official_ids:
            raise SystemExit(f"native id collision: {native_id}")
        row = {"id": native_id, "image": image, "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": answer}]}
        if set(row) != top_keys:
            raise SystemExit("source train has unsupported extra top-level fields")
        native_rows.append(row)
        overlay_src = Path(candidate["overlay_path"])
        overlay_dst = overlay_out / f"{safe_id(candidate_id)}.jpg"; overlay_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(overlay_src, overlay_dst)
        audits.append({
            "id": native_id, "candidate_id": candidate_id, "source": "sam3", "class": cls,
            "score": candidate["score"], "bbox_xyxy_px": candidate["bbox_xyxy_px"],
            "mask_path": candidate["mask_path"], "overlay_path": str(overlay_dst.resolve()),
            "max_same_class_official_iou": candidate["max_same_class_official_iou"],
            "question_review_reason": decision.get("reason", ""),
        })

    write_jsonl(out / "synthetic_grounding.jsonl", native_rows)
    write_jsonl(out / "synthetic_audit.jsonl", audits)
    write_jsonl(out / "rejected_question_candidates.jsonl", rejected)
    (out / "synthetic_grounding.json").write_text(json.dumps(native_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for image in selected_images:
        image_dst = image_out / image
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        if not image_dst.exists():
            shutil.copy2(Path(args.image_root) / image, image_dst)

    cards = []
    row_by_id = {row["id"]: row for row in native_rows}
    audits_by_image = defaultdict(list)
    for audit in audits:
        audits_by_image[row_by_id[audit["id"]]["image"]].append(audit)
    for image in selected_images:
        official_cards = []
        for ordinal, official_row in enumerate(official_rows[image], 1):
            official_question, official_answer = question_answer(official_row)
            official_name = f"{safe_id(Path(image).stem)}_official_{ordinal:02d}.jpg"
            official_path = official_overlay_out / official_name
            if not official_path.exists():
                render_official_overlay(
                    Path(args.image_root) / image, official_answer, official_path, f"official {ordinal}"
                )
            official_cards.append(
                "<div class='item official'><img src='official_overlays/{}'><p><b>原问题 {}</b>：{}</p><code>{}</code></div>".format(
                    html.escape(official_name), ordinal,
                    html.escape(official_question.replace("<image>\n", "", 1)),
                    html.escape(official_answer),
                )
            )
        synthetic_cards = []
        for audit in audits_by_image.get(image, []):
            row = row_by_id[audit["id"]]
            question, answer = question_answer(row)
            overlay_rel = Path(audit["overlay_path"]).relative_to(out)
            clean_new_question = question.replace("<image>\n", "", 1)
            synthetic_cards.append(
                f"<div class='item synthetic'><img src='{html.escape(str(overlay_rel))}'><p>SAM3 mask + bbox，score={audit['score']:.3f}</p><p><b>新问题：</b>{html.escape(clean_new_question)}</p><code>{html.escape(answer)}</code></div>"
            )
        synthetic_html = "".join(synthetic_cards) or "<p class='empty'>本图没有通过全部门禁的新实例。</p>"
        cards.append(
            f"<section><h2>{html.escape(image)}</h2>"
            f"<h3>原图</h3><div class='grid'><div class='item'><img src='images/{html.escape(image)}'><p>无标记原图</p></div></div>"
            f"<h3>官方原问题及对应框</h3><div class='grid'>{''.join(official_cards)}</div>"
            f"<h3>新构造问题及 SAM3 框</h3><div class='grid'>{synthetic_html}</div></section>"
        )
    gallery = """<!doctype html><meta charset='utf-8'><title>SAM3 Grounding dry run</title><style>body{font-family:system-ui;background:#171717;color:#eee;margin:24px}section{border:1px solid #555;padding:14px;margin:16px 0}h3{margin-top:18px}.grid{display:flex;gap:18px;flex-wrap:wrap}.item{width:384px}.item img{width:384px;height:384px;object-fit:contain;background:#222}.item p{line-height:1.4;margin:6px 0}.official{border-top:3px solid #28dd65;padding-top:8px}.synthetic{border-top:3px solid #ffd629;padding-top:8px}.empty{color:#bbb;font-style:italic}code{color:#9fe}</style><h1>SAM3 Grounding augmentation</h1><p>绿色框为官方原问题；红色 mask 与黄色框为 SAM3 新候选。所有选中图片均展示；无通过候选的图片会明确标注。所有 overlay 仅用于审核，不进入训练数据。</p>""" + "\n".join(cards)
    (out / "gallery.html").write_text(gallery, encoding="utf-8")
    summary = {
        "dataset_profile": args.dataset_profile, "allowed_classes": list(allowed_classes),
        "train_sha256": sha256(args.train), "test_split_opened": False,
        "sam3_manifest": str(Path(args.sam3_manifest).resolve()), "sam3_checkpoint": sam3_manifest.get("checkpoint"),
        "sam3_checkpoint_sha256": sam3_manifest.get("checkpoint_sha256"), "fixture_only": bool(sam3_manifest.get("fixture_only")),
        "selected_images": len(selected_images), "candidate_records": len(candidates), "accepted_records": len(native_rows), "rejected_for_question_ambiguity": len(rejected),
        "train_modified": False, "test_used_for_generation": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
