from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .bbox import area, clip_xyxy, xywh_to_xyxy, xyxy_to_xywh
from .config import workspace
from .io import read_jsonl, write_json, write_jsonl

GENERIC_NEGATIVES = (
    "microscope objective lens",
    "airport baggage carousel",
    "laboratory centrifuge rotor",
    "submarine periscope",
    "snowmobile track",
    "industrial wind turbine nacelle",
    "tennis court umpire chair",
    "railway semaphore signal",
)


def _resolve_opd_artifact(config: dict[str, Any], value: str, field: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        data_root = config.get("data", {}).get("root")
        if not data_root:
            raise ValueError(f"Relative {field} requires data.root: {value}")
        path = Path(data_root).expanduser() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing OPSD {field}: {path}")
    return str(path)


def _validated_negatives(
    config: dict[str, Any], split_name: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    method = config.get("validated_sam_method", {})
    key = "train_negative_manifest" if split_name == "train" else "valid_negative_manifest"
    path = method.get(key)
    if not path:
        return {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("accepted", True) is not True:
            continue
        image_path = row.get("image_path") or row.get("original_path")
        if not image_path:
            raise ValueError(f"Validated negative row has no image path in {path}")
        source_id = str(row.get("source_id", "")).strip()
        positive_prompt = str(row.get("positive_prompt", "")).strip()
        if not source_id or not positive_prompt:
            raise ValueError(f"Validated negative row needs source_id and positive_prompt in {path}")
        key = (source_id, positive_prompt)
        if key in rows:
            raise ValueError(f"Validated negative manifest has duplicate target {key}: {path}")
        rows[key] = row
    return rows


def _build_tree_node_sam_dataset(
    config: dict[str, Any], trajectories: str | Path, stage: str, round_id: int,
) -> dict[str, str]:
    """Build the original bbox-free weak-box dataset from unique final tree nodes.

    This is deliberately separate from the mask-recovery builder below.  A target
    is usable only when stock SAM failed initially and CVSearch ended at exactly
    one valid node.  Dataset GT boxes are never consulted.
    """
    root = workspace(config)
    rows = list(read_jsonl(trajectories))
    split_name = "train" if Path(trajectories).name.startswith("train") else "valid"
    output_root = root / "data" / "sam_coco" / stage / f"round_{round_id:02d}"
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    images, annotations, negative_rows, rejected = [], [], [], []
    validated = _validated_negatives(config, split_name)
    min_area_ratio = float(config.get("filter", {}).get("min_area_ratio", 0.001))
    max_area_ratio = float(config.get("filter", {}).get("max_area_ratio", 0.85))

    for row in rows:
        if int(row.get("search_mode", 3)) != 2 or not row.get("answer_correct"):
            continue
        for target in row.get("target_traces", []):
            target_index = int(target.get("target_index", -1))
            target_text = str(target.get("target_text", "")).strip()
            reason = None
            if target.get("initial_sam_success"):
                reason = "initial_sam_success"
            elif not target_text:
                reason = "empty_target_text"
            final_nodes = list(target.get("final_nodes", []))
            if reason is None and not final_nodes:
                reason = "missing_final_node"
            elif reason is None and len(final_nodes) != 1:
                reason = "ambiguous_multiple_final_nodes"

            bbox_xyxy = None
            ratio = None
            node = final_nodes[0] if len(final_nodes) == 1 else None
            if reason is None:
                try:
                    bbox_xyxy = clip_xyxy(
                        xywh_to_xyxy(node["bbox_xywh"]), int(row["width"]), int(row["height"])
                    )
                    ratio = area(bbox_xyxy) / float(int(row["width"]) * int(row["height"]))
                except (KeyError, TypeError, ValueError) as exc:
                    reason = f"invalid_final_node_bbox:{type(exc).__name__}"
            if reason is None and not min_area_ratio <= float(ratio) <= max_area_ratio:
                reason = "pseudo_bbox_area_out_of_range"
            if reason is not None:
                rejected.append({
                    "source_id": row["sample_id"], "target_index": target_index,
                    "target_text": target_text, "reason": reason,
                })
                continue

            image_id = len(images)
            source = Path(row["image_path"])
            suffix = source.suffix.lower() or ".jpg"
            file_name = f"{image_id:06d}_{row['sample_id']}_target{target_index}{suffix}"
            target_path = split_dir / file_name
            if target_path.exists() or target_path.is_symlink():
                if target_path.resolve() != source.resolve():
                    raise FileExistsError(target_path)
            else:
                target_path.symlink_to(source)
            images.append({
                "id": image_id, "file_name": file_name,
                "width": int(row["width"]), "height": int(row["height"]),
                "source_id": row["sample_id"], "target_index": target_index,
                "pseudo_bbox_source": "tree_final_node",
            })
            bbox = xyxy_to_xywh(bbox_xyxy)
            annotations.append({
                "id": len(annotations), "image_id": image_id, "category_id": 1,
                "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
                "prompt": target_text, "raw_question": row["question"],
                "source_id": row["sample_id"], "target_index": target_index,
                "pseudo_bbox_source": "tree_final_node",
                "pseudo_bbox_node_source": node.get("source"),
                "pseudo_bbox_area_ratio": ratio,
            })
            verified = validated.get((str(row["sample_id"]), target_text))
            if validated and verified is None:
                raise ValueError(
                    f"No validated {split_name} negative for {row['sample_id']} / {target_text!r}"
                )
            if verified is not None:
                absent = float(verified.get("absent_probability", 0.0))
                if verified.get("accepted") is not True or absent < 0.9:
                    raise ValueError(f"Unverified negative for {source.name}: absent_probability={absent}")
                negative_rows.append({
                    **verified, "image_id": image_id, "image_path": str(source),
                    "positive_prompt": target_text, "source_id": row["sample_id"],
                })

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "evidence region"}],
        "info": {
            "bbox_format": "xywh", "round": round_id, "stage": stage,
            "supervision": "cvsearch_tree_final_node_pseudo_bbox",
            "uses_dataset_gt_bbox": False, "uses_tree_node_bbox": True,
        },
    }
    write_json(split_dir / "_annotations.coco.json", coco)
    negative_path = output_root / f"{split_name}_negatives.jsonl"
    write_jsonl(negative_path, negative_rows)
    rejection_path = output_root / f"{split_name}_pseudo_bbox_rejections.jsonl"
    write_jsonl(rejection_path, rejected)
    negative_types: dict[str, int] = {}
    for row in negative_rows:
        kind = row["negative_type"]
        negative_types[kind] = negative_types.get(kind, 0) + 1
    return {
        "data_root": str(output_root), "negative_manifest": str(negative_path),
        "split_dir": str(split_dir), "samples": len(images), "instances": len(annotations),
        "negative_types": negative_types,
        "rejected": len(rejected), "rejection_manifest": str(rejection_path),
    }


def build_sam_dataset(config: dict[str, Any], trajectories: str | Path, stage: str, round_id: int) -> dict[str, str]:
    supervision = config.get("sam_pseudo_supervision", {}).get("source", "tree_final_node")
    if supervision == "tree_final_node":
        return _build_tree_node_sam_dataset(config, trajectories, stage, round_id)
    if supervision != "local_sam_recovery_mask":
        raise ValueError(f"Unsupported sam_pseudo_supervision.source: {supervision!r}")
    root = workspace(config)
    rows = list(read_jsonl(trajectories))
    split_name = "train" if Path(trajectories).name.startswith("train") else "valid"
    output_root = root / "data" / "sam_coco" / stage / f"round_{round_id:02d}"
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    images, annotations, negative_rows, rejected = [], [], [], []
    validated = _validated_negatives(config, split_name)
    filter_config = {
        **config.get("filter", {}),
        **{
            key: value for key, value in config.get("sam_pseudo_supervision", {}).items()
            if key in {"min_area_ratio", "max_area_ratio"}
        },
    }
    min_area_ratio = float(filter_config.get("min_area_ratio", 0.001))
    max_area_ratio = float(filter_config.get("max_area_ratio", 0.85))
    recovery_stages = list(
        config.get("sam_pseudo_supervision", {}).get(
            "recovery_stages", ["local_retry_1", "local_retry_2"]
        )
    )
    if not recovery_stages:
        raise ValueError("sam_pseudo_supervision.recovery_stages must not be empty")
    recovery_order = {stage_name: index for index, stage_name in enumerate(recovery_stages)}
    for row in rows:
        if int(row.get("search_mode", 3)) != 2 or not row.get("answer_correct"):
            continue
        for target in row.get("target_traces", []):
            target_index = int(target.get("target_index", -1))
            target_text = str(target.get("target_text", "")).strip()
            reason = None
            if target.get("initial_sam_success"):
                reason = "initial_sam_success"
            elif not target_text:
                reason = "empty_target_text"
            attempts = sorted(
                (
                    attempt for attempt in target.get("sam_attempts", [])
                    if attempt.get("stage") in recovery_order
                ),
                key=lambda attempt: (
                    recovery_order[attempt.get("stage")],
                    int(attempt.get("call_index", 0)),
                ),
            )
            recovery = next((attempt for attempt in attempts if attempt.get("sam_success")), None)
            if reason is None and recovery is None:
                reason = "missing_local_sam_recovery"
            valid_instances = []
            if reason is None:
                for instance in recovery.get("instances", []):
                    if not instance.get("retained_by_cvsearch"):
                        continue
                    instance_reason = None
                    bbox_xyxy = None
                    ratio = None
                    if not instance.get("valid_mask"):
                        instance_reason = str(instance.get("rejection_reason", "invalid_mask"))
                    elif not instance.get("mask_path") or not Path(instance["mask_path"]).is_file():
                        instance_reason = "missing_mask_artifact"
                    else:
                        try:
                            bbox_xyxy = clip_xyxy(
                                instance["pseudo_bbox_xyxy"], int(row["width"]), int(row["height"])
                            )
                            ratio = area(bbox_xyxy) / float(int(row["width"]) * int(row["height"]))
                        except (KeyError, TypeError, ValueError) as exc:
                            instance_reason = f"invalid_mask_bbox:{type(exc).__name__}"
                    if instance_reason is None and not min_area_ratio <= float(ratio) <= max_area_ratio:
                        instance_reason = "pseudo_bbox_area_out_of_range"
                    if instance_reason is not None:
                        rejected.append({
                            "source_id": row["sample_id"], "target_index": target_index,
                            "target_text": target_text, "stage": recovery.get("stage"),
                            "prediction_index": instance.get("prediction_index"),
                            "reason": instance_reason,
                        })
                        continue
                    valid_instances.append((instance, bbox_xyxy, ratio))
                if not valid_instances:
                    reason = "no_valid_recovered_mask"
            if reason is not None:
                rejected.append({
                    "source_id": row["sample_id"], "target_index": target_index,
                    "target_text": target_text, "reason": reason,
                })
                continue
            image_id = len(images)
            source = Path(row["image_path"])
            suffix = source.suffix.lower() or ".jpg"
            file_name = f"{image_id:06d}_{row['sample_id']}_target{target_index}{suffix}"
            target_path = split_dir / file_name
            if target_path.exists() or target_path.is_symlink():
                if target_path.resolve() != source.resolve():
                    raise FileExistsError(target_path)
            else:
                target_path.symlink_to(source)
            images.append({
                "id": image_id, "file_name": file_name,
                "width": int(row["width"]), "height": int(row["height"]),
                "source_id": row["sample_id"], "target_index": target_index,
                "prompt": target_text,
                "pseudo_bbox_source": "sam_success_mask",
                "sam_recovery_stage": recovery["stage"],
                "num_instances": len(valid_instances),
            })
            for instance, bbox_xyxy, ratio in valid_instances:
                bbox = xyxy_to_xywh(bbox_xyxy)
                annotations.append({
                    "id": len(annotations), "image_id": image_id, "category_id": 1,
                    "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
                    "prompt": target_text, "raw_question": row["question"],
                    "source_id": row["sample_id"], "target_index": target_index,
                    "pseudo_bbox_source": "sam_success_mask",
                    "sam_recovery_stage": recovery["stage"],
                    "sam_call_index": recovery.get("call_index"),
                    "sam_prediction_index": instance.get("prediction_index"),
                    "sam_score": instance.get("score"),
                    "pseudo_mask_path": instance.get("mask_path"),
                    "pseudo_mask_pixel_sha256": instance.get("mask_pixel_sha256"),
                    "pseudo_bbox_area_ratio": ratio,
                })
            verified = validated.get((str(row["sample_id"]), target_text))
            if validated and verified is None:
                raise ValueError(
                    f"No validated {split_name} negative for {row['sample_id']} / {target_text!r}"
                )
            if verified is not None:
                absent = float(verified.get("absent_probability", 0.0))
                if verified.get("accepted") is not True or absent < 0.9:
                    raise ValueError(f"Unverified negative for {source.name}: absent_probability={absent}")
                negative_rows.append({
                    **verified, "image_id": image_id, "image_path": str(source),
                    "positive_prompt": target_text, "source_id": row["sample_id"],
                })
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "evidence region"}],
        "info": {
            "bbox_format": "xywh", "round": round_id, "stage": stage,
            "supervision": "cvsearch_local_sam_recovery_mask_derived_multi_bbox",
            "uses_dataset_gt_bbox": False, "uses_tree_node_bbox": False,
            "uses_mask_loss": False,
            "recovery_stages": recovery_stages,
        },
    }
    write_json(split_dir / "_annotations.coco.json", coco)
    negative_path = output_root / f"{split_name}_negatives.jsonl"
    write_jsonl(negative_path, negative_rows)
    rejection_path = output_root / f"{split_name}_pseudo_bbox_rejections.jsonl"
    write_jsonl(rejection_path, rejected)
    negative_types: dict[str, int] = {}
    for row in negative_rows:
        kind = row["negative_type"]
        negative_types[kind] = negative_types.get(kind, 0) + 1
    return {
        "data_root": str(output_root), "negative_manifest": str(negative_path),
        "split_dir": str(split_dir), "samples": len(images), "instances": len(annotations),
        "negative_types": negative_types,
        "rejected": len(rejected), "rejection_manifest": str(rejection_path),
    }


def build_opd_parquet(config: dict[str, Any], trajectories: str | Path, stage: str, round_id: int) -> dict[str, Any]:
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("The active environment needs the `datasets` package to write parquet") from exc
    output = workspace(config) / "data" / "opd_parquet" / stage / f"round_{round_id:02d}" / "train.parquet"
    records = []
    for row in read_jsonl(trajectories):
        if not row.get("accepted_for_opsd"):
            continue
        image_path = _resolve_opd_artifact(config, row["image_path"], "image_path")
        teacher_image_path = _resolve_opd_artifact(
            config, row["teacher_image_path"], "teacher_image_path"
        )
        records.append({
            "data_source": "cvsearch_self_evolution",
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "images": [{"path": image_path}],
            "bbox_images": [{"path": teacher_image_path}],
            "ability": "visual_question_answering",
            "reward_model": {"style": "none", "ground_truth": row["gt_answer"]},
            "extra_info": {
                "sample_id": row["sample_id"], "round": round_id,
                "answer": row["gt_answer"], "question": row["question"],
                "search_bboxes_xyxy": row["searched_bboxes_xyxy"],
                "targets": row["extracted_targets"], "search_mode": row["search_mode"],
                "num_pop": row["num_pop"],
                "final_mllm_crop": row.get("final_mllm_crop"),
                "final_mllm_pixel_sha256": row.get("final_mllm_pixel_sha256"),
                "optional_gt_bbox_xyxy": row.get("optional_gt_bbox_xyxy"),
                "trace_path": str(Path(trajectories).resolve()),
            },
        })
    if not records:
        raise ValueError("No trajectories passed the OPSD filter")
    accepted = len(records)
    # The distributed smoke test needs at least one sample per GPU. Repeating
    # accepted rows is confined to smoke plumbing and is recorded explicitly;
    # full-stage datasets are never padded or duplicated.
    if stage == "smoke" and len(records) < int(config["opd_training"]["gpus"]):
        originals = list(records)
        target = int(config["opd_training"]["gpus"])
        while len(records) < target:
            repeated = deepcopy(originals[len(records) % len(originals)])
            repeated["extra_info"]["smoke_repeat_index"] = len(records)
            records.append(repeated)
    output.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(records).to_parquet(str(output))
    summary = {
        "output": str(output), "accepted": accepted,
        "training_records": len(records), "source": str(Path(trajectories).resolve()),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    return summary
