from __future__ import annotations

import gc
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from .answers import evaluate_answer, normalize_answer
from .bbox import xywh_to_xyxy
from .config import workspace
from .io import atomic_write_text, read_jsonl, sha256_file, write_jsonl
from .models import add_cvsearch_import_path, build_cvsearch_mllm, build_tracing_sam


TRACE_SCHEMA_VERSION = 5


def pixel_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("utf-8"))
    digest.update(f"{image.width}x{image.height}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _pop_value(value: Any) -> float:
    flat = []
    def visit(item):
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, (int, float)):
            flat.append(float(item))
    visit(value)
    return sum(flat)


def _serialize_sam_calls(
    calls: list[dict[str, Any]], sample_id: str, mask_root: Path, image_size: tuple[int, int]
) -> list[dict[str, Any]]:
    """Persist traced masks and replace private tensors with JSON-safe metadata."""
    original_width, original_height = image_size
    safe_sample_id = str(sample_id).replace("/", "_")
    serialized: list[dict[str, Any]] = []
    for raw_call in calls:
        call = {key: value for key, value in raw_call.items() if key != "predictions"}
        crop = [int(value) for value in call.get("crop_xyxy", [0, 0, original_width, original_height])]
        crop_width = max(0, crop[2] - crop[0])
        crop_height = max(0, crop[3] - crop[1])
        predictions = []
        for raw_prediction in raw_call.get("predictions", []):
            prediction = {
                key: value for key, value in raw_prediction.items() if key != "_mask_tensors"
            }
            masks = raw_prediction.get("_mask_tensors")
            retained = set(int(index) for index in prediction.get("retained_prediction_indices", []))
            instances = []
            count = len(prediction.get("boxes_xyxy", []))
            for instance_index in range(count):
                instance = {
                    "prediction_index": instance_index,
                    "score": float(prediction["joint_scores"][instance_index]),
                    "sam_box_xyxy_local": [
                        float(value) for value in prediction["boxes_xyxy"][instance_index]
                    ],
                    "retained_by_cvsearch": instance_index in retained,
                    "valid_mask": False,
                    "mask_path": None,
                    "mask_pixel_sha256": None,
                    "pseudo_bbox_xyxy": None,
                }
                if masks is None or instance_index >= int(masks.shape[0]):
                    instance["rejection_reason"] = "missing_mask"
                    instances.append(instance)
                    continue
                mask = masks[instance_index]
                while mask.ndim > 2 and mask.shape[0] == 1:
                    mask = mask[0]
                if mask.ndim != 2:
                    instance["rejection_reason"] = "invalid_mask_rank"
                    instances.append(instance)
                    continue
                mask_image = Image.fromarray((mask.numpy().astype("uint8") * 255), mode="L")
                if mask_image.size != (crop_width, crop_height):
                    if crop_width <= 0 or crop_height <= 0:
                        instance["rejection_reason"] = "invalid_crop_extent"
                        instances.append(instance)
                        continue
                    mask_image = mask_image.resize((crop_width, crop_height), Image.Resampling.NEAREST)
                binary = torch.from_numpy(np.array(mask_image, dtype="uint8")) > 0
                coordinates = torch.nonzero(binary, as_tuple=False)
                if coordinates.numel() == 0:
                    instance["rejection_reason"] = "empty_mask"
                    instances.append(instance)
                    continue
                y1 = int(coordinates[:, 0].min().item())
                y2 = int(coordinates[:, 0].max().item()) + 1
                x1 = int(coordinates[:, 1].min().item())
                x2 = int(coordinates[:, 1].max().item()) + 1
                original_bbox = [
                    max(0, min(original_width, crop[0] + x1)),
                    max(0, min(original_height, crop[1] + y1)),
                    max(0, min(original_width, crop[0] + x2)),
                    max(0, min(original_height, crop[1] + y2)),
                ]
                if original_bbox[2] <= original_bbox[0] or original_bbox[3] <= original_bbox[1]:
                    instance["rejection_reason"] = "degenerate_original_bbox"
                    instances.append(instance)
                    continue
                target_index = int(prediction.get("target_index", -1))
                mask_path = (
                    mask_root / safe_sample_id /
                    f"target{target_index}_{call.get('stage', 'unknown')}_call{call.get('call_index', 0)}_"
                    f"instance{instance_index}.png"
                )
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                mask_image.save(mask_path, format="PNG")
                with Image.open(mask_path) as saved:
                    saved.load()
                    mask_sha = pixel_sha256(saved)
                instance.update({
                    "valid_mask": True,
                    "mask_path": str(mask_path),
                    "mask_pixel_sha256": mask_sha,
                    "mask_size": [mask_image.width, mask_image.height],
                    "mask_bbox_xyxy_local": [x1, y1, x2, y2],
                    "pseudo_bbox_xyxy": original_bbox,
                    "mask_area_pixels": int(binary.sum().item()),
                    "pseudo_bbox_source": "sam_success_mask",
                })
                instances.append(instance)
            prediction["instances"] = instances
            prediction["sam_success"] = bool(prediction.get("retained_prediction_indices"))
            predictions.append(prediction)
        call["predictions"] = predictions
        call["postprocessor_returns_input_size_masks"] = True
        serialized.append(call)
    return serialized


class TrajectoryGenerator:
    def __init__(self, config: dict[str, Any], sam_path: str, mllm_path: str, round_id: int):
        add_cvsearch_import_path(config)
        import spacy
        self.config = config
        self.round_id = round_id
        self.sam_path = str(Path(sam_path).resolve())
        self.mllm_path = str(Path(mllm_path).resolve())
        self.mllm = build_cvsearch_mllm(config, self.mllm_path)
        self.sam = build_tracing_sam(config, self.sam_path)
        self.nlp = spacy.load(config["models"]["spacy"])
        with Path(config["cvsearch"]["ic_examples"]).open(encoding="utf-8") as handle:
            self.ic_examples = json.load(handle)

    def generate_one(
        self, record: dict[str, Any], teacher_root: Path, mask_root: Path
    ) -> dict[str, Any]:
        from CVSearch import get_cvsearch_response
        annotation = {
            "input_image": record["image_path"],
            "question": record["question"],
            "options": "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", record["options"])),
            "answer_type": "option_single",
        }
        self.sam.reset_trace()
        cv_cfg = self.config["cvsearch"]
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                response, artifacts = get_cvsearch_response(
                    sam_model=self.sam,
                    zoom_model=self.mllm,
                    nlp_model=self.nlp,
                    annotation=annotation,
                    ic_examples=self.ic_examples,
                    decomposed_question_template="What is the appearance of the {}?",
                    answering_confidence_threshold_upper=float(cv_cfg["answering_confidence_threshold_upper"]),
                    answering_confidence_threshold_lower=float(cv_cfg["answering_confidence_threshold_lower"]),
                    fast_threshold=float(cv_cfg["fast_threshold"]),
                    pop_limit=lambda depth: depth * int(cv_cfg["pop_multiplier"]),
                    threshold_descrease=list(cv_cfg["threshold_descrease"]),
                    image_folder=None,
                    return_artifacts=True,
                    fine_sam_policy=str(cv_cfg.get("fine_sam_policy", "off")),
                )
        finally:
            atomic_write_text(teacher_root / "_search_logs" / f"{record['sample_id']}.log", captured.getvalue())
        prediction = normalize_answer(response)
        correct = evaluate_answer(response, record["gt_answer"])
        boxes_xywh = annotation.get("searched_bbox", [])
        search_mode = int(annotation.get("search_mode", 3))
        if bool(cv_cfg.get("force_visual_search", False)) and search_mode == 0:
            raise AssertionError(
                f"Quick Answer was not disabled for {record['sample_id']}; "
                f"lower={cv_cfg['answering_confidence_threshold_lower']} fast={cv_cfg['fast_threshold']}"
            )
        teacher_path = teacher_root / f"{record['sample_id']}.png"
        reject_reasons = []
        if not correct:
            reject_reasons.append("cvsearch_answer_incorrect")
        if search_mode not in {1, 2}:
            reject_reasons.append(f"search_mode_{search_mode}_not_trainable")
        teacher = artifacts.get("final_mllm_image")
        teacher_pixel_sha = None
        if not isinstance(teacher, Image.Image) or teacher.width <= 0 or teacher.height <= 0:
            reject_reasons.append("invalid_final_mllm_visual_input")
        else:
            teacher_root.mkdir(parents=True, exist_ok=True)
            teacher.save(teacher_path, format="PNG")
            teacher_pixel_sha = pixel_sha256(teacher)
            with Image.open(teacher_path) as saved:
                saved.load()
                if pixel_sha256(saved) != teacher_pixel_sha:
                    raise AssertionError(f"Teacher pixel mismatch after PNG save: {teacher_path}")
        targets = annotation.get("targets") or []
        sam_calls = _serialize_sam_calls(
            self.sam.calls,
            record["sample_id"],
            mask_root,
            (int(record["width"]), int(record["height"])),
        )
        target_traces = annotation.get("target_traces") or []
        initial_predictions = sam_calls[0].get("predictions", []) if sam_calls else []
        for target in target_traces:
            index = int(target["target_index"])
            target["initial_sam_prediction"] = initial_predictions[index] if index < len(initial_predictions) else None
            target["sam_attempts"] = [
                {
                    "call_index": call["call_index"],
                    "stage": call["stage"],
                    "crop_xyxy": call["crop_xyxy"],
                    "image_size": call["image_size"],
                    **prediction,
                }
                for call in sam_calls
                for prediction in call.get("predictions", [])
                if int(prediction.get("target_index", -1)) == index
            ]
        boxes_xyxy = [xywh_to_xyxy(box) for box in boxes_xywh]
        return {
            **record,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "round": self.round_id,
            "image_sha256": sha256_file(record["image_path"]),
            "extracted_targets": targets,
            "root_answer_confidence": annotation.get("root_ans_conf"),
            "sam_predictions": sam_calls,
            "target_traces": target_traces,
            "search_mode": search_mode,
            "searched_bboxes_xywh": boxes_xywh,
            "searched_bboxes_xyxy": boxes_xyxy,
            "num_pop": _pop_value(annotation.get("num_pop", [])),
            "sam_route_flags": annotation.get("sam", []),
            "final_output": response,
            "normalized_prediction": prediction,
            "answer_correct": correct,
            "teacher_image_path": str(teacher_path) if teacher_path.exists() else None,
            "final_mllm_crop": artifacts.get("final_mllm_visual_metadata"),
            "final_mllm_pixel_sha256": teacher_pixel_sha,
            "teacher_prediction": prediction,
            "teacher_gt_logprob": None,
            "student_gt_logprob": None,
            "accepted_for_opsd": not reject_reasons,
            "reject_reasons": sorted(set(reject_reasons)),
            "sam_model_sha256": sha256_file(self.sam_path),
            "mllm_model_sha256": sha256_file(Path(self.mllm_path) / "config.json"),
            "config_sha256": self.config["_config_sha256"],
        }

    def close(self) -> None:
        del self.sam, self.mllm
        gc.collect()
        torch.cuda.empty_cache()


def generate_trajectories(
    config: dict[str, Any], split: str, stage: str, round_id: int, sam_path: str, mllm_path: str,
    limit: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    tag: str | None = None,
) -> Path:
    root = workspace(config)
    split_name = f"smoke_{split}" if stage == "smoke" else split
    records = list(read_jsonl(root / "data" / "splits" / f"{split_name}.jsonl"))
    if limit is not None:
        records = records[:limit]
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"Invalid shard {shard_index}/{num_shards}")
    records = [record for index, record in enumerate(records) if index % num_shards == shard_index]
    suffix = f"_{tag}" if tag else ""
    suffix += f"_limit{limit}" if limit is not None else ""
    if num_shards > 1:
        suffix += f".shard{shard_index:03d}-of-{num_shards:03d}"
    output = root / "data" / "trajectories" / stage / f"round_{round_id:02d}" / f"{split}{suffix}.jsonl"
    # Candidate combinations must never overwrite the baseline artifact for
    # the same sample ID. Shards are disjoint, so the model tag is sufficient.
    teacher_namespace = f"{split}_{tag}" if tag else split
    teacher_root = root / "data" / "teacher_images" / stage / f"round_{round_id:02d}" / teacher_namespace
    mask_root = root / "data" / "sam_masks" / stage / f"round_{round_id:02d}" / teacher_namespace
    existing_rows = [
        row for row in read_jsonl(output) if int(row.get("trace_schema_version", 0)) == TRACE_SCHEMA_VERSION
    ] if output.is_file() else []
    completed_ids = {row["sample_id"] for row in existing_rows}
    rows = list(existing_rows)
    pending = [record for record in records if record["sample_id"] not in completed_ids]
    if not pending:
        return output
    generator = TrajectoryGenerator(config, sam_path, mllm_path, round_id)
    try:
        for pending_index, record in enumerate(pending, 1):
            completed = len(existing_rows) + pending_index
            print(f"trajectory {split} {completed}/{len(records)} sample={record['sample_id']}", flush=True)
            rows.append(generator.generate_one(record, teacher_root, mask_root))
            write_jsonl(output, rows)
    finally:
        generator.close()
    return output


def generate_trajectories_parallel(
    config: dict[str, Any], split: str, stage: str, round_id: int,
    sam_path: str, mllm_path: str, workers: int, tag: str | None = None,
) -> Path:
    root = workspace(config)
    split_name = f"smoke_{split}" if stage == "smoke" else split
    expected = list(read_jsonl(root / "data" / "splits" / f"{split_name}.jsonl"))
    suffix = f"_{tag}" if tag else ""
    main_output = root / "data" / "trajectories" / stage / f"round_{round_id:02d}" / f"{split}{suffix}.jsonl"
    if main_output.is_file():
        existing = list(read_jsonl(main_output))
        if (
            len(existing) == len(expected)
            and {row["sample_id"] for row in existing} == {row["sample_id"] for row in expected}
            and all(int(row.get("trace_schema_version", 0)) == TRACE_SCHEMA_VERSION for row in existing)
        ):
            return main_output
    if workers <= 1:
        return generate_trajectories(config, split, stage, round_id, sam_path, mllm_path, tag=tag)
    code_root = Path(__file__).resolve().parents[2]
    log_root = root / "runs" / stage / f"round_{round_id:02d}" / "trajectory_logs" / f"{split}{'_' + tag if tag else ''}"
    log_root.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    for shard in range(workers):
        log_path = log_root / f"shard_{shard:03d}.log"
        handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard)
        env["PYTHONPATH"] = str(code_root / "src")
        command = [
            sys.executable, "-m", "evolve", "--config", config["_config_path"], "generate",
            "--split", split, "--stage", stage, "--round", str(round_id),
            "--sam-path", sam_path, "--mllm-path", mllm_path,
            "--shard-index", str(shard), "--num-shards", str(workers),
        ]
        if tag:
            command.extend(["--tag", tag])
        processes.append((shard, subprocess.Popen(command, cwd=code_root, env=env, stdout=handle, stderr=subprocess.STDOUT)))
        handles.append(handle)
    failures = []
    try:
        while processes:
            remaining = []
            for shard, process in processes:
                return_code = process.poll()
                if return_code is None:
                    remaining.append((shard, process))
                elif return_code != 0:
                    failures.append((shard, return_code))
                else:
                    print(f"trajectory shard complete {split} {shard + 1}/{workers}", flush=True)
            if failures:
                for _, process in remaining:
                    process.terminate()
                break
            processes = remaining
            if processes:
                import time
                time.sleep(2)
    finally:
        for handle in handles:
            handle.close()
    if failures:
        raise RuntimeError(f"Trajectory shards failed: {failures}; inspect {log_root}")
    row_by_id = {}
    if main_output.is_file():
        row_by_id.update({row["sample_id"]: row for row in read_jsonl(main_output)})
    for shard in range(workers):
        shard_path = main_output.with_name(f"{split}{suffix}.shard{shard:03d}-of-{workers:03d}.jsonl")
        row_by_id.update({row["sample_id"]: row for row in read_jsonl(shard_path)})
    missing = [row["sample_id"] for row in expected if row["sample_id"] not in row_by_id]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} merged trajectories; first={missing[:5]}")
    write_jsonl(main_output, [row_by_id[row["sample_id"]] for row in expected])
    return main_output


def generate_direct_answers(
    config: dict[str, Any], split: str, stage: str, round_id: int, mllm_path: str, tag: str,
) -> Path:
    add_cvsearch_import_path(config)
    from CVSearch import get_direct_response
    root = workspace(config)
    split_name = f"smoke_{split}" if stage == "smoke" else split
    records = list(read_jsonl(root / "data" / "splits" / f"{split_name}.jsonl"))
    output = root / "data" / "trajectories" / stage / f"round_{round_id:02d}" / f"direct_{split}_{tag}.jsonl"
    existing = list(read_jsonl(output)) if output.is_file() else []
    if len(existing) == len(records) and {row["sample_id"] for row in existing} == {row["sample_id"] for row in records}:
        return output
    model = build_cvsearch_mllm(config, mllm_path)
    rows = []
    try:
        for index, record in enumerate(records, 1):
            print(f"direct {tag} {split} {index}/{len(records)}", flush=True)
            annotation = {
                "input_image": record["image_path"], "question": record["question"],
                "options": "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", record["options"])),
                "answer_type": "option_single",
            }
            response = get_direct_response(model, annotation, image_folder=None)
            prediction = normalize_answer(response)
            rows.append({
                **record, "round": round_id, "evaluation_mode": "direct", "model_tag": tag,
                "final_output": response, "normalized_prediction": prediction,
                "answer_correct": evaluate_answer(response, record["gt_answer"]), "num_pop": 0.0,
                "search_mode": -1, "teacher_prediction": None, "accepted_for_opsd": False,
                "accepted_for_sam_correction": False,
            })
            write_jsonl(output, rows)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return output
