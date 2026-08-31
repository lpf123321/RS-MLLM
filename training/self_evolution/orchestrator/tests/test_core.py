from __future__ import annotations

import json
import tempfile
import unittest
import sys
from pathlib import Path

import torch
from PIL import Image

from evolve.answers import build_prompt, evaluate_answer, normalize_answer, normalize_teacher_answer, split_question_options
from evolve.bbox import iou, xywh_to_xyxy, xyxy_to_xywh
from evolve.metrics import paired_trajectory_comparison, sam_trace_metrics, trajectory_metrics
from evolve.promotion import evaluate_candidate
from evolve.teacher import make_montage
from evolve.data import _frozen_split, load_source_records, prepare
from evolve.datasets import build_opd_parquet, build_sam_dataset
from evolve.trajectory import _serialize_sam_calls, pixel_sha256
from evolve.system import train_mllm


SELF_EVOLUTION_ROOT = Path(__file__).resolve().parents[2]
CVSEARCH_ROOT = SELF_EVOLUTION_ROOT / "cvsearch"
SAM_BRIDGE_SCRIPTS = SELF_EVOLUTION_ROOT / "sam_bridge" / "scripts"


class CoreTests(unittest.TestCase):
    def test_controlled_variant_can_reuse_mllm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            parquet = root / "train.parquet"
            parquet.touch()
            config = {
                "workspace": str(root / "workspace"),
                "repositories": {"vision_opd": str(root)},
                "opd_training": {"reuse_models": {"smoke": str(model)}},
            }
            result = train_mllm(config, "smoke", 0, str(model), str(parquet))
            provenance = json.loads(
                (root / "workspace/runs/smoke/round_00/opd/reuse_model.json").read_text()
            )
        self.assertEqual(result["model"], str(model.resolve()))
        self.assertEqual(result["raw_checkpoint"], "reused_controlled_variant")
        self.assertEqual(provenance["stage"], "smoke")

    def test_fine_sam_policies_trace_and_replace(self):
        sys.path[:0] = [str(CVSEARCH_ROOT / "cvsearch"), str(CVSEARCH_ROOT)]
        from CVSearch import _apply_fine_sam_policy
        from models.tree import NodeA, NodeState

        class FakeSAM:
            def __init__(self):
                self.contexts = []

            def set_trace_context(self, **context):
                self.contexts.append(context)

            def batch_inference(self, image, prompts):
                result = {
                    "boxes": torch.tensor([[2.0, 3.0, 12.0, 13.0]]),
                    "scores": torch.tensor([0.95]),
                }
                return {}, {0: result}, [0]

        image = Image.new("RGB", (100, 80), "white")
        fine_node = NodeA(NodeState(image, [10, 20, 40, 30]))
        fine_node.search_source = "fine"
        fine_node.target_index = 2

        trace_sam = FakeSAM()
        traced = _apply_fine_sam_policy(
            trace_sam, image, [fine_node], "car", 2, "fine_probe_1", "trace_only"
        )
        self.assertIs(traced[0], fine_node)
        self.assertEqual(trace_sam.contexts[0]["crop_xyxy"], [10, 20, 50, 50])
        self.assertEqual(trace_sam.contexts[0]["stage"], "fine_probe_1")

        replace_sam = FakeSAM()
        replaced = _apply_fine_sam_policy(
            replace_sam, image, [fine_node], "car", 2, "fine_probe_1", "replace"
        )
        self.assertEqual(replaced[0].state.bbox, [12, 23, 10, 10])
        self.assertEqual(replaced[0].search_source, "fine_sam")
        self.assertEqual(replaced[0].target_index, 2)

        off_sam = FakeSAM()
        untouched = _apply_fine_sam_policy(
            off_sam, image, [fine_node], "car", 2, "fine_probe_1", "off"
        )
        self.assertIs(untouched[0], fine_node)
        self.assertEqual(off_sam.contexts, [])

    def test_forced_visual_search_root_boundaries(self):
        sys.path[:0] = [str(CVSEARCH_ROOT / "cvsearch"), str(CVSEARCH_ROOT)]
        from CVSearch import should_take_quick_answer
        self.assertEqual(
            [should_take_quick_answer(confidence, 0.0, 1.1) for confidence in (-1.0, 0.0, 1.0)],
            [False, False, False],
        )

    def test_bbox_roundtrip_and_iou(self):
        box = [10, 20, 80, 100]
        self.assertEqual(xywh_to_xyxy(xyxy_to_xywh(box)), [float(v) for v in box])
        self.assertEqual(iou(box, box), 1.0)
        self.assertEqual(iou(box, [100, 100, 110, 110]), 0.0)

    def test_question_and_answer_parsing(self):
        problem = """<image>\nWhat color is it?\n\nOnly focus on the objects inside the red bounding box in the image to answer this question.\n\nA. red\nB. blue\nC. green\nD. white\n\nAnswer with the option's letter from the given choices."""
        question, options = split_question_options(problem)
        self.assertEqual(question, "What color is it?")
        self.assertEqual(options, ["red", "blue", "green", "white"])
        self.assertEqual(build_prompt(question, options).count("<image>"), 1)
        self.assertEqual(normalize_answer("The answer is (C)."), "C")
        self.assertIsNone(normalize_answer("unknown"))
        self.assertIsNone(normalize_answer("B. blue"))
        self.assertEqual(normalize_teacher_answer("B. blue"), "B")
        self.assertTrue(evaluate_answer("A", "A"))
        with self.assertRaises(NotImplementedError):
            evaluate_answer("cat", "cat", evaluator="cider")

    def test_montage(self):
        first = Image.new("RGB", (100, 50), "red")
        second = Image.new("RGB", (30, 90), "blue")
        montage = make_montage([first, second], tile_size=64)
        self.assertEqual(montage.size, (128, 64))
        self.assertEqual(make_montage([first]).size, first.size)

    def test_perfect_and_empty_sam_metrics(self):
        base = {
            "sample_id": "x", "search_mode": 2, "answer_correct": True, "target_traces": [{
                "target_index": 0, "target_text": "thing", "initial_sam_success": False,
                "sam_attempts": [{
                    "stage": "local_retry_1", "call_index": 1, "sam_success": True,
                    "instances": [{
                        "retained_by_cvsearch": True, "valid_mask": True,
                        "pseudo_bbox_xyxy": [0, 0, 10, 10],
                    }],
                }],
            }],
        }
        perfect = json.loads(json.dumps(base))
        perfect["target_traces"][0]["initial_sam_prediction"] = {
            "joint_scores": [0.9], "boxes_xyxy": [[0, 0, 10, 10]],
        }
        empty = json.loads(json.dumps(base))
        empty["target_traces"][0]["initial_sam_prediction"] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(perfect) + "\n", encoding="utf-8")
            self.assertEqual(sam_trace_metrics(path)["pseudo_ap50"], 1.0)
            path.write_text(json.dumps(empty) + "\n", encoding="utf-8")
            self.assertEqual(sam_trace_metrics(path)["pseudo_recall_at_0_5"], 0.0)

    def test_promotion(self):
        baseline = {"accuracy": 0.5, "correct": 10, "mean_pop": 1.0, "tree_search_samples": 5}
        candidate = {"accuracy": 0.55, "correct": 11, "mean_pop": 1.1, "tree_search_samples": 6}
        sam = {"negative_fpr_at_0_5": 0.2}
        rules = {"max_direct_accuracy_drop": 0.01}
        self.assertTrue(evaluate_candidate(baseline, candidate, sam, sam, rules)["accepted"])
        equal_better = {"accuracy": 0.5, "correct": 10, "mean_pop": 0.95, "tree_search_samples": 5}
        self.assertTrue(evaluate_candidate(baseline, equal_better, sam, sam, rules)["accepted"])
        equal_more_tree = {**equal_better, "tree_search_samples": 6}
        self.assertFalse(evaluate_candidate(baseline, equal_more_tree, sam, sam, rules)["accepted"])
        high_fpr = {"negative_fpr_at_0_5": 0.21}
        self.assertFalse(evaluate_candidate(baseline, candidate, sam, high_fpr, rules)["accepted"])
        self.assertFalse(evaluate_candidate(baseline, candidate, sam, sam, rules, regression_drop=0.011)["accepted"])

    def test_bbox_free_standard_mcq_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (32, 24), "white").save(root / "image.png")
            source = root / "source.jsonl"
            source.write_text(json.dumps({
                "image_path": "image.png", "question": "What color is it?",
                "options": ["white", "red", "blue", "green"], "answer": "A",
            }) + "\n", encoding="utf-8")
            rows = load_source_records({"data": {"root": str(root), "source_jsonl": str(source)}})
        self.assertIsNone(rows[0]["optional_gt_bbox_xyxy"])
        self.assertNotIn("gt_bbox_xyxy", rows[0])
        self.assertEqual(rows[0]["stratum"], "what color")

    def test_bbox_deleted_prepare_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (32, 24), "white").save(root / "image.png")
            source = root / "source.jsonl"
            source.write_text(json.dumps({
                "image_path": "image.png", "question": "What color is it?",
                "options": ["white", "red", "blue", "green"], "answer": "A",
            }) + "\n", encoding="utf-8")
            config = {
                "workspace": str(root / "workspace"), "seed": 42,
                "data": {
                    "root": str(root), "source_jsonl": str(source),
                    "train_size": 1, "valid_size": 0, "test_size": 0,
                    "smoke_train_size": 1, "smoke_valid_size": 0,
                },
            }
            summary = prepare(config)
            prepared = json.loads((root / "workspace/data/splits/train.jsonl").read_text())
        self.assertEqual(summary["rows_with_optional_gt_bbox"], 0)
        self.assertIsNone(prepared["optional_gt_bbox_xyxy"])

    def test_prepare_accepts_portable_train_valid_bundle_without_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            Image.new("RGB", (32, 24), "white").save(root / "images" / "train.png")
            Image.new("RGB", (40, 30), "blue").save(root / "images" / "valid.png")
            split_root = root / "published_splits"
            split_root.mkdir()
            for split, image, sample_id in (
                ("train", "images/train.png", "train-id"),
                ("valid", "images/valid.png", "valid-id"),
            ):
                row = {
                    "sample_id": sample_id, "source_index": 1 if split == "train" else 2,
                    "image_relative_path": image, "question": "What color is it?",
                    "options": ["white", "blue", "red", "green"],
                    "prompt": "<image>\nWhat color is it?", "gt_answer": "A" if split == "train" else "B",
                    "question_family": "what color", "stratum": "what color",
                }
                (split_root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            config = {
                "workspace": str(root / "workspace"), "seed": 42,
                "data": {
                    "root": str(root), "prepared_split_root": str(split_root),
                    "train_size": 1, "valid_size": 1, "test_size": 0,
                    "smoke_train_size": 1, "smoke_valid_size": 1,
                },
            }
            summary = prepare(config)
            prepared_test = (root / "workspace" / "data" / "splits" / "test.jsonl").read_text()
            prepared_train = json.loads(
                (root / "workspace" / "data" / "splits" / "train.jsonl").read_text()
            )
        self.assertEqual(summary["split_counts"], {"train": 1, "valid": 1, "test": 0})
        self.assertFalse(summary["test_labels_published"])
        self.assertEqual(prepared_train["image_relative_path"], "images/train.png")
        self.assertTrue(Path(prepared_train["image_path"]).is_absolute())
        self.assertEqual(prepared_test, "")

    def test_bbox_free_dual_dataset_builders_and_pixel_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            teacher = root / "teacher.png"
            Image.new("RGB", (100, 80), "white").save(source)
            teacher_image = Image.new("RGB", (40, 30), "blue")
            teacher_image.save(teacher)
            mask_one = root / "mask_one.png"
            mask_two = root / "mask_two.png"
            Image.new("L", (20, 20), 255).save(mask_one)
            Image.new("L", (10, 10), 255).save(mask_two)
            row = {
                "sample_id": "sample", "image_path": str(source), "width": 100, "height": 80,
                "question": "Where is it?", "options": ["a", "b", "c", "d"],
                "prompt": "<image>\nWhere is it?", "gt_answer": "A", "answer_correct": True,
                "search_mode": 2, "accepted_for_opsd": True, "teacher_image_path": str(teacher),
                "final_mllm_pixel_sha256": pixel_sha256(teacher_image), "final_mllm_crop": {"source_crop_xyxy": [1, 2, 41, 32]},
                "searched_bboxes_xyxy": [[1, 2, 21, 22]], "extracted_targets": ["first", "second", "third"],
                "num_pop": 3, "optional_gt_bbox_xyxy": None,
                "target_traces": [
                    {"target_index": 0, "target_text": "first", "initial_sam_success": True,
                     "final_nodes": [{"bbox_xywh": [1, 2, 20, 20], "source": "fast"}]},
                    {"target_index": 1, "target_text": "second", "initial_sam_success": False,
                     "sam_attempts": [{
                         "call_index": 1, "stage": "local_retry_1", "sam_success": True,
                         "instances": [
                             {"prediction_index": 0, "score": 0.9, "retained_by_cvsearch": True,
                              "valid_mask": True, "mask_path": str(mask_one),
                              "mask_pixel_sha256": "one", "pseudo_bbox_xyxy": [10, 10, 30, 30]},
                             {"prediction_index": 1, "score": 0.8, "retained_by_cvsearch": True,
                              "valid_mask": True, "mask_path": str(mask_two),
                              "mask_pixel_sha256": "two", "pseudo_bbox_xyxy": [40, 40, 50, 50]},
                         ],
                     }]},
                    {"target_index": 2, "target_text": "third", "initial_sam_success": False,
                     "sam_attempts": [],
                     "final_nodes": [{"bbox_xywh": [20, 20, 10, 10], "source": "fine_fallback"}]},
                ],
            }
            trajectories = root / "train.jsonl"
            trajectories.write_text(json.dumps(row) + "\n", encoding="utf-8")
            config = {
                "workspace": str(root / "workspace"), "opd_training": {"gpus": 1},
                "sam_pseudo_supervision": {"source": "local_sam_recovery_mask"},
            }
            sam = build_sam_dataset(config, trajectories, "smoke", 0)
            coco = json.loads((Path(sam["split_dir"]) / "_annotations.coco.json").read_text())
            self.assertEqual(sam["samples"], 1)
            self.assertEqual(sam["instances"], 2)
            self.assertFalse(coco["info"]["uses_dataset_gt_bbox"])
            self.assertFalse(coco["info"]["uses_tree_node_bbox"])
            self.assertEqual({ann["prompt"] for ann in coco["annotations"]}, {"second"})
            self.assertEqual({ann["image_id"] for ann in coco["annotations"]}, {0})
            self.assertEqual(
                {ann["pseudo_bbox_source"] for ann in coco["annotations"]}, {"sam_success_mask"}
            )
            opd = build_opd_parquet(config, trajectories, "smoke", 0)
            self.assertEqual(opd["accepted"], 1)
            with Image.open(teacher) as reopened:
                self.assertEqual(pixel_sha256(reopened), row["final_mllm_pixel_sha256"])

    def test_opsd_builder_rebases_published_relative_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "published"
            image = data_root / "vision_opd/images/source.png"
            teacher = data_root / "teacher_crops/teacher.png"
            image.parent.mkdir(parents=True)
            teacher.parent.mkdir(parents=True)
            Image.new("RGB", (16, 16), "white").save(image)
            Image.new("RGB", (8, 8), "blue").save(teacher)
            row = {
                "sample_id": "portable",
                "image_path": "vision_opd/images/source.png",
                "teacher_image_path": "teacher_crops/teacher.png",
                "prompt": "<image>\nWhat is visible?",
                "gt_answer": "A",
                "question": "What is visible?",
                "accepted_for_opsd": True,
                "searched_bboxes_xyxy": [],
                "extracted_targets": [],
                "search_mode": 1,
                "num_pop": 0,
            }
            trajectories = root / "train.jsonl"
            trajectories.write_text(json.dumps(row) + "\n", encoding="utf-8")
            config = {
                "workspace": str(root / "workspace"),
                "data": {"root": str(data_root)},
                "opd_training": {"gpus": 1},
            }
            report = build_opd_parquet(config, trajectories, "full", 0)
            import pyarrow.parquet as pq

            table = pq.read_table(report["output"])
            self.assertEqual(table.column("images").to_pylist()[0][0]["path"], str(image))
            self.assertEqual(
                table.column("bbox_images").to_pylist()[0][0]["path"], str(teacher)
            )

    def test_tree_node_weak_box_builder_is_a_distinct_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (100, 80), "white").save(source)
            row = {
                "sample_id": "tree", "image_path": str(source), "width": 100, "height": 80,
                "question": "Where is it?", "answer_correct": True, "search_mode": 2,
                # This GT box must not influence the weak-box output.
                "optional_gt_bbox_xyxy": [0, 0, 100, 80],
                "target_traces": [
                    {"target_index": 0, "target_text": "usable", "initial_sam_success": False,
                     "final_nodes": [{"bbox_xywh": [10, 20, 30, 20], "source": "fine"}]},
                    {"target_index": 1, "target_text": "already found", "initial_sam_success": True,
                     "final_nodes": [{"bbox_xywh": [1, 1, 5, 5], "source": "fast"}]},
                    {"target_index": 2, "target_text": "missing", "initial_sam_success": False,
                     "final_nodes": []},
                    {"target_index": 3, "target_text": "ambiguous", "initial_sam_success": False,
                     "final_nodes": [
                         {"bbox_xywh": [1, 1, 5, 5], "source": "fine"},
                         {"bbox_xywh": [6, 6, 5, 5], "source": "fine"},
                     ]},
                ],
            }
            trajectories = root / "train.jsonl"
            trajectories.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = build_sam_dataset({
                "workspace": str(root / "workspace"),
                "filter": {"min_area_ratio": 0.001, "max_area_ratio": 0.85},
                "sam_pseudo_supervision": {"source": "tree_final_node"},
            }, trajectories, "full", 0)
            coco = json.loads((Path(result["split_dir"]) / "_annotations.coco.json").read_text())
            rejected = [json.loads(line) for line in Path(result["rejection_manifest"]).read_text().splitlines()]
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["instances"], 1)
        self.assertEqual(coco["annotations"][0]["bbox"], [10.0, 20.0, 30.0, 20.0])
        self.assertEqual(coco["annotations"][0]["pseudo_bbox_node_source"], "fine")
        self.assertTrue(coco["info"]["uses_tree_node_bbox"])
        self.assertFalse(coco["info"]["uses_dataset_gt_bbox"])
        self.assertEqual(
            {row["reason"] for row in rejected},
            {"initial_sam_success", "missing_final_node", "ambiguous_multiple_final_nodes"},
        )

    def test_mode1_produces_no_sam_training_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (32, 32), "white").save(source)
            row = {
                "sample_id": "mode1", "image_path": str(source), "width": 32, "height": 32,
                "question": "What?", "answer_correct": True, "search_mode": 1,
                "target_traces": [{"target_index": 0, "target_text": "thing", "initial_sam_success": True}],
            }
            trajectories = root / "train.jsonl"
            trajectories.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = build_sam_dataset({"workspace": str(root / "workspace")}, trajectories, "smoke", 0)
        self.assertEqual(result["samples"], 0)
        self.assertEqual(result["instances"], 0)

    def test_local_mask_is_mapped_to_original_coordinates(self):
        mask = torch.zeros((1, 20, 20), dtype=torch.uint8)
        mask[0, 2:5, 3:7] = 1
        calls = [{
            "call_index": 1, "stage": "local_retry_1", "crop_xyxy": [10, 20, 30, 40],
            "image_size": [20, 20],
            "predictions": [{
                "prompt": "thing", "target_index": 0,
                "boxes_xyxy": [[3, 2, 7, 5]], "joint_scores": [0.9],
                "retained_prediction_indices": [0], "_mask_tensors": mask,
            }],
        }]
        with tempfile.TemporaryDirectory() as directory:
            serialized = _serialize_sam_calls(calls, "sample", Path(directory), (100, 80))
            instance = serialized[0]["predictions"][0]["instances"][0]
            self.assertTrue(Path(instance["mask_path"]).is_file())
        self.assertEqual(instance["mask_bbox_xyxy_local"], [3, 2, 7, 5])
        self.assertEqual(instance["pseudo_bbox_xyxy"], [13, 22, 17, 25])

    def test_validated_bbox_loader_supports_one_prompt_multiple_objects(self):
        scripts = str(SAM_BRIDGE_SCRIPTS)
        for import_root in (scripts, str(CVSEARCH_ROOT)):
            if import_root not in sys.path:
                sys.path.insert(0, import_root)
        from vopd_bbox_common import VOPDBBoxDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (100, 80), "white").save(root / "image.png")
            (root / "_annotations.coco.json").write_text(json.dumps({
                "images": [{"id": 0, "file_name": "image.png", "width": 100, "height": 80}],
                "annotations": [
                    {"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 10, 20, 20], "prompt": "car"},
                    {"id": 1, "image_id": 0, "category_id": 1, "bbox": [50, 20, 15, 25], "prompt": "car"},
                ],
                "categories": [{"id": 1, "name": "evidence region"}],
            }), encoding="utf-8")
            dataset = VOPDBBoxDataset(root)
            datapoint = dataset[0]
        self.assertEqual(len(datapoint.images[0].objects), 2)
        self.assertEqual(datapoint.find_queries[0].object_ids_output, [0, 1])
        self.assertTrue(datapoint.find_queries[0].is_exhaustive)
        self.assertTrue(all(obj.segment is None for obj in datapoint.images[0].objects))

    def test_frozen_split_is_exact_and_disjoint(self):
        records = [{"source_index": index, "value": index} for index in range(6)]
        payload = {"splits": {"train": [4, 0, 2], "valid": [5], "test": [1, 3]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            split = _frozen_split(records, path)
        self.assertEqual([row["source_index"] for row in split["train"]], [4, 0, 2])
        self.assertEqual({row["split"] for row in split["valid"]}, {"valid"})
        self.assertEqual(sum(map(len, split.values())), 6)

    def test_teacher_prediction_without_logprob_is_valid_metric_input(self):
        row = {
            "sample_id": "x", "answer_correct": True, "normalized_prediction": "A",
            "num_pop": 0, "search_mode": 0, "question_family": "what is",
            "teacher_prediction": "A", "gt_answer": "A",
            "teacher_gt_logprob": None, "student_gt_logprob": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            metrics = trajectory_metrics(path)
        self.assertEqual(metrics["teacher_crop_accuracy"], 1.0)
        self.assertIsNone(metrics["teacher_logprob_gain_mean"])

    def test_paired_trajectory_comparison(self):
        baseline = [
            {"sample_id": "a", "answer_correct": False},
            {"sample_id": "b", "answer_correct": True},
        ]
        candidate = [
            {"sample_id": "a", "answer_correct": True},
            {"sample_id": "b", "answer_correct": True},
        ]
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left.jsonl", Path(directory) / "right.jsonl"
            left.write_text("".join(json.dumps(row) + "\n" for row in baseline), encoding="utf-8")
            right.write_text("".join(json.dumps(row) + "\n" for row in candidate), encoding="utf-8")
            result = paired_trajectory_comparison(left, right)
        self.assertEqual(result["pairs"], 2)
        self.assertEqual(result["accuracy_delta"], 0.5)


if __name__ == "__main__":
    unittest.main()
