from __future__ import annotations

import json
import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from codex_runner import enforce_expected_identity, validate_inputs, validate_output  # noqa: E402
from collect_results import collect_general, collect_grounding, native_option_labels  # noqa: E402
import prepare_general_jobs  # noqa: E402
import prepare_grounding_jobs  # noqa: E402
import prepare_class_jobs  # noqa: E402
import collect_class_results  # noqa: E402
from stage_generation_audit import portable_path, sanitize  # noqa: E402
from sam3_pipeline import finalize_vrsbench, finalize_xlrs  # noqa: E402
from sam3_pipeline import normalize_xlrs_fixed5, postprocess_vrsbench, postprocess_xlrs  # noqa: E402


class DataGenerationTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def invoke_main(self, module: object, argv: list[str]) -> str:
        with mock.patch.object(sys, "argv", [getattr(module, "__file__", "tool")] + argv):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                module.main()
        return output.getvalue()

    def test_identity_change_is_rejected(self) -> None:
        job = {"job_id": "sample", "payload": {"expected_identity": {"sample_id": "sample"}}}
        with self.assertRaisesRegex(ValueError, "frozen identity"):
            enforce_expected_identity({"sample_id": "changed"}, job)

    def test_provenance_is_not_part_of_model_schema(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {"value": {"type": "string"}},
        }
        validate_output({"value": "ok", "_provenance": {"job_id": "one"}}, schema)

    def test_general_collector_normalizes_answer_letters(self) -> None:
        job = {"job_id": "s1", "images": ["images/a.jpg"], "payload": {}}
        result = {
            "dataset": "xlrs", "source_id": "x", "sample_id": "s1",
            "canonical_source_key": "xlrs:train:x", "instruction": "Question?\n(A) one\n(B) two",
            "response": "B. (B) two", "style_audit": {},
            "direct_session_review": {
                "mode": "direct_session", "full_image_answerable": True,
                "answer_factually_supported": True, "answer_complete": True,
                "style_matched": True, "no_annotation_leakage": True,
            },
        }
        keep, row = collect_general(job, result)
        self.assertTrue(keep)
        self.assertEqual(row["conversations"][1]["value"], "B")
        self.assertEqual(row["conversations"][0]["value"].count("<image>"), 1)

    def test_general_collector_rejects_style_profile_drift(self) -> None:
        job = {
            "job_id": "s1", "images": ["images/a.jpg"],
            "payload": {"style_profile": {"profile_id": "frozen", "option_count": 4}},
        }
        result = {
            "dataset": "xlrs", "source_id": "x", "sample_id": "s1",
            "canonical_source_key": "xlrs:train:x",
            "instruction": "Question?\n(A) one\n(B) two\n(C) three\n(D) four",
            "response": "B. two", "style_audit": {"profile_id": "changed"},
            "direct_session_review": {
                "mode": "direct_session", "full_image_answerable": True,
                "answer_factually_supported": True, "answer_complete": True,
                "style_matched": True, "no_annotation_leakage": True,
            },
        }
        with self.assertRaisesRegex(ValueError, "style audit changed frozen"):
            collect_general(job, result)

    def test_native_option_layouts_cover_retained_xlrs_styles(self) -> None:
        self.assertEqual(
            native_option_labels("Question? (A) one (B) two (C) three (D) four", 4),
            ["A", "B", "C", "D"],
        )
        self.assertEqual(
            native_option_labels("Question?\nA. (A) one\nB. (B) two\nC. (C) three\nD. (D) four", 4),
            ["A", "B", "C", "D"],
        )
        self.assertEqual(
            native_option_labels("Question?\nA. one\nB. two\nC. three\nD. four", 4),
            ["A", "B", "C", "D"],
        )
        self.assertEqual(
            native_option_labels("Question? A. one B. two C. three D. four", 4),
            ["A", "B", "C", "D"],
        )

    def test_grounding_language_cannot_return_a_box(self) -> None:
        job = {"job_id": "c1", "payload": {"dataset_profile": "vrsbench"}}
        result = {
            "candidate_id": "c1", "accepted": True, "reason": "visible",
            "language_text": "[refer] Where is <p>the highlighted ship</p>?",
            "box_visually_supported": True, "description_unique": True,
            "source_style_matched": True,
        }
        keep, row = collect_grounding(job, result)
        self.assertFalse(keep)
        self.assertEqual(row["reason"], "language_meta_leak")

    def test_rejected_grounding_result_must_not_carry_language(self) -> None:
        job = {"job_id": "c1", "payload": {"dataset_profile": "vrsbench"}}
        result = {
            "candidate_id": "c1", "accepted": False, "reason": "ambiguous",
            "language_text": "[refer] leaked proposal", "box_visually_supported": False,
            "description_unique": False, "source_style_matched": False,
        }
        keep, row = collect_grounding(job, result)
        self.assertFalse(keep)
        self.assertEqual(row["reason"], "rejected_result_has_language")

    def test_vrsbench_collector_adds_required_image_token(self) -> None:
        job = {"job_id": "c1", "payload": {"dataset_profile": "vrsbench"}}
        result = {
            "candidate_id": "c1", "accepted": True, "reason": "unique",
            "language_text": "[refer] where is <p>the left ship</p>?",
            "box_visually_supported": True, "description_unique": True,
            "source_style_matched": True,
        }
        keep, row = collect_grounding(job, result)
        self.assertTrue(keep)
        self.assertTrue(row["question"].startswith("<image>\n[refer]"))

    def test_mme_new_and_xlrs_new_job_fixtures(self) -> None:
        profiles = {
            "mme-new": {
                "profile_id": "mme:mcq:color:en:v1", "source_task_category": "color",
                "task_type": "multiple_choice_vqa", "language": "en", "option_count": 5,
                "prompt_pattern": "What color is {target}?", "answer_form": "letter",
                "answer_pattern": "X", "lexical_register": "remote-sensing MCQ",
            },
            "xlrs-new": {
                "profile_id": "xlrs:mcq:classification:en:v1",
                "source_task_category": "Object properties/Object classification",
                "task_type": "multiple_choice_vqa", "language": "en", "option_count": 4,
                "prompt_pattern": "Identify {target}.", "answer_form": "letter and label",
                "answer_pattern": "X. (X) label", "lexical_register": "concise directive",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "train.jpg"
            image.write_bytes(b"fixture-image")
            for dataset, profile in profiles.items():
                source = root / f"{dataset}.sources.jsonl"
                questions = root / f"{dataset}.questions.jsonl"
                style = root / f"{dataset}.style.json"
                jobs = root / f"{dataset}.jobs.jsonl"
                self.write_jsonl(source, [{
                    "dataset": dataset, "split": "train", "source_id": "source-1",
                    "public_source_key": f"{dataset}:train:source-1",
                    "original_images": [image.name],
                    "source_record": {"split": "train", "hidden_official_answer": "DO_NOT_COPY"},
                }])
                self.write_jsonl(questions, [{
                    "dataset": dataset, "split": "train", "source_id": "source-1",
                    "sample_id": f"{dataset}-1",
                    "source_task_category": profile["source_task_category"],
                }])
                style.write_text(json.dumps({"dataset": dataset, "profiles": [profile]}), encoding="utf-8")
                self.invoke_main(prepare_general_jobs, [
                    "--source-manifest", str(source), "--question-manifest", str(questions),
                    "--style-profile", str(style), "--split", "train", "--output", str(jobs),
                ])
                row = json.loads(jobs.read_text(encoding="utf-8"))
                self.assertEqual(row["payload"]["split"], "train")
                self.assertFalse(row["payload"]["source_annotations_included"])
                self.assertNotIn("DO_NOT_COPY", json.dumps(row))
                self.assertEqual(row["payload"]["style_profile"]["option_count"], profile["option_count"])

    def test_vrsbench_new_and_xlrs_grounding_new_job_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            image_root.mkdir()
            (image_root / "train.jpg").write_bytes(b"raw")
            (root / "mask.png").write_bytes(b"mask")
            (root / "overlay.jpg").write_bytes(b"overlay")
            candidates = root / "candidates.jsonl"
            self.write_jsonl(candidates, [{
                "candidate_id": "candidate-1", "split": "train", "image": "train.jpg",
                "class": "ship", "bbox_xyxy_px": [10, 20, 30, 40],
                "image_size": [100, 100], "score": 0.9,
                "mask_path": "mask.png", "overlay_path": "overlay.jpg",
            }])
            for profile in ("vrsbench", "xlrs"):
                official = root / f"{profile}.official.jsonl"
                jobs = root / f"{profile}.jobs.jsonl"
                question = (
                    "<image>\n[refer] where can I locate <p>the ship</p>?"
                    if profile == "vrsbench" else "Official XLRS Description: style"
                )
                self.write_jsonl(official, [{"split": "train", "image": "train.jpg", "question": question}])
                self.invoke_main(prepare_grounding_jobs, [
                    "--candidates", str(candidates), "--image-root", str(image_root),
                    "--official-train-records", str(official), "--dataset-profile", profile,
                    "--split", "train", "--boxes-are-frozen-postprocessed", "--output", str(jobs),
                ])
                row = json.loads(jobs.read_text(encoding="utf-8"))
                frozen = row["payload"]["frozen_candidate"]
                self.assertEqual(frozen["bbox_xyxy_px"], [10, 20, 30, 40])
                self.assertTrue(row["payload"]["boxes_frozen_before_language"])
                self.assertNotIn("bbox_xyxy_px", json.dumps(json.loads(
                    (MODULE_ROOT / "schemas/grounding_language_review.schema.json").read_text()
                )))
                if profile == "vrsbench":
                    self.assertIn("exactly one <p>...</p>", row["prompt"])
                else:
                    self.assertIn("35-100 words", row["prompt"])

    def test_declared_test_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.jpg"
            image.write_bytes(b"fixture")
            sources = root / "sources.jsonl"
            questions = root / "questions.jsonl"
            profile = root / "style.json"
            self.write_jsonl(sources, [{
                "dataset": "mme-new", "split": "test", "source_id": "s",
                "original_images": [image.name],
            }])
            self.write_jsonl(questions, [{
                "dataset": "mme-new", "split": "train", "source_id": "s", "sample_id": "q",
                "source_task_category": "color",
            }])
            profile.write_text(json.dumps({"profiles": [{
                "profile_id": "p", "source_task_category": "color",
            }]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-train"):
                self.invoke_main(prepare_general_jobs, [
                    "--source-manifest", str(sources), "--question-manifest", str(questions),
                    "--style-profile", str(profile), "--split", "train",
                    "--output", str(root / "jobs.jsonl"),
                ])

    def test_offline_validate_does_not_check_login(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.jpg"
            image.write_bytes(b"fixture")
            jobs = root / "jobs.jsonl"
            self.write_jsonl(jobs, [{"job_id": "one", "prompt": "inspect", "images": [str(image)]}])
            schema = root / "schema.json"
            schema.write_text(json.dumps({"type": "object"}), encoding="utf-8")
            args = argparse.Namespace(
                jobs=jobs, schema=schema, run_root=root / "run", stage="smoke",
                smoke_size=1, limit=None,
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                validate_inputs(args)
            report = json.loads(output.getvalue())
            self.assertTrue(report["valid"])
            self.assertFalse(report["codex_invoked"])
            self.assertFalse(report["login_checked"])

    def test_generation_audit_rewrites_local_paths(self) -> None:
        value = sanitize({
            "mask_path": "/home/person/run/masks/one.png",
            "source": json.dumps({
                "archive": "/users/shared/VRSBench/Images_train.zip",
                "member": "/data/VRSBench/Images_train/P0001.png",
            }),
        })
        encoded = json.dumps(value)
        self.assertNotIn("/home/", encoded)
        self.assertNotIn("/users/", encoded)
        self.assertNotIn("/data/", encoded)
        self.assertEqual(value["mask_path"], "omitted-local-artifact://one.png")
        self.assertIn("official://xiang709/VRSBench@", value["source"])
        self.assertTrue(portable_path("/tmp/x.jpg").endswith("x.jpg"))

    def test_class_screening_jobs_and_collection_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            (images / "train.jpg").write_bytes(b"fixture")
            selection = root / "selection.jsonl"
            self.write_jsonl(selection, [{"image": "train.jpg", "split": "train"}])
            jobs = root / "jobs.jsonl"
            self.invoke_main(prepare_class_jobs, [
                "--selection", str(selection), "--image-root", str(images),
                "--dataset-profile", "vrsbench", "--split", "train", "--output", str(jobs),
            ])
            job = json.loads(jobs.read_text(encoding="utf-8"))
            results = root / "results"
            results.mkdir()
            (results / f"{job['job_id']}.json").write_text(json.dumps({
                "image": "train.jpg", "classes": ["ship"], "reason": "visible vessel",
            }), encoding="utf-8")
            output = root / "classes.jsonl"
            self.invoke_main(collect_class_results, [
                "--jobs", str(jobs), "--results-dir", str(results),
                "--dataset-profile", "vrsbench", "--output", str(output),
            ])
            self.assertEqual(json.loads(output.read_text())["classes"], ["ship"])

    def test_vrsbench_deterministic_postprocess_and_finalize_fixture(self) -> None:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency error is actionable
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (100, 100), "gray").save(images / "train.png")
            mask = root / "mask.png"
            Image.new("L", (100, 100), 255).save(mask)
            train = root / "train.json"
            train.write_text(json.dumps([{
                "id": "official-1", "image": "train.png",
                "conversations": [
                    {"from": "human", "value": "<image>\n[refer] where is <p>the right ship</p>?"},
                    {"from": "gpt", "value": "{<70><70><90><90>}"},
                ],
            }]), encoding="utf-8")
            annotations = root / "annotations"
            annotations.mkdir()
            (annotations / "train.json").write_text(json.dumps({"objects": [{
                "obj_cls": "ship", "obj_coord": [0.7, 0.7, 0.9, 0.9],
            }]}), encoding="utf-8")
            detections = root / "detections.jsonl"
            self.write_jsonl(detections, [
                {"detection_id": "keep", "image": "train.png", "class": "ship", "score": 0.9,
                 "bbox_xyxy_px": [10, 10, 30, 30], "image_size": [100, 100],
                 "mask_path": str(mask), "source": "sam3"},
                {"detection_id": "nms", "image": "train.png", "class": "ship", "score": 0.8,
                 "bbox_xyxy_px": [11, 11, 31, 31], "image_size": [100, 100],
                 "mask_path": str(mask), "source": "sam3"},
                {"detection_id": "official", "image": "train.png", "class": "ship", "score": 0.95,
                 "bbox_xyxy_px": [70, 70, 90, 90], "image_size": [100, 100],
                 "mask_path": str(mask), "source": "sam3"},
            ])
            candidate_root = root / "candidates"
            self.invoke_main(postprocess_vrsbench, [
                "--detections", str(detections), "--dataset-profile", "vrsbench",
                "--train", str(train), "--annotations-root", str(annotations),
                "--image-root", str(images), "--output-dir", str(candidate_root),
            ])
            candidate = json.loads((candidate_root / "candidates.jsonl").read_text())
            self.assertEqual(candidate["candidate_id"], "keep")
            decisions = root / "decisions.jsonl"
            self.write_jsonl(decisions, [{
                "candidate_id": "keep", "status": "accepted",
                "question": "<image>\n[refer] where is <p>the left ship</p>?", "reason": "unique",
            }])
            selection = root / "selection.jsonl"
            self.write_jsonl(selection, [{"image": "train.png"}])
            manifest = root / "sam3_manifest.json"
            manifest.write_text(json.dumps({
                "dataset_profile": "vrsbench", "fixture_only": True,
                "checkpoint": "fixture", "checkpoint_sha256": "fixture",
            }), encoding="utf-8")
            final = root / "final"
            self.invoke_main(finalize_vrsbench, [
                "--train", str(train), "--dataset-profile", "vrsbench",
                "--candidates", str(candidate_root / "candidates.jsonl"),
                "--question-decisions", str(decisions), "--sam3-manifest", str(manifest),
                "--selection", str(selection), "--image-root", str(images),
                "--output-dir", str(final),
            ])
            row = json.loads((final / "synthetic_grounding.jsonl").read_text())
            self.assertEqual(row["conversations"][1]["value"], "{<10><10><30><30>}")
            self.assertFalse(json.loads((final / "summary.json").read_text())["test_split_opened"])

    def test_xlrs_deterministic_postprocess_finalize_and_fixed5_fixture(self) -> None:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (100, 100), "gray").save(images / "train.jpg")
            mask = root / "mask.png"
            Image.new("L", (100, 100), 255).save(mask)
            official = root / "official.jsonl"
            self.write_jsonl(official, [{
                "image": "train.jpg", "inferred_class": "ship", "bbox_xyxy_px": [70, 70, 90, 90],
                "question": (
                    "Given a 100 x 100 pixel satellite image, identify the bounding box of the object "
                    "in the format [xmin, ymin, xmax, ymax]. Description: official target"
                ),
            }])
            detections = root / "detections.jsonl"
            self.write_jsonl(detections, [{
                "detection_id": "keep", "image": "train.jpg", "class": "ship", "score": 0.9,
                "bbox_xyxy_px": [10, 20, 30, 40], "image_size": [100, 100],
                "mask_path": str(mask), "source": "sam3",
            }])
            candidate_root = root / "candidates"
            self.invoke_main(postprocess_xlrs, [
                "--detections", str(detections), "--official-records", str(official),
                "--image-root", str(images), "--output-dir", str(candidate_root),
            ])
            decisions = root / "decisions.jsonl"
            description = (
                "The target ship is in the upper-left portion of the image beside a narrow road. "
                "Below it lies an open gray area, while to the right are several small buildings "
                "and trees that distinguish this vessel from other objects nearby."
            )
            self.write_jsonl(decisions, [{
                "candidate_id": "keep", "status": "accepted", "description": description,
                "reason": "unique by visible relations",
            }])
            selection = root / "selection.jsonl"
            self.write_jsonl(selection, [{"image": "train.jpg"}])
            final = root / "final"
            self.invoke_main(finalize_xlrs, [
                "--selection", str(selection), "--official-records", str(official),
                "--candidates", str(candidate_root / "candidates.jsonl"),
                "--question-decisions", str(decisions), "--image-root", str(images),
                "--output-dir", str(final),
            ])
            fixed = root / "fixed5.json"
            self.invoke_main(normalize_xlrs_fixed5, [
                "--synthetic-jsonl", str(final / "synthetic_grounding.jsonl"),
                "--output", str(fixed),
            ])
            row = json.loads(fixed.read_text())[0]
            self.assertEqual(row["conversations"][1]["value"], "[0.10000,0.20000,0.30000,0.40000]")
            self.assertFalse(json.loads(fixed.with_suffix(".summary.json").read_text())["test_split_opened"])


if __name__ == "__main__":
    unittest.main()
