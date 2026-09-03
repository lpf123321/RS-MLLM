from __future__ import annotations

import sys
import tempfile
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from caption_metrics import summarize_reported_caption_metrics
from run_eval import _max_new_tokens
from schema import ImageRef, Sample
from scripts.build_sample_manifest import (
    XLRS_CAPTION_PROMPT_PATH,
    convert_flat_caption,
    convert_flat_grounding,
)
from scripts import fetch_models
from vision_opd_vllm_eval import VLLMBatchAdapter
from vision_opd_eval import intentional_length_constraint
from rsmllm.router_eval import evaluation_runtime_config
from rsmllm import console
from rsmllm.console import _single_task_plan
from rsmllm.eval_reporting import key_metric_lines, print_combined_key_metrics


class _RecordingProcessor:
    def __init__(self) -> None:
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "rendered-user-turn\n"


class ProtocolFixTests(unittest.TestCase):
    def test_explicit_model_fetch_reuses_console_cache_without_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "cache" / "models" / "repo" / "snapshots" / "master"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"weights")
            cache_root = root / "cache"
            with patch.object(
                fetch_models, "get_model", return_value=str(snapshot)
            ) as get_model:
                rc = fetch_models.fetch("expert_general", cache_root, False)

            self.assertEqual(rc, 0)
            get_model.assert_called_once_with(
                "expert_general", cache_dir=str(cache_root)
            )
            self.assertFalse((root / "models" / "expert_general").exists())

    def test_hidden_console_services_remain_registered(self) -> None:
        self.assertEqual(console.HIDDEN_MENU_KEYS, {"4", "5", "6", "7", "8"})
        self.assertTrue(console.HIDDEN_MENU_KEYS <= console.MAIN_MENU.keys())

    def test_console_key_metrics_cover_discrete_bbox_and_caption(self) -> None:
        summary = {
            "prediction_rows": 30,
            "generation_errors": 0,
            "generation_truncations": 1,
            "groups": {
                "demo/vqa": {
                    "accuracy": 0.75,
                    "correct": 3,
                    "eligible_scoreable_samples": 4,
                },
                "demo/grounding": {
                    "bbox_metrics": {
                        "accuracy_at_0_5": 0.32,
                        "accuracy_at_0_7": 0.2,
                        "mean_iou": 0.28,
                        "format_validity": 0.99,
                    }
                },
                "demo/caption": {
                    "caption_metrics": {
                        "percent": {
                            "BLEU-4": 6.12,
                            "METEOR": 30.61,
                            "ROUGE-L": 20.55,
                            "CIDEr-D": 134.61,
                        }
                    }
                },
            },
        }
        rendered = "\n".join(key_metric_lines(summary))
        self.assertIn("Accuracy=75.00% (3/4)", rendered)
        self.assertIn("Acc@0.5=32.00%", rendered)
        self.assertIn("BLEU-4=6.12%", rendered)
        self.assertIn("CIDEr-D=134.61%", rendered)

    def test_combined_metrics_repeat_every_completed_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first_task"
            second = root / "second_task"
            for path, accuracy in ((first, 0.75), (second, 0.5)):
                path.mkdir()
                summary = {
                    "prediction_rows": 4,
                    "generation_errors": 0,
                    "generation_truncations": 0,
                    "groups": {
                        path.name: {
                            "accuracy": accuracy,
                            "correct": round(accuracy * 4),
                            "eligible_scoreable_samples": 4,
                        }
                    },
                }
                (path / "clean_summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )
            output = io.StringIO()
            with redirect_stdout(output):
                print_combined_key_metrics([first, second])

        rendered = output.getvalue()
        self.assertIn("本轮全部已完成任务汇总", rendered)
        self.assertIn("first_task: Accuracy=75.00%", rendered)
        self.assertIn("second_task: Accuracy=50.00%", rendered)

    def test_single_mode_rejects_tasks_owned_by_other_experts(self) -> None:
        self.assertEqual(
            [task[0] for task in _single_task_plan("grounding", "all")],
            ["vrsbench-referring", "xlrs-bench-grounding-en"],
        )
        self.assertEqual(
            _single_task_plan("caption", "xlrs_caption")[0][0],
            "xlrs-bench-caption",
        )
        with self.assertRaisesRegex(ValueError, "不属于 grounding 专家"):
            _single_task_plan("grounding", "xlrs_caption")

    def test_single_grounding_uses_router_task_and_runner(self) -> None:
        manifest = Path("/tmp/xlrs_grounding_test.jsonl")
        output_dir = Path("/tmp/xlrs_grounding_result")
        with (
            patch.object(
                console,
                "_ask",
                side_effect=["single", "grounding", "bf16", "xlrs_grounding"],
            ),
            patch.object(console, "resolve_model", return_value="/models/ground"),
            patch("rsmllm.data.prepare_eval") as prepare_eval,
            patch.object(
                console, "build_subtask_manifest", return_value=manifest
            ) as build_manifest,
            patch.object(
                console, "new_evaluation_output_dir", return_value=output_dir
            ),
            patch.object(console, "run_eval", return_value=0) as routed_eval,
            patch.object(console, "print_combined_key_metrics") as combined,
        ):
            console._cmd_eval()

        prepare_eval.assert_called_once_with("xlrs_grounding")
        build_manifest.assert_called_once_with(
            "xlrs_grounding_test.jsonl", None, "xlrs_bench_grounding_en"
        )
        routed_eval.assert_called_once_with(
            "/models/ground",
            "expert_ground",
            manifest,
            None,
            "bf16",
            output_dir=output_dir,
        )
        combined.assert_called_once_with([output_dir])

    def test_xlrs_manifest_uses_training_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "x.png"
            Image.new("RGB", (16, 12)).save(image_path)
            row = convert_flat_caption(
                {"id": "x", "image": str(image_path), "references": ["reference"]},
                "xlrs_caption",
                1,
                source_dir=Path(tmp),
                images_root=None,
            )
        self.assertEqual(
            row["prompt"], XLRS_CAPTION_PROMPT_PATH.read_text().strip()
        )
        self.assertEqual(
            row["metadata"]["prompt_protocol"],
            "xlrs_caption_nine_grid_user_v1",
        )

    def test_xlrs_caption_decode_budget(self) -> None:
        sample = Sample(
            id="x",
            dataset="xlrs_caption",
            subtask="caption",
            task_type="caption",
            prompt="prompt",
            images=[ImageRef("/missing.png", "none", 1, 1)],
            references=["reference"],
        )
        self.assertEqual(_max_new_tokens(sample), 550)
        self.assertTrue(
            intentional_length_constraint(
                sample, truncated=True, max_new_tokens=550
            )
        )
        self.assertFalse(
            intentional_length_constraint(
                sample, truncated=True, max_new_tokens=1100
            )
        )

    def test_grounding_decode_budget_matches_exp5(self) -> None:
        sample = Sample(
            id="g",
            dataset="xlrs_grounding",
            subtask="grounding",
            task_type="bbox",
            prompt="locate",
            images=[ImageRef("/missing.png", "none", 4096, 4096)],
            references=["[0.1,0.2,0.3,0.4]"],
        )
        self.assertEqual(_max_new_tokens(sample), 64)

    def test_route_and_single_share_xlrs_grounding_runtime(self) -> None:
        runtime = evaluation_runtime_config(
            Path("xlrs_grounding_test__xlrs_grounding_referring.jsonl")
        )
        self.assertEqual(runtime["max_pixels"], 16_777_216)
        self.assertEqual(runtime["max_model_len"], 32_768)
        self.assertEqual(runtime["batch_size"], 4)

    def test_grounding_profile_uses_historical_assistant_prefix(self) -> None:
        adapter = object.__new__(VLLMBatchAdapter)
        adapter.profile_key = "expert_ground"
        adapter.processor = _RecordingProcessor()
        adapter.min_pixels = 1
        adapter.max_pixels = 1024
        rendered = adapter._prompt_text(["/missing.png"], "locate object")
        self.assertFalse(adapter.processor.kwargs["add_generation_prompt"])
        self.assertTrue(
            rendered.endswith("<|im_start|>assistant\n thinking\n\n response\n\n")
        )

    def test_xlrs_grounding_uses_reported_4096_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            full_res = (
                Path(tmp) / "images_exported_test_4096" / "xlrs_vg_00000.jpg"
            )
            full_res.parent.mkdir()
            Image.new("RGB", (32, 32)).save(full_res)
            source_path = Path(tmp) / "images_exported_test" / "xlrs_vg_00844.jpg"
            row = convert_flat_grounding(
                {
                    "id": "x",
                    "path": "original-image.jpg",
                    "image": str(source_path),
                    "question": "locate it",
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "image_width": 4096,
                    "image_height": 4096,
                },
                "xlrs_grounding",
                1,
                source_dir=Path(tmp),
                images_root=None,
                image_index_by_source={},
            )
        self.assertEqual(
            Path(row["images"][0]["path"]).name,
            "xlrs_vg_00000.jpg",
        )
        self.assertIn("images_exported_test_4096", row["images"][0]["path"])
        self.assertEqual(row["images"][0]["width"], 4096)
        self.assertEqual(
            row["metadata"]["image_protocol"],
            "xlrs_grounding_4096_export_v1",
        )

    def test_reported_caption_metrics_include_percent_units(self) -> None:
        result = summarize_reported_caption_metrics(
            ["a small red building"], [["a small red building"]]
        )
        self.assertEqual(result["protocol"], "project_report_caption_metrics_v1")
        self.assertAlmostEqual(
            result["percent"]["ROUGE-L"], result["raw"]["ROUGE-L"] * 100
        )


if __name__ == "__main__":
    unittest.main()
