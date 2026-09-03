from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest

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
from vision_opd_vllm_eval import VLLMBatchAdapter
from vision_opd_eval import intentional_length_constraint
from rsmllm.router_eval import evaluation_runtime_config


class _RecordingProcessor:
    def __init__(self) -> None:
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "rendered-user-turn\n"


class ProtocolFixTests(unittest.TestCase):
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
            full_res = Path(tmp) / "images_exported_test_4096" / "x.jpg"
            full_res.parent.mkdir()
            Image.new("RGB", (32, 32)).save(full_res)
            source_path = Path(tmp) / "images_exported_test" / "x.jpg"
            row = convert_flat_grounding(
                {
                    "id": "x",
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
            )
        self.assertEqual(
            Path(row["images"][0]["path"]).name,
            "x.jpg",
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
