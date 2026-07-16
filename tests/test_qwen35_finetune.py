from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate_swift_finetune as esf  # noqa: E402
import prepare_qwen35_rs_data as prep  # noqa: E402


def row(*images: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "answer"},
        ],
        "images": list(images),
    }


class Qwen35DataSplitTest(unittest.TestCase):
    def test_split_keeps_train_validation_images_disjoint_across_sources(self):
        buckets = {
            "caption": [row("shared.png"), row("caption-only.png")],
            "vqa": [row("shared.png"), row("vqa-only.png")],
        }
        train, validation = prep.split_buckets_image_disjoint(
            buckets, val_per_source=1, rng=random.Random(1)
        )
        self.assertEqual({item["images"][0] for item in train["caption"]}, {"shared.png"})
        self.assertEqual({item["images"][0] for item in train["vqa"]}, {"shared.png"})
        self.assertEqual(
            {item["images"][0] for item in validation["caption"]},
            {"caption-only.png"},
        )
        self.assertEqual(
            {item["images"][0] for item in validation["vqa"]},
            {"vqa-only.png"},
        )

    def test_split_moves_every_record_for_selected_image_group(self):
        buckets = {
            "caption": [row("same.png"), row("same.png"), row("shared.png")],
            "vqa": [row("shared.png"), row("vqa-only.png"), row("other.png")],
        }
        train, validation = prep.split_buckets_image_disjoint(
            buckets, val_per_source=2, rng=random.Random(2)
        )
        self.assertEqual(len(validation["caption"]), 2)
        self.assertEqual({item["images"][0] for item in validation["caption"]}, {"same.png"})
        self.assertNotIn("same.png", {item["images"][0] for item in train["caption"]})


class Qwen35EvaluationTest(unittest.TestCase):
    def test_change_caption_markers_receive_caption_metrics(self):
        for marker in ("[CD]", "[CCD]"):
            task = esf.task_name(f"<image><image>{marker} Describe the changes")
            self.assertEqual(task, "change_caption")
            metrics = esf.metric_values(task, "a new road", "a new road")
            self.assertEqual(metrics["token_f1"], 1.0)
            self.assertEqual(metrics["rouge_l_f1"], 1.0)

    def test_clean_response_removes_thinking_block(self):
        self.assertEqual(esf.clean_response("<think>hidden</think> Yes"), "Yes")

    def test_unknown_task_marker_fails_explicitly(self):
        with self.assertRaises(ValueError):
            esf.task_name("unmarked prompt")


if __name__ == "__main__":
    unittest.main()
