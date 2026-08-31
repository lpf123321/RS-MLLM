from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from flashopd.config import OPDConfig
from flashopd.cli import parse_args
from flashopd.loss import (
    jsd_divergence,
    kl_divergence,
    kl_from_sparse_logprobs,
)
from flashopd.multimodal import (
    JsonlOffsetDataset,
    load_multimodal_dataset,
    normalize_multimodal_example,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = REPOSITORY_ROOT / "training/distillation/flashopd_mm/configs"


class LossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.student = torch.tensor(
            [[[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]]], requires_grad=True
        )
        self.teacher = torch.tensor([[[1.0, 0.0, -2.0], [2.0, 1.0, 0.0]]])
        self.mask = torch.tensor([[1.0, 0.0]])

    def test_dense_losses_are_finite_and_differentiable(self) -> None:
        losses = [
            kl_divergence(
                self.student,
                self.teacher,
                kl_type=kind,
                token_mask=self.mask,
                chunk_size=1,
            )
            for kind in ("forward", "reverse")
        ]
        losses.append(jsd_divergence(self.student, self.teacher, token_mask=self.mask))
        for loss in losses:
            self.assertTrue(torch.isfinite(loss))
            self.assertGreaterEqual(float(loss.detach()), 0.0)
        sum(losses).backward()
        self.assertTrue(torch.isfinite(self.student.grad).all())

    def test_sparse_teacher_loss_is_finite(self) -> None:
        top_ids = torch.tensor([[[0, 1], [1, 2]]])
        top_logprobs = torch.log_softmax(self.teacher, -1).gather(-1, top_ids)
        loss = kl_from_sparse_logprobs(
            self.student,
            top_ids,
            top_logprobs,
            token_mask=self.mask,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.detach()), 0.0)


class ConfigurationTests(unittest.TestCase):
    def test_all_published_configs_load(self) -> None:
        paths = sorted(CONFIG_ROOT.glob("*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            config = OPDConfig.from_yaml(str(path))
            self.assertIn(config.kl_type, {"forward", "reverse", "jsd"})
            self.assertIn(config.teacher_backend, {"local", "api"})
            self.assertGreater(config.max_new_tokens, 0)

    def test_optional_numeric_cli_overrides_keep_numeric_types(self) -> None:
        argv = [
            "flashopd",
            "--max_samples",
            "1",
            "--max_pixels",
            "262144",
            "--eval_steps",
            "3",
            "--learning_rate",
            "0.0001",
            "--use_lora",
            "true",
        ]
        with patch.object(sys, "argv", argv):
            config = parse_args()
        self.assertEqual(config.max_samples, 1)
        self.assertEqual(config.max_pixels, 262144)
        self.assertEqual(config.eval_steps, 3)
        self.assertEqual(config.learning_rate, 0.0001)
        self.assertIs(config.use_lora, True)


class MultimodalDatasetTests(unittest.TestCase):
    def test_portable_vrsbench_conversation_and_jsonl_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images" / "sample.png"
            image.parent.mkdir()
            Image.new("RGB", (8, 8), color=(20, 40, 60)).save(image)
            row = {
                "id": "fixture-1",
                "image": "images/sample.png",
                "conversations": [
                    {"from": "human", "value": "<image>\nWhat is visible?"},
                    {"from": "gpt", "value": "A road."},
                ],
            }
            data = root / "train.jsonl"
            data.write_text(json.dumps(row) + "\n\n", encoding="utf-8")

            dataset = JsonlOffsetDataset(str(data))
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["id"], "fixture-1")
            subset = load_multimodal_dataset(str(data), max_samples=1)
            self.assertEqual(len(subset), 1)

            messages, images = normalize_multimodal_example(row, str(root))
            self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
            self.assertEqual(images, [str(image)])
            self.assertEqual(messages[0]["content"][0], {"type": "image"})


if __name__ == "__main__":
    unittest.main()
