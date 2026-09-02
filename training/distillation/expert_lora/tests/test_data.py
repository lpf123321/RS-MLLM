from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from expert_lora.data import normalize_paths, validate_records
from expert_lora.runner import load_config


def row(row_id: str, image: str, answer: str) -> dict:
    return {
        "id": row_id,
        "image": [image],
        "conversations": [
            {"from": "human", "value": "<image>\nquestion"},
            {"from": "gpt", "value": answer},
        ],
    }


class DataValidationTest(unittest.TestCase):
    def test_portable_mcq_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images" / "a.png"
            image.parent.mkdir()
            image.write_bytes(b"not-a-real-png")
            report = validate_records(
                [row("a", "images/a.png", "B")],
                schema="mcq",
                image_root=root,
            )
            self.assertTrue(report["passed"], report)
            self.assertTrue(report["portable_paths"])

    def test_absolute_path_is_rejected(self):
        report = validate_records(
            [row("a", "/tmp/a.png", "A")],
            schema="mcq",
            image_root=Path("/tmp"),
            check_images=False,
        )
        self.assertFalse(report["passed"])
        self.assertIn("not portable", report["errors"][0])

    def test_grounding_coordinate_grammars(self):
        vrs = validate_records(
            [row("v", "v.png", "{<1><2><90><99>}")],
            schema="vrs_grounding",
            image_root=Path("."),
            check_images=False,
        )
        xlrs = validate_records(
            [row("x", "x.png", "[0.10000,0.20000,0.90000,1.00000]")],
            schema="xlrs_grounding",
            image_root=Path("."),
            check_images=False,
        )
        self.assertTrue(vrs["passed"], vrs)
        self.assertTrue(xlrs["passed"], xlrs)

    def test_normalize_absolute_prefix(self):
        records = [row("a", "/old/root/folder/a.png", "A")]
        normalized = normalize_paths(records, [("/old/root", "images/mme")])
        self.assertEqual(normalized[0]["image"], ["images/mme/folder/a.png"])
        self.assertEqual(records[0]["image"], ["/old/root/folder/a.png"])


class ConfigTest(unittest.TestCase):
    def test_all_retained_configs_load(self):
        config_root = Path(__file__).resolve().parents[1] / "configs"
        names = {
            "general_exp3.json",
            "general_exp7.json",
            "grounding_bootstrap_942.json",
            "grounding_exp1.json",
            "grounding_exp4.json",
            "grounding_exp5.json",
        }
        for name in names:
            config = load_config(config_root / name)
            self.assertGreater(config["expected_records"], 0)
            self.assertEqual(config["effective_batch_size"], 32)


if __name__ == "__main__":
    unittest.main()
