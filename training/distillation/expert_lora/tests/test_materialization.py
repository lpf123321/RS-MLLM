from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"


class MaterializationTest(unittest.TestCase):
    def test_resolves_by_basename_and_verifies_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            source = root / "official" / "nested"
            dataset.mkdir()
            source.mkdir(parents=True)
            payload = b"official-image-bytes"
            original = source / "sample.png"
            original.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            row = {
                "path": f"images/{digest[:24]}.png",
                "sha256": digest,
                "bytes": len(payload),
                "upstream_dataset": "vrsbench",
                "upstream_path": "Images_train/sample.png",
            }
            (dataset / "IMAGE_INDEX.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "materialize_images.py"),
                    "--dataset-root",
                    str(dataset),
                    "--source",
                    f"vrsbench={root / 'official'}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual((dataset / row["path"]).read_bytes(), payload)

    def test_test_directories_are_never_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            source = root / "official"
            train = source / "train"
            test = source / "test"
            dataset.mkdir()
            train.mkdir(parents=True)
            test.mkdir(parents=True)
            payload = b"same-public-image-bytes"
            train_image = train / "sample.png"
            test_image = test / "sample.png"
            train_image.write_bytes(payload)
            test_image.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            row = {
                "path": f"images/{digest[:24]}.png",
                "sha256": digest,
                "bytes": len(payload),
                "upstream_dataset": "xlrs_grounding",
                "upstream_path": "sample.png",
            }
            (dataset / "IMAGE_INDEX.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "materialize_images.py"),
                    "--dataset-root",
                    str(dataset),
                    "--source",
                    f"xlrs_grounding={source}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            materialized = dataset / row["path"]
            self.assertEqual(materialized.stat().st_ino, train_image.stat().st_ino)
            self.assertNotEqual(materialized.stat().st_ino, test_image.stat().st_ino)

    def test_hash_mismatch_fails_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            source = root / "official"
            dataset.mkdir()
            source.mkdir()
            (source / "sample.png").write_bytes(b"wrong")
            row = {
                "path": "images/expected.png",
                "sha256": hashlib.sha256(b"expected").hexdigest(),
                "bytes": len(b"expected"),
                "upstream_dataset": "vrsbench",
                "upstream_path": "sample.png",
            }
            (dataset / "IMAGE_INDEX.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "materialize_images.py"),
                    "--dataset-root",
                    str(dataset),
                    "--source",
                    f"vrsbench={source}",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((dataset / row["path"]).exists())


if __name__ == "__main__":
    unittest.main()
