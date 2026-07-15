from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate_results as er  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "eval_sample.jsonl"


class EvaluateResultsTest(unittest.TestCase):
    def load_fixture_units(self):
        runs = er.load_runs([f"fixture:{FIXTURE}"])
        return er.group_units(runs["fixture"])

    def test_xlrs_accuracy_fixture(self):
        units = self.load_fixture_units()[("xlrs_lite", "vqa")]
        values = er.score_accuracy(units)
        summary = er.summarize_metric(
            "fixture", "xlrs_lite", "vqa", "accuracy", values,
            row_count=sum(unit["row_count"] for unit in units),
            bootstrap=0,
            seed=1,
        )
        self.assertEqual(summary["observed"], 0.5)
        self.assertEqual(summary["unit_count"], 2)
        self.assertEqual(summary["row_count"], 2)

    def test_caption_text_metrics_exact_match(self):
        units = self.load_fixture_units()[("vrsbench", "caption")]
        scores = er.score_text(units)
        self.assertEqual(len(units), 1)
        self.assertEqual(scores["rouge_l_f1"][0], 1.0)
        self.assertEqual(scores["token_f1"][0], 1.0)

    def test_refer_normalizes_qwen_coordinate_scale(self):
        units = self.load_fixture_units()[("vrsbench", "refer")]
        scores = er.score_grounding(units)
        self.assertEqual(scores["format_validity"], [1.0, 0.0])
        self.assertEqual(scores["accuracy_at_0_5"], [1.0, 0.0])
        self.assertGreater(scores["mean_iou"][0], 0.8)

    def test_levir_collapses_references(self):
        units = self.load_fixture_units()[("levir_cc", "change_caption")]
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["unit_id"], "pair1")
        self.assertEqual(units[0]["reference_count"], 2)
        self.assertEqual(units[0]["prediction_variant_count"], 1)

    def test_cli_writes_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "report"
            cmd = [
                sys.executable,
                str(ROOT / "evaluate_results.py"),
                "--run",
                f"fixture:{FIXTURE}",
                "--out-dir",
                str(out_dir),
                "--bootstrap",
                "20",
                "--seed",
                "1",
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)
            for name in [
                "summary.json",
                "summary.csv",
                "summary.md",
                "unit_scores.jsonl",
                "run_manifest.json",
            ]:
                self.assertTrue((out_dir / name).exists(), name)
            summary = json.loads((out_dir / "summary.json").read_text())
            xlrs = [
                row for row in summary
                if row["dataset"] == "xlrs_lite"
                and row["task"] == "vqa"
                and row["metric"] == "accuracy"
            ][0]
            self.assertEqual(xlrs["observed"], 0.5)


if __name__ == "__main__":
    unittest.main()
