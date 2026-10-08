import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from cargo_baseline.diagnostics.model_compatibility import (
    inspect_checkpoint, validate_strict_state_dict)
from cargo_baseline.diagnostics.checkpoint_diagnostics import parse_stage1_log


class CargoCheckpointDiagnosticsTests(unittest.TestCase):
    def test_missing_checkpoint_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            inspect_checkpoint("definitely_missing_checkpoint.pth.tar")

    def test_incompatible_checkpoint_is_rejected_by_strict_load(self):
        model = nn.Linear(4, 2)
        incompatible = {"weight": torch.zeros(3, 4)}
        result = validate_strict_state_dict(model, incompatible)
        self.assertFalse(result["strict_load_succeeded"])
        self.assertTrue(result["shape_mismatches"])
        self.assertIn("bias", result["missing_keys"])

    def test_checkpoint_metadata_and_nonfinite_values_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth.tar"
            torch.save({
                "epoch": 5,
                "backbone": "agw",
                "state_dict": {"weight": torch.tensor([1.0, float("nan")])},
            }, path)
            _, _, report = inspect_checkpoint(str(path))
        self.assertEqual(report["epoch"], 5)
        self.assertEqual(report["backbone_metadata"], "agw")
        self.assertEqual(report["nan_values"], 1)

    def test_stage1_log_parser_reports_unknown_fields_truthfully(self):
        content = """Aerial clustering criterion: eps: 0.500
Ground clustering criterion: eps: 0.500
==> Statistics for aerial epoch 0: 409 clusters
==> Statistics for ground epoch 0: 493 clusters
Epoch: [0][400/400] Time 0.3 (0.4) Loss 10.532 (11.093) Loss ir 4.117 Loss rgb 6.415
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "1log.txt"
            path.write_text(content, encoding="utf-8")
            report = parse_stage1_log(str(path), max_epoch=0)
        self.assertEqual(report["dbscan_eps"]["aerial"], 0.5)
        self.assertEqual(report["epochs"]["0"]["aerial_clusters"], 409)
        self.assertEqual(report["epochs"]["0"]["average_loss"], 11.093)
        self.assertEqual(report["epochs"]["0"]["noise_ratio"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
