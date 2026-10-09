import tempfile
import unittest
from pathlib import Path

from cargo_baseline.diagnose_stage1_followup import build_parser
from cargo_baseline.diagnostics.pseudo_label_audit import (
    NO_HISTORY, RECLUSTER_MARKER, compute_pseudo_label_quality,
    inventory_historical_artifacts)


class PseudoLabelAuditTests(unittest.TestCase):
    def test_noise_purity_pairwise_and_camera_metrics(self):
        labels = [0, 0, 1, 1, -1]
        pids = [10, 10, 20, 21, 20]
        camids = [0, 1, 0, 1, 2]
        report = compute_pseudo_label_quality(labels, pids, camids)
        self.assertEqual(report["marker"], RECLUSTER_MARKER)
        self.assertEqual(report["assigned_samples"], 4)
        self.assertEqual(report["noise_samples"], 1)
        self.assertAlmostEqual(report["coverage"], 0.8)
        self.assertAlmostEqual(report["cluster_purity"], 0.75)
        self.assertEqual(report["pairwise"]["true_positive_pairs"], 1)
        self.assertEqual(report["pairwise"]["false_positive_pairs"], 1)
        self.assertEqual(report["merged_cluster_count"], 1)
        self.assertEqual(report["cross_camera_clusters"], 2)

    def test_inventory_does_not_claim_history_when_none_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "checkpoint.pth.tar"
            checkpoint.write_bytes(b"placeholder")
            report = inventory_historical_artifacts(str(checkpoint))
            self.assertEqual(report["status"], NO_HISTORY)

    def test_full_reclustering_is_not_default_mode(self):
        parser = build_parser()
        args = parser.parse_args([
            "--legacy-code-dir", "legacy", "--checkpoint", "checkpoint",
            "--data-dir", "data", "--device", "cpu", "--output-dir", "out"])
        self.assertEqual(list(args.mode), ["audit", "sample"])
        self.assertFalse(args.enable_full_reclustering)


if __name__ == "__main__":
    unittest.main()
