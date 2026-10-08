import unittest

import numpy as np

from cargo_baseline.diagnostics.feature_health import summarize_feature_health


class CargoFeatureHealthTests(unittest.TestCase):
    def test_detects_constant_feature_collapse(self):
        features = np.ones((24, 8), dtype=np.float32)
        pids = np.repeat(np.arange(6), 4)
        domains = np.asarray(["aerial", "ground"] * 12)
        result = summarize_feature_health(features, pids, domains, seed=7)
        self.assertTrue(result["near_constant_features"])
        self.assertLess(result["effective_rank"]["value"], 1.5)
        self.assertEqual(result["nonfinite_rows"], 0)

    def test_reports_discriminative_synthetic_features(self):
        features = np.repeat(np.eye(6, dtype=np.float32), 4, axis=0)
        pids = np.repeat(np.arange(6), 4)
        result = summarize_feature_health(features, pids, seed=3)
        self.assertFalse(result["near_constant_features"])
        self.assertGreater(result["identity_separability"]["mean_cosine_gap"], 0.9)


if __name__ == "__main__":
    unittest.main()
