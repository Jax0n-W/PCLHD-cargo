import unittest

import numpy as np

from cargo_baseline.diagnostics.train_feature_probe import (
    DIAGNOSTIC_GT_ONLY, evaluate_domain_probe, parse_cargo_path,
    select_domain_probe)


class TrainFeatureProbeTests(unittest.TestCase):
    def test_filename_parsing_preserves_raw_pid_and_domain(self):
        aerial = parse_cargo_path("/tmp/Cam5_day_2501_9.jpg")
        ground = parse_cargo_path("/tmp/Cam6_night_2501_10.jpg")
        self.assertEqual(aerial["pid"], 2501)
        self.assertEqual(aerial["camid"], 4)
        self.assertEqual(aerial["domain"], "aerial")
        self.assertEqual(ground["domain"], "ground")

    def test_sampling_is_deterministic_cross_camera_and_disjoint(self):
        samples = []
        for pid in range(1, 6):
            for cam in (0, 1, 2):
                for index in range(3):
                    samples.append({
                        "path": "a_{}_{}_{}.jpg".format(pid, cam, index),
                        "pid": pid, "camid": cam, "domain": "aerial"})
        first = select_domain_probe(samples, "aerial", max_identities=3,
                                    gallery_images_per_camera=1, seed=7)
        second = select_domain_probe(samples, "aerial", max_identities=3,
                                     gallery_images_per_camera=1, seed=7)
        self.assertEqual(first["marker"], DIAGNOSTIC_GT_ONLY)
        self.assertEqual(first["query"], second["query"])
        self.assertEqual(first["gallery"], second["gallery"])
        self.assertFalse({x["path"] for x in first["query"]} &
                         {x["path"] for x in first["gallery"]})
        for query in first["query"]:
            positives = [x for x in first["gallery"] if x["pid"] == query["pid"]]
            self.assertTrue(positives)
            self.assertTrue(all(x["camid"] != query["camid"] for x in positives))

    def test_metrics_filter_same_pid_same_camera(self):
        query = [{"path": "q", "pid": 1, "camid": 0}]
        gallery = [
            {"path": "invalid", "pid": 1, "camid": 0},
            {"path": "positive", "pid": 1, "camid": 1},
            {"path": "negative", "pid": 2, "camid": 1},
        ]
        features = {
            "q": np.array([1.0, 0.0]),
            "invalid": np.array([1.0, 0.0]),
            "positive": np.array([0.9, 0.1]),
            "negative": np.array([0.0, 1.0]),
        }
        result = evaluate_domain_probe(features, {
            "query": query, "gallery": gallery, "selected_identities": 1,
            "eligible_cross_camera_identities": 1,
            "eligible_identity_coverage": 1.0,
        })
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["rank1"], 1.0)
        self.assertEqual(result["mAP"], 1.0)


if __name__ == "__main__":
    unittest.main()
