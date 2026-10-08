import unittest

from cargo_baseline.diagnostics.protocol_oracle import evaluate_oracle_samples


def sample(path, pid, camid, domain):
    return {"path": path, "pid": pid, "camid": camid, "domain": domain}


class CargoProtocolOracleTests(unittest.TestCase):
    def test_all_protocols_are_perfect_for_valid_queries(self):
        query = [
            sample("qa1", 1, 0, "aerial"),
            sample("qg2", 2, 5, "ground"),
            sample("qa3-no-positive", 3, 0, "aerial"),
        ]
        gallery = [
            sample("ga1", 1, 1, "aerial"),
            sample("gg1", 1, 6, "ground"),
            sample("ga2", 2, 1, "aerial"),
            sample("gg2", 2, 6, "ground"),
            sample("negative", 99, 7, "ground"),
        ]
        results = evaluate_oracle_samples(query, gallery)
        self.assertEqual(set(results), {"all", "aa", "gg", "ag", "g2ag"})
        for protocol, result in results.items():
            self.assertAlmostEqual(result["rank1"], 1.0, msg=protocol)
            self.assertAlmostEqual(result["mAP"], 1.0, msg=protocol)
            self.assertAlmostEqual(result["mINP"], 1.0, msg=protocol)
        self.assertGreater(results["all"]["skipped_queries"], 0)
        self.assertGreater(results["aa"]["skipped_queries"], 0)


if __name__ == "__main__":
    unittest.main()
