import tempfile
import unittest

from cargo_baseline.diagnostics.pipeline_consistency import audit_pipeline_consistency


class PipelineConsistencyTests(unittest.TestCase):
    def test_missing_legacy_tree_is_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            report = audit_pipeline_consistency(temp + "/missing")
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["checks"], [])


if __name__ == "__main__":
    unittest.main()
