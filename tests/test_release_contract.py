import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "cargo_baseline" / "_train_cargo_engine.py"
REQUIRED_DATA_SOURCE = (
    "clustercontrast/datasets/__init__.py",
    "clustercontrast/datasets/cargo_aerial.py",
    "clustercontrast/datasets/cargo_common.py",
    "clustercontrast/datasets/cargo_ground.py",
    "clustercontrast/utils/data/__init__.py",
    "clustercontrast/utils/data/base_dataset.py",
    "clustercontrast/utils/data/preprocessor.py",
    "clustercontrast/utils/data/sampler.py",
    "clustercontrast/utils/data/transforms.py",
)


class ReleaseContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine_text = ENGINE.read_text(encoding="utf-8")
        cls.engine_ast = ast.parse(cls.engine_text)

    def test_engine_is_valid_python(self):
        self.assertIsInstance(self.engine_ast, ast.Module)

    def test_fixed_ablation_contract_is_visible(self):
        for expected in (
            "Stage 2 memory: CMhard",
            "ChannelAdapGray: False",
            "ChannelExchange: False",
            "Dynamic AGVA: False",
            "Three-domain matching: False",
            "EMA loss included in backward: False",
            "ALL-memory second optimization: True",
        ):
            self.assertIn(expected, self.engine_text)

    def test_channel_augmentation_is_not_imported(self):
        imported_modules = []
        for node in ast.walk(self.engine_ast):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        self.assertNotIn("ChannelAug", imported_modules)

    def test_release_contains_only_cargo_dataset_factories(self):
        namespace = {}
        source = (ROOT / "clustercontrast" / "datasets" / "__init__.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        factory_keys = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        factory_keys.add(key.value)
        self.assertEqual(factory_keys, {"cargo_aerial", "cargo_ground"})

    def test_data_source_directories_are_published(self):
        for relative_path in REQUIRED_DATA_SOURCE:
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

        ignore_rules = {
            line.strip()
            for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("/datasets/", ignore_rules)
        self.assertIn("/data/", ignore_rules)
        self.assertNotIn("datasets/", ignore_rules)
        self.assertNotIn("data/", ignore_rules)


if __name__ == "__main__":
    unittest.main()
