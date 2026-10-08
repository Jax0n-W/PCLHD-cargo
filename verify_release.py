#!/usr/bin/env python3
"""Static pre-publication checks for this source-only release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = {
    "README.md",
    "LICENSE",
    "requirements.txt",
    "environment.yml",
    "cargo_baseline/train_cargo.py",
    "cargo_baseline/_train_cargo_engine.py",
    "cargo_baseline/evaluate_cargo.py",
    "cargo_baseline/diagnose_stage1.py",
    "cargo_baseline/diagnostics/legacy_worker.py",
    "cargo_baseline/diagnostics/protocol_oracle.py",
    "cargo_baseline/diagnostics/feature_health.py",
    "cargo_baseline/diagnostics/model_compatibility.py",
    "cargo_baseline/diagnostics/checkpoint_diagnostics.py",
    "clustercontrast/datasets/__init__.py",
    "clustercontrast/datasets/cargo_aerial.py",
    "clustercontrast/datasets/cargo_common.py",
    "clustercontrast/datasets/cargo_ground.py",
    "clustercontrast/models/agw_cargo_stable.py",
    "clustercontrast/utils/data/__init__.py",
    "clustercontrast/utils/data/base_dataset.py",
    "clustercontrast/utils/data/preprocessor.py",
    "clustercontrast/utils/data/sampler.py",
    "clustercontrast/utils/data/transforms.py",
}
FORBIDDEN_TEXT = (
    re.compile(r"/home/", re.IGNORECASE),
    re.compile(r"/data/home/", re.IGNORECASE),
    re.compile(r"[A-Z]:\\Users\\", re.IGNORECASE),
)
FORBIDDEN_SUFFIXES = {".pth", ".pt", ".ckpt", ".pyc"}


def main() -> None:
    missing = sorted(path for path in REQUIRED if not (ROOT / path).is_file())
    if missing:
        raise RuntimeError("Missing required release files: {}".format(missing))

    manifest = {}
    violations = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file()):
        relative = path.relative_to(ROOT).as_posix()
        if relative == "MANIFEST_SHA256.json":
            continue
        relative_parts = Path(relative).parts
        if ".git" in relative_parts or "__pycache__" in relative_parts:
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append("forbidden artifact: {}".format(relative))
            continue
        data = path.read_bytes()
        manifest[relative] = hashlib.sha256(data).hexdigest()
        if (relative != "verify_release.py" and
                path.suffix.lower() in {".py", ".md", ".txt", ".yml", ".yaml", ".sh"}):
            text = data.decode("utf-8")
            for pattern in FORBIDDEN_TEXT:
                if pattern.search(text):
                    violations.append(
                        "private absolute path in {}: {}".format(relative, pattern.pattern))
        if path.suffix.lower() == ".py":
            compile(data.decode("utf-8"), str(path), "exec")

    if violations:
        raise RuntimeError("\n".join(violations))

    output = ROOT / "MANIFEST_SHA256.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Release verification passed: {} files".format(len(manifest)))
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
