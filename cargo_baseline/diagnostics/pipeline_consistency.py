"""Static, evidence-backed audit of a legacy CARGO Stage-1 pipeline."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable


def _read(path: Path) -> tuple[str, list[str]]:
    if not path.is_file():
        return "", []
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, text.splitlines()


def _evidence(path: Path, lines: list[str], patterns: Iterable[str], limit: int = 8) -> list[dict]:
    compiled = [re.compile(pattern) for pattern in patterns]
    result = []
    for number, line in enumerate(lines, 1):
        if any(pattern.search(line) for pattern in compiled):
            result.append({"file": str(path), "line": number, "text": line.strip()})
            if len(result) >= limit:
                break
    return result


def _check(item: str, conclusion: str, evidence: list[dict],
           status: str | None = None, note: str = "") -> dict:
    return {
        "item": item,
        "status": status or ("VERIFIED" if evidence else "UNKNOWN"),
        "conclusion": conclusion,
        "evidence": evidence,
        "note": note,
    }


def audit_pipeline_consistency(legacy_code_dir: str) -> dict:
    root = Path(legacy_code_dir).expanduser().resolve()
    engine = root / "cargo_baseline" / "_train_cargo_engine.py"
    evaluator = root / "cargo_baseline" / "cargo_evaluation.py"
    feature_extractor = root / "clustercontrast" / "evaluators.py"
    model_candidates = (
        root / "clustercontrast" / "models" / "agw.py",
        root / "clustercontrast" / "models" / "agw_lag.py",
        root / "clustercontrast" / "models" / "agw_lag_stable.py",
        root / "clustercontrast" / "models" / "resnet_agw.py",
    )
    model = next((path for path in model_candidates if path.is_file()), model_candidates[0])
    dataset_candidates = (
        root / "clustercontrast" / "datasets" / "cargo_common.py",
        root / "clustercontrast" / "datasets" / "cargo_aerial.py",
        root / "clustercontrast" / "datasets" / "cargo_ground.py",
    )
    dataset = next((path for path in dataset_candidates if path.is_file()), dataset_candidates[0])
    files = {"engine": engine, "evaluator": evaluator,
             "feature_extractor": feature_extractor, "model": model,
             "dataset": dataset}
    contents = {name: _read(path) for name, path in files.items()}

    if not root.is_dir():
        return {
            "status": "BLOCKED",
            "source_root": str(root),
            "reason": "Legacy source directory is not accessible",
            "audited_files": {name: {"path": str(path), "exists": False}
                              for name, path in files.items()},
            "checks": [],
        }

    e_text, e_lines = contents["engine"]
    v_text, v_lines = contents["evaluator"]
    x_text, x_lines = contents["feature_extractor"]
    m_text, m_lines = contents["model"]
    d_text, d_lines = contents["dataset"]
    checks = []

    aerial = _evidence(engine, e_lines, [r"extract_features\(.*mode\s*=\s*2", r"modal\s*=\s*2"])
    checks.append(_check("aerial forward modal", "Aerial clustering uses modal=2.", aerial))
    ground = _evidence(engine, e_lines, [r"extract_features\(.*mode\s*=\s*1", r"modal\s*=\s*1"])
    checks.append(_check("ground forward modal", "Ground clustering uses modal=1.", ground))

    route_ev = _evidence(evaluator, v_lines, [r"aerial.*2.*ground.*1", r"modal\s*=\s*2", r"modal\s*=\s*1", r"model\("])
    route_verified = (bool(re.search(r"aerial.*2.*ground.*1", v_text)) or (
        bool(re.search(r"modal\s*=\s*2|domain.*aerial", v_text, re.I))
        and bool(re.search(r"modal\s*=\s*1|domain.*ground", v_text, re.I))))
    checks.append(_check(
        "evaluation modal routing",
        ("Evaluation routes aerial samples to modal=2 and ground samples to modal=1."
         if route_verified else
         "Evaluation modal routing could not be proven from the located source."),
        route_ev, "VERIFIED" if route_verified else "UNKNOWN"))

    cluster_pre = _evidence(engine, e_lines, [r"def get_test_loader", r"T\.Resize", r"T\.Normalize", r"interpolation\s*=\s*3"])
    checks.append(_check("clustering preprocessing",
                         "Clustering resize/interpolation/normalization are recorded from source.", cluster_pre))
    eval_pre = _evidence(evaluator, v_lines, [r"T\.Resize", r"T\.Normalize", r"interpolation\s*=\s*3"])
    checks.append(_check("evaluation preprocessing",
                         "Evaluation resize/interpolation/normalization are recorded from source.", eval_pre))

    embedding = _evidence(model, m_lines, [r"bottleneck", r"return\s+self\.l2norm", r"F\.normalize"])
    checks.append(_check("embedding definition",
                         "The returned evaluation embedding and its normalization must be read from the legacy model.", embedding))
    bn = _evidence(model, m_lines, [r"bottleneck", r"x_pool", r"feat\s*=", r"return\s+self\.l2norm"])
    checks.append(_check("BN-neck boundary",
                         "Evidence identifies pooled features, BN neck, and returned representation.", bn))

    norm = (_evidence(engine, e_lines, [r"F\.normalize", r"Normalize\("])
            + _evidence(evaluator, v_lines, [r"F\.normalize", r"Normalize\("]))
    checks.append(_check("normalization scope",
                         "Input and embedding normalization sites are listed; runtime equality is not inferred.", norm))

    default_flip = bool(re.search(r"def\s+extract_features\([^\n]*flip\s*=\s*True", x_text))
    explicit_flip = bool(re.search(r"extract_features\([^\n]*flip\s*=", e_text))
    unconditional_flip = bool(re.search(r"flip\s*=\s*fliplr\(", x_text)) and not bool(
        re.search(r"if\s+flip\s*:", x_text))
    tta_ev = _evidence(feature_extractor, x_lines, [r"def extract_features", r"flip", r"torch\.flip"])
    tta_engine = _evidence(engine, e_lines, [r"extract_features\("])
    if unconditional_flip:
        tta_conclusion = (
            "extract_features computes and averages horizontal-flip features "
            "unconditionally; Stage1 clustering calls therefore use flip TTA as well."
        )
    elif default_flip and not explicit_flip:
        tta_conclusion = (
            "Stage1 clustering inherits extract_features(flip=True) because calls do not override it."
        )
    else:
        tta_conclusion = "TTA behavior is explicitly controlled or could not be proven from static source."
    checks.append(_check("horizontal-flip TTA scope", tta_conclusion,
                         tta_ev + tta_engine[:4],
                         "VERIFIED" if default_flip or unconditional_flip else "UNKNOWN",
                         "This is an observed implementation choice, not automatically a defect."))

    order = _evidence(engine, e_lines, [r"sorted\(dataset.*\.train", r"torch\.cat\(\[features", r"zip\(sorted"])
    checks.append(_check("sample/feature ordering",
                         "Sorting and concatenation/zip sites used for label alignment are listed.", order))

    pid_ev = (_evidence(evaluator, v_lines, [r"pid", r"same.*cam", r"same.*camera", r"_read_split"])
              + _evidence(dataset, d_lines, [r"query\s*=.*relabel\s*=\s*False",
                                             r"gallery\s*=.*relabel\s*=\s*False",
                                             r"if\s+relabel", r"pid2label"]))
    pid_conclusion = (
        "Query/gallery construction explicitly disables relabeling; raw test PIDs are preserved."
        if re.search(r"query\s*=.*relabel\s*=\s*False", d_text)
        and re.search(r"gallery\s*=.*relabel\s*=\s*False", d_text)
        else "Raw evaluation PID preservation could not be proven from the located source."
    )
    checks.append(_check("evaluation PID and camera semantics",
                         pid_conclusion, pid_ev,
                         "VERIFIED" if "explicitly disables" in pid_conclusion else "UNKNOWN"))

    checks.append(_check(
        "checkpoint/runtime topology",
        "Static source cannot prove checkpoint compatibility; the isolated worker must strict-load the checkpoint into this legacy AGW.",
        _evidence(model, m_lines, [r"class", r"def forward"]),
        "REQUIRES_RUNTIME" if model.is_file() else "UNKNOWN"))

    missing = [name for name, path in files.items() if not path.is_file()]
    return {
        "status": "COMPLETED_WITH_UNKNOWNS" if missing or any(
            check["status"] in {"UNKNOWN", "REQUIRES_RUNTIME"} for check in checks) else "COMPLETED",
        "source_root": str(root),
        "audited_files": {name: {"path": str(path), "exists": path.is_file()}
                          for name, path in files.items()},
        "missing_files": missing,
        "checks": checks,
        "policy": (
            "Conclusions are based only on the supplied legacy source tree. Missing evidence is "
            "reported as UNKNOWN; ordinary training augmentation differences are not classified as bugs."
        ),
    }
