#!/usr/bin/env python3
"""Read-only follow-up diagnosis for CARGO Stage1 validity and pseudo labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cargo_baseline.diagnostics.pipeline_consistency import audit_pipeline_consistency
from cargo_baseline.diagnostics.pseudo_label_audit import (
    NO_HISTORY, estimate_distance_memory, inventory_historical_artifacts)


OUTPUTS = {
    "pipeline": "pipeline_consistency.json",
    "sample": "train_feature_comparison.json",
    "inventory": "pseudo_artifact_inventory.json",
    "cluster": "pseudo_label_quality.json",
}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "BLOCKED", "reason": "Could not read report: {}".format(exc)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the legacy Stage1 source, compare ImageNet/Stage1 train-split "
            "features, and optionally recompute DBSCAN labels without training."))
    parser.add_argument("--legacy-code-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--imagenet-weights", default="")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", nargs="+", choices=("audit", "sample", "full-cluster"),
                        default=("audit", "sample"))
    parser.add_argument("--arch", default="agw")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--max-identities", type=int, default=64)
    parser.add_argument("--gallery-images-per-camera", type=int, default=2)
    parser.add_argument("--artifact-root", action="append", default=[])
    parser.add_argument("--artifact-scan-limit", type=int, default=10000)
    parser.add_argument("--enable-full-reclustering", action="store_true",
                        help="second explicit gate required with --mode full-cluster")
    parser.add_argument("--aerial-eps", type=float)
    parser.add_argument("--ground-eps", type=float)
    parser.add_argument("--k1", type=int)
    parser.add_argument("--k2", type=int)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--memory-overhead-factor", type=float, default=2.5)
    parser.add_argument("--max-memory-fraction", type=float, default=0.70)
    parser.add_argument("--force-memory-risk", action="store_true")
    return parser


def _worker_command(args, output_dir: Path, stages: list[str]) -> list[str]:
    worker = HERE / "diagnostics" / "followup_worker.py"
    command = [
        sys.executable, str(worker),
        "--legacy-code-dir", args.legacy_code_dir,
        "--checkpoint", args.checkpoint,
        "--data-dir", args.data_dir,
        "--output-dir", str(output_dir),
        "--device", args.device,
        "--arch", args.arch,
        "--height", str(args.height), "--width", str(args.width),
        "--batch-size", str(args.batch_size), "--workers", str(args.workers),
        "--print-freq", str(args.print_freq), "--seed", str(args.seed),
        "--max-identities", str(args.max_identities),
        "--gallery-images-per-camera", str(args.gallery_images_per_camera),
        "--stages", *stages,
    ]
    if args.imagenet_weights:
        command.extend(["--imagenet-weights", args.imagenet_weights])
    for option in ("aerial_eps", "ground_eps", "k1", "k2", "min_samples"):
        value = getattr(args, option)
        if value is not None:
            command.extend(["--" + option.replace("_", "-"), str(value)])
    command.extend([
        "--memory-overhead-factor", str(args.memory_overhead_factor),
        "--max-memory-fraction", str(args.max_memory_fraction),
    ])
    if args.force_memory_risk:
        command.append("--force-memory-risk")
    return command


def _write_summary(output_dir: Path, args, reports: dict, worker: dict) -> None:
    lines = [
        "# CARGO Stage1 Follow-up Diagnosis",
        "",
        "This diagnosis is read-only: no training, dataset mutation, checkpoint mutation, "
        "or automatic full reclustering was performed.",
        "",
        "## Inputs",
        "",
        "- Legacy source: `{}`".format(args.legacy_code_dir),
        "- Stage1 checkpoint: `{}`".format(args.checkpoint),
        "- ImageNet AGW initialization: `{}`".format(args.imagenet_weights or "NOT SUPPLIED"),
        "- CARGO dataset: `{}`".format(args.data_dir),
        "- Modes: `{}`".format(", ".join(args.mode)),
        "- Device: `{}`".format(args.device),
        "",
        "## Status",
        "",
        "| Deliverable | Status |",
        "|---|---|",
    ]
    for name in ("pipeline", "sample", "inventory", "cluster"):
        lines.append("| {} | {} |".format(
            OUTPUTS[name], reports[name].get("status", "UNKNOWN")))
    lines.extend([
        "",
        "## Interpretation boundaries",
        "",
        "- Train identities are used only after inference for `DIAGNOSTIC_GT_ONLY` metrics.",
        "- A checkpoint reclustering report, if requested, is marked "
        "`CHECKPOINT_RECLUSTER_NOT_HISTORICAL_EPOCH_LABELS`.",
        "- Absence of saved labels is reported as `{}`; recomputed labels are not substituted "
        "for historical epoch labels.".format(NO_HISTORY),
        "- Full reclustering is disabled unless both `--mode full-cluster` and "
        "`--enable-full-reclustering` are supplied with all DBSCAN/Jaccard parameters.",
        "- Runtime worker return code: `{}`.".format(worker.get("returncode", "NOT RUN")),
        "",
        "## Resource estimate for the known CARGO train split",
        "",
        "| Domain | Samples | One float32 distance matrix | Conservative peak (2.5x) |",
        "|---|---:|---:|---:|",
    ])
    for domain, count in (("Aerial", 22338), ("Ground", 29113)):
        estimate = estimate_distance_memory(count)
        lines.append("| {} | {:,} | {:.2f} GiB | {:.2f} GiB |".format(
            domain, count, estimate["single_float32_matrix_gib"],
            estimate["estimated_peak_gib"]))
    lines.extend([
        "",
        "Actual FAISS/Jaccard working memory can exceed this estimate. Domains are processed "
        "sequentially and a preflight gate checks available host memory.",
        "",
    ])
    (output_dir / "diagnosis_summary.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    args.mode = list(dict.fromkeys(args.mode))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = {
        "pipeline": {"status": "NOT RUN", "reason": "audit mode was not requested"},
        "sample": {"status": "NOT RUN", "reason": "sample mode was not requested"},
        "inventory": {"status": "NOT RUN", "reason": "audit mode was not requested"},
        "cluster": {"status": "DISABLED", "reason": "full reclustering is opt-in"},
    }
    for key, value in defaults.items():
        _write(output_dir / OUTPUTS[key], value)

    if "audit" in args.mode:
        _write(output_dir / OUTPUTS["pipeline"],
               audit_pipeline_consistency(args.legacy_code_dir))
        try:
            inventory = inventory_historical_artifacts(
                args.checkpoint, args.artifact_root, args.artifact_scan_limit)
        except Exception as exc:
            inventory = {"status": "BLOCKED", "reason": "{}: {}".format(
                type(exc).__name__, exc)}
        _write(output_dir / OUTPUTS["inventory"], inventory)

    worker_stages = []
    if "sample" in args.mode:
        worker_stages.append("sample")
    if "full-cluster" in args.mode:
        if not args.enable_full_reclustering:
            _write(output_dir / OUTPUTS["cluster"], {
                "status": "BLOCKED",
                "reason": ("Full reclustering needs the explicit "
                           "--enable-full-reclustering safety gate."),
            })
        else:
            required = ("aerial_eps", "ground_eps", "k1", "k2", "min_samples")
            missing = [name for name in required if getattr(args, name) is None]
            if missing:
                _write(output_dir / OUTPUTS["cluster"], {
                    "status": "BLOCKED",
                    "reason": "Missing explicit frozen parameters: {}".format(
                        ", ".join(missing)),
                })
            else:
                worker_stages.append("full-cluster")

    worker_result = {"returncode": "NOT RUN"}
    if worker_stages:
        command = _worker_command(args, output_dir, worker_stages)
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        worker_result = {"returncode": completed.returncode, "command": command}

    reports = {key: _read(output_dir / filename) for key, filename in OUTPUTS.items()}
    if reports["sample"].get("stage1_checkpoint_compatibility"):
        pipeline = reports["pipeline"]
        pipeline["runtime_checkpoint_compatibility"] = reports["sample"][
            "stage1_checkpoint_compatibility"]
        _write(output_dir / OUTPUTS["pipeline"], pipeline)
        reports["pipeline"] = pipeline
    _write_summary(output_dir, args, reports, worker_result)
    return 0 if worker_result.get("returncode") in {"NOT RUN", 0} else 2


if __name__ == "__main__":
    raise SystemExit(main())
