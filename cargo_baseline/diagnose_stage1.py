#!/usr/bin/env python3
"""Read-only, offline-deployable diagnosis for a legacy CARGO Stage-1 run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cargo_baseline.diagnostics.checkpoint_diagnostics import (
    parse_stage1_log, read_json, write_json)
from cargo_baseline.diagnostics.protocol_oracle import run_protocol_oracle


OUTPUT_FILES = {
    "oracle": "protocol_oracle.json",
    "checkpoint": "checkpoint_compatibility.json",
    "feature": "feature_health.json",
    "initialization": "initialization_comparison.json",
    "numerical": "numerical_diagnostics.json",
    "log": "training_log_summary.json",
}
MODEL_STAGES = {"checkpoint", "feature", "initialization", "numerical"}


def _status(status: str, reason: str) -> dict:
    return {"status": status, "reason": reason}


def _load_outputs(output_dir: Path) -> Dict[str, dict]:
    reports = {}
    for stage, filename in OUTPUT_FILES.items():
        path = output_dir / filename
        reports[stage] = read_json(str(path)) if path.is_file() else {
            "status": "NOT RUN", "reason": "No report file was generated"}
    return reports


def _candidate_causes(reports: Dict[str, dict]) -> list[dict]:
    causes = []
    oracle = reports["oracle"]
    if oracle.get("status") == "COMPLETED" and oracle.get("all_protocols_passed"):
        causes.append({
            "candidate": "CARGO protocol filtering/metric implementation",
            "classification": "NOT SUPPORTED",
            "evidence": "Identity oracle reached 100% Rank-1/mAP for every valid query.",
        })
    elif oracle.get("status") == "FAILED":
        causes.append({
            "candidate": "CARGO protocol filtering/metric implementation",
            "classification": "CONFIRMED",
            "evidence": "Identity oracle did not reach its exact pass condition.",
        })
    else:
        causes.append({
            "candidate": "CARGO protocol filtering/metric implementation",
            "classification": "UNKNOWN",
            "evidence": oracle.get("reason", "Oracle was not completed."),
        })

    compatibility = reports["checkpoint"]
    if compatibility.get("strict_load_succeeded") is False and compatibility.get(
            "strict_load_attempted"):
        classification = "CONFIRMED"
    elif compatibility.get("strict_load_succeeded") is True:
        classification = "NOT SUPPORTED"
    else:
        classification = "UNKNOWN"
    causes.append({
        "candidate": "Checkpoint/legacy-AGW structural incompatibility",
        "classification": classification,
        "evidence": compatibility.get("strict_load_error") or compatibility.get(
            "reason", "See checkpoint_compatibility.json"),
    })
    if compatibility.get("nonfinite_values", 0):
        causes.append({
            "candidate": "NaN/Inf already stored in checkpoint parameters",
            "classification": "CONFIRMED",
            "evidence": "Checkpoint contains {} non-finite values.".format(
                compatibility["nonfinite_values"]),
        })
    elif compatibility.get("status") in {"COMPATIBLE", "INSPECTED"}:
        causes.append({
            "candidate": "NaN/Inf already stored in checkpoint parameters",
            "classification": "NOT SUPPORTED",
            "evidence": "No non-finite checkpoint parameter values were found.",
        })

    health = reports["feature"]
    if health.get("status") == "COMPLETED":
        if health.get("nonfinite_rows", 0):
            causes.append({
                "candidate": "Non-finite Stage1 output features",
                "classification": "CONFIRMED",
                "evidence": "{} sampled feature rows are non-finite.".format(
                    health["nonfinite_rows"]),
            })
        else:
            causes.append({
                "candidate": "Non-finite Stage1 output features",
                "classification": "NOT SUPPORTED",
                "evidence": "All sampled feature rows are finite.",
            })
        causes.append({
            "candidate": "Near-constant/collapsed Stage1 representation",
            "classification": ("CONFIRMED" if health.get("near_constant_features")
                               else "NOT SUPPORTED"),
            "evidence": (
                "effective_rank={}, mean_dimension_variance={}".format(
                    health.get("effective_rank", {}).get("value"),
                    health.get("dimension_variance", {}).get("mean"))
            ),
        })
    else:
        causes.append({
            "candidate": "Stage1 output feature failure/collapse",
            "classification": "UNKNOWN",
            "evidence": health.get("reason", "Feature diagnosis was not completed."),
        })

    numerical = reports["numerical"]
    if numerical.get("status") == "COMPLETED":
        risk = (numerical.get("nonfinite_values_after_direct_cube", 0) > 0
                or numerical.get("cube_overflow_risk_values", 0) > 0)
        causes.append({
            "candidate": "Legacy direct-cubic GeM numerical overflow",
            "classification": "STRONGLY SUSPECTED" if risk else "NOT SUPPORTED",
            "evidence": (
                "cube_overflow_risk_values={}, nonfinite_after_cube={}".format(
                    numerical.get("cube_overflow_risk_values"),
                    numerical.get("nonfinite_values_after_direct_cube"))
            ),
        })
    else:
        causes.append({
            "candidate": "Legacy direct-cubic GeM numerical overflow",
            "classification": "UNKNOWN",
            "evidence": numerical.get("reason", "Numerical hooks were not completed."),
        })
    return causes


def _write_summary(output_dir: Path, args, reports: Dict[str, dict],
                   worker_result: dict) -> None:
    causes = _candidate_causes(reports)
    lines = [
        "# CARGO Stage1 Diagnosis Summary",
        "",
        "This report is read-only. It does not train a model or modify the dataset/checkpoint.",
        "",
        "## Inputs",
        "",
        "- Legacy code: `{}`".format(args.legacy_code_dir),
        "- Checkpoint: `{}`".format(args.checkpoint),
        "- Dataset: `{}`".format(args.data_dir),
        "- Device: `{}`".format(args.device),
        "- Seed: `{}`".format(args.seed),
        "- Requested stages: `{}`".format(", ".join(args.stages)),
        "",
        "## Stage status",
        "",
        "| Stage | Status |",
        "|---|---|",
    ]
    for stage in OUTPUT_FILES:
        lines.append("| {} | {} |".format(stage, reports[stage].get("status", "UNKNOWN")))
    lines.extend([
        "",
        "## Candidate causes",
        "",
        "| Candidate | Classification | Evidence |",
        "|---|---|---|",
    ])
    for item in causes:
        evidence = str(item["evidence"]).replace("|", "\\|").replace("\n", " ")
        lines.append("| {} | **{}** | {} |".format(
            item["candidate"], item["classification"], evidence))
    lines.extend([
        "",
        "## Fixed-subset policy",
        "",
        "AA, GG and AG queries are sampled with a fixed seed only after confirming "
        "that each query has a legal positive under the existing evaluator. One legal "
        "positive is retained before seeded distractor sampling. No full-gallery "
        "similarity matrix is constructed for feature-health statistics.",
        "",
        "## Worker process",
        "",
        "- Return code: `{}`".format(worker_result.get("returncode", "NOT RUN")),
        "- Legacy import isolation: a separate Python process places the legacy code "
        "directory before the release checkout on `sys.path`.",
        "- Strict loading: required; no missing/unexpected parameters are ignored.",
        "- Stable AGW fallback: disabled.",
        "- Automatic ImageNet download: disabled.",
        "",
    ])
    (output_dir / "diagnosis_summary.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only diagnosis of a legacy CARGO PCLHD Stage1 checkpoint")
    parser.add_argument("--legacy-code-dir", required=True,
                        help="original code checkout that defines legacy arch=agw")
    parser.add_argument("--checkpoint", required=True,
                        help="Stage1 checkpoint; opened read-only")
    parser.add_argument("--data-dir", required=True,
                        help="CARGO root containing query/ and gallery/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True,
                        help="inference device, for example cuda:0 or cpu")
    parser.add_argument("--stage1-log", default="")
    parser.add_argument("--imagenet-weights", default="",
                        help="optional exact full legacy-AGW initialization state")
    parser.add_argument("--arch", default="agw",
                        help="legacy architecture to verify structurally (default: agw)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-query-per-protocol", type=int, default=32)
    parser.add_argument("--max-gallery-per-protocol", type=int, default=256)
    parser.add_argument("--max-pairs", type=int, default=5000)
    parser.add_argument(
        "--stages", nargs="+", choices=tuple(OUTPUT_FILES) + ("all",),
        default=["all"], help="run all diagnostics or selected independent stages")
    args = parser.parse_args()

    selected = set(OUTPUT_FILES) if "all" in args.stages else set(args.stages)
    args.stages = [stage for stage in OUTPUT_FILES if stage in selected]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for stage, filename in OUTPUT_FILES.items():
        write_json(str(output_dir / filename), _status(
            "NOT RUN", "Stage was not requested" if stage not in selected
            else "Stage has not started"))

    if "oracle" in selected:
        try:
            write_json(str(output_dir / OUTPUT_FILES["oracle"]),
                       run_protocol_oracle(str(Path(args.data_dir).resolve())))
        except Exception as exc:
            write_json(str(output_dir / OUTPUT_FILES["oracle"]), _status(
                "BLOCKED", "{}: {}".format(type(exc).__name__, exc)))

    if "log" in selected:
        if args.stage1_log:
            write_json(str(output_dir / OUTPUT_FILES["log"]),
                       parse_stage1_log(args.stage1_log, max_epoch=4))
        else:
            write_json(str(output_dir / OUTPUT_FILES["log"]), _status(
                "SKIPPED", "No --stage1-log path was supplied"))

    worker_result = {"returncode": "NOT RUN"}
    requested_model_stages = selected & MODEL_STAGES
    if requested_model_stages:
        worker = HERE / "diagnostics" / "legacy_worker.py"
        command = [
            sys.executable, str(worker),
            "--legacy-code-dir", args.legacy_code_dir,
            "--checkpoint", args.checkpoint,
            "--data-dir", args.data_dir,
            "--output-dir", str(output_dir),
            "--device", args.device,
            "--arch", args.arch,
            "--height", str(args.height),
            "--width", str(args.width),
            "--batch-size", str(args.batch_size),
            "--max-query-per-protocol", str(args.max_query_per_protocol),
            "--max-gallery-per-protocol", str(args.max_gallery_per_protocol),
            "--max-pairs", str(args.max_pairs),
            "--seed", str(args.seed),
            "--stages", *sorted(requested_model_stages),
        ]
        if args.imagenet_weights:
            command.extend(["--imagenet-weights", args.imagenet_weights])
        completed = subprocess.run(
            command, cwd=str(PROJECT_ROOT), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        worker_result = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        (output_dir / "legacy_worker.log").write_text(
            "STDOUT\n{}\nSTDERR\n{}".format(
                completed.stdout, completed.stderr), encoding="utf-8")

    reports = _load_outputs(output_dir)
    _write_summary(output_dir, args, reports, worker_result)
    print("CARGO Stage1 diagnosis reports: {}".format(output_dir))
    for stage in OUTPUT_FILES:
        print("  {}: {}".format(stage, reports[stage].get("status", "UNKNOWN")))
    if isinstance(worker_result.get("returncode"), int) and worker_result["returncode"] != 0:
        print("Legacy worker failed; see legacy_worker.log and compatibility report.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
