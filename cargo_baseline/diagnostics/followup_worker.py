#!/usr/bin/env python3
"""Isolated worker for the Stage1 train-split probe and optional reclustering."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import random
import sys
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from legacy_worker import _configure_imports, _extract, _make_model, _strict_load


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_engine(legacy_root: Path):
    path = legacy_root / "cargo_baseline" / "_train_cargo_engine.py"
    if not path.is_file():
        raise FileNotFoundError("Legacy CARGO engine is missing: {}".format(path))
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_cargo_followup_legacy_engine", str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _strict_model(weight_path, args, models, nn, inspect_checkpoint, validate):
    _, state, metadata = inspect_checkpoint(weight_path)
    base = _make_model(models, args.arch)
    model, strict = _strict_load(base, state, validate, nn)
    if not strict["strict_load_succeeded"]:
        raise RuntimeError(
            "Strict legacy AGW load failed for {}: {}".format(
                weight_path, strict.get("strict_load_error")))
    metadata.update(strict)
    metadata["status"] = "COMPATIBLE"
    return model.to(args.device), metadata


def _metric_delta(after: dict, before: dict) -> dict:
    keys = ("rank1", "rank5", "mAP", "mean_similarity_gap")
    return {key: after[key] - before[key] for key in keys
            if isinstance(after.get(key), (int, float))
            and isinstance(before.get(key), (int, float))}


def _run_sample(args, torch, models, nn, inspect_checkpoint, validate,
                read_split, parse_cargo_path, select_train_probe,
                evaluate_domain_probe) -> dict:
    if not args.imagenet_weights:
        raise ValueError("--imagenet-weights is required for sample comparison")
    train = [parse_cargo_path(item["path"])
             for item in read_split(args.data_dir, "train")]
    probe = select_train_probe(
        train, max_identities=args.max_identities,
        gallery_images_per_camera=args.gallery_images_per_camera,
        seed=args.seed)
    selected = {}
    for domain in ("aa", "gg"):
        for sample in probe[domain]["query"] + probe[domain]["gallery"]:
            selected[sample["path"]] = sample
    samples = [selected[path] for path in sorted(selected)]

    stage1_model, stage1_meta = _strict_model(
        args.checkpoint, args, models, nn, inspect_checkpoint, validate)
    stage1_features = _extract(stage1_model, samples, args, torch)
    del stage1_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    imagenet_model, imagenet_meta = _strict_model(
        args.imagenet_weights, args, models, nn, inspect_checkpoint, validate)
    imagenet_features = _extract(imagenet_model, samples, args, torch)
    del imagenet_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = {
        "status": "COMPLETED",
        "marker": "DIAGNOSTIC_GT_ONLY",
        "ground_truth_boundary": (
            "Real train PIDs and camera IDs are used only for deterministic probe "
            "construction and post-inference metrics; they are not passed to either model."
        ),
        "sampling": probe,
        "unique_inference_images": len(samples),
        "stage1_checkpoint_compatibility": stage1_meta,
        "imagenet_initialization_compatibility": imagenet_meta,
        "imagenet_initialization": {},
        "stage1_checkpoint": {},
        "delta_stage1_minus_imagenet": {},
        "inference_contract": {
            "aerial_modal": 2,
            "ground_modal": 1,
            "resize": [args.height, args.width],
            "horizontal_flip_tta": True,
            "fusion": "mean(original, horizontal_flip), then L2 normalization",
        },
    }
    for domain in ("aa", "gg"):
        before = evaluate_domain_probe(imagenet_features, probe[domain])
        after = evaluate_domain_probe(stage1_features, probe[domain])
        result["imagenet_initialization"][domain] = before
        result["stage1_checkpoint"][domain] = after
        result["delta_stage1_minus_imagenet"][domain] = _metric_delta(after, before)
    return result


def _memory_gate(name, sample_count, args, estimate_distance_memory,
                 available_memory) -> dict:
    estimate = estimate_distance_memory(sample_count, overhead_factor=args.memory_overhead_factor)
    available = available_memory()
    result = {"domain": name, "estimate": estimate, "available_host_memory": available,
              "threshold_fraction": args.max_memory_fraction}
    if not available.get("bytes"):
        result["status"] = "OVERRIDDEN" if args.force_memory_risk else "REFUSED"
        if not args.force_memory_risk:
            raise MemoryError(
                "Available host memory could not be measured. Install psutil or "
                "use --force-memory-risk only after manual capacity review.")
    elif estimate["estimated_peak_bytes"] > (
            available["bytes"] * args.max_memory_fraction):
        result["status"] = "OVERRIDDEN" if args.force_memory_risk else "REFUSED"
        if not args.force_memory_risk:
            raise MemoryError(
                "{} reclustering estimated peak {:.2f} GiB exceeds {:.0%} of "
                "available host memory ({:.2f} GiB). Use --force-memory-risk only "
                "after manual review.".format(
                    name, estimate["estimated_peak_gib"], args.max_memory_fraction,
                    available["gib"]))
    else:
        result["status"] = "PASSED"
    return result


def _run_full_cluster(args, torch, model, legacy_root,
                      compute_pseudo_label_quality, estimate_distance_memory,
                      available_memory, parse_cargo_path) -> dict:
    required = {
        "aerial_eps": args.aerial_eps, "ground_eps": args.ground_eps,
        "k1": args.k1, "k2": args.k2, "min_samples": args.min_samples,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Full reclustering requires explicit frozen parameters: {}".format(
                ", ".join(missing)))
    engine = _load_engine(legacy_root)
    from sklearn.cluster import DBSCAN

    reports = {
        "status": "COMPLETED",
        "marker": "CHECKPOINT_RECLUSTER_NOT_HISTORICAL_EPOCH_LABELS",
        "warning": (
            "Labels were recomputed from the supplied checkpoint. They are not the "
            "historical pseudo labels used during the original training epoch."
        ),
        "configuration": required,
        "configuration_provenance": (
            "All eps/k1/k2/min_samples values were supplied explicitly by the "
            "operator. They must be checked against the historical Stage1 command/log; "
            "the diagnostic does not infer or tune them."
        ),
        "domains": {},
    }
    domains = (
        ("aerial", "cargo_aerial", 2, args.aerial_eps),
        ("ground", "cargo_ground", 1, args.ground_eps),
    )
    for domain, dataset_name, modal, eps in domains:
        dataset = engine.get_data(dataset_name, args.data_dir, trial=1)
        ordered = sorted(dataset.train)
        reports["domains"][domain] = {
            "memory_gate": _memory_gate(
                domain, len(ordered), args, estimate_distance_memory, available_memory)}
        loader = engine.get_test_loader(
            dataset, args.height, args.width, args.batch_size, args.workers,
            testset=ordered)
        feature_dict, _ = engine.extract_features(
            model, loader, print_freq=args.print_freq, mode=modal)
        features = torch.cat(
            [feature_dict[path].unsqueeze(0) for path, _, _ in ordered], dim=0)
        if not bool(torch.isfinite(features).all()):
            raise RuntimeError("Non-finite {} features block reclustering".format(domain))
        distance = engine.compute_jaccard_distance(
            features, k1=args.k1, k2=args.k2, search_option=3)
        labels = DBSCAN(
            eps=eps, min_samples=args.min_samples,
            metric="precomputed", n_jobs=-1).fit_predict(distance)
        parsed = [parse_cargo_path(path) for path, _, _ in ordered]
        reports["domains"][domain].update(compute_pseudo_label_quality(
            labels, [item["pid"] for item in parsed],
            [item["camid"] for item in parsed]))
        del distance, features, feature_dict, loader, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-code-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--imagenet-weights", default="")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--arch", default="agw")
    parser.add_argument("--stages", nargs="+", choices=("sample", "full-cluster"), required=True)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-identities", type=int, default=64)
    parser.add_argument("--gallery-images-per-camera", type=int, default=2)
    parser.add_argument("--aerial-eps", type=float)
    parser.add_argument("--ground-eps", type=float)
    parser.add_argument("--k1", type=int)
    parser.add_argument("--k2", type=int)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--memory-overhead-factor", type=float, default=2.5)
    parser.add_argument("--max-memory-fraction", type=float, default=0.70)
    parser.add_argument("--force-memory-risk", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_path = output / "train_feature_comparison.json"
    cluster_path = output / "pseudo_label_quality.json"
    try:
        legacy_root = _configure_imports(args.legacy_code_dir)
        import torch
        from torch import nn
        from clustercontrast import models
        from cargo_baseline.cargo_evaluation import _read_split
        from cargo_baseline.diagnostics.model_compatibility import (
            inspect_checkpoint, validate_strict_state_dict)
        from cargo_baseline.diagnostics.pseudo_label_audit import (
            available_memory, compute_pseudo_label_quality,
            estimate_distance_memory)
        from cargo_baseline.diagnostics.train_feature_probe import (
            evaluate_domain_probe, parse_cargo_path, select_train_probe)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if torch.cuda.is_available():
            requested_device = torch.device(args.device)
            if requested_device.type == "cuda":
                torch.cuda.set_device(requested_device)
            torch.cuda.manual_seed_all(args.seed)

        if "sample" in args.stages:
            _write(sample_path, _run_sample(
                args, torch, models, nn, inspect_checkpoint,
                validate_strict_state_dict, _read_split, parse_cargo_path,
                select_train_probe, evaluate_domain_probe))

        if "full-cluster" in args.stages:
            model, compatibility = _strict_model(
                args.checkpoint, args, models, nn,
                inspect_checkpoint, validate_strict_state_dict)
            report = _run_full_cluster(
                args, torch, model, legacy_root, compute_pseudo_label_quality,
                estimate_distance_memory, available_memory, parse_cargo_path)
            report["checkpoint_compatibility"] = compatibility
            _write(cluster_path, report)
        return 0
    except Exception as exc:
        report = {"status": "BLOCKED", "error": "{}: {}".format(
            type(exc).__name__, exc), "traceback": traceback.format_exc()}
        target = cluster_path if "full-cluster" in args.stages else sample_path
        _write(target, report)
        print(report["traceback"], file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
