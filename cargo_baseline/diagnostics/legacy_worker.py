#!/usr/bin/env python3
"""Isolated legacy-AGW worker used by :mod:`diagnose_stage1`.

This process imports ``clustercontrast`` from ``--legacy-code-dir`` before the
release checkout.  It never trains and never writes to the dataset/checkpoint.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import traceback

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _blocked(reason: str, details: str = "") -> dict:
    result = {"status": "BLOCKED", "reason": reason}
    if details:
        result["details"] = details
    return result


def _configure_imports(legacy_code_dir: str) -> Path:
    legacy = Path(legacy_code_dir).expanduser().resolve()
    expected = legacy / "clustercontrast" / "models" / "__init__.py"
    if not expected.is_file():
        raise FileNotFoundError(
            "Legacy model registry is missing: {}".format(expected))
    sys.path.insert(0, str(legacy))
    sys.path.insert(1, str(PROJECT_ROOT))
    return legacy


def _disable_implicit_downloads() -> None:
    """Force legacy AGW construction to random init before strict loading."""
    legacy_agw = importlib.import_module("clustercontrast.models.agw")

    original = legacy_agw.resnet50_agw

    def no_download_resnet50(*args, **kwargs):
        kwargs["pretrained"] = False
        return original(*args, **kwargs)

    legacy_agw.resnet50_agw = no_download_resnet50


def _make_model(models, arch: str):
    _disable_implicit_downloads()
    if arch not in models.names():
        raise KeyError(
            "Legacy model registry does not provide {!r}; available={}".format(
                arch, models.names()))
    return models.create(
        arch, num_features=0, norm=True, dropout=0,
        num_classes=0, pooling_type="gem")


def _strict_load(base_model, state_dict, validate_strict_state_dict, nn):
    keys = list(state_dict)
    has_module = [key.startswith("module.") for key in keys]
    if any(has_module) and not all(has_module):
        return base_model, {
            "strict_load_attempted": False,
            "strict_load_succeeded": False,
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatches": [],
            "strict_load_error": "Mixed module-prefixed and unprefixed keys",
        }
    candidate = nn.DataParallel(base_model) if keys and all(has_module) else base_model
    result = validate_strict_state_dict(candidate, state_dict)
    loaded = candidate.module if isinstance(candidate, nn.DataParallel) else candidate
    return loaded, result


def _legal_positive(query, gallery, domain_camids: bool) -> bool:
    if int(query["pid"]) != int(gallery["pid"]):
        return False
    q_cam = (0 if query["domain"] == "aerial" else 1) \
        if domain_camids else int(query["camid"])
    g_cam = (0 if gallery["domain"] == "aerial" else 1) \
        if domain_camids else int(gallery["camid"])
    return q_cam != g_cam


def _select_protocol_subset(query, gallery, protocol, max_queries,
                            max_gallery, seed, protocol_samples):
    q_pool, g_pool, domain_camids = protocol_samples(query, gallery, protocol)
    rng = random.Random(seed + sum(map(ord, protocol)))
    candidates = []
    positives = {}
    for sample in q_pool:
        matches = [item for item in g_pool
                   if _legal_positive(sample, item, domain_camids)]
        if matches:
            candidates.append(sample)
            positives[sample["path"]] = matches
    rng.shuffle(candidates)
    selected_queries = candidates[:max_queries]
    required = {}
    for sample in selected_queries:
        matches = list(positives[sample["path"]])
        rng.shuffle(matches)
        required[matches[0]["path"]] = matches[0]
    remaining = [sample for sample in g_pool if sample["path"] not in required]
    rng.shuffle(remaining)
    target = max(max_gallery, len(required))
    selected_gallery = list(required.values()) + remaining[:max(0, target - len(required))]
    return selected_queries, selected_gallery, domain_camids, {
        "eligible_query_images": len(candidates),
        "selected_query_images": len(selected_queries),
        "selected_gallery_images": len(selected_gallery),
        "required_positive_images": len(required),
        "domain_camids": domain_camids,
    }


def _build_fixed_sample(data_dir, max_queries, max_gallery, seed,
                        read_split, protocol_samples):
    query = read_split(data_dir, "query")
    gallery = read_split(data_dir, "gallery")
    selections = {}
    samples = {}
    for protocol in ("aa", "gg", "ag"):
        selected_q, selected_g, domain_camids, metadata = _select_protocol_subset(
            query, gallery, protocol, max_queries, max_gallery, seed,
            protocol_samples)
        if not selected_q:
            raise RuntimeError("No legal query can be sampled for {}".format(protocol))
        selections[protocol] = (selected_q, selected_g, domain_camids)
        metadata["query_paths_sha_order"] = [item["path"] for item in selected_q]
        selections[protocol + "_metadata"] = metadata
        for sample in selected_q + selected_g:
            samples[sample["path"]] = sample
    ordered = [samples[path] for path in sorted(samples)]
    return ordered, selections


def _pil_to_tensor(path, height, width, torch):
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    image = Image.open(path).convert("RGB").resize((width, height), resampling)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


class _TensorAccumulator:
    def __init__(self):
        self.values = 0
        self.finite_values = 0
        self.nan = 0
        self.posinf = 0
        self.neginf = 0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.abs_max = 0.0
        self.sum = 0.0
        self.sum_sq = 0.0

    def update(self, tensor, torch):
        value = tensor.detach().float()
        self.values += int(value.numel())
        self.nan += int(torch.isnan(value).sum().item())
        self.posinf += int(torch.isposinf(value).sum().item())
        self.neginf += int(torch.isneginf(value).sum().item())
        finite = value[torch.isfinite(value)]
        self.finite_values += int(finite.numel())
        if finite.numel():
            self.minimum = min(self.minimum, float(finite.min().item()))
            self.maximum = max(self.maximum, float(finite.max().item()))
            self.abs_max = max(self.abs_max, float(finite.abs().max().item()))
            self.sum += float(finite.double().sum().item())
            self.sum_sq += float(finite.double().square().sum().item())

    def result(self):
        mean = self.sum / self.finite_values if self.finite_values else None
        variance = (self.sum_sq / self.finite_values - mean * mean
                    if self.finite_values else None)
        return {
            "values": self.values,
            "finite_values": self.finite_values,
            "nan_values": self.nan,
            "positive_inf_values": self.posinf,
            "negative_inf_values": self.neginf,
            "min": None if self.minimum == math.inf else self.minimum,
            "max": None if self.maximum == -math.inf else self.maximum,
            "abs_max": self.abs_max,
            "mean": mean,
            "std": math.sqrt(max(variance, 0.0)) if variance is not None else None,
        }


class _NumericalRecorder:
    def __init__(self, model, torch):
        self.torch = torch
        self.gem_input = _TensorAccumulator()
        self.gem_output = _TensorAccumulator()
        self.bn_input = _TensorAccumulator()
        self.bn_output = _TensorAccumulator()
        self.cube_nonfinite = 0
        self.cube_risk_values = 0
        self.gem_bn_max_abs_difference = 0.0
        self.latest_recomputed_gem = None
        self.handles = []
        modules = dict(model.named_modules())
        layer4 = []
        for name, module in modules.items():
            match = re.fullmatch(r"base_resnet\.base\.layer4\.(\d+)", name)
            if match:
                layer4.append((int(match.group(1)), name, module))
        if not layer4 or "bottleneck" not in modules:
            raise RuntimeError("Could not locate legacy layer4 output and bottleneck")
        _, self.gem_module_name, gem_module = max(layer4)
        self.bn_module_name = "bottleneck"
        self.handles.append(gem_module.register_forward_hook(self._gem_hook))
        self.handles.append(modules["bottleneck"].register_forward_pre_hook(self._bn_pre_hook))
        self.handles.append(modules["bottleneck"].register_forward_hook(self._bn_post_hook))
        self.bn = modules["bottleneck"]

    def _gem_hook(self, module, inputs, output):
        torch = self.torch
        value = output if isinstance(output, torch.Tensor) else output[0]
        self.gem_input.update(value, torch)
        threshold = float(torch.finfo(torch.float32).max ** (1.0 / 3.0))
        self.cube_risk_values += int((value.detach().abs() > threshold).sum().item())
        cube = value.detach().float().pow(3)
        self.cube_nonfinite += int((~torch.isfinite(cube)).sum().item())
        pooled = (torch.mean(cube.flatten(2), dim=-1) + 1e-12).pow(1.0 / 3.0)
        self.latest_recomputed_gem = pooled
        self.gem_output.update(pooled, torch)

    def _bn_pre_hook(self, module, inputs):
        value = inputs[0]
        self.bn_input.update(value, self.torch)
        with self.torch.no_grad():
            cube = self.latest_recomputed_gem
            if cube is not None and cube.shape == value.shape:
                finite = self.torch.isfinite(cube) & self.torch.isfinite(value)
                if bool(finite.any()):
                    difference = float((cube[finite] - value[finite]).abs().max().item())
                    self.gem_bn_max_abs_difference = max(
                        self.gem_bn_max_abs_difference, difference)

    def _bn_post_hook(self, module, inputs, output):
        self.bn_output.update(output, self.torch)

    def close(self):
        for handle in self.handles:
            handle.remove()

    def result(self, model):
        running_mean = self.bn.running_mean.detach().float().cpu().numpy()
        running_var = self.bn.running_var.detach().float().cpu().numpy()
        try:
            source = inspect.getsource(model.forward)
        except (OSError, TypeError):
            source = ""
        threshold = float(np.finfo(np.float32).max ** (1.0 / 3.0))
        return {
            "status": "COMPLETED",
            "hook_modules": {
                "gem_input": self.gem_module_name,
                "bn_neck": self.bn_module_name,
            },
            "legacy_forward_contains_direct_cube": "x**p" in source.replace(" ", ""),
            "float32_cube_overflow_abs_threshold": threshold,
            "gem_input": self.gem_input.result(),
            "recomputed_legacy_gem_output": self.gem_output.result(),
            "bn_input": self.bn_input.result(),
            "bn_output": self.bn_output.result(),
            "cube_overflow_risk_values": self.cube_risk_values,
            "nonfinite_values_after_direct_cube": self.cube_nonfinite,
            "recomputed_gem_vs_bn_input_max_abs_difference": (
                self.gem_bn_max_abs_difference),
            "bn_running_mean": {
                "finite": bool(np.isfinite(running_mean).all()),
                "min": float(np.nanmin(running_mean)),
                "max": float(np.nanmax(running_mean)),
                "mean": float(np.nanmean(running_mean)),
                "std": float(np.nanstd(running_mean)),
            },
            "bn_running_var": {
                "finite": bool(np.isfinite(running_var).all()),
                "min": float(np.nanmin(running_var)),
                "max": float(np.nanmax(running_var)),
                "mean": float(np.nanmean(running_var)),
                "std": float(np.nanstd(running_var)),
            },
            "interpretation_rule": (
                "Finite activations alone do not establish healthy representation; "
                "feature variance, effective rank and identity cosine gap must also be inspected."
            ),
        }


def _extract(model, samples, args, torch, recorder=None):
    model.eval()
    outputs = {}
    batch_size = args.batch_size
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            images = torch.stack([
                _pil_to_tensor(sample["path"], args.height, args.width, torch)
                for sample in batch
            ]).to(args.device)
            for domain, modal in (("aerial", 2), ("ground", 1)):
                positions = [index for index, sample in enumerate(batch)
                             if sample["domain"] == domain]
                if not positions:
                    continue
                selected = images[positions]
                original = model(selected, selected, modal)
                flipped_images = torch.flip(selected, dims=(3,))
                flipped = model(flipped_images, flipped_images, modal)
                if not isinstance(original, torch.Tensor) or original.ndim != 2:
                    raise TypeError("Legacy model eval output is not a 2-D tensor")
                fused = (original + flipped) / 2.0
                fused = torch.nn.functional.normalize(fused, dim=1)
                for local_index, position in enumerate(positions):
                    outputs[batch[position]["path"]] = fused[local_index].cpu().numpy()
    return outputs


def _evaluate_selected(feature_map, selections, metrics):
    reports = {}
    for protocol in ("aa", "gg", "ag"):
        query, gallery, domain_camids = selections[protocol]
        q_features = np.stack([feature_map[item["path"]] for item in query])
        g_features = np.stack([feature_map[item["path"]] for item in gallery])
        result = metrics(q_features, g_features, query, gallery,
                         use_domain_camids=domain_camids)
        reports[protocol] = {
            "rank1": float(result["cmc"][0]),
            "mAP": float(result["mAP"]),
            "mINP": float(result["mINP"]),
            "valid_queries": int(result["valid_queries"]),
            "skipped_queries": int(result["skipped_queries"]),
            "query_images": int(result["query_images"]),
            "gallery_images": int(result["gallery_images"]),
        }
    return reports


def _health(feature_map, samples, summarize_feature_health, seed, max_pairs):
    matrix = np.stack([feature_map[item["path"]] for item in samples])
    return summarize_feature_health(
        matrix,
        [item["pid"] for item in samples],
        [item["domain"] for item in samples],
        seed=seed,
        max_pairs=max_pairs,
    )


def _load_exact_model(models, arch, state_dict, validate, nn):
    model = _make_model(models, arch)
    model, result = _strict_load(model, state_dict, validate, nn)
    return model, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-code-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--arch", default="agw")
    parser.add_argument("--imagenet-weights", default="")
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-query-per-protocol", type=int, default=32)
    parser.add_argument("--max-gallery-per-protocol", type=int, default=256)
    parser.add_argument("--max-pairs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--stages", nargs="+",
        choices=("checkpoint", "feature", "initialization", "numerical"),
        default=("checkpoint", "feature", "initialization", "numerical"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compatibility_path = output_dir / "checkpoint_compatibility.json"
    feature_path = output_dir / "feature_health.json"
    initialization_path = output_dir / "initialization_comparison.json"
    numerical_path = output_dir / "numerical_diagnostics.json"
    compatibility = None
    try:
        _configure_imports(args.legacy_code_dir)
        import torch
        from torch import nn
        from clustercontrast import models
        from cargo_baseline.cargo_evaluation import (
            _metrics, _protocol_samples, _read_split)
        from cargo_baseline.diagnostics.feature_health import summarize_feature_health
        from cargo_baseline.diagnostics.model_compatibility import (
            inspect_checkpoint, validate_strict_state_dict)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        checkpoint, state_dict, compatibility = inspect_checkpoint(args.checkpoint)
        metadata_backbone = compatibility["backbone_metadata"]
        compatibility.update({
            "requested_legacy_arch": args.arch,
            "legacy_code_dir": str(Path(args.legacy_code_dir).resolve()),
            "architecture_selection": (
                "checkpoint metadata plus strict structural validation"
                if metadata_backbone != "UNKNOWN"
                else "explicit/default --arch plus strict structural validation"
            ),
            "stable_agw_load_attempted": False,
            "implicit_weight_downloads_disabled": True,
        })
        if metadata_backbone != "UNKNOWN" and metadata_backbone != args.arch:
            raise RuntimeError(
                "Checkpoint backbone metadata {!r} conflicts with --arch {!r}".format(
                    metadata_backbone, args.arch))
        base_model = _make_model(models, args.arch)
        model, strict = _strict_load(
            base_model, state_dict, validate_strict_state_dict, nn)
        compatibility.update(strict)
        compatibility["status"] = ("COMPATIBLE" if strict["strict_load_succeeded"]
                                   else "INCOMPATIBLE")
        _write(compatibility_path, compatibility)
        if not strict["strict_load_succeeded"]:
            raise RuntimeError("Checkpoint failed strict=True legacy AGW loading")
        model = model.to(device)

        requested = set(args.stages)
        if requested == {"checkpoint"}:
            return 0

        samples, selections = _build_fixed_sample(
            str(Path(args.data_dir).resolve()),
            args.max_query_per_protocol,
            args.max_gallery_per_protocol,
            args.seed,
            _read_split,
            _protocol_samples,
        )
        recorder = _NumericalRecorder(model, torch) if "numerical" in requested else None
        try:
            feature_map = _extract(model, samples, args, torch, recorder)
        finally:
            if recorder is not None:
                recorder.close()
        if recorder is not None:
            numerical = recorder.result(model)
            numerical["sample_count"] = len(samples)
            _write(numerical_path, numerical)

        health = _health(
            feature_map, samples, summarize_feature_health,
            args.seed, args.max_pairs)
        if health["nonfinite_rows"]:
            health["protocol_performance_on_fixed_subset"] = {
                "status": "BLOCKED",
                "reason": "Retrieval metrics are not reported for non-finite features.",
            }
        else:
            health["protocol_performance_on_fixed_subset"] = _evaluate_selected(
                feature_map, selections, _metrics)
        health["sampling"] = {
            "seed": args.seed,
            "rule": (
                "For AA, GG and AG, seeded query sampling retains only queries "
                "with at least one legal positive; one legal positive per query "
                "is forced into the gallery before seeded distractor sampling."
            ),
            "unique_inference_images": len(samples),
            "max_query_per_protocol": args.max_query_per_protocol,
            "max_gallery_per_protocol": args.max_gallery_per_protocol,
            "protocols": {
                key: selections[key + "_metadata"] for key in ("aa", "gg", "ag")
            },
        }
        if "feature" in requested:
            _write(feature_path, health)

        if "initialization" not in requested:
            initialization = None
        elif not args.imagenet_weights:
            initialization = {
                "status": "SKIPPED",
                "reason": (
                    "No --imagenet-weights path was supplied. Random initialization "
                    "is not used as a substitute for ImageNet pretraining."
                ),
                "stage1_checkpoint": {
                    "protocols": health["protocol_performance_on_fixed_subset"],
                    "effective_rank": health["effective_rank"],
                    "dimension_variance": health["dimension_variance"],
                    "identity_separability": health["identity_separability"],
                },
            }
        else:
            try:
                _, init_state, init_meta = inspect_checkpoint(args.imagenet_weights)
                init_model, init_strict = _load_exact_model(
                    models, args.arch, init_state, validate_strict_state_dict, nn)
                if not init_strict["strict_load_succeeded"]:
                    raise RuntimeError(init_strict["strict_load_error"])
                init_model = init_model.to(device)
                init_features = _extract(init_model, samples, args, torch)
                init_health = _health(
                    init_features, samples, summarize_feature_health,
                    args.seed, args.max_pairs)
                initialization = {
                    "status": "COMPLETED",
                    "weight_file": str(Path(args.imagenet_weights).resolve()),
                    "weight_metadata": init_meta,
                    "load_policy": "Exact same legacy AGW topology with strict=True.",
                    "fixed_sample_reused": True,
                    "imagenet_initialization": {
                        "protocols": _evaluate_selected(
                            init_features, selections, _metrics),
                        "effective_rank": init_health["effective_rank"],
                        "dimension_variance": init_health["dimension_variance"],
                        "identity_separability": init_health["identity_separability"],
                    },
                    "stage1_checkpoint": {
                        "protocols": health["protocol_performance_on_fixed_subset"],
                        "effective_rank": health["effective_rank"],
                        "dimension_variance": health["dimension_variance"],
                        "identity_separability": health["identity_separability"],
                    },
                }
            except Exception as exc:
                initialization = {
                    "status": "SKIPPED",
                    "reason": (
                        "Provided initialization weights could not be verified as an "
                        "exact legacy AGW initialization under strict=True."
                    ),
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }
        if initialization is not None:
            _write(initialization_path, initialization)
        return 0
    except Exception as exc:
        detail = traceback.format_exc()
        if compatibility is None:
            compatibility = _blocked(
                "Legacy checkpoint/model compatibility could not be established", detail)
        else:
            compatibility["status"] = "INCOMPATIBLE"
            compatibility["fatal_error"] = "{}: {}".format(type(exc).__name__, exc)
            compatibility["traceback"] = detail
        _write(compatibility_path, compatibility)
        for path, reason in (
            (feature_path, "Feature inference requires a strictly compatible legacy model"),
            (numerical_path, "Numerical hooks require a strictly compatible legacy model"),
            (initialization_path, "Initialization comparison requires Stage1 inference"),
        ):
            if not path.exists():
                _write(path, _blocked(reason, "{}: {}".format(type(exc).__name__, exc)))
        print(detail, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
