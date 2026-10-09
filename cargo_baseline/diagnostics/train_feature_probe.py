"""Ground-truth-only train-split probe for representation diagnosis.

Ground-truth PIDs are used only after inference for sampling and metrics. They
must never enter model forward, clustering, training, or hyperparameter choice.
"""

from __future__ import annotations

from collections import defaultdict
import os.path as osp
import random
import re
from typing import Dict, Mapping, Sequence

import numpy as np


DIAGNOSTIC_GT_ONLY = "DIAGNOSTIC_GT_ONLY"
_PATTERN = re.compile(
    r"^Cam(?P<cam>\d+)_(?P<time>day|night)_(?P<pid>\d+)_(?P<index>\d+)\.jpg$",
    re.IGNORECASE,
)


def parse_cargo_path(path: str) -> dict:
    match = _PATTERN.match(osp.basename(path))
    if match is None:
        raise ValueError("Invalid CARGO image name: {}".format(path))
    camera = int(match.group("cam"))
    return {
        "path": path,
        "pid": int(match.group("pid")),
        "camid": camera - 1,
        "domain": "aerial" if camera <= 5 else "ground",
        "time": match.group("time").lower(),
        "index": int(match.group("index")),
    }


def select_domain_probe(
    samples: Sequence[Mapping[str, object]],
    domain: str,
    max_identities: int = 64,
    gallery_images_per_camera: int = 2,
    seed: int = 1,
) -> dict:
    domain_samples = [sample for sample in samples if sample["domain"] == domain]
    by_pid_camera = defaultdict(lambda: defaultdict(list))
    for sample in domain_samples:
        by_pid_camera[int(sample["pid"])][int(sample["camid"])].append(dict(sample))
    eligible = sorted(pid for pid, cameras in by_pid_camera.items()
                      if len(cameras) >= 2)
    rng = random.Random(seed + (101 if domain == "aerial" else 211))
    rng.shuffle(eligible)
    selected_pids = eligible[:max_identities]
    query = []
    gallery_by_path = {}
    for pid in selected_pids:
        cameras = sorted(by_pid_camera[pid])
        rng.shuffle(cameras)
        query_camera = cameras[0]
        query_candidates = sorted(
            by_pid_camera[pid][query_camera], key=lambda item: item["path"])
        query_sample = query_candidates[rng.randrange(len(query_candidates))]
        query.append(query_sample)
        for camera in cameras[1:]:
            candidates = sorted(
                by_pid_camera[pid][camera], key=lambda item: item["path"])
            rng.shuffle(candidates)
            for sample in candidates[:gallery_images_per_camera]:
                gallery_by_path[sample["path"]] = sample

    gallery = [gallery_by_path[path] for path in sorted(gallery_by_path)]
    query_paths = {sample["path"] for sample in query}
    if query_paths & set(gallery_by_path):
        raise AssertionError("Query and gallery paths overlap")
    return {
        "marker": DIAGNOSTIC_GT_ONLY,
        "domain": domain,
        "query": query,
        "gallery": gallery,
        "total_domain_identities": len(by_pid_camera),
        "eligible_cross_camera_identities": len(eligible),
        "selected_identities": len(selected_pids),
        "valid_query_images": len(query),
        "gallery_images": len(gallery),
        "eligible_identity_coverage": (
            len(selected_pids) / len(eligible) if eligible else 0.0),
        "sampling": {
            "seed": seed,
            "max_identities": max_identities,
            "gallery_images_per_camera": gallery_images_per_camera,
            "rule": (
                "One query is selected per identity from one camera; gallery "
                "positives are selected only from other cameras."
            ),
        },
    }


def select_train_probe(samples: Sequence[Mapping[str, object]], **kwargs) -> dict:
    aerial_pids = {int(item["pid"]) for item in samples if item["domain"] == "aerial"}
    ground_pids = {int(item["pid"]) for item in samples if item["domain"] == "ground"}
    return {
        "marker": DIAGNOSTIC_GT_ONLY,
        "global_pid_semantics": {
            "aerial_identity_count": len(aerial_pids),
            "ground_identity_count": len(ground_pids),
            "shared_raw_pid_count": len(aerial_pids & ground_pids),
            "raw_pid_overlap_present": bool(aerial_pids & ground_pids),
            "cross_domain_probe_run": False,
            "requires_dataset_semantics_confirmation": True,
            "note": (
                "Raw filename PIDs are inspected before any domain-wise relabeling. "
                "PID overlap alone does not prove global cross-domain identity "
                "semantics; no AG train diagnostic is reported without independent "
                "dataset-documentation confirmation."
            ),
        },
        "aa": select_domain_probe(samples, "aerial", **kwargs),
        "gg": select_domain_probe(samples, "ground", **kwargs),
    }


def _summary(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None,
                "min": None, "median": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def evaluate_domain_probe(feature_map: Mapping[str, np.ndarray], probe: dict) -> dict:
    query = probe["query"]
    gallery = probe["gallery"]
    if not query or not gallery:
        return {"status": "BLOCKED", "reason": "No legal query/gallery sample"}
    q = np.stack([feature_map[item["path"]] for item in query]).astype(np.float64)
    g = np.stack([feature_map[item["path"]] for item in gallery]).astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    similarity = q @ g.T
    g_pid = np.asarray([int(item["pid"]) for item in gallery])
    g_cam = np.asarray([int(item["camid"]) for item in gallery])
    rank1 = 0
    rank5 = 0
    aps = []
    positive_similarities = []
    negative_similarities = []
    hardest_positives = []
    hardest_negatives = []
    legal_positive_counts = []
    valid = 0
    for index, sample in enumerate(query):
        same_pid = g_pid == int(sample["pid"])
        same_cam = g_cam == int(sample["camid"])
        legal_positive = same_pid & ~same_cam
        legal_gallery = ~(same_pid & same_cam)
        if not np.any(legal_positive):
            continue
        scores = similarity[index][legal_gallery]
        matches = legal_positive[legal_gallery]
        order = np.argsort(-scores)
        ranked_matches = matches[order]
        positive_positions = np.flatnonzero(ranked_matches)
        rank1 += int(positive_positions[0] < 1)
        rank5 += int(positive_positions[0] < 5)
        cumulative = np.cumsum(ranked_matches, dtype=np.float64)
        aps.append(float(np.mean(
            cumulative[positive_positions] / (positive_positions + 1.0))))
        positives = similarity[index][legal_positive]
        negatives = similarity[index][~same_pid]
        legal_positive_counts.append(int(legal_positive.sum()))
        positive_similarities.extend(positives.tolist())
        negative_similarities.extend(negatives.tolist())
        hardest_positives.append(float(positives.min()))
        hardest_negatives.append(float(negatives.max()))
        valid += 1
    if not valid:
        return {"status": "BLOCKED", "reason": "No query retained a legal positive"}
    return {
        "status": "COMPLETED",
        "marker": DIAGNOSTIC_GT_ONLY,
        "rank1": rank1 / valid,
        "rank5": rank5 / valid,
        "mAP": float(np.mean(aps)),
        "valid_queries": valid,
        "skipped_queries": len(query) - valid,
        "gallery_images": len(gallery),
        "positive_similarity": _summary(positive_similarities),
        "negative_similarity": _summary(negative_similarities),
        "same_pid_cross_camera_similarity": _summary(positive_similarities),
        "different_pid_similarity": _summary(negative_similarities),
        "legal_positive_images_per_query": _summary(legal_positive_counts),
        "mean_similarity_gap": (
            float(np.mean(positive_similarities) - np.mean(negative_similarities))),
        "hardest_positive_similarity": _summary(hardest_positives),
        "hardest_negative_similarity": _summary(hardest_negatives),
        "selected_identities": probe["selected_identities"],
        "eligible_cross_camera_identities": probe[
            "eligible_cross_camera_identities"],
        "eligible_identity_coverage": probe["eligible_identity_coverage"],
        "filter_rule": "same PID and same camera are excluded; query paths are disjoint from gallery",
    }
