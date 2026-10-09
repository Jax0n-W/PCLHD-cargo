"""Historical-artifact inventory and post-hoc pseudo-label quality metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import os
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np


RECLUSTER_MARKER = "CHECKPOINT_RECLUSTER_NOT_HISTORICAL_EPOCH_LABELS"
NO_HISTORY = "HISTORICAL_PSEUDO_LABELS_NOT_AVAILABLE"
_ARTIFACT_SUFFIXES = {".npy", ".npz", ".pkl", ".pickle", ".pt", ".pth", ".json"}
_ARTIFACT_WORDS = ("pseudo", "cluster", "label", "feature", "jaccard")


def inventory_historical_artifacts(
    checkpoint: str,
    extra_roots: Iterable[str] = (),
    max_files: int = 10000,
) -> dict:
    roots = [Path(checkpoint).expanduser().resolve().parent]
    roots.extend(Path(root).expanduser().resolve() for root in extra_roots)
    found = []
    scanned = 0
    for root in roots:
        if not root.is_dir():
            continue
        for current, dirs, files in os.walk(root):
            dirs[:] = [name for name in dirs if name not in {".git", "__pycache__"}]
            for filename in files:
                scanned += 1
                if scanned > max_files:
                    break
                path = Path(current) / filename
                lower = filename.lower()
                if (path.suffix.lower() in _ARTIFACT_SUFFIXES
                        and any(word in lower for word in _ARTIFACT_WORDS)
                        and path.resolve() != Path(checkpoint).expanduser().resolve()):
                    found.append({
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "possible_epoch": (
                            int(match.group(1)) if (match := re.search(
                                r"(?:epoch|ep)[_-]?(\d+)", lower)) else None),
                        "classification": "CANDIDATE_ONLY_NOT_VALIDATED",
                    })
            if scanned > max_files:
                break
    return {
        "status": "CANDIDATES_FOUND" if found else NO_HISTORY,
        "roots": [str(root) for root in roots],
        "files_scanned": scanned,
        "scan_limit_reached": scanned > max_files,
        "candidate_count": len(found),
        "epoch_0_to_4_candidate_count": sum(
            item["possible_epoch"] in range(5) for item in found
            if item["possible_epoch"] is not None),
        "candidates": found,
        "warning": (
            "A filename match does not prove that an artifact contains historical "
            "epoch pseudo labels; provenance must be validated before reuse."
        ),
    }


def estimate_distance_memory(sample_count: int, bytes_per_value: int = 4,
                             overhead_factor: float = 2.5) -> dict:
    matrix_bytes = int(sample_count) * int(sample_count) * bytes_per_value
    return {
        "sample_count": int(sample_count),
        "matrix_shape": [int(sample_count), int(sample_count)],
        "single_float32_matrix_bytes": matrix_bytes,
        "single_float32_matrix_gib": matrix_bytes / (1024 ** 3),
        "estimated_peak_bytes": int(math.ceil(matrix_bytes * overhead_factor)),
        "estimated_peak_gib": matrix_bytes * overhead_factor / (1024 ** 3),
        "overhead_factor": overhead_factor,
    }


def available_memory() -> dict:
    try:
        import psutil
        value = int(psutil.virtual_memory().available)
        return {"status": "AVAILABLE", "bytes": value,
                "gib": value / (1024 ** 3), "source": "psutil"}
    except Exception:
        return {"status": "UNKNOWN", "bytes": None, "gib": None,
                "source": "psutil unavailable"}


def _comb2(value: int) -> int:
    return value * (value - 1) // 2


def compute_pseudo_label_quality(
    labels: Sequence[int],
    pids: Sequence[int],
    camids: Sequence[int],
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    pids = np.asarray(pids, dtype=np.int64)
    camids = np.asarray(camids, dtype=np.int64)
    if not (len(labels) == len(pids) == len(camids)):
        raise ValueError("labels, pids and camids must have equal length")
    assigned = labels != -1
    assigned_count = int(assigned.sum())
    clusters = sorted(int(label) for label in np.unique(labels[assigned]))
    sizes = [int(np.sum(labels == label)) for label in clusters]

    purity_numerator = 0
    predicted_pairs = 0
    true_positive_pairs = 0
    merged_identities = {}
    camera_counts = {}
    for label in clusters:
        mask = labels == label
        pid_counts = Counter(pids[mask].tolist())
        purity_numerator += max(pid_counts.values())
        predicted_pairs += _comb2(int(mask.sum()))
        true_positive_pairs += sum(_comb2(count) for count in pid_counts.values())
        if len(pid_counts) > 1:
            merged_identities[str(label)] = {
                "identity_count": len(pid_counts),
                "sample_count": int(mask.sum()),
                "pids": sorted(pid_counts),
            }
        camera_counts[str(label)] = len(np.unique(camids[mask]))

    total_true_pairs = sum(_comb2(count) for count in Counter(pids.tolist()).values())
    false_positive_pairs = predicted_pairs - true_positive_pairs
    false_negative_pairs = total_true_pairs - true_positive_pairs
    precision = (true_positive_pairs / predicted_pairs if predicted_pairs else 0.0)
    recall = (true_positive_pairs / total_true_pairs if total_true_pairs else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    pid_clusters = defaultdict(set)
    for label, pid in zip(labels, pids):
        if label != -1:
            pid_clusters[int(pid)].add(int(label))
    split_identities = {
        str(pid): sorted(cluster_ids)
        for pid, cluster_ids in pid_clusters.items() if len(cluster_ids) > 1
    }

    ari = None
    nmi = None
    metric_scope = "assigned samples only; DBSCAN noise (-1) excluded"
    if assigned_count >= 2 and len(clusters) >= 1:
        try:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            ari = float(adjusted_rand_score(pids[assigned], labels[assigned]))
            nmi = float(normalized_mutual_info_score(pids[assigned], labels[assigned]))
        except Exception:
            pass
    size_array = np.asarray(sizes, dtype=np.float64)
    return {
        "marker": RECLUSTER_MARKER,
        "samples": int(len(labels)),
        "assigned_samples": assigned_count,
        "noise_samples": int((~assigned).sum()),
        "coverage": assigned_count / len(labels) if len(labels) else 0.0,
        "noise_rate": float((~assigned).mean()) if len(labels) else 0.0,
        "clusters": len(clusters),
        "cluster_size": {
            "min": int(size_array.min()) if sizes else None,
            "mean": float(size_array.mean()) if sizes else None,
            "median": float(np.median(size_array)) if sizes else None,
            "max": int(size_array.max()) if sizes else None,
            "p95": float(np.quantile(size_array, 0.95)) if sizes else None,
        },
        "largest_cluster_ratio": (max(sizes) / assigned_count
                                  if assigned_count and sizes else 0.0),
        "cluster_purity": (purity_numerator / assigned_count
                           if assigned_count else 0.0),
        "pairwise": {
            "true_positive_pairs": true_positive_pairs,
            "false_positive_pairs": false_positive_pairs,
            "false_negative_pairs": false_negative_pairs,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "ARI": ari,
        "NMI": nmi,
        "ARI_NMI_scope": metric_scope,
        "split_identity_count": len(split_identities),
        "split_identities": split_identities,
        "merged_cluster_count": len(merged_identities),
        "merged_clusters": merged_identities,
        "cross_camera_clusters": sum(count >= 2 for count in camera_counts.values()),
        "cross_camera_cluster_ratio": (
            sum(count >= 2 for count in camera_counts.values()) / len(clusters)
            if clusters else 0.0),
        "cluster_camera_counts": camera_counts,
        "ground_truth_policy": (
            "Ground truth is read only after DBSCAN labels are fixed and is used "
            "solely for DIAGNOSTIC_GT_ONLY quality metrics."
        ),
    }
