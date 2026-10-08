"""CPU-only oracle for the exact CARGO protocol and junk-filter logic."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from cargo_baseline.cargo_evaluation import (
    PROTOCOLS,
    _metrics,
    _protocol_samples,
    _read_split,
    _subset_features,
)


def _identity_features(samples: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Assign each PID a deterministic unit vector shared by all its images."""
    pids = sorted({int(sample["pid"]) for sample in samples})
    pid_to_index = {pid: index for index, pid in enumerate(pids)}
    denominator = max(len(pids), 1)
    rows = []
    for sample in samples:
        angle = 2.0 * np.pi * pid_to_index[int(sample["pid"])] / denominator
        rows.append((np.cos(angle), np.sin(angle)))
    return np.asarray(rows, dtype=np.float64)


def evaluate_oracle_samples(
    query: Sequence[Mapping[str, object]],
    gallery: Sequence[Mapping[str, object]],
    protocols: Iterable[str] = PROTOCOLS,
) -> Dict[str, dict]:
    """Evaluate perfect identity features without changing protocol semantics."""
    protocols = tuple(protocols)
    unknown = sorted(set(protocols) - set(PROTOCOLS))
    if unknown:
        raise ValueError("Unknown CARGO protocols: {}".format(unknown))

    all_samples = list(query) + list(gallery)
    all_features = _identity_features(all_samples)
    query_features = all_features[:len(query)]
    gallery_features = all_features[len(query):]
    results: Dict[str, dict] = {}
    for protocol in protocols:
        q_samples, g_samples, domain_camids = _protocol_samples(
            list(query), list(gallery), protocol)
        q_features = _subset_features(list(query), query_features, q_samples)
        g_features = _subset_features(list(gallery), gallery_features, g_samples)
        result = _metrics(
            q_features,
            g_features,
            q_samples,
            g_samples,
            use_domain_camids=domain_camids,
        )
        results[protocol] = {
            "query_images": result["query_images"],
            "gallery_images": result["gallery_images"],
            "valid_queries": result["valid_queries"],
            "skipped_queries": result["skipped_queries"],
            "rank1": float(result["cmc"][0]),
            "mAP": float(result["mAP"]),
            "mINP": float(result["mINP"]),
        }
    return results


def run_protocol_oracle(data_root: str) -> dict:
    query = _read_split(data_root, "query")
    gallery = _read_split(data_root, "gallery")
    results = evaluate_oracle_samples(query, gallery)
    passed = all(
        abs(item["rank1"] - 1.0) < 1e-12
        and abs(item["mAP"] - 1.0) < 1e-12
        for item in results.values()
    )
    return {
        "status": "COMPLETED" if passed else "FAILED",
        "purpose": "Validate protocol filtering independently of a model.",
        "oracle_definition": (
            "All images with the same PID receive an identical unit feature; "
            "different PIDs receive distinct unit features."
        ),
        "pass_condition": "Rank-1 and mAP equal 1.0 for every valid query.",
        "all_protocols_passed": passed,
        "protocols": results,
    }
