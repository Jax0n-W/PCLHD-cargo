"""Numerical and discriminative health summaries for feature matrices."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None,
                "min": None, "p05": None, "median": None,
                "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def _effective_rank(features: np.ndarray, seed: int, max_rows: int = 512) -> dict:
    if features.shape[0] < 2:
        return {"value": 0.0, "rows_used": int(features.shape[0])}
    rng = np.random.default_rng(seed)
    if features.shape[0] > max_rows:
        indices = np.sort(rng.choice(features.shape[0], max_rows, replace=False))
        features = features[indices]
    centered = features - features.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    if total <= 0:
        value = 0.0
    else:
        probabilities = eigenvalues[eigenvalues > 0] / total
        value = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    return {"value": value, "rows_used": int(features.shape[0])}


def _sample_pairs(
    pids: np.ndarray,
    rng: np.random.Generator,
    max_pairs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    groups = {pid: np.flatnonzero(pids == pid) for pid in np.unique(pids)}
    eligible = [indices for indices in groups.values() if len(indices) >= 2]
    same = []
    if eligible:
        for _ in range(max_pairs):
            group = eligible[int(rng.integers(len(eligible)))]
            pair = rng.choice(group, 2, replace=False)
            same.append((int(pair[0]), int(pair[1])))

    different = []
    if len(groups) >= 2:
        attempts = 0
        while len(different) < max_pairs and attempts < max_pairs * 20:
            pair = rng.integers(0, len(pids), size=2)
            attempts += 1
            if pair[0] != pair[1] and pids[pair[0]] != pids[pair[1]]:
                different.append((int(pair[0]), int(pair[1])))
    return (np.asarray(same, dtype=np.int64).reshape(-1, 2),
            np.asarray(different, dtype=np.int64).reshape(-1, 2))


def _pair_cosines(normalized: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.sum(normalized[pairs[:, 0]] * normalized[pairs[:, 1]], axis=1)


def _separability(
    features: np.ndarray,
    pids: np.ndarray,
    seed: int,
    max_pairs: int,
) -> dict:
    normalized = _normalize(features)
    same_pairs, different_pairs = _sample_pairs(
        pids, np.random.default_rng(seed), max_pairs)
    same = _pair_cosines(normalized, same_pairs)
    different = _pair_cosines(normalized, different_pairs)
    gap = None
    if same.size and different.size:
        gap = float(same.mean() - different.mean())
    return {
        "same_pid_cosine": _distribution(same),
        "different_pid_cosine": _distribution(different),
        "mean_cosine_gap": gap,
        "sampling": "Seeded pair sampling; no full pairwise matrix is built.",
    }


def summarize_feature_health(
    features: np.ndarray,
    pids: Sequence[int],
    domains: Optional[Sequence[str]] = None,
    seed: int = 1,
    max_pairs: int = 5000,
) -> dict:
    features = np.asarray(features, dtype=np.float64)
    pids = np.asarray(pids)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if len(features) != len(pids):
        raise ValueError("features and pids must contain the same number of rows")
    if domains is None:
        domains = np.asarray(["all"] * len(features))
    else:
        domains = np.asarray(domains)
    if len(domains) != len(features):
        raise ValueError("domains and features must contain the same number of rows")

    finite_rows = np.isfinite(features).all(axis=1)
    finite_features = features[finite_rows]
    finite_pids = pids[finite_rows]
    finite_domains = domains[finite_rows]
    norms = np.linalg.norm(finite_features, axis=1) if len(finite_features) else np.array([])
    zero_vectors = int(np.sum(norms <= 1e-12))
    variances = (np.var(finite_features, axis=0)
                 if len(finite_features) else np.array([]))
    effective_rank = _effective_rank(finite_features, seed)
    overall = _separability(finite_features, finite_pids, seed, max_pairs) \
        if len(finite_features) else {
            "same_pid_cosine": _distribution(np.array([])),
            "different_pid_cosine": _distribution(np.array([])),
            "mean_cosine_gap": None,
            "sampling": "No finite rows.",
        }

    domain_reports: Dict[str, dict] = {}
    for offset, domain in enumerate(sorted(set(finite_domains.tolist()))):
        mask = finite_domains == domain
        subset = finite_features[mask]
        subset_pids = finite_pids[mask]
        domain_reports[str(domain)] = {
            "rows": int(mask.sum()),
            "l2_norm": _distribution(np.linalg.norm(subset, axis=1)),
            "effective_rank": _effective_rank(subset, seed + offset + 17),
            "identity_separability": _separability(
                subset, subset_pids, seed + offset + 29, max_pairs),
        }

    mean_variance = float(variances.mean()) if variances.size else None
    normalized = _normalize(finite_features) if len(finite_features) else finite_features
    adjacent_distance = (np.linalg.norm(np.diff(normalized, axis=0), axis=1)
                         if len(normalized) > 1 else np.array([]))
    near_constant = bool(
        len(finite_features) > 1
        and (mean_variance is not None and mean_variance < 1e-8
             or effective_rank["value"] < 1.5)
    )
    return {
        "status": "COMPLETED",
        "rows": int(features.shape[0]),
        "dimensions": int(features.shape[1]),
        "finite_rows": int(finite_rows.sum()),
        "nonfinite_rows": int((~finite_rows).sum()),
        "nan_values": int(np.isnan(features).sum()),
        "positive_inf_values": int(np.isposinf(features).sum()),
        "negative_inf_values": int(np.isneginf(features).sum()),
        "zero_vectors": zero_vectors,
        "l2_norm": _distribution(norms),
        "dimension_variance": {
            "mean": mean_variance,
            "min": float(variances.min()) if variances.size else None,
            "max": float(variances.max()) if variances.size else None,
            "zero_dimensions": int(np.sum(variances <= 1e-15)),
        },
        "effective_rank": effective_rank,
        "identity_separability": overall,
        "domains": domain_reports,
        "adjacent_normalized_feature_distance": _distribution(adjacent_distance),
        "near_constant_features": near_constant,
        "collapse_thresholds": {
            "mean_dimension_variance_below": 1e-8,
            "effective_rank_below": 1.5,
        },
        "ordering_note": (
            "Adjacent-row distance is reported for inspection but is excluded "
            "from collapse classification because sample ordering may group identities."
        ),
    }
