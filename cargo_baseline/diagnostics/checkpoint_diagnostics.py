"""Training-log parser and report helpers for Stage-1 diagnostics."""

from __future__ import annotations

import json
import os
import re
from typing import Dict


_EPS = re.compile(r"(Aerial|Ground) clustering criterion:\s*eps:\s*([0-9.eE+-]+)", re.I)
_CLUSTERS = re.compile(
    r"Statistics for (aerial|ground) epoch\s+(\d+):\s+(\d+) clusters", re.I)
_LOSS = re.compile(
    r"Epoch:\s*\[(\d+)\]\[(\d+)/(\d+)\].*?Loss\s+"
    r"([0-9.eE+-]+)\s+\(([0-9.eE+-]+)\).*?Loss ir\s+"
    r"([0-9.eE+-]+)\s+Loss rgb\s+([0-9.eE+-]+)", re.I)
_VALID = re.compile(
    r"(?:valid|effective|training)\s*(?:samples|images)?\s*[:=]\s*(\d+)", re.I)
_NOISE = re.compile(r"noise(?: ratio)?\s*[:=]\s*([0-9.eE+%-]+)", re.I)
_LARGEST = re.compile(r"(?:largest|max) cluster(?: ratio)?\s*[:=]\s*([0-9.eE+%-]+)", re.I)


def _number_or_percent(value: str) -> float:
    if value.endswith("%"):
        return float(value[:-1]) / 100.0
    return float(value)


def parse_stage1_log(path: str, max_epoch: int = 4) -> dict:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        return {"status": "BLOCKED", "reason": "Log file does not exist",
                "path": resolved, "epochs": {}}
    epochs: Dict[str, dict] = {
        str(epoch): {
            "aerial_clusters": "UNKNOWN",
            "ground_clusters": "UNKNOWN",
            "last_reported_iteration": "UNKNOWN",
            "loss": "UNKNOWN",
            "average_loss": "UNKNOWN",
            "loss_aerial": "UNKNOWN",
            "loss_ground": "UNKNOWN",
            "effective_training_samples": "UNKNOWN",
            "noise_ratio": "UNKNOWN",
            "largest_cluster_ratio": "UNKNOWN",
        }
        for epoch in range(max_epoch + 1)
    }
    eps = {"aerial": "UNKNOWN", "ground": "UNKNOWN"}
    with open(resolved, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _EPS.search(line)
            if match:
                eps[match.group(1).lower()] = float(match.group(2))
            match = _CLUSTERS.search(line)
            if match and int(match.group(2)) <= max_epoch:
                epochs[match.group(2)][match.group(1).lower() + "_clusters"] = int(match.group(3))
            match = _LOSS.search(line)
            if match and int(match.group(1)) <= max_epoch:
                entry = epochs[match.group(1)]
                entry.update({
                    "last_reported_iteration": int(match.group(2)),
                    "configured_iterations": int(match.group(3)),
                    "loss": float(match.group(4)),
                    "average_loss": float(match.group(5)),
                    "loss_aerial": float(match.group(6)),
                    "loss_ground": float(match.group(7)),
                })
            for regex, key in ((_VALID, "effective_training_samples"),
                               (_NOISE, "noise_ratio"),
                               (_LARGEST, "largest_cluster_ratio")):
                match = regex.search(line)
                if match:
                    epoch_matches = re.findall(r"epoch\s*(\d+)", line, re.I)
                    if epoch_matches and int(epoch_matches[-1]) <= max_epoch:
                        value = (int(match.group(1)) if key == "effective_training_samples"
                                 else _number_or_percent(match.group(1)))
                        epochs[epoch_matches[-1]][key] = value
    return {
        "status": "COMPLETED",
        "path": resolved,
        "reported_epoch_range": [0, max_epoch],
        "dbscan_eps": eps,
        "epochs": epochs,
        "unknown_policy": "Metrics absent from the log are reported as UNKNOWN.",
    }


def write_json(path: str, value: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
