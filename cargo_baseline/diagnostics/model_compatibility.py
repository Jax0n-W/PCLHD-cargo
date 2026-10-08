"""Read-only checkpoint inspection and strict model compatibility checks."""

from __future__ import annotations

import os
from typing import Mapping, Tuple

import torch


def _extract_state_dict(checkpoint) -> Tuple[Mapping[str, torch.Tensor], str]:
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("state_dict"), Mapping):
        return checkpoint["state_dict"], "checkpoint['state_dict']"
    if isinstance(checkpoint, Mapping) and checkpoint and all(
            isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint, "checkpoint"
    raise ValueError("Checkpoint does not contain a recognizable state_dict")


def inspect_checkpoint(path: str) -> tuple[object, Mapping[str, torch.Tensor], dict]:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError("Checkpoint does not exist: {}".format(resolved))
    checkpoint = torch.load(
        resolved, map_location=torch.device("cpu"), weights_only=False)
    state_dict, state_source = _extract_state_dict(checkpoint)
    tensor_items = [(key, value) for key, value in state_dict.items()
                    if isinstance(value, torch.Tensor)]
    nonfinite = 0
    nan = 0
    posinf = 0
    neginf = 0
    parameter_count = 0
    shapes = {}
    for key, tensor in tensor_items:
        cpu = tensor.detach().cpu()
        shapes[key] = list(cpu.shape)
        parameter_count += int(cpu.numel())
        if cpu.is_floating_point() or cpu.is_complex():
            nan += int(torch.isnan(cpu).sum().item())
            posinf += int(torch.isposinf(cpu).sum().item())
            neginf += int(torch.isneginf(cpu).sum().item())
    nonfinite = nan + posinf + neginf
    metadata = checkpoint if isinstance(checkpoint, Mapping) else {}
    report = {
        "status": "INSPECTED",
        "checkpoint": resolved,
        "epoch": metadata.get("epoch", "UNKNOWN"),
        "backbone_metadata": metadata.get("backbone", "UNKNOWN"),
        "state_dict_source": state_source,
        "state_dict_keys": len(state_dict),
        "tensor_keys": len(tensor_items),
        "parameter_values": parameter_count,
        "nan_values": nan,
        "positive_inf_values": posinf,
        "negative_inf_values": neginf,
        "nonfinite_values": nonfinite,
        "tensor_shapes": shapes,
        "strict_load_attempted": False,
        "strict_load_succeeded": False,
    }
    return checkpoint, state_dict, report


def validate_strict_state_dict(model, state_dict: Mapping[str, torch.Tensor]) -> dict:
    model_state = model.state_dict()
    model_keys = set(model_state)
    checkpoint_keys = set(state_dict)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    shape_mismatches = []
    for key in sorted(model_keys & checkpoint_keys):
        if tuple(model_state[key].shape) != tuple(state_dict[key].shape):
            shape_mismatches.append({
                "key": key,
                "model_shape": list(model_state[key].shape),
                "checkpoint_shape": list(state_dict[key].shape),
            })
    result = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "strict_load_attempted": True,
        "strict_load_succeeded": False,
        "strict_load_error": None,
    }
    try:
        model.load_state_dict(state_dict, strict=True)
        result["strict_load_succeeded"] = True
    except Exception as exc:  # the exact RuntimeError varies across torch versions
        result["strict_load_error"] = "{}: {}".format(type(exc).__name__, exc)
    return result
