"""Read-only diagnostics for the CARGO Stage-1 checkpoint."""

from .checkpoint_diagnostics import parse_stage1_log
from .feature_health import summarize_feature_health
from .model_compatibility import inspect_checkpoint, validate_strict_state_dict
from .protocol_oracle import run_protocol_oracle

__all__ = [
    "inspect_checkpoint",
    "parse_stage1_log",
    "run_protocol_oracle",
    "summarize_feature_health",
    "validate_strict_state_dict",
]
