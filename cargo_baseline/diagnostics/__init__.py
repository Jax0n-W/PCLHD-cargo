"""Read-only diagnostics for the CARGO Stage-1 checkpoint."""

from .checkpoint_diagnostics import parse_stage1_log
from .feature_health import summarize_feature_health
from .model_compatibility import inspect_checkpoint, validate_strict_state_dict
from .pipeline_consistency import audit_pipeline_consistency
from .protocol_oracle import run_protocol_oracle
from .pseudo_label_audit import compute_pseudo_label_quality
from .train_feature_probe import evaluate_domain_probe, select_train_probe

__all__ = [
    "audit_pipeline_consistency",
    "compute_pseudo_label_quality",
    "evaluate_domain_probe",
    "inspect_checkpoint",
    "parse_stage1_log",
    "run_protocol_oracle",
    "select_train_probe",
    "summarize_feature_health",
    "validate_strict_state_dict",
]
