from __future__ import annotations

from dataset_tools.manager import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    export_dataset_csv,
    export_dataset_jsonl,
    register_sample,
    summarize_dataset,
    validate_dataset,
)
from dataset_tools.feature_audit import FEATURE_AUDIT_SCHEMA_VERSION, print_audit_summary, run_feature_audit
from dataset_tools.feature_preparation import (
    FEATURE_SCHEMA_VERSION,
    REGISTRY_VERSION,
    check_feature_compatibility,
    generate_feature_docs,
    generate_feature_schema,
    load_feature_registry,
    print_preparation_summary,
    run_feature_preparation,
    validate_feature_registry,
)
from dataset_tools.models import DatasetManifestEntry, DatasetSummary
from dataset_tools.source_adapters import (
    audit_faceforensics_metadata,
    match_faceforensics_video,
    plan_faceforensics_samples,
    validate_plan,
    write_faceforensics_audit,
)

__all__ = [
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "FEATURE_AUDIT_SCHEMA_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "REGISTRY_VERSION",
    "DatasetManifestEntry",
    "DatasetSummary",
    "check_feature_compatibility",
    "export_dataset_csv",
    "export_dataset_jsonl",
    "generate_feature_docs",
    "generate_feature_schema",
    "load_feature_registry",
    "print_audit_summary",
    "print_preparation_summary",
    "register_sample",
    "run_feature_audit",
    "run_feature_preparation",
    "summarize_dataset",
    "validate_feature_registry",
    "validate_dataset",
    "audit_faceforensics_metadata",
    "write_faceforensics_audit",
    "plan_faceforensics_samples",
    "validate_plan",
    "match_faceforensics_video",
]
