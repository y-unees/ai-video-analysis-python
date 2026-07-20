from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import numpy as np

from config import APP_VERSION
from dataset_tools.manager import ALLOWED_LABELS, DEFAULT_DATASET_ROOT, export_dataset_csv


FEATURE_AUDIT_SCHEMA_VERSION = "0.9.3"
DEFAULT_EXPORT_PATH = DEFAULT_DATASET_ROOT / "exports" / "dataset_features.csv"
DEFAULT_STATISTICS_DIRECTORY = DEFAULT_DATASET_ROOT / "statistics"
TARGET_COLUMN = "expected_label"
TARGET_CLASSES = ["real", "ai_generated"]
MISSING_TOKENS = {"", "null", "none", "nan", "na", "n/a"}
BOOLEAN_TOKENS = {"true", "false"}
LEAKAGE_TERMS = {
    "real",
    "ai",
    "fake",
    "generated",
    "ai_generated",
    "deepfake",
    "kaggle",
    "personal",
    "external",
    "manual_source",
    "self_recorded",
}


class FeatureAuditError(ValueError):
    pass


def run_feature_audit(
    input_path: Path | None = None,
    output_dir: Path = DEFAULT_STATISTICS_DIRECTORY,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    missing_threshold: float = 0.40,
    near_constant_threshold: float = 0.95,
    correlation_threshold: float = 0.95,
    overwrite: bool = True,
    include_plots: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    source_path = _resolve_input(input_path, dataset_root)
    rows, columns = load_feature_export(source_path)
    if not rows:
        raise FeatureAuditError("Dataset export contains no rows.")
    if TARGET_COLUMN not in columns:
        raise FeatureAuditError(f"Dataset export is missing target column: {TARGET_COLUMN}")

    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise FeatureAuditError(f"Output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "input_path": source_path,
        "output_dir": output_dir,
        "missing_threshold": missing_threshold,
        "near_constant_threshold": near_constant_threshold,
        "correlation_threshold": correlation_threshold,
        "strict": strict,
        "include_plots": include_plots,
    }
    audit = _build_audit(rows, columns, context)
    _write_outputs(audit, output_dir, overwrite)
    return audit


def load_feature_export(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    path = Path(path)
    if not path.exists():
        raise FeatureAuditError(f"Dataset export does not exist: {path}")
    if path.stat().st_size == 0:
        raise FeatureAuditError(f"Dataset export is empty: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".jsonl":
        return _load_jsonl(path)
    raise FeatureAuditError("Dataset export must be CSV or JSONL.")


def _resolve_input(input_path: Path | None, dataset_root: Path) -> Path:
    if input_path:
        return Path(input_path)
    csv_path = Path(dataset_root) / "exports" / "dataset_features.csv"
    jsonl_path = Path(dataset_root) / "exports" / "dataset_features.jsonl"
    if csv_path.exists():
        return csv_path
    if jsonl_path.exists():
        return jsonl_path
    try:
        return export_dataset_csv(Path(dataset_root), csv_path)
    except Exception as error:
        raise FeatureAuditError(f"Default export is missing and could not be created: {error}") from error


def _load_csv(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise FeatureAuditError(f"CSV export is missing a header: {path}")
            columns = list(reader.fieldnames)
            rows: list[dict[str, Any]] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise FeatureAuditError(f"CSV row {line_number} has too many fields.")
                missing = [column for column in columns if column not in row]
                if missing:
                    raise FeatureAuditError(f"CSV row {line_number} is missing fields: {missing}")
                rows.append({column: _normalize_cell(row[column]) for column in columns})
            return rows, columns
    except UnicodeDecodeError as error:
        raise FeatureAuditError(f"CSV export is not valid UTF-8: {path}") from error
    except csv.Error as error:
        raise FeatureAuditError(f"CSV parsing failed: {error}") from error


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    seen_columns: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise FeatureAuditError(f"JSONL line {line_number} is malformed: {error}") from error
            if not isinstance(value, dict):
                raise FeatureAuditError(f"JSONL line {line_number} must contain an object.")
            row = {str(key): _normalize_cell(child) for key, child in value.items()}
            rows.append(row)
            for key in sorted(row):
                if key not in seen_columns:
                    seen_columns.add(key)
                    columns.append(key)
    if rows:
        for row in rows:
            for column in columns:
                row.setdefault(column, None)
    return rows, columns


def _build_audit(rows: list[dict[str, Any]], columns: list[str], context: dict[str, Any]) -> dict[str, Any]:
    labels = [str(row.get(TARGET_COLUMN)) for row in rows]
    label_counts = Counter(labels)
    unexpected_labels = sorted(label for label in label_counts if label not in ALLOWED_LABELS)
    missing_labels = [label for label in TARGET_CLASSES if label_counts.get(label, 0) == 0]
    class_counts = {label: label_counts.get(label, 0) for label in TARGET_CLASSES}
    class_percentages = {label: _safe_ratio(class_counts[label], len(rows)) for label in TARGET_CLASSES}

    duplicate_report = _duplicate_report(rows)
    column_profiles = [_profile_column(column, rows, context) for column in columns]
    profile_by_name = {profile["column"]: profile for profile in column_profiles}
    leakage_findings = _leakage_findings(column_profiles, rows)
    for finding in leakage_findings:
        profile = profile_by_name.get(finding["column"])
        if profile and profile["semantic_category"] != "target":
            profile["semantic_category"] = "possible_leakage"
            profile["flags"] = sorted(set(profile["flags"] + ["possible_leakage"]))
            profile["model_eligible"] = False
            profile["exclusion_reason"] = finding["leakage_reason"]

    quality = [_quality_record(profile, context) for profile in column_profiles]
    quality_by_name = {record["column"]: record for record in quality}
    for profile in column_profiles:
        quality_record = quality_by_name[profile["column"]]
        profile["primary_status"] = quality_record["primary_status"]
        profile["flags"] = quality_record["flags"]
        profile["model_eligible"] = quality_record["model_eligible"]
        profile["exclusion_reason"] = quality_record["exclusion_reason"]

    numeric_columns = [
        profile["column"]
        for profile in column_profiles
        if profile["detected_data_type"] == "numeric" and profile["semantic_category"] == "numeric_feature"
    ]
    candidate_features = [
        profile["column"]
        for profile in column_profiles
        if profile["model_eligible"] and profile["semantic_category"] == "numeric_feature"
    ]
    statistics_rows = _statistics_rows(rows, candidate_features)
    comparison_rows = _class_comparisons(rows, candidate_features)
    correlation_matrix, high_correlation_pairs = _correlations(rows, candidate_features, context["correlation_threshold"])
    readiness, readiness_reasons = _model_readiness(rows, class_counts, column_profiles, candidate_features, duplicate_report)
    warnings = _warnings(unexpected_labels, missing_labels, readiness_reasons, context)
    recommendations = _recommendations(readiness, column_profiles, candidate_features)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    dataset_profile = {
        "schema_version": FEATURE_AUDIT_SCHEMA_VERSION,
        "project_version": APP_VERSION,
        "audit_timestamp": generated_at,
        "input_dataset_path": _relative_path(context["input_path"]),
        "total_samples": len(rows),
        "total_columns": len(columns),
        "discovered_class_labels": sorted(label_counts),
        "target_column": TARGET_COLUMN,
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "class_balance_ratio": _class_balance_ratio(class_counts),
        "duplicate_full_rows": duplicate_report["duplicate_full_rows"],
        "duplicate_sample_ids": duplicate_report["duplicate_sample_ids"],
        "duplicate_source_hashes": duplicate_report["duplicate_source_hashes"],
        "duplicate_source_paths": duplicate_report["duplicate_source_paths"],
        "duplicate_feature_file_references": duplicate_report["duplicate_feature_file_references"],
        "numeric_columns": sorted(numeric_columns),
        "categorical_columns": sorted(_columns_by_category(column_profiles, "categorical_feature")),
        "identifier_columns": sorted(_columns_by_category(column_profiles, "identifier")),
        "metadata_columns": sorted(_columns_by_category(column_profiles, "source_metadata") + _columns_by_category(column_profiles, "analysis_metadata")),
        "target_columns": sorted(_columns_by_category(column_profiles, "target")),
        "unsupported_columns": sorted(_columns_by_category(column_profiles, "unsupported")),
        "fully_empty_columns": sorted(profile["column"] for profile in column_profiles if "fully_empty" in profile["flags"]),
        "constant_columns": sorted(profile["column"] for profile in column_profiles if "constant" in profile["flags"]),
        "near_constant_columns": sorted(profile["column"] for profile in column_profiles if "near_constant" in profile["flags"]),
        "columns_with_missing_values": sorted(profile["column"] for profile in column_profiles if profile["missing_count"] > 0),
        "columns_with_invalid_numeric_values": sorted(profile["column"] for profile in column_profiles if profile["invalid_numeric_count"] > 0),
        "candidate_model_feature_count": len(candidate_features),
        "excluded_feature_count": len(columns) - len(candidate_features) - 1,
        "audit_status": "completed_with_warnings" if warnings else "completed",
        "model_readiness_status": readiness,
        "warnings": warnings,
        "recommendations": recommendations,
    }

    model_schema = _model_schema(
        generated_at=generated_at,
        dataset_path=_relative_path(context["input_path"]),
        candidate_features=candidate_features,
        column_profiles=column_profiles,
        class_counts=class_counts,
        readiness=readiness,
        warnings=warnings,
        sample_count=len(rows),
    )
    report_text = _human_report(dataset_profile, column_profiles, leakage_findings, comparison_rows, high_correlation_pairs, model_schema, readiness_reasons)
    return {
        "dataset_profile": dataset_profile,
        "column_profiles": column_profiles,
        "feature_statistics": statistics_rows,
        "class_comparison": comparison_rows,
        "missing_values": _missing_rows(column_profiles),
        "invalid_values": _invalid_rows(column_profiles),
        "constant_features": _constant_rows(column_profiles),
        "correlation_matrix": correlation_matrix,
        "high_correlation_pairs": high_correlation_pairs,
        "leakage_report": {
            "schema_version": FEATURE_AUDIT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "findings": leakage_findings,
        },
        "feature_quality": {
            "schema_version": FEATURE_AUDIT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "features": quality,
        },
        "model_feature_schema": model_schema,
        "statistics_report": report_text,
    }


def _profile_column(column: str, rows: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    values = [row.get(column) for row in rows]
    non_missing = [value for value in values if not _is_missing(value)]
    missing_count = len(values) - len(non_missing)
    unique_values = sorted({_display_value(value) for value in non_missing})
    parsed = [_parse_number(value) for value in non_missing]
    numeric_success = [result for result in parsed if result["status"] == "numeric"]
    invalid_numeric = [result for result in parsed if result["status"] == "invalid_numeric"]
    detected_type = _detected_type(non_missing, numeric_success, invalid_numeric)
    semantic = _semantic_category(column, non_missing, detected_type, missing_count == len(values))
    dominant_value = None
    dominant_count = 0
    dominant_percentage = None
    if non_missing:
        counts = Counter(_display_value(value) for value in non_missing)
        dominant_value, dominant_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        dominant_percentage = dominant_count / len(non_missing)
    flags: list[str] = []
    if missing_count == len(values):
        flags.append("fully_empty")
    if missing_count / len(values) > context["missing_threshold"]:
        flags.append("missing_heavy")
    if semantic == "numeric_feature" and invalid_numeric:
        flags.append("invalid_numeric")
    if non_missing and len(unique_values) == 1 and "fully_empty" not in flags:
        flags.append("constant")
    elif dominant_percentage is not None and dominant_percentage >= context["near_constant_threshold"]:
        flags.append("near_constant")
    if detected_type in {"numeric", "mixed_numeric"}:
        impossible = _impossible_numeric_values(column, values)
        invalid_rows = _invalid_numeric_rows(column, rows, invalid_numeric, impossible)
    else:
        impossible = []
        invalid_rows = []
    if impossible:
        flags.append("invalid_numeric")
    missing_by_class = {
        label: sum(1 for row in rows if row.get(TARGET_COLUMN) == label and _is_missing(row.get(column)))
        for label in TARGET_CLASSES
    }
    return {
        "column": column,
        "semantic_category": semantic,
        "detected_data_type": detected_type,
        "non_null_count": len(non_missing),
        "missing_count": missing_count,
        "missing_percentage": _safe_ratio(missing_count, len(values)),
        "unique_count": len(unique_values),
        "sample_values": unique_values[:5],
        "missing_by_class": missing_by_class,
        "invalid_numeric_count": len(invalid_rows),
        "invalid_numeric_examples": invalid_rows[:10],
        "dominant_value": dominant_value,
        "dominant_value_frequency": dominant_count,
        "dominant_value_percentage": dominant_percentage,
        "flags": sorted(set(flags)),
        "model_eligible": False,
        "exclusion_reason": None,
        "primary_status": "review_required",
        "recommended_action": "review",
    }


def _semantic_category(column: str, non_missing: list[Any], detected_type: str, fully_empty: bool) -> str:
    lowered = column.lower()
    if fully_empty:
        return "fully_empty"
    if column == TARGET_COLUMN:
        return "target"
    if lowered == "identity.expected_label" or "label" in lowered or "verdict" in lowered or "classification" in lowered:
        return "possible_leakage"
    if "path" in lowered or "file" in lowered or "video_name" in lowered or lowered.endswith(".name") or lowered.endswith("_name"):
        return "path"
    if lowered in {"sample_id"} or lowered.endswith("sha256") or lowered.endswith(".analysis_id") or lowered.endswith(".video_sha256"):
        return "identifier"
    if lowered.endswith("_at") or "timestamp" in lowered or "creation_time" in lowered:
        return "timestamp"
    if lowered in {"source", "generator_or_camera", "notes"} or "generator" in lowered or "camera" in lowered:
        return "source_metadata"
    if "schema_version" in lowered or "application_version" in lowered or lowered.startswith("d3_summary.d3_encoder") or lowered.startswith("d3_summary.d3_distance") or "calibration" in lowered or "score_direction" in lowered or lowered.endswith(".d3_status"):
        return "analysis_metadata"
    if detected_type in {"numeric", "mixed_numeric"}:
        return "numeric_feature"
    if detected_type in {"boolean", "categorical"}:
        return "categorical_feature"
    return "unsupported"


def _quality_record(profile: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    semantic = profile["semantic_category"]
    flags = sorted(set(profile["flags"]))
    status = semantic
    reason = None
    model_eligible = False
    if semantic == "target":
        status = "target"
        reason = "target column"
    elif semantic in {"identifier", "path", "timestamp", "source_metadata", "analysis_metadata", "possible_leakage", "unsupported", "fully_empty"}:
        status = {
            "source_metadata": "metadata_only",
            "analysis_metadata": "metadata_only",
        }.get(semantic, semantic)
        reason = f"semantic category is {semantic}"
    elif "fully_empty" in flags:
        status = "fully_empty"
        reason = "column has no non-missing values"
    elif "invalid_numeric" in flags:
        status = "invalid_numeric"
        reason = "invalid or impossible numeric values present"
    elif "constant" in flags:
        status = "constant"
        reason = "column has one unique non-missing value"
    elif "near_constant" in flags:
        status = "near_constant"
        reason = f"dominant value ratio >= {context['near_constant_threshold']}"
    elif "missing_heavy" in flags:
        status = "missing_heavy"
        reason = f"missing value ratio > {context['missing_threshold']}"
    elif semantic == "numeric_feature":
        status = "candidate"
        model_eligible = True
    elif semantic == "categorical_feature":
        status = "review_required"
        reason = "categorical fields are not included until encoding is designed"
    return {
        "column": profile["column"],
        "primary_status": status,
        "flags": flags,
        "model_eligible": model_eligible,
        "exclusion_reason": None if model_eligible else reason,
        "recommended_action": "candidate for future modeling" if model_eligible else "exclude from future model schema",
    }


def _leakage_findings(column_profiles: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for profile in column_profiles:
        column = profile["column"]
        lowered = column.lower()
        if column == TARGET_COLUMN:
            continue
        reasons: list[tuple[str, str]] = []
        if lowered == "identity.expected_label" or "label" in lowered:
            reasons.append(("critical", "duplicates or exposes the target label"))
        if lowered in {"sample_id"}:
            reasons.append(("high", "sample IDs include class prefixes in the current manifest"))
        if "video_name" in lowered or "file" in lowered or "path" in lowered:
            reasons.append(("high", "filename or path may contain class-bearing conventions such as real__ or ai__"))
        if lowered in {"source", "generator_or_camera", "notes"} or "generator" in lowered:
            reasons.append(("medium", "source notes or generator metadata can directly reveal dataset origin or class"))
        values = [_display_value(row.get(column)).lower() for row in rows if not _is_missing(row.get(column))]
        evidence_terms = sorted({term for term in LEAKAGE_TERMS if any(term in value for value in values)})
        if evidence_terms:
            reasons.append(("medium", f"values contain class/source terms: {', '.join(evidence_terms[:8])}"))
        if reasons:
            severity = _max_severity([severity for severity, _reason in reasons])
            findings.append(
                {
                    "column": column,
                    "severity": severity,
                    "leakage_reason": "; ".join(reason for _severity, reason in reasons),
                    "evidence": profile["sample_values"],
                    "required_action": "exclude from future model training input",
                }
            )
    return sorted(findings, key=lambda item: (["critical", "high", "medium", "low"].index(item["severity"]), item["column"]))


def _statistics_rows(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for column in columns:
        output.append({"column": column, "group": "all", **_numeric_stats([row.get(column) for row in rows])})
        for label in TARGET_CLASSES:
            output.append({"column": column, "group": label, **_numeric_stats([row.get(column) for row in rows if row.get(TARGET_COLUMN) == label])})
    return output


def _class_comparisons(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for column in columns:
        real_values = _numeric_values([row.get(column) for row in rows if row.get(TARGET_COLUMN) == "real"])
        ai_values = _numeric_values([row.get(column) for row in rows if row.get(TARGET_COLUMN) == "ai_generated"])
        if not real_values or not ai_values:
            output.append(_comparison_unavailable(column, real_values, ai_values, "insufficient class data"))
            continue
        real_mean = mean(real_values)
        ai_mean = mean(ai_values)
        real_median = median(real_values)
        ai_median = median(ai_values)
        pooled = _pooled_std(real_values, ai_values)
        effect = None if pooled in {None, 0.0} else (ai_mean - real_mean) / pooled
        point_biserial = _point_biserial(real_values, ai_values)
        p_value, test_name = _mann_whitney_approx(real_values, ai_values)
        output.append(
            {
                "column": column,
                "real_sample_count": len(real_values),
                "ai_generated_sample_count": len(ai_values),
                "real_mean": real_mean,
                "ai_generated_mean": ai_mean,
                "mean_difference": ai_mean - real_mean,
                "absolute_mean_difference": abs(ai_mean - real_mean),
                "real_median": real_median,
                "ai_generated_median": ai_median,
                "median_difference": ai_median - real_median,
                "pooled_standard_deviation": pooled,
                "standardized_effect_size": effect,
                "point_biserial_correlation": point_biserial,
                "class_overlap_indicator": _overlap_indicator(real_values, ai_values),
                "statistical_test_result": "available" if p_value is not None else "unavailable",
                "p_value": p_value,
                "test_name": test_name,
                "interpretation_warning": "exploratory only; pilot sample is too small for reliable generalization",
            }
        )
    return sorted(output, key=lambda row: (-(row.get("absolute_mean_difference") or 0), row["column"]))


def _correlations(rows: list[dict[str, Any]], columns: list[str], threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for left in columns:
        row: dict[str, Any] = {"feature": left}
        for right in columns:
            value = _pearson_pair(rows, left, right)
            row[right] = value
            if left < right and value is not None and abs(value) >= threshold:
                pairs.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "correlation": value,
                        "warning": "possible redundant features; review before future modeling",
                    }
                )
        matrix.append(row)
    return matrix, sorted(pairs, key=lambda item: (-abs(item["correlation"]), item["feature_a"], item["feature_b"]))


def _model_schema(
    generated_at: str,
    dataset_path: str,
    candidate_features: list[str],
    column_profiles: list[dict[str, Any]],
    class_counts: dict[str, int],
    readiness: str,
    warnings: list[str],
    sample_count: int,
) -> dict[str, Any]:
    included = sorted(candidate_features)
    excluded = sorted(profile["column"] for profile in column_profiles if profile["column"] not in included and profile["semantic_category"] != "target")
    exclusion_reasons = {profile["column"]: profile["exclusion_reason"] for profile in column_profiles if profile["column"] in excluded}
    feature_metadata = {
        profile["column"]: {
            "semantic_category": profile["semantic_category"],
            "detected_data_type": profile["detected_data_type"],
            "missing_count": profile["missing_count"],
            "missing_percentage": profile["missing_percentage"],
            "primary_status": profile["primary_status"],
            "flags": profile["flags"],
        }
        for profile in column_profiles
    }
    schema = {
        "schema_version": FEATURE_AUDIT_SCHEMA_VERSION,
        "project_version": APP_VERSION,
        "generated_at": generated_at,
        "dataset_path": dataset_path,
        "target": {
            "column": TARGET_COLUMN,
            "classes": TARGET_CLASSES,
            "encoding": {"real": 0, "ai_generated": 1},
        },
        "included_features": included,
        "excluded_features": excluded,
        "exclusion_reasons": exclusion_reasons,
        "feature_metadata": feature_metadata,
        "dataset_sample_count": sample_count,
        "class_counts": class_counts,
        "warnings": warnings,
        "model_readiness": readiness,
    }
    _validate_model_schema(schema)
    return schema


def _model_readiness(
    rows: list[dict[str, Any]],
    class_counts: dict[str, int],
    column_profiles: list[dict[str, Any]],
    candidate_features: list[str],
    duplicate_report: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if len(rows) < 100 or min(class_counts.values() or [0]) < 50:
        reasons.append("pilot dataset has fewer than 100 total samples or fewer than 50 samples per class")
    if any(count == 0 for count in class_counts.values()):
        reasons.append("one or more expected classes are missing")
    if duplicate_report["duplicate_full_rows"] or duplicate_report["duplicate_sample_ids"] or duplicate_report["duplicate_source_hashes"]:
        reasons.append("duplicates require review")
    if any(profile["invalid_numeric_count"] for profile in column_profiles):
        reasons.append("invalid numeric values require cleanup")
    if not candidate_features:
        reasons.append("no eligible numeric candidate features")
    if reasons and reasons[0].startswith("pilot dataset"):
        return "insufficient_dataset_size", reasons
    if reasons:
        return "requires_cleanup", reasons
    if len(rows) < 500:
        return "audit_passed_for_experimentation", ["dataset is structurally clean but still small"]
    return "ready_for_baseline_experiment", ["structural audit passed with enough samples for a baseline experiment"]


def _write_outputs(audit: dict[str, Any], output_dir: Path, overwrite: bool) -> None:
    json_outputs = {
        "dataset_profile.json": audit["dataset_profile"],
        "leakage_report.json": audit["leakage_report"],
        "feature_quality.json": audit["feature_quality"],
        "model_feature_schema.json": audit["model_feature_schema"],
    }
    csv_outputs = {
        "column_profile.csv": audit["column_profiles"],
        "feature_statistics.csv": audit["feature_statistics"],
        "class_comparison.csv": audit["class_comparison"],
        "missing_values.csv": audit["missing_values"],
        "invalid_values.csv": audit["invalid_values"],
        "constant_features.csv": audit["constant_features"],
        "correlation_matrix.csv": audit["correlation_matrix"],
        "high_correlation_pairs.csv": audit["high_correlation_pairs"],
    }
    for filename, value in json_outputs.items():
        _write_json(output_dir / filename, value, overwrite)
    for filename, rows in csv_outputs.items():
        _write_csv(output_dir / filename, rows, overwrite)
    _write_text(output_dir / "statistics_report.txt", audit["statistics_report"], overwrite)


def print_audit_summary(audit: dict[str, Any]) -> None:
    profile = audit["dataset_profile"]
    print("Dataset feature audit complete.")
    print(f"Input samples: {profile['total_samples']}")
    print(f"Real samples: {profile['class_counts'].get('real', 0)}")
    print(f"AI-generated samples: {profile['class_counts'].get('ai_generated', 0)}")
    print(f"Total columns: {profile['total_columns']}")
    print(f"Numeric candidate features: {profile['candidate_model_feature_count']}")
    print(f"Leakage-risk columns: {len(audit['leakage_report']['findings'])}")
    print(f"Fully empty columns: {len(profile['fully_empty_columns'])}")
    print(f"Constant columns: {len(profile['constant_columns'])}")
    print(f"Near-constant columns: {len(profile['near_constant_columns'])}")
    print(f"Missing-heavy columns: {sum(1 for item in audit['feature_quality']['features'] if item['primary_status'] == 'missing_heavy')}")
    print(f"Invalid numeric columns: {len(profile['columns_with_invalid_numeric_values'])}")
    print(f"Duplicate samples: {profile['duplicate_full_rows'] + profile['duplicate_sample_ids'] + profile['duplicate_source_hashes']}")
    print(f"Model readiness: {profile['model_readiness_status']}")
    print(f"Report path: {DEFAULT_STATISTICS_DIRECTORY / 'statistics_report.txt'}")
    print(f"Model schema path: {DEFAULT_STATISTICS_DIRECTORY / 'model_feature_schema.json'}")


def _human_report(
    profile: dict[str, Any],
    columns: list[dict[str, Any]],
    leakage_findings: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    model_schema: dict[str, Any],
    readiness_reasons: list[str],
) -> str:
    top_differences = [row for row in comparisons if row.get("standardized_effect_size") is not None][:10]
    lines = [
        "Dataset Statistics and Feature Audit",
        "",
        f"Project version: {profile['project_version']}",
        f"Audit timestamp: {profile['audit_timestamp']}",
        f"Input dataset: {profile['input_dataset_path']}",
        "",
        "Dataset Dimensions",
        f"Samples: {profile['total_samples']}",
        f"Columns: {profile['total_columns']}",
        f"Classes: {profile['class_counts']}",
        f"Class balance ratio: {profile['class_balance_ratio']}",
        "",
        "Structural Validation",
        f"Audit status: {profile['audit_status']}",
        f"Model readiness: {profile['model_readiness_status']}",
        "",
        "Missing Values",
        f"Columns with missing values: {len(profile['columns_with_missing_values'])}",
        f"Fully empty columns: {', '.join(profile['fully_empty_columns']) or 'none'}",
        "",
        "Constant and Near-Constant Features",
        f"Constant columns: {', '.join(profile['constant_columns']) or 'none'}",
        f"Near-constant columns: {', '.join(profile['near_constant_columns']) or 'none'}",
        "",
        "Invalid Numeric Findings",
        f"Columns with invalid numeric values: {', '.join(profile['columns_with_invalid_numeric_values']) or 'none'}",
        "",
        "Duplicate Findings",
        f"Duplicate full rows: {profile['duplicate_full_rows']}",
        f"Duplicate sample IDs: {profile['duplicate_sample_ids']}",
        f"Duplicate source hashes: {profile['duplicate_source_hashes']}",
        "",
        "Leakage Warnings",
    ]
    if leakage_findings:
        for finding in leakage_findings:
            lines.append(f"- {finding['severity']}: {finding['column']} - {finding['leakage_reason']}")
    else:
        lines.append("none")
    lines.extend(["", "Top Exploratory Class Differences"])
    if top_differences:
        for row in top_differences:
            lines.append(
                f"- {row['column']}: mean difference={_format(row['mean_difference'])}, "
                f"effect size={_format(row['standardized_effect_size'])}"
            )
    else:
        lines.append("none")
    lines.extend(["", "Highly Correlated Features"])
    if correlations:
        for row in correlations[:10]:
            lines.append(f"- {row['feature_a']} / {row['feature_b']}: r={_format(row['correlation'])}")
    else:
        lines.append("none")
    lines.extend(
        [
            "",
            "Candidate Model Features",
            ", ".join(model_schema["included_features"]) or "none",
            "",
            "Excluded Features",
            f"Excluded feature count: {len(model_schema['excluded_features'])}",
            "",
            "Model-Readiness Assessment",
        ]
    )
    for reason in readiness_reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "Limitations",
            "- The current dataset is a pilot dataset and is not sufficient for reliable generalization.",
            "- Feature differences are exploratory and may reflect dataset source, compression, duration, missingness, or acquisition artifacts.",
            "- Audio missingness cannot always be separated into expected absence, unsupported extraction, parsing failure, or unexplained missingness from the flattened export alone.",
            "",
            "Recommendations",
        ]
    )
    for recommendation in profile["recommendations"]:
        lines.append(f"- {recommendation}")
    lines.extend(["", "No model was trained, calibrated, evaluated, or used for prediction in this audit."])
    return "\n".join(lines) + "\n"


def _missing_rows(column_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "column": profile["column"],
            "missing_count": profile["missing_count"],
            "missing_percentage": profile["missing_percentage"],
            "non_missing_count": profile["non_null_count"],
            "fully_empty": "fully_empty" in profile["flags"],
            "missing_real": profile["missing_by_class"].get("real", 0),
            "missing_ai_generated": profile["missing_by_class"].get("ai_generated", 0),
            "missingness_differs_by_class": profile["missing_by_class"].get("real", 0) != profile["missing_by_class"].get("ai_generated", 0),
            "interpretation": _missing_interpretation(profile),
        }
        for profile in column_profiles
        if profile["missing_count"] > 0
    ]


def _invalid_rows(column_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in column_profiles:
        for example in profile["invalid_numeric_examples"]:
            rows.append({"column": profile["column"], **example})
    return rows


def _constant_rows(column_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "column": profile["column"],
            "unique_value_count": profile["unique_count"],
            "dominant_value": profile["dominant_value"],
            "dominant_value_frequency": profile["dominant_value_frequency"],
            "dominant_value_percentage": profile["dominant_value_percentage"],
            "constant": "constant" in profile["flags"],
            "near_constant": "near_constant" in profile["flags"],
            "recommended_action": profile["recommended_action"],
        }
        for profile in column_profiles
        if "constant" in profile["flags"] or "near_constant" in profile["flags"] or "fully_empty" in profile["flags"]
    ]


def _numeric_stats(values: list[Any]) -> dict[str, Any]:
    numeric = _numeric_values(values)
    missing_count = len([value for value in values if _is_missing(value)])
    if not numeric:
        return {
            "count": 0,
            "missing_count": missing_count,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
            "first_quartile": None,
            "third_quartile": None,
            "interquartile_range": None,
            "unique_count": 0,
        }
    q1, q3 = _quartiles(numeric)
    return {
        "count": len(numeric),
        "missing_count": missing_count,
        "mean": mean(numeric),
        "median": median(numeric),
        "standard_deviation": pstdev(numeric) if len(numeric) > 1 else 0.0,
        "minimum": min(numeric),
        "maximum": max(numeric),
        "first_quartile": q1,
        "third_quartile": q3,
        "interquartile_range": q3 - q1,
        "unique_count": len(set(numeric)),
    }


def _numeric_values(values: list[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        parsed = _parse_number(value)
        if parsed["status"] == "numeric":
            output.append(float(parsed["value"]))
    return output


def _parse_number(value: Any) -> dict[str, Any]:
    if _is_missing(value):
        return {"status": "missing", "value": None}
    if isinstance(value, bool):
        return {"status": "not_numeric", "value": value}
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return {"status": "numeric", "value": float(value)}
        return {"status": "invalid_numeric", "value": _display_value(value), "reason": "NaN or infinity"}
    text = str(value).strip()
    lowered = text.lower()
    if lowered in BOOLEAN_TOKENS:
        return {"status": "not_numeric", "value": value}
    try:
        number = float(text)
    except ValueError:
        if re.search(r"\d", text):
            return {"status": "invalid_numeric", "value": text, "reason": "mixed numeric and non-numeric value"}
        return {"status": "not_numeric", "value": value}
    if not math.isfinite(number):
        return {"status": "invalid_numeric", "value": text, "reason": "NaN or infinity"}
    return {"status": "numeric", "value": number}


def _detected_type(non_missing: list[Any], numeric_success: list[dict[str, Any]], invalid_numeric: list[dict[str, Any]]) -> str:
    if not non_missing:
        return "empty"
    lowered = {str(value).strip().lower() for value in non_missing}
    if lowered and lowered <= BOOLEAN_TOKENS:
        return "boolean"
    if len(numeric_success) == len(non_missing) and not invalid_numeric:
        return "numeric"
    if numeric_success and invalid_numeric:
        return "mixed_numeric"
    return "categorical"


def _impossible_numeric_values(column: str, values: list[Any]) -> list[dict[str, Any]]:
    lowered = column.lower()
    if not _column_has(lowered, ("duration", "width", "height", "count", "ratio", "percentage", "frame_rate", "sample")):
        return []
    output: list[dict[str, Any]] = []
    for index, value in enumerate(values, start=1):
        parsed = _parse_number(value)
        if parsed["status"] != "numeric":
            continue
        number = float(parsed["value"])
        if _column_has(lowered, ("duration", "width", "height", "count", "frame_rate", "sample")) and number < 0:
            output.append({"row_number": index, "value": number, "reason": "negative value is not valid for this column"})
        if _column_has(lowered, ("ratio", "percentage")) and (number < 0 or number > 1):
            output.append({"row_number": index, "value": number, "reason": "ratio/percentage outside [0, 1]"})
    return output


def _column_has(column: str, terms: tuple[str, ...]) -> bool:
    tokens = set(re.split(r"[^a-z0-9]+", column))
    return any(term in tokens for term in terms)


def _invalid_numeric_rows(column: str, rows: list[dict[str, Any]], invalid_numeric: list[dict[str, Any]], impossible: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    invalid_iter = iter(invalid_numeric)
    for index, row in enumerate(rows, start=1):
        parsed = _parse_number(row.get(column))
        if parsed["status"] == "invalid_numeric":
            output.append({"row_number": index, "sample_id": row.get("sample_id"), "value": parsed["value"], "reason": parsed["reason"]})
            next(invalid_iter, None)
    for item in impossible:
        sample_id = rows[item["row_number"] - 1].get("sample_id") if item["row_number"] - 1 < len(rows) else None
        output.append({"row_number": item["row_number"], "sample_id": sample_id, "value": item["value"], "reason": item["reason"]})
    return output


def _duplicate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full_rows = Counter(json.dumps(row, sort_keys=True, default=str) for row in rows)
    return {
        "duplicate_full_rows": sum(count - 1 for count in full_rows.values() if count > 1),
        "duplicate_sample_ids": _duplicate_count(row.get("sample_id") for row in rows),
        "duplicate_source_hashes": _duplicate_count(row.get("identity.video_sha256") or row.get("video_sha256") for row in rows),
        "duplicate_source_paths": _duplicate_count(row.get("source_path") or row.get("stored_source_path") for row in rows),
        "duplicate_feature_file_references": _duplicate_count(row.get("feature_file") for row in rows),
    }


def _duplicate_count(values: Any) -> int:
    filtered = [value for value in values if not _is_missing(value)]
    counts = Counter(filtered)
    return sum(count - 1 for count in counts.values() if count > 1)


def _warnings(unexpected_labels: list[str], missing_labels: list[str], readiness_reasons: list[str], context: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if unexpected_labels:
        warnings.append(f"Unexpected labels found: {unexpected_labels}")
    if missing_labels:
        warnings.append(f"Expected labels missing: {missing_labels}")
    warnings.extend(readiness_reasons)
    if context.get("include_plots"):
        warnings.append("Plot generation is not implemented in v0.9.3 core audit; textual and tabular outputs were generated.")
    return sorted(set(warnings))


def _recommendations(readiness: str, column_profiles: list[dict[str, Any]], candidate_features: list[str]) -> list[str]:
    recommendations = [
        "Keep filename, path, label, source, and identifier fields excluded from future model inputs.",
        "Collect more manually labeled videos before classifier development.",
        "Review missing-heavy and constant features before using the model schema.",
    ]
    if readiness == "insufficient_dataset_size":
        recommendations.append("Treat the current dataset as a workflow pilot only, not as evidence of generalizable detection.")
    if not candidate_features:
        recommendations.append("No eligible numeric features were found; inspect export and validation outputs.")
    if any("audio_summary" in profile["column"] and profile["missing_count"] for profile in column_profiles):
        recommendations.append("Record whether each source video contains usable audio so audio missingness can be interpreted later.")
    return recommendations


def _validate_model_schema(schema: dict[str, Any]) -> None:
    forbidden = ("label", "path", "file", "video_name", "sample_id", "sha256", "hash")
    for feature in schema["included_features"]:
        lowered = feature.lower()
        if any(term in lowered for term in forbidden):
            raise FeatureAuditError(f"Model schema included forbidden leakage feature: {feature}")


def _write_json(path: Path, value: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FeatureAuditError(f"Output already exists: {path}")
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_text(path: Path, value: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FeatureAuditError(f"Output already exists: {path}")
    path.write_text(value, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FeatureAuditError(f"Output already exists: {path}")
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" if row.get(column) is None else row.get(column) for column in columns})


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def _columns_by_category(column_profiles: list[dict[str, Any]], category: str) -> list[str]:
    return [profile["column"] for profile in column_profiles if profile["semantic_category"] == category]


def _class_balance_ratio(class_counts: dict[str, int]) -> float | None:
    counts = [count for count in class_counts.values() if count > 0]
    if len(counts) < 2:
        return None
    return min(counts) / max(counts)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _quartiles(values: list[float]) -> tuple[float, float]:
    array = np.array(sorted(values), dtype=float)
    return float(np.percentile(array, 25)), float(np.percentile(array, 75))


def _pooled_std(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(right) < 2:
        return None
    numerator = (len(left) - 1) * (pstdev(left) ** 2) + (len(right) - 1) * (pstdev(right) ** 2)
    denominator = len(left) + len(right) - 2
    if denominator <= 0:
        return None
    return math.sqrt(numerator / denominator)


def _point_biserial(real_values: list[float], ai_values: list[float]) -> float | None:
    values = real_values + ai_values
    labels = [0.0] * len(real_values) + [1.0] * len(ai_values)
    return _pearson(values, labels)


def _pearson_pair(rows: list[dict[str, Any]], left: str, right: str) -> float | None:
    pairs = []
    for row in rows:
        left_parsed = _parse_number(row.get(left))
        right_parsed = _parse_number(row.get(right))
        if left_parsed["status"] == "numeric" and right_parsed["status"] == "numeric":
            pairs.append((float(left_parsed["value"]), float(right_parsed["value"])))
    if len(pairs) < 2:
        return None
    return _pearson([pair[0] for pair in pairs], [pair[1] for pair in pairs])


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_den = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_den = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_den == 0 or right_den == 0:
        return None
    return numerator / (left_den * right_den)


def _mann_whitney_approx(left: list[float], right: list[float]) -> tuple[float | None, str]:
    if len(left) < 2 or len(right) < 2:
        return None, "mann_whitney_u_normal_approx_unavailable"
    combined = sorted((value, "left") for value in left) + sorted((value, "right") for value in right)
    combined = sorted(combined, key=lambda item: item[0])
    ranks: dict[int, float] = {}
    index = 0
    while index < len(combined):
        end = index + 1
        while end < len(combined) and combined[end][0] == combined[index][0]:
            end += 1
        rank = (index + 1 + end) / 2
        for rank_index in range(index, end):
            ranks[rank_index] = rank
        index = end
    rank_sum_left = sum(ranks[index] for index, item in enumerate(combined) if item[1] == "left")
    n1 = len(left)
    n2 = len(right)
    u1 = rank_sum_left - n1 * (n1 + 1) / 2
    mean_u = n1 * n2 / 2
    std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if std_u == 0:
        return None, "mann_whitney_u_normal_approx_unavailable"
    z = abs((u1 - mean_u) / std_u)
    p_value = math.erfc(z / math.sqrt(2))
    return p_value, "mann_whitney_u_normal_approx"


def _comparison_unavailable(column: str, real_values: list[float], ai_values: list[float], reason: str) -> dict[str, Any]:
    return {
        "column": column,
        "real_sample_count": len(real_values),
        "ai_generated_sample_count": len(ai_values),
        "real_mean": None,
        "ai_generated_mean": None,
        "mean_difference": None,
        "absolute_mean_difference": None,
        "real_median": None,
        "ai_generated_median": None,
        "median_difference": None,
        "pooled_standard_deviation": None,
        "standardized_effect_size": None,
        "point_biserial_correlation": None,
        "class_overlap_indicator": "unavailable",
        "statistical_test_result": reason,
        "p_value": None,
        "test_name": None,
        "interpretation_warning": "insufficient data; no predictive claim",
    }


def _overlap_indicator(left: list[float], right: list[float]) -> str:
    if max(left) < min(right) or max(right) < min(left):
        return "no_observed_range_overlap"
    return "observed_ranges_overlap"


def _missing_interpretation(profile: dict[str, Any]) -> str:
    if profile["column"].startswith("audio_summary."):
        return "audio absence, unsupported extraction, parsing failure, or unexplained missingness cannot be separated from this flattened export alone"
    if profile["missing_count"] == profile["non_null_count"] + profile["missing_count"]:
        return "fully empty"
    return "review missingness before future modeling"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in MISSING_TOKENS:
        return True
    return False


def _normalize_cell(value: Any) -> Any:
    if _is_missing(value):
        return None
    return value


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _relative_path(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return Path(path).name


def _max_severity(values: list[str]) -> str:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(values, key=lambda item: order[item])[0]


def _format(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
