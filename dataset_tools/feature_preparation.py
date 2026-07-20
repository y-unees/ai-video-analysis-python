from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import APP_VERSION
from dataset_tools.feature_audit import TARGET_CLASSES, TARGET_COLUMN, load_feature_export
from dataset_tools.manager import DEFAULT_DATASET_ROOT, export_dataset_csv


FEATURE_SCHEMA_VERSION = "0.9.4"
REGISTRY_VERSION = "0.9.4"
DEFAULT_REGISTRY_PATH = Path("schemas") / "feature_registry.json"
DEFAULT_SCHEMA_PATH = Path("schemas") / "feature_schema.json"
DEFAULT_LINEAGE_PATH = Path("schemas") / "feature_lineage.json"
DEFAULT_EVOLUTION_PATH = Path("schemas") / "feature_evolution.json"
DEFAULT_FEATURE_DOCS_PATH = Path("docs") / "FEATURES.md"
DEFAULT_STANDARDIZED_DIR = DEFAULT_DATASET_ROOT / "standardized"
DEFAULT_INPUT_PATH = DEFAULT_DATASET_ROOT / "exports" / "dataset_features.csv"

DATA_TYPES = {"integer", "float", "boolean", "string", "categorical"}
CATEGORIES = {
    "raw_numeric",
    "raw_boolean",
    "raw_categorical",
    "derived_numeric",
    "identifier",
    "target",
    "path",
    "timestamp",
    "source_metadata",
    "analysis_metadata",
    "leakage",
    "unsupported",
    "deprecated",
}
MODEL_FORBIDDEN_CATEGORIES = {"identifier", "target", "path", "timestamp", "source_metadata", "analysis_metadata", "leakage", "unsupported", "deprecated"}
LEAKAGE_EXPORT_NAMES = {
    "expected_label",
    "identity.expected_label",
    "identity.video_name",
    "identity.analysis_id",
    "feature_file",
    "sample_id",
    "source",
    "generator_or_camera",
    "notes",
}


class FeaturePreparationError(ValueError):
    pass


def run_feature_preparation(
    input_path: Path | None = None,
    output_dir: Path = DEFAULT_STANDARDIZED_DIR,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    strict: bool = False,
    overwrite: bool = True,
    allow_unregistered: bool = False,
    report_only: bool = False,
) -> dict[str, Any]:
    input_path = _resolve_input(input_path)
    registry = load_feature_registry(registry_path)
    registry_validation = validate_feature_registry(registry)
    if registry_validation["errors"]:
        raise FeaturePreparationError(f"Invalid feature registry: {'; '.join(registry_validation['errors'])}")
    rows, columns = load_feature_export(input_path)
    value_validation = validate_feature_values(rows, columns, registry, strict=strict, allow_unregistered=allow_unregistered)
    if value_validation["errors"] and strict:
        raise FeaturePreparationError(f"Feature validation failed: {'; '.join(value_validation['errors'][:5])}")
    engineered_rows, engineering_report = engineer_feature_rows(rows, registry)
    schema = generate_feature_schema(registry, schema_path=schema_path)
    compatibility = check_feature_compatibility(new_schema=schema)
    standardized = standardize_feature_rows(engineered_rows, registry, schema)
    output_paths: dict[str, str] = {}
    if not report_only:
        output_paths = write_standardized_outputs(
            rows=standardized,
            profile=_standardized_profile(input_path, rows, columns, registry, schema, value_validation, engineering_report, compatibility),
            validation=value_validation,
            engineering_report=engineering_report,
            compatibility=compatibility,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        write_schema_files(registry, schema, registry_path, schema_path, overwrite=True)
        generate_feature_docs(registry, schema, DEFAULT_FEATURE_DOCS_PATH, overwrite=True)
    result = {
        "project_version": APP_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "registry_version": REGISTRY_VERSION,
        "input_path": _relative(input_path),
        "input_sample_count": len(rows),
        "raw_field_count": len(columns),
        "registered_feature_count": len(registry["features"]),
        "unregistered_field_count": len(value_validation["unregistered_fields"]),
        "raw_candidate_count": len(schema["raw_model_candidate_features"]),
        "derived_candidate_count": len(schema["derived_model_candidate_features"]),
        "standardized_feature_count": len(schema["feature_order"]),
        "excluded_field_count": len(schema["excluded_fields"]),
        "validation_error_count": len(value_validation["errors"]),
        "validation_warning_count": len(value_validation["warnings"]),
        "missing_dependency_count": engineering_report["missing_dependency_count"],
        "out_of_range_count": value_validation["out_of_range_count"],
        "invalid_type_count": value_validation["invalid_type_count"],
        "compatibility_status": compatibility["status"],
        "schema_fingerprint": schema["schema_fingerprint"],
        "output_paths": output_paths,
        "registry_validation": registry_validation,
        "feature_validation": value_validation,
        "engineering_report": engineering_report,
        "compatibility_report": compatibility,
        "schema": schema,
    }
    return result


def load_feature_registry(path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FeaturePreparationError(f"Feature registry is missing: {path}")
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FeaturePreparationError(f"Feature registry is malformed JSON: {error}") from error
    if not isinstance(registry, dict) or not isinstance(registry.get("features"), list):
        raise FeaturePreparationError("Feature registry must be an object with a features list.")
    return registry


def validate_feature_registry(registry: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    information: list[str] = []
    required = {
        "id",
        "export_name",
        "display_name",
        "description",
        "category",
        "source_module",
        "data_type",
        "units",
        "nullable",
        "required",
        "minimum",
        "maximum",
        "allowed_values",
        "model_candidate",
        "leakage_risk",
        "introduced_version",
        "deprecated_version",
        "derived",
        "dependencies",
        "transformation",
        "aliases",
        "notes",
    }
    features = registry.get("features", [])
    ids = [feature.get("id") for feature in features]
    exports = [feature.get("export_name") for feature in features]
    aliases: list[str] = []
    by_id = {feature.get("id"): feature for feature in features}
    for value, count in Counter(ids).items():
        if value and count > 1:
            errors.append(f"Duplicate feature id: {value}")
    for value, count in Counter(exports).items():
        if value and count > 1:
            errors.append(f"Duplicate export_name: {value}")
    for feature in features:
        missing = sorted(required - set(feature))
        if missing:
            errors.append(f"{feature.get('id', '<unknown>')}: missing required fields {missing}")
            continue
        aliases.extend(feature.get("aliases") or [])
        _validate_registry_feature(feature, errors, warnings)
        for dependency in feature.get("dependencies") or []:
            if dependency not in by_id:
                errors.append(f"{feature['id']}: unknown dependency {dependency}")
    for alias, count in Counter(aliases).items():
        if alias and count > 1:
            errors.append(f"Duplicate alias: {alias}")
    errors.extend(_cycle_errors(features))
    deprecated_model = [feature["id"] for feature in features if feature.get("deprecated_version") and feature.get("model_candidate")]
    for feature_id in deprecated_model:
        errors.append(f"{feature_id}: deprecated features cannot be active model candidates")
    information.append(f"Registered features: {len(features)}")
    return {"status": "valid" if not errors else "invalid", "errors": errors, "warnings": warnings, "information": information}


def _validate_registry_feature(feature: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    feature_id = feature["id"]
    if feature["category"] not in CATEGORIES:
        errors.append(f"{feature_id}: invalid category {feature['category']}")
    if feature["data_type"] not in DATA_TYPES:
        errors.append(f"{feature_id}: invalid data_type {feature['data_type']}")
    if not _valid_version(feature["introduced_version"]):
        errors.append(f"{feature_id}: invalid introduced_version")
    if feature["deprecated_version"] is not None and not _valid_version(feature["deprecated_version"]):
        errors.append(f"{feature_id}: invalid deprecated_version")
    minimum = feature.get("minimum")
    maximum = feature.get("maximum")
    if minimum is not None and maximum is not None and minimum > maximum:
        errors.append(f"{feature_id}: minimum is greater than maximum")
    if feature.get("derived") and not feature.get("dependencies"):
        errors.append(f"{feature_id}: derived feature must declare dependencies")
    if feature.get("derived") and not feature.get("transformation"):
        errors.append(f"{feature_id}: derived feature must declare transformation")
    if feature.get("transformation") and feature["transformation"].get("type") not in {"safe_ratio", "sum", "availability_count", "availability_indicator"}:
        errors.append(f"{feature_id}: unsupported transformation type")
    if feature.get("model_candidate") and (feature.get("category") in MODEL_FORBIDDEN_CATEGORIES or feature.get("leakage_risk")):
        errors.append(f"{feature_id}: leakage/identifier/path/target/metadata fields cannot be model candidates")
    if feature.get("export_name") in LEAKAGE_EXPORT_NAMES and feature.get("model_candidate"):
        errors.append(f"{feature_id}: known leakage export cannot be a model candidate")
    if feature.get("derived") and any(dependency == "expected_label" for dependency in feature.get("dependencies", [])):
        errors.append(f"{feature_id}: derived feature cannot depend on target")
    if feature.get("category") == "deprecated" and not feature.get("deprecated_version"):
        warnings.append(f"{feature_id}: deprecated category without deprecated_version")


def validate_feature_values(
    rows: list[dict[str, Any]],
    columns: list[str],
    registry: dict[str, Any],
    strict: bool = False,
    allow_unregistered: bool = False,
) -> dict[str, Any]:
    by_export = {feature["export_name"]: feature for feature in registry["features"]}
    errors: list[str] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    unregistered = sorted(column for column in columns if column not in by_export)
    if unregistered and not allow_unregistered:
        target = errors if strict else warnings
        target.append(f"Unregistered exported fields: {unregistered}")
    missing_required: list[str] = []
    invalid_type_count = 0
    out_of_range_count = 0
    for feature in registry["features"]:
        export_name = feature["export_name"]
        if feature.get("required") and export_name not in columns:
            missing_required.append(export_name)
    if missing_required:
        errors.append(f"Missing required features: {missing_required}")
    for row_index, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        for column in columns:
            feature = by_export.get(column)
            if not feature:
                continue
            value = row.get(column)
            finding = _validate_one_value(feature, value, row_index, sample_id)
            if finding:
                records.append(finding)
                if finding["finding_type"] == "invalid_type":
                    invalid_type_count += 1
                if finding["finding_type"] == "out_of_range":
                    out_of_range_count += 1
                if finding["severity"] == "error":
                    errors.append(f"{column} row {row_index}: {finding['message']}")
                else:
                    warnings.append(f"{column} row {row_index}: {finding['message']}")
    deprecated_fields = sorted(column for column in columns if by_export.get(column, {}).get("deprecated_version"))
    if deprecated_fields:
        warnings.append(f"Deprecated fields present: {deprecated_fields}")
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "project_version": APP_VERSION,
        "status": "valid" if not errors else "invalid",
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "records": records,
        "unregistered_fields": unregistered,
        "missing_required_fields": missing_required,
        "deprecated_fields": deprecated_fields,
        "invalid_type_count": invalid_type_count,
        "out_of_range_count": out_of_range_count,
    }


def engineer_feature_rows(rows: list[dict[str, Any]], registry: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features = registry["features"]
    by_id = {feature["id"]: feature for feature in features}
    by_export = {feature["export_name"]: feature for feature in features}
    engineered: list[dict[str, Any]] = []
    missing_dependency_count = 0
    derivations: list[dict[str, Any]] = []
    derived_features = [feature for feature in features if feature.get("derived")]
    for row in rows:
        new_row = dict(row)
        for feature in derived_features:
            value, missing = _derive_feature(feature, new_row, by_id, by_export)
            new_row[feature["export_name"]] = value
            missing_dependency_count += missing
        engineered.append(new_row)
    for feature in derived_features:
        derivations.append(
            {
                "feature_id": feature["id"],
                "export_name": feature["export_name"],
                "dependencies": feature["dependencies"],
                "transformation": feature["transformation"],
                "uses_target": False,
                "fitted": False,
            }
        )
    return engineered, {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "project_version": APP_VERSION,
        "derived_feature_count": len(derived_features),
        "missing_dependency_count": missing_dependency_count,
        "derivations": derivations,
        "fitted_transformations": False,
        "uses_target": False,
        "no_model_trained": True,
    }


def generate_feature_schema(registry: dict[str, Any], schema_path: Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    raw_candidates = [feature for feature in registry["features"] if feature["model_candidate"] and not feature["derived"]]
    derived_candidates = [feature for feature in registry["features"] if feature["model_candidate"] and feature["derived"]]
    excluded = [feature for feature in registry["features"] if not feature["model_candidate"] and feature["category"] != "target"]
    schema: dict[str, Any] = {
        "project_version": APP_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "registry_version": REGISTRY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "registry_path": _relative(DEFAULT_REGISTRY_PATH),
        "target": {"column": TARGET_COLUMN, "classes": TARGET_CLASSES, "encoding": {"real": 0, "ai_generated": 1}},
        "raw_model_candidate_features": [_schema_feature(feature) for feature in raw_candidates],
        "derived_model_candidate_features": [_schema_feature(feature) for feature in derived_candidates],
        "excluded_fields": [_excluded_feature(feature) for feature in excluded],
        "feature_order": [feature["export_name"] for feature in raw_candidates + derived_candidates],
        "deprecated_fields": [_schema_feature(feature) for feature in registry["features"] if feature.get("deprecated_version")],
        "compatibility_rules": {
            "description_only_change": "non-breaking",
            "new_model_feature": "requires_regeneration",
            "removed_model_feature": "breaking",
            "type_or_unit_change": "breaking",
            "derivation_change": "breaking",
            "nullable_false_to_true": "potentially_breaking",
        },
    }
    schema["schema_fingerprint"] = schema_fingerprint(schema)
    return schema


def schema_fingerprint(schema: dict[str, Any]) -> str:
    canonical = {key: value for key, value in schema.items() if key not in {"generated_at", "schema_fingerprint"}}
    payload = json.dumps(_json_safe(canonical), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def standardize_feature_rows(rows: list[dict[str, Any]], registry: dict[str, Any], schema: dict[str, Any]) -> list[dict[str, Any]]:
    by_export = {feature["export_name"]: feature for feature in registry["features"]}
    output: list[dict[str, Any]] = []
    for row in rows:
        features: dict[str, Any] = {}
        for export_name in schema["feature_order"]:
            value = _coerce_value(row.get(export_name), by_export[export_name])
            if isinstance(value, float) and not math.isfinite(value):
                raise FeaturePreparationError(f"Non-finite standardized output for {export_name}")
            features[export_name] = value
        output.append(
            {
                "sample": {"sample_id": row.get("sample_id"), "label": row.get(TARGET_COLUMN)},
                "schema": {
                    "project_version": APP_VERSION,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "schema_fingerprint": schema["schema_fingerprint"],
                },
                "features": features,
            }
        )
    return output


def write_standardized_outputs(
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    validation: dict[str, Any],
    engineering_report: dict[str, Any],
    compatibility: dict[str, Any],
    output_dir: Path = DEFAULT_STANDARDIZED_DIR,
    overwrite: bool = True,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "standardized_features.csv"
    jsonl_path = output_dir / "standardized_features.jsonl"
    profile_path = output_dir / "standardized_dataset_profile.json"
    validation_json = output_dir / "feature_validation.json"
    validation_csv = output_dir / "feature_validation.csv"
    engineering_json = output_dir / "feature_engineering_report.json"
    engineering_txt = output_dir / "feature_engineering_report.txt"
    compatibility_json = output_dir / "compatibility_report.json"
    compatibility_txt = output_dir / "compatibility_report.txt"
    for path in (csv_path, jsonl_path, profile_path, validation_json, validation_csv, engineering_json, engineering_txt, compatibility_json, compatibility_txt):
        if path.exists() and not overwrite:
            raise FeaturePreparationError(f"Output already exists: {path}")
    feature_order = list(rows[0]["features"]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "expected_label", "feature_schema_version", "schema_fingerprint", *feature_order], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample"]["sample_id"],
                    "expected_label": row["sample"]["label"],
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "schema_fingerprint": row["schema"]["schema_fingerprint"],
                    **{key: "" if row["features"][key] is None else row["features"][key] for key in feature_order},
                }
            )
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")
    _write_json(profile_path, profile)
    _write_json(validation_json, validation)
    _write_csv(validation_csv, validation.get("records", []))
    _write_json(engineering_json, engineering_report)
    _write_text(engineering_txt, _engineering_text(profile, validation, engineering_report, compatibility))
    _write_json(compatibility_json, compatibility)
    _write_text(compatibility_txt, _compatibility_text(compatibility))
    return {
        "standardized_csv": _relative(csv_path),
        "standardized_jsonl": _relative(jsonl_path),
        "standardized_profile": _relative(profile_path),
        "feature_validation_json": _relative(validation_json),
        "feature_validation_csv": _relative(validation_csv),
        "feature_engineering_report_json": _relative(engineering_json),
        "feature_engineering_report_txt": _relative(engineering_txt),
        "compatibility_report_json": _relative(compatibility_json),
        "compatibility_report_txt": _relative(compatibility_txt),
    }


def write_schema_files(registry: dict[str, Any], schema: dict[str, Any], registry_path: Path, schema_path: Path, overwrite: bool = True) -> dict[str, str]:
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    lineage = feature_lineage(registry)
    evolution = feature_evolution(registry)
    _write_json(schema_path, schema)
    _write_json(DEFAULT_LINEAGE_PATH, lineage)
    _write_json(DEFAULT_EVOLUTION_PATH, evolution)
    return {"schema": _relative(schema_path), "lineage": _relative(DEFAULT_LINEAGE_PATH), "evolution": _relative(DEFAULT_EVOLUTION_PATH)}


def check_feature_compatibility(
    old_schema_path: Path | None = None,
    new_schema_path: Path | None = None,
    dataset_path: Path | None = None,
    new_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    status = "compatible"
    if new_schema is None:
        if new_schema_path is None:
            new_schema_path = DEFAULT_SCHEMA_PATH
        if not Path(new_schema_path).exists():
            return {"schema_version": FEATURE_SCHEMA_VERSION, "status": "unknown", "findings": [{"severity": "informational", "message": "new schema is missing"}]}
        new_schema = json.loads(Path(new_schema_path).read_text(encoding="utf-8"))
    if old_schema_path and Path(old_schema_path).exists():
        old_schema = json.loads(Path(old_schema_path).read_text(encoding="utf-8"))
        findings.extend(_schema_diff(old_schema, new_schema))
    if dataset_path and Path(dataset_path).exists():
        rows, columns = load_feature_export(Path(dataset_path))
        missing = [feature for feature in new_schema["feature_order"] if feature not in columns]
        added = [column for column in columns if column not in set(new_schema["feature_order"]) | {"sample_id", "expected_label", "feature_schema_version", "schema_fingerprint"}]
        if missing:
            findings.append({"severity": "breaking", "message": f"dataset is missing standardized features: {missing}"})
        if added:
            findings.append({"severity": "informational", "message": f"dataset has extra columns: {added}"})
        fingerprints = {row.get("schema_fingerprint") for row in rows if row.get("schema_fingerprint")}
        if fingerprints and fingerprints != {new_schema["schema_fingerprint"]}:
            findings.append({"severity": "potentially breaking", "message": "dataset schema fingerprint differs from current schema"})
    severities = {finding["severity"] for finding in findings}
    if "breaking" in severities:
        status = "incompatible"
    elif "potentially breaking" in severities:
        status = "requires_regeneration"
    elif findings:
        status = "compatible_with_warnings"
    return {"schema_version": FEATURE_SCHEMA_VERSION, "status": status, "findings": findings, "new_schema_fingerprint": new_schema.get("schema_fingerprint")}


def generate_feature_docs(registry: dict[str, Any], schema: dict[str, Any], output_path: Path = DEFAULT_FEATURE_DOCS_PATH, overwrite: bool = True) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FeaturePreparationError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Feature Registry and Standardized Schema",
        "",
        f"Project version: `{APP_VERSION}`",
        f"Feature schema version: `{FEATURE_SCHEMA_VERSION}`",
        f"Registry version: `{REGISTRY_VERSION}`",
        f"Schema fingerprint: `{schema['schema_fingerprint']}`",
        "",
        "This document is generated from `schemas/feature_registry.json` and `schemas/feature_schema.json`.",
        "",
        "No fitted normalization, target-aware selection, probabilities, thresholds, or model training are performed in v0.9.4.",
        "",
        "## Feature Categories",
    ]
    for category in sorted(CATEGORIES):
        lines.append(f"- `{category}`")
    lines.extend(["", "## Null Policy", "", "JSON uses `null`; CSV uses empty fields. Missing values are not imputed or replaced with fitted values.", "", "## Standardized Model Features"])
    for export_name in schema["feature_order"]:
        feature = _feature_by_export(registry, export_name)
        lines.extend(_feature_doc_lines(feature))
    lines.extend(["", "## Excluded Fields"])
    for item in schema["excluded_fields"]:
        feature = _feature_by_export(registry, item["export_name"])
        lines.extend(_feature_doc_lines(feature, excluded_reason=item["reason"]))
    lines.extend(["", "## Lineage", "", "All v0.9.4 derived features are same-sample, non-fitted, and do not use the target label.", "", "## Compatibility", "", "Description-only changes are non-breaking. Type, unit, derivation, removed-feature, or feature-order changes can require regeneration or be breaking."])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def print_preparation_summary(result: dict[str, Any]) -> None:
    print("Feature preparation complete.")
    print(f"Project version: {result['project_version']}")
    print(f"Feature schema version: {result['feature_schema_version']}")
    print(f"Input samples: {result['input_sample_count']}")
    print(f"Registered features: {result['registered_feature_count']}")
    print(f"Raw candidate features: {result['raw_candidate_count']}")
    print(f"Derived candidate features: {result['derived_candidate_count']}")
    print(f"Standardized model features: {result['standardized_feature_count']}")
    print(f"Excluded fields: {result['excluded_field_count']}")
    print(f"Validation errors: {result['validation_error_count']}")
    print(f"Validation warnings: {result['validation_warning_count']}")
    print(f"Unregistered fields: {result['unregistered_field_count']}")
    print(f"Out-of-range values: {result['out_of_range_count']}")
    print(f"Missing derived dependencies: {result['missing_dependency_count']}")
    print(f"Compatibility status: {result['compatibility_status']}")
    print(f"Schema fingerprint: {result['schema_fingerprint']}")
    print(f"Standardized CSV: {result['output_paths'].get('standardized_csv')}")
    print(f"Standardized JSONL: {result['output_paths'].get('standardized_jsonl')}")
    print(f"Feature documentation: {DEFAULT_FEATURE_DOCS_PATH}")
    print("No fitted normalization performed.")
    print("No model was trained.")


def _derive_feature(feature: dict[str, Any], row: dict[str, Any], by_id: dict[str, dict[str, Any]], by_export: dict[str, dict[str, Any]]) -> tuple[Any, int]:
    transformation = feature["transformation"]
    transform_type = transformation["type"]
    dependencies = feature["dependencies"]
    values = []
    missing = 0
    for dependency in dependencies:
        export_name = by_id[dependency]["export_name"]
        value = row.get(export_name)
        if _is_missing(value):
            missing += 1
        values.append(value)
    if transform_type == "safe_ratio":
        numerator = _number(values[0])
        denominator = _number(values[1])
        if numerator is None or denominator in {None, 0.0}:
            return None, missing
        return numerator / denominator, missing
    if transform_type == "sum":
        numbers = [_number(value) for value in values]
        if any(value is None for value in numbers):
            return None, missing
        return sum(numbers), missing
    if transform_type == "availability_indicator":
        value = values[0]
        return 0 if _is_missing(value) or value in {False, "False", "false", 0, "0"} else 1, missing
    if transform_type == "availability_count":
        count = 0
        for dependency, value in zip(dependencies, values):
            export = by_id[dependency]["export_name"]
            if export.endswith("d3_status"):
                count += 1 if str(value) == "completed" else 0
            else:
                number = _number(value)
                count += 1 if number is not None and number > 0 else 0
        return count / len(values) if values else None, missing
    raise FeaturePreparationError(f"Unsupported transformation: {transform_type}")


def _standardized_profile(
    input_path: Path,
    raw_rows: list[dict[str, Any]],
    raw_columns: list[str],
    registry: dict[str, Any],
    schema: dict[str, Any],
    validation: dict[str, Any],
    engineering_report: dict[str, Any],
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    return {
        "project_version": APP_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "registry_version": REGISTRY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_path": _relative(input_path),
        "input_sample_count": len(raw_rows),
        "raw_field_count": len(raw_columns),
        "registered_feature_count": len(registry["features"]),
        "unregistered_feature_count": len(validation["unregistered_fields"]),
        "raw_candidate_count": len(schema["raw_model_candidate_features"]),
        "derived_feature_count": engineering_report["derived_feature_count"],
        "standardized_feature_count": len(schema["feature_order"]),
        "excluded_field_count": len(schema["excluded_fields"]),
        "validation_error_count": len(validation["errors"]),
        "validation_warning_count": len(validation["warnings"]),
        "missing_dependency_count": engineering_report["missing_dependency_count"],
        "out_of_range_count": validation["out_of_range_count"],
        "invalid_type_count": validation["invalid_type_count"],
        "compatibility_status": compatibility["status"],
        "schema_fingerprint": schema["schema_fingerprint"],
        "no_fitted_normalization": True,
        "no_model_trained": True,
    }


def _schema_feature(feature: dict[str, Any]) -> dict[str, Any]:
    keys = ["id", "export_name", "data_type", "units", "nullable", "minimum", "maximum", "source_module", "dependencies", "derived", "model_candidate", "introduced_version", "deprecated_version"]
    return {key: feature.get(key) for key in keys}


def _excluded_feature(feature: dict[str, Any]) -> dict[str, Any]:
    reason = "not a model candidate"
    if feature["category"] in MODEL_FORBIDDEN_CATEGORIES:
        reason = f"category is {feature['category']}"
    if feature.get("leakage_risk"):
        reason = "leakage risk"
    if feature.get("deprecated_version"):
        reason = "deprecated"
    return {"id": feature["id"], "export_name": feature["export_name"], "category": feature["category"], "reason": reason}


def feature_lineage(registry: dict[str, Any]) -> dict[str, Any]:
    records = []
    for feature in registry["features"]:
        if feature.get("derived"):
            records.append(
                {
                    "feature_id": feature["id"],
                    "export_name": feature["export_name"],
                    "dependencies": feature["dependencies"],
                    "dependency_order": feature["dependencies"],
                    "source_modules": sorted({_feature_by_id(registry, dependency)["source_module"] for dependency in feature["dependencies"]}),
                    "formula": feature["transformation"],
                    "null_behavior": "returns null when required numeric inputs are missing",
                    "zero_denominator_behavior": feature["transformation"].get("zero_denominator"),
                    "introduced_version": feature["introduced_version"],
                    "explanation": feature["description"],
                    "uses_target": False,
                    "fitted": False,
                }
            )
    return {"schema_version": FEATURE_SCHEMA_VERSION, "project_version": APP_VERSION, "features": records}


def feature_evolution(registry: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "project_version": APP_VERSION,
        "features": [
            {
                "id": feature["id"],
                "introduced_version": feature["introduced_version"],
                "aliases": feature.get("aliases", []),
                "renamed_from": None,
                "deprecated_version": feature.get("deprecated_version"),
                "removed_version": None,
                "type_changes": [],
                "unit_changes": [],
                "category_changes": [],
                "formula_changes": [],
            }
            for feature in registry["features"]
        ],
    }


def _schema_diff(old_schema: dict[str, Any], new_schema: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    old_order = old_schema.get("feature_order", [])
    new_order = new_schema.get("feature_order", [])
    if old_schema.get("schema_fingerprint") == new_schema.get("schema_fingerprint") and old_order == new_order:
        return findings
    if old_order != new_order:
        findings.append({"severity": "potentially breaking", "message": "feature order changed"})
    removed = sorted(set(old_order) - set(new_order))
    added = sorted(set(new_order) - set(old_order))
    if removed:
        findings.append({"severity": "breaking", "message": f"model features removed: {removed}"})
    if added:
        findings.append({"severity": "potentially breaking", "message": f"model features added: {added}"})
    old_features = {item["export_name"]: item for item in old_schema.get("raw_model_candidate_features", []) + old_schema.get("derived_model_candidate_features", [])}
    new_features = {item["export_name"]: item for item in new_schema.get("raw_model_candidate_features", []) + new_schema.get("derived_model_candidate_features", [])}
    for name in sorted(set(old_features) & set(new_features)):
        for key in ("data_type", "units", "nullable", "minimum", "maximum", "dependencies"):
            if old_features[name].get(key) != new_features[name].get(key):
                findings.append({"severity": "breaking", "message": f"{name}: {key} changed"})
    return findings


def _validate_one_value(feature: dict[str, Any], value: Any, row_index: int, sample_id: Any) -> dict[str, Any] | None:
    if _is_missing(value):
        if feature["required"] or not feature["nullable"]:
            return {"row_number": row_index, "sample_id": sample_id, "column": feature["export_name"], "finding_type": "invalid_null", "severity": "error", "message": "null value is not allowed"}
        return None
    coerced = _coerce_value(value, feature, invalid_as_none=True)
    if coerced is None:
        return {"row_number": row_index, "sample_id": sample_id, "column": feature["export_name"], "finding_type": "invalid_type", "severity": "error", "message": f"value does not match {feature['data_type']}"}
    if isinstance(coerced, float) and not math.isfinite(coerced):
        return {"row_number": row_index, "sample_id": sample_id, "column": feature["export_name"], "finding_type": "invalid_type", "severity": "error", "message": "non-finite numeric value"}
    if feature["allowed_values"] is not None and coerced not in feature["allowed_values"]:
        return {"row_number": row_index, "sample_id": sample_id, "column": feature["export_name"], "finding_type": "invalid_type", "severity": "error", "message": "value not in allowed_values"}
    if isinstance(coerced, (int, float)):
        minimum = feature.get("minimum")
        maximum = feature.get("maximum")
        if minimum is not None and coerced < minimum:
            return {"row_number": row_index, "sample_id": sample_id, "column": feature["export_name"], "finding_type": "out_of_range", "severity": "error", "message": "value below minimum"}
        if maximum is not None and coerced > maximum:
            return {"row_number": row_index, "sample_id": sample_id, "column": feature["export_name"], "finding_type": "out_of_range", "severity": "error", "message": "value above maximum"}
    return None


def _coerce_value(value: Any, feature: dict[str, Any], invalid_as_none: bool = False) -> Any:
    if _is_missing(value):
        return None
    data_type = feature["data_type"]
    try:
        if data_type == "float":
            number = float(value)
            if not math.isfinite(number):
                return None if invalid_as_none else FeaturePreparationError("non-finite")
            return number
        if data_type == "integer":
            if isinstance(value, str) and re.search(r"[.eE]", value):
                number = float(value)
                if not number.is_integer():
                    return None
                return int(number)
            return int(value)
        if data_type == "boolean":
            if isinstance(value, bool):
                return value
            lowered = str(value).strip().lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            return None
        if data_type in {"string", "categorical"}:
            return str(value)
    except (TypeError, ValueError):
        return None
    return None


def _number(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _valid_version(value: str) -> bool:
    return bool(re.match(r"^\d+\.\d+\.\d+$", str(value)))


def _cycle_errors(features: list[dict[str, Any]]) -> list[str]:
    graph = {feature.get("id"): feature.get("dependencies", []) for feature in features}
    errors: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, stack: list[str]) -> None:
        if node in visiting:
            errors.append(f"Circular dependency detected: {' -> '.join(stack + [node])}")
            return
        if node in visited:
            return
        visiting.add(node)
        for child in graph.get(node, []):
            visit(child, stack + [node])
        visiting.remove(node)
        visited.add(node)

    for feature_id in graph:
        visit(feature_id, [])
    return errors


def _resolve_input(input_path: Path | None) -> Path:
    if input_path:
        return Path(input_path)
    if DEFAULT_INPUT_PATH.exists():
        return DEFAULT_INPUT_PATH
    return export_dataset_csv(DEFAULT_DATASET_ROOT)


def _feature_by_export(registry: dict[str, Any], export_name: str) -> dict[str, Any]:
    for feature in registry["features"]:
        if feature["export_name"] == export_name:
            return feature
    raise KeyError(export_name)


def _feature_by_id(registry: dict[str, Any], feature_id: str) -> dict[str, Any]:
    for feature in registry["features"]:
        if feature["id"] == feature_id:
            return feature
    raise KeyError(feature_id)


def _feature_doc_lines(feature: dict[str, Any], excluded_reason: str | None = None) -> list[str]:
    lines = [
        "",
        f"### `{feature['export_name']}`",
        "",
        f"- ID: `{feature['id']}`",
        f"- Display name: {feature['display_name']}",
        f"- Description: {feature['description']}",
        f"- Category: `{feature['category']}`",
        f"- Type: `{feature['data_type']}`",
        f"- Units: `{feature['units']}`",
        f"- Nullable: `{feature['nullable']}`",
        f"- Range: `{feature['minimum']}` to `{feature['maximum']}`",
        f"- Source module: `{feature['source_module']}`",
        f"- Model candidate: `{feature['model_candidate']}`",
        f"- Introduced: `{feature['introduced_version']}`",
        f"- Deprecated: `{feature['deprecated_version']}`",
        f"- Dependencies: `{feature['dependencies']}`",
        f"- Transformation: `{feature['transformation']}`",
        f"- Aliases: `{feature['aliases']}`",
        f"- Notes: {feature['notes']}",
    ]
    if excluded_reason:
        lines.append(f"- Exclusion reason: {excluded_reason}")
    return lines


def _engineering_text(profile: dict[str, Any], validation: dict[str, Any], engineering: dict[str, Any], compatibility: dict[str, Any]) -> str:
    lines = [
        "Feature Engineering and Standardization Report",
        "",
        f"Project version: {profile['project_version']}",
        f"Feature schema version: {profile['feature_schema_version']}",
        f"Input path: {profile['input_path']}",
        f"Input samples: {profile['input_sample_count']}",
        f"Raw fields: {profile['raw_field_count']}",
        f"Registered features: {profile['registered_feature_count']}",
        f"Unregistered features: {profile['unregistered_feature_count']}",
        f"Raw candidates: {profile['raw_candidate_count']}",
        f"Derived features: {profile['derived_feature_count']}",
        f"Standardized features: {profile['standardized_feature_count']}",
        f"Excluded fields: {profile['excluded_field_count']}",
        f"Validation errors: {profile['validation_error_count']}",
        f"Validation warnings: {profile['validation_warning_count']}",
        f"Missing dependencies: {profile['missing_dependency_count']}",
        f"Out-of-range values: {profile['out_of_range_count']}",
        f"Invalid type values: {profile['invalid_type_count']}",
        f"Compatibility status: {compatibility['status']}",
        f"Schema fingerprint: {profile['schema_fingerprint']}",
        f"Generated documentation path: {DEFAULT_FEATURE_DOCS_PATH}",
        "",
        "Limitations",
        "- Missing-value causes are preserved as null where the source export cannot distinguish unavailable, unsupported, failed extraction, or not applicable states.",
        "- Fitted scaling, fitted imputation, target-aware feature selection, and model training are deferred until group-safe dataset splitting exists.",
        "",
        "No fitted normalization occurred.",
        "No model was trained.",
    ]
    return "\n".join(lines) + "\n"


def _compatibility_text(compatibility: dict[str, Any]) -> str:
    lines = ["Feature Compatibility Report", "", f"Status: {compatibility['status']}", f"Schema fingerprint: {compatibility.get('new_schema_fingerprint')}", ""]
    if compatibility["findings"]:
        for finding in compatibility["findings"]:
            lines.append(f"- {finding['severity']}: {finding['message']}")
    else:
        lines.append("No compatibility findings.")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return value


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return Path(path).name
