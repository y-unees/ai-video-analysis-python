from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset_tools.models import DatasetManifestEntry, DatasetSummary
from outcome import OUTCOME_FEATURE_SCHEMA_VERSION, validate_outcome_features


DATASET_MANIFEST_SCHEMA_VERSION = "0.9.1"
DEFAULT_DATASET_ROOT = Path("dataset")
MANIFEST_FILENAME = "manifest.jsonl"
FEATURES_DIRECTORY = "features"
EXPORTS_DIRECTORY = "exports"
ALLOWED_LABELS = {"real", "ai_generated"}
FORBIDDEN_FIELD_NAMES = {
    "probability",
    "threshold",
    "confidence",
    "confidence_percentage",
    "percentage",
    "classification",
    "verdict",
    "authenticity_verdict",
    "manipulation_verdict",
    "fake_real_verdict",
    "real_fake_verdict",
}


def register_sample(
    feature_path: Path,
    expected_label: str,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    source: str | None = None,
    generator_or_camera: str | None = None,
    notes: str | None = None,
    allow_duplicate: bool = False,
) -> DatasetManifestEntry:
    if expected_label not in ALLOWED_LABELS:
        raise ValueError("expected_label must be real or ai_generated.")
    dataset_root = Path(dataset_root)
    manifest_path = dataset_root / MANIFEST_FILENAME
    features_dir = dataset_root / FEATURES_DIRECTORY
    feature = _load_json(Path(feature_path))
    feature["identity"]["expected_label"] = expected_label
    validation = validate_outcome_features(feature, feature.get("identity", {}).get("video_sha256"))
    if validation["errors"]:
        raise ValueError(f"Invalid outcome feature artifact: {'; '.join(validation['errors'])}")
    existing = _read_manifest(manifest_path)
    video_sha256 = feature["identity"]["video_sha256"]
    duplicate_count = sum(1 for entry in existing if entry.get("video_sha256") == video_sha256)
    if duplicate_count and not allow_duplicate:
        raise ValueError(f"Duplicate video SHA-256 rejected: {video_sha256}")

    sample_id = _sample_id(expected_label, video_sha256, duplicate_count if allow_duplicate else 0)
    relative_feature_file = f"{FEATURES_DIRECTORY}/{sample_id}.json"
    destination = _safe_dataset_path(dataset_root, relative_feature_file)
    features_dir.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(feature, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")

    entry = DatasetManifestEntry(
        sample_id=sample_id,
        video_name=feature["identity"].get("video_name"),
        video_sha256=video_sha256,
        expected_label=expected_label,
        feature_schema_version=feature["identity"].get("feature_schema_version"),
        application_version=feature["identity"].get("application_version"),
        feature_file=relative_feature_file,
        source=source,
        generator_or_camera=generator_or_camera,
        notes=notes,
        added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    dataset_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
    return entry


def validate_dataset(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    allow_duplicate_hashes: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    manifest_path = dataset_root / MANIFEST_FILENAME
    errors: list[str] = []
    warnings: list[str] = []
    entries = _read_manifest(manifest_path, errors)
    sample_ids: set[str] = set()
    video_hashes: dict[str, int] = {}
    valid_samples = 0
    invalid_samples = 0
    for index, entry in enumerate(entries, start=1):
        prefix = f"manifest line {index}"
        entry_errors = 0
        sample_id = entry.get("sample_id")
        if not sample_id:
            errors.append(f"{prefix}: sample_id is missing.")
            entry_errors += 1
        elif sample_id in sample_ids:
            errors.append(f"{prefix}: duplicate sample_id {sample_id}.")
            entry_errors += 1
        else:
            sample_ids.add(sample_id)
        label = entry.get("expected_label")
        if label not in ALLOWED_LABELS:
            errors.append(f"{prefix}: invalid expected_label.")
            entry_errors += 1
        video_sha = entry.get("video_sha256")
        if video_sha:
            video_hashes[video_sha] = video_hashes.get(video_sha, 0) + 1
        feature_file = entry.get("feature_file")
        if not feature_file:
            errors.append(f"{prefix}: feature_file is missing.")
            entry_errors += 1
            invalid_samples += 1
            continue
        try:
            path = _safe_dataset_path(dataset_root, str(feature_file))
        except ValueError as error:
            errors.append(f"{prefix}: {error}")
            entry_errors += 1
            invalid_samples += 1
            continue
        if not path.exists():
            errors.append(f"{prefix}: feature file is missing: {feature_file}")
            entry_errors += 1
            invalid_samples += 1
            continue
        try:
            feature = _load_json(path)
        except ValueError as error:
            errors.append(f"{prefix}: {error}")
            entry_errors += 1
            invalid_samples += 1
            continue
        feature_identity = feature.get("identity", {})
        if feature_identity.get("video_sha256") != video_sha:
            errors.append(f"{prefix}: manifest and feature SHA-256 values do not match.")
            entry_errors += 1
        if feature_identity.get("expected_label") != label:
            errors.append(f"{prefix}: manifest and feature labels do not match.")
            entry_errors += 1
        validation = validate_outcome_features(feature, video_sha)
        if validation["errors"]:
            errors.extend(f"{prefix}: {error}" for error in validation["errors"])
            entry_errors += len(validation["errors"])
        forbidden_error_count = len(errors)
        _validate_no_forbidden_fields(feature, errors, f"{prefix}.feature")
        entry_errors += len(errors) - forbidden_error_count
        if entry_errors:
            invalid_samples += 1
        else:
            valid_samples += 1
    duplicates = {sha: count for sha, count in video_hashes.items() if count > 1}
    if duplicates and not allow_duplicate_hashes:
        errors.append(f"Duplicate video SHA-256 values found: {sorted(duplicates)}")
    return {
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "warnings": warnings,
        "sample_count": len(entries),
        "valid_sample_count": valid_samples,
        "invalid_sample_count": invalid_samples,
        "duplicate_video_sha256_count": sum(count - 1 for count in duplicates.values()),
    }


def export_dataset_jsonl(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_path: Path | None = None,
) -> Path:
    dataset_root = Path(dataset_root)
    output_path = output_path or dataset_root / EXPORTS_DIRECTORY / "dataset_features.jsonl"
    rows = _export_rows(dataset_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return output_path


def export_dataset_csv(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_path: Path | None = None,
) -> Path:
    dataset_root = Path(dataset_root)
    output_path = output_path or dataset_root / EXPORTS_DIRECTORY / "dataset_features.csv"
    rows = _export_rows(dataset_root)
    columns = sorted({key for row in rows for key in row})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in columns})
    return output_path


def summarize_dataset(dataset_root: Path = DEFAULT_DATASET_ROOT) -> DatasetSummary:
    dataset_root = Path(dataset_root)
    validation = validate_dataset(dataset_root, allow_duplicate_hashes=True)
    entries = _read_manifest(dataset_root / MANIFEST_FILENAME)
    labels = [entry.get("expected_label") for entry in entries]
    versions: set[str] = set()
    d3_counts: dict[str, int] = {"completed": 0, "failed": 0, "unavailable": 0, "missing": 0}
    missing_values: dict[str, int] = {}
    video_hash_counts: dict[str, int] = {}
    for entry in entries:
        sha = entry.get("video_sha256")
        if sha:
            video_hash_counts[sha] = video_hash_counts.get(sha, 0) + 1
        try:
            feature = _load_json(_safe_dataset_path(dataset_root, str(entry.get("feature_file"))))
        except ValueError:
            continue
        versions.add(str(feature.get("identity", {}).get("feature_schema_version")))
        flat = _flatten_features(feature)
        for key, value in flat.items():
            if value is None:
                missing_values[key] = missing_values.get(key, 0) + 1
        status = feature.get("d3_summary", {}).get("d3_status")
        if status == "completed":
            d3_counts["completed"] += 1
        elif status == "failed":
            d3_counts["failed"] += 1
        elif status == "unavailable":
            d3_counts["unavailable"] += 1
        else:
            d3_counts["missing"] += 1
    duplicate_count = sum(count - 1 for count in video_hash_counts.values() if count > 1)
    return DatasetSummary(
        total_samples=len(entries),
        real_sample_count=labels.count("real"),
        ai_generated_sample_count=labels.count("ai_generated"),
        duplicate_count=duplicate_count,
        valid_sample_count=int(validation["valid_sample_count"]),
        invalid_sample_count=int(validation["invalid_sample_count"]),
        feature_schema_versions_present=sorted(version for version in versions if version != "None"),
        d3_status_counts=d3_counts,
        missing_value_count_per_feature=dict(sorted(missing_values.items())),
    )


def _export_rows(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in sorted(_read_manifest(dataset_root / MANIFEST_FILENAME), key=lambda item: item.get("sample_id", "")):
        feature = _load_json(_safe_dataset_path(dataset_root, str(entry["feature_file"])))
        validation = validate_outcome_features(feature, entry.get("video_sha256"))
        if validation["errors"]:
            raise ValueError(f"Invalid feature file for {entry.get('sample_id')}: {'; '.join(validation['errors'])}")
        row = _flatten_features(feature)
        row.update(
            {
                "sample_id": entry.get("sample_id"),
                "expected_label": entry.get("expected_label"),
                "feature_file": entry.get("feature_file"),
                "source": entry.get("source"),
                "generator_or_camera": entry.get("generator_or_camera"),
                "notes": entry.get("notes"),
                "added_at": entry.get("added_at"),
            }
        )
        rows.append(dict(sorted(row.items())))
    return rows


def _flatten_features(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten_features(value[key], child_prefix))
        return output
    return {prefix: value}


def _read_manifest(path: Path, errors: list[str] | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as error:
                if errors is not None:
                    errors.append(f"manifest line {line_number}: invalid JSONL record: {error}")
    return entries


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Feature file is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Feature file is not valid JSON: {path}") from error
    if not isinstance(data, dict):
        raise ValueError("Feature file must contain a JSON object.")
    return data


def _sample_id(label: str, video_sha256: str, duplicate_index: int) -> str:
    suffix = f"-dup{duplicate_index + 1:02d}" if duplicate_index else ""
    return f"{label}-{video_sha256[:16]}{suffix}"


def _safe_dataset_path(dataset_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or _looks_absolute(str(relative_path)):
        raise ValueError(f"Dataset path must be relative: {relative_path}")
    resolved_root = dataset_root.resolve()
    resolved = (dataset_root / path).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Dataset path escapes the dataset root: {relative_path}")
    return resolved


def _looks_absolute(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("/") or value.startswith("\\\\")


def _validate_no_forbidden_fields(value: Any, errors: list[str], path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_FIELD_NAMES:
                errors.append(f"Forbidden field present: {path}.{key}")
            _validate_no_forbidden_fields(child, errors, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_forbidden_fields(child, errors, f"{path}[{index}]")
