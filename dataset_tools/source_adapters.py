from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_DATASET = "faceforensics_pp"
ALLOWED_NORMALIZED_LABELS = {"real", "ai_generated"}
FORBIDDEN_PLAN_FIELDS = {
    "probability",
    "threshold",
    "confidence",
    "classification",
    "verdict",
    "authenticity_verdict",
    "manipulation_verdict",
}
LABEL_COLUMNS = ("label", "class", "category", "target", "type")
PATH_COLUMNS = ("path", "filepath", "file_path", "filename", "file", "video", "video_path")
MANIPULATION_COLUMNS = ("manipulation", "method", "manipulation_method", "attack", "generator")
COMPRESSION_COLUMNS = ("compression", "quality", "level")


def audit_faceforensics_metadata(
    metadata_path: Path,
    dataset_source_root: Path = Path("dataset_sources") / "faceforensics_pp",
    videos_root: Path | None = None,
) -> dict[str, Any]:
    videos_root = videos_root or dataset_source_root / "videos"
    sniffed = _read_csv(metadata_path)
    rows = sniffed["rows"]
    columns = sniffed["columns"]
    duplicate_rows = len(rows) - len({_row_key(row, columns) for row in rows})
    normalized = normalize_faceforensics_rows(metadata_path)
    availability = [_availability(record, videos_root) for record in normalized]
    path_values = [_first_value(row, _matching_columns(columns, PATH_COLUMNS)) for row in rows]
    label_values = [_first_value(row, _matching_columns(columns, LABEL_COLUMNS)) for row in rows]
    return {
        "source_dataset": SOURCE_DATASET,
        "metadata_source_file": _safe_relative(metadata_path),
        "resolved_file_path": str(metadata_path.resolve()),
        "encoding": sniffed["encoding"],
        "delimiter": sniffed["delimiter"],
        "row_count": len(rows),
        "column_names": columns,
        "inferred_column_types": {column: _infer_type([row.get(column) for row in rows]) for column in columns},
        "duplicate_row_count": duplicate_rows,
        "duplicate_filename_or_path_count": _duplicate_count([value for value in path_values if value]),
        "missing_value_count_per_column": {
            column: sum(1 for row in rows if _blank(row.get(column)))
            for column in columns
        },
        "unique_label_values": sorted({value for value in label_values if value}),
        "label_counts": dict(sorted(Counter(value for value in label_values if value).items())),
        "unique_manipulation_or_category_values": _unique_values(rows, columns, MANIPULATION_COLUMNS),
        "row_representation": _row_representation(path_values),
        "path_style": _path_style(path_values),
        "video_availability": dict(sorted(Counter(item["status"] for item in availability).items())),
        "trusted_label_mapping": _trusted_label_mapping_summary(normalized),
        "normalized_record_count": len(normalized),
    }


def write_faceforensics_audit(
    metadata_path: Path,
    dataset_source_root: Path = Path("dataset_sources") / "faceforensics_pp",
) -> dict[str, Any]:
    audit = audit_faceforensics_metadata(metadata_path, dataset_source_root)
    output = dataset_source_root / "audits" / "metadata_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit | {"audit_report_path": _safe_relative(output)}


def normalize_faceforensics_rows(metadata_path: Path) -> list[dict[str, Any]]:
    sniffed = _read_csv(metadata_path)
    rows = sniffed["rows"]
    columns = sniffed["columns"]
    label_columns = _matching_columns(columns, LABEL_COLUMNS)
    path_columns = _matching_columns(columns, PATH_COLUMNS)
    manipulation_columns = _matching_columns(columns, MANIPULATION_COLUMNS)
    compression_columns = _matching_columns(columns, COMPRESSION_COLUMNS)
    output = []
    for index, row in enumerate(rows, start=1):
        original_label = _first_value(row, label_columns)
        source_path = _first_value(row, path_columns)
        normalized_label = _normalize_label(original_label)
        video_name = Path(str(source_path).replace("\\", "/")).name if source_path else None
        output.append(
            {
                "source_dataset": SOURCE_DATASET,
                "source_row_id": str(index),
                "video_name": video_name,
                "source_video_path": _clean_path(source_path),
                "source_video_path_invalid": bool(source_path and _unsafe_relative_path(source_path.strip().replace("\\", "/"))),
                "normalized_label": normalized_label,
                "original_label": original_label,
                "manipulation_method": _first_value(row, manipulation_columns),
                "compression": _first_value(row, compression_columns),
                "frame_count": _to_int(_value_by_contains(row, columns, "frame")),
                "width": _to_int(_value_by_contains(row, columns, "width")),
                "height": _to_int(_value_by_contains(row, columns, "height")),
                "codec": _value_by_contains(row, columns, "codec"),
                "file_size_mb": _to_float(_value_by_contains(row, columns, "size")),
                "metadata_source_file": _safe_relative(metadata_path),
            }
        )
    return output


def plan_faceforensics_samples(
    metadata_path: Path,
    dataset_source_root: Path = Path("dataset_sources") / "faceforensics_pp",
    real_count: int = 10,
    ai_count: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    videos_root = dataset_source_root / "videos"
    records = normalize_faceforensics_rows(metadata_path)
    annotated = [{**record, **_availability(record, videos_root)} for record in records]
    selected = _select_balanced(annotated, real_count, ai_count, seed)
    plan_path = dataset_source_root / "plans" / "pilot_plan.jsonl"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8", newline="\n") as file:
        for index, record in enumerate(selected, start=1):
            entry = {
                "plan_index": index,
                "source_dataset": SOURCE_DATASET,
                "metadata_source_file": _safe_relative(metadata_path),
                "source_row_id": record["source_row_id"],
                "video_name": record["video_name"],
                "resolved_video_path": record.get("resolved_video_path"),
                "expected_label": record["normalized_label"],
                "original_label": record["original_label"],
                "manipulation_method": record.get("manipulation_method"),
                "compression": record.get("compression"),
                "selection_seed": seed,
                "video_available": record["status"] == "video_found",
                "planning_notes": record.get("planning_notes", []),
            }
            file.write(json.dumps(entry, sort_keys=True) + "\n")
    counts = Counter(record["normalized_label"] for record in selected)
    return {
        "plan_path": _safe_relative(plan_path),
        "selected_count": len(selected),
        "selected_real_count": counts.get("real", 0),
        "selected_ai_generated_count": counts.get("ai_generated", 0),
        "requested_real_count": real_count,
        "requested_ai_generated_count": ai_count,
        "shortages": {
            "real": max(0, real_count - counts.get("real", 0)),
            "ai_generated": max(0, ai_count - counts.get("ai_generated", 0)),
        },
    }


def validate_plan(plan_path: Path, project_root: Path = Path("."), requested_real_count: int | None = None, requested_ai_count: int | None = None) -> dict[str, Any]:
    errors: list[str] = []
    entries = _read_jsonl(plan_path, errors)
    previous = 0
    indices: set[int] = set()
    paths: set[str] = set()
    for entry in entries:
        for key in ("plan_index", "source_dataset", "metadata_source_file", "source_row_id", "video_name", "expected_label", "selection_seed", "video_available"):
            if key not in entry:
                errors.append(f"Plan entry missing required field: {key}")
        index = entry.get("plan_index")
        if not isinstance(index, int) or index <= previous:
            errors.append("Plan entries are not deterministically ordered by increasing plan_index.")
        if isinstance(index, int):
            if index in indices:
                errors.append(f"Duplicate plan_index: {index}")
            indices.add(index)
            previous = index
        if entry.get("expected_label") not in ALLOWED_NORMALIZED_LABELS:
            errors.append("Plan entry has invalid expected_label.")
        metadata = entry.get("metadata_source_file")
        if metadata and not (project_root / metadata).exists():
            errors.append(f"Plan metadata source file is missing: {metadata}")
        resolved = entry.get("resolved_video_path")
        if resolved:
            if _unsafe_relative_path(resolved):
                errors.append(f"Plan path is unsafe: {resolved}")
            if resolved in paths:
                errors.append(f"Duplicate selected video path: {resolved}")
            paths.add(resolved)
            if entry.get("video_available") and not (project_root / resolved).exists():
                errors.append(f"Available plan video no longer exists: {resolved}")
        _check_forbidden(entry, errors)
    labels = Counter(entry.get("expected_label") for entry in entries)
    if requested_real_count is not None and labels.get("real", 0) != requested_real_count:
        errors.append("Plan does not match requested real sample count.")
    if requested_ai_count is not None and labels.get("ai_generated", 0) != requested_ai_count:
        errors.append("Plan does not match requested AI-generated sample count.")
    return {"status": "valid" if not errors else "invalid", "errors": errors, "sample_count": len(entries)}


def match_faceforensics_video(video_path: Path, metadata_path: Path | None) -> dict[str, Any] | None:
    if metadata_path is None or not metadata_path.exists():
        return None
    name = video_path.name.lower()
    matches = [record for record in normalize_faceforensics_rows(metadata_path) if record.get("video_name", "").lower() == name]
    trusted = [record for record in matches if record.get("normalized_label") in ALLOWED_NORMALIZED_LABELS]
    if len(trusted) == 1:
        return trusted[0]
    return None


def _read_csv(path: Path) -> dict[str, Any]:
    encoding = "utf-8-sig"
    try:
        text = path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        encoding = "latin-1"
        text = path.read_text(encoding=encoding)
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
    columns = list(rows[0].keys()) if rows else []
    return {"encoding": encoding, "delimiter": delimiter, "rows": rows, "columns": columns}


def _availability(record: dict[str, Any], videos_root: Path) -> dict[str, Any]:
    if record.get("normalized_label") not in ALLOWED_NORMALIZED_LABELS or not record.get("video_name"):
        return {"status": "invalid_metadata", "resolved_video_path": None, "planning_notes": ["invalid_or_untrusted_metadata"]}
    if record.get("source_video_path_invalid"):
        return {"status": "invalid_metadata", "resolved_video_path": None, "planning_notes": ["unsafe_source_video_path"]}
    candidates: list[Path] = []
    source_path = record.get("source_video_path")
    if source_path and not _unsafe_relative_path(source_path):
        exact = videos_root / source_path
        if exact.exists():
            candidates.append(exact)
    direct = videos_root / record["video_name"]
    if direct.exists():
        candidates.append(direct)
    recursive = list(videos_root.rglob(record["video_name"])) if videos_root.exists() else []
    candidates.extend(recursive)
    unique = sorted({path.resolve() for path in candidates})
    if len(unique) == 1:
        return {"status": "video_found", "resolved_video_path": _safe_relative(unique[0]), "planning_notes": []}
    if len(unique) > 1:
        return {"status": "ambiguous_video_match", "resolved_video_path": None, "planning_notes": ["multiple_files_with_same_name"]}
    return {"status": "video_missing", "resolved_video_path": None, "planning_notes": ["video_not_found"]}


def _select_balanced(records: list[dict[str, Any]], real_count: int, ai_count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected = []
    for label, count in (("real", real_count), ("ai_generated", ai_count)):
        eligible = [record for record in records if record.get("normalized_label") == label]
        eligible.sort(key=lambda item: (0 if item["status"] == "video_found" else 1, item.get("manipulation_method") or "", item.get("video_name") or "", item["source_row_id"]))
        buckets: dict[str, list[dict[str, Any]]] = {}
        for record in eligible:
            buckets.setdefault(record.get("manipulation_method") or "unknown", []).append(record)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        ordered: list[dict[str, Any]] = []
        while any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key]:
                    ordered.append(buckets[key].pop(0))
        seen_names: set[str] = set()
        seen_paths: set[str] = set()
        for record in ordered:
            name = record.get("video_name")
            path = record.get("resolved_video_path")
            if name in seen_names or (path and path in seen_paths):
                continue
            seen_names.add(name)
            if path:
                seen_paths.add(path)
            selected.append(record)
            if sum(1 for item in selected if item.get("normalized_label") == label) >= count:
                break
    return sorted(selected, key=lambda item: (item["normalized_label"], item.get("manipulation_method") or "", item.get("video_name") or "", item["source_row_id"]))


def _normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"real", "original", "authentic", "pristine", "0"}:
        return "real"
    if lowered in {"fake", "manipulated", "ai_generated", "ai-generated", "synthetic", "generated", "deepfake", "1"}:
        return "ai_generated"
    return None


def _matching_columns(columns: list[str], candidates: tuple[str, ...]) -> list[str]:
    return [column for column in columns if any(candidate in column.lower().replace(" ", "_") for candidate in candidates)]


def _first_value(row: dict[str, Any], columns: list[str]) -> str | None:
    for column in columns:
        value = row.get(column)
        if not _blank(value):
            return str(value).strip()
    return None


def _value_by_contains(row: dict[str, Any], columns: list[str], pattern: str) -> str | None:
    return _first_value(row, [column for column in columns if pattern in column.lower()])


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _row_key(row: dict[str, Any], columns: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")).strip() for column in columns)


def _infer_type(values: list[Any]) -> str:
    clean = [str(value).strip() for value in values if not _blank(value)]
    if not clean:
        return "empty"
    if all(_to_int(value) is not None for value in clean):
        return "integer"
    if all(_to_float(value) is not None for value in clean):
        return "number"
    lowered = {value.lower() for value in clean}
    if lowered <= {"true", "false", "0", "1", "yes", "no"}:
        return "boolean_like"
    return "string"


def _duplicate_count(values: list[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def _unique_values(rows: list[dict[str, Any]], columns: list[str], candidates: tuple[str, ...]) -> dict[str, list[str]]:
    return {
        column: sorted({str(row.get(column)).strip() for row in rows if not _blank(row.get(column))})
        for column in _matching_columns(columns, candidates)
    }


def _row_representation(paths: list[str | None]) -> str:
    clean = [path for path in paths if path]
    if not clean:
        return "unknown"
    frame_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    suffixes = {Path(path.replace("\\", "/")).suffix.lower() for path in clean}
    if suffixes & video_extensions:
        return "videos"
    if suffixes & frame_extensions:
        return "extracted_frames"
    return "unknown"


def _path_style(paths: list[str | None]) -> str:
    clean = [path for path in paths if path]
    if not clean:
        return "unknown"
    if any(re.match(r"^[A-Za-z]:[\\/]", path) for path in clean):
        return "windows_absolute"
    if any(path.startswith("/") for path in clean):
        return "unix_absolute"
    if any("\\" in path for path in clean):
        return "windows_relative"
    if any("/" in path for path in clean):
        return "unix_relative"
    return "filename_only"


def _trusted_label_mapping_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record.get("normalized_label") or "unmapped" for record in records).items()))


def _clean_path(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().replace("\\", "/")
    if _unsafe_relative_path(cleaned):
        return None
    return cleaned


def _unsafe_relative_path(value: str) -> bool:
    path = Path(value)
    return path.is_absolute() or bool(re.match(r"^[A-Za-z]:[\\/]", value)) or ".." in path.parts


def _safe_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    entries = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as error:
                errors.append(f"Plan line {line_number} is invalid JSON: {error}")
    return entries


def _check_forbidden(value: Any, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_PLAN_FIELDS:
                errors.append(f"Forbidden plan field: {key}")
            _check_forbidden(child, errors)
    elif isinstance(value, list):
        for child in value:
            _check_forbidden(child, errors)


def _to_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None
