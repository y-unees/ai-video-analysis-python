from __future__ import annotations

import math
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from config import APP_VERSION, SCHEMA_VERSION
from frame_analyzer import summarize_frame_analysis


def validate_report(report: dict[str, Any]) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append("Schema version is missing or does not match the application schema.")
    if report.get("analysis_environment", {}).get("application_version") != APP_VERSION:
        errors.append("Application version is missing or inconsistent.")

    frames = report.get("frame_analysis", {}).get("frames", [])
    comparisons = report.get("frame_analysis", {}).get("comparisons", [])
    sampling = report.get("sampling", {})
    summary = report.get("frame_analysis", {}).get("summary", {})

    if sampling.get("decoded_frame_count") != len(frames):
        errors.append("Decoded-frame count does not equal the number of frame records.")

    indices = [frame.get("sample_index") for frame in frames]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        errors.append("Frame sample indices are not unique and ordered.")

    expected_comparisons = max(len(frames) - 1, 0)
    if len(comparisons) != expected_comparisons:
        errors.append("Comparison count does not match consecutive frame count.")

    frame_by_index = {frame.get("sample_index"): frame for frame in frames}
    for comparison in comparisons:
        from_index = comparison.get("from_sample_index")
        to_index = comparison.get("to_sample_index")
        from_frame = frame_by_index.get(from_index)
        to_frame = frame_by_index.get(to_index)
        if from_frame is None or to_frame is None:
            errors.append("A comparison references an unknown frame index.")
            continue
        if comparison.get("start_timestamp_seconds") != from_frame.get("decoded_timestamp_seconds"):
            errors.append("A comparison start timestamp does not match the referenced frame.")
        if comparison.get("end_timestamp_seconds") != to_frame.get("decoded_timestamp_seconds"):
            errors.append("A comparison end timestamp does not match the referenced frame.")

    recomputed_summary = summarize_frame_analysis(frames, comparisons)
    for key, expected_value in recomputed_summary.items():
        actual_value = summary.get(key)
        if not _same_value(actual_value, expected_value):
            errors.append(f"Frame-analysis summary value is inconsistent: {key}.")

    observation_ids = _collect_observation_ids(report.get("observations", {}))
    if len(observation_ids) != len(set(observation_ids)):
        errors.append("Observation IDs are not unique.")

    for path_value in _relative_paths(report):
        if _is_absolute_path(path_value):
            errors.append(f"Report contains an absolute path: {path_value}")

    _check_numeric_values(report, errors)
    _check_non_negative_counts(report, errors)
    _check_analysis_times(report, errors)

    artifacts = report.get("artifacts", {})
    if artifacts.get("paths_relative_to") != "report_directory":
        warnings.append("Artifact path base is missing or unclear.")
    for artifact_key in ("raw_ffprobe_report", "frames_directory"):
        artifact = artifacts.get(artifact_key)
        if not artifact or _is_absolute_path(str(artifact)):
            errors.append(f"Artifact reference is missing or not relative: {artifact_key}.")

    return {"errors": errors, "warnings": warnings}


def _same_value(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 0.001
    return actual == expected


def _collect_observation_ids(observations: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for value in observations.values():
        if isinstance(value, list):
            ids.extend(item.get("observation_id") for item in value if isinstance(item, dict))
    return [item for item in ids if item is not None]


def _relative_paths(report: dict[str, Any]) -> list[str]:
    paths = [str(report.get("source", {}).get("path", ""))]
    for frame in report.get("frame_analysis", {}).get("frames", []):
        if frame.get("artifact_path"):
            paths.append(str(frame["artifact_path"]))
    artifacts = report.get("artifacts", {})
    for key in ("raw_ffprobe_report", "frames_directory"):
        if artifacts.get(key):
            paths.append(str(artifacts[key]))
    return paths


def _is_absolute_path(value: str) -> bool:
    path = PurePosixPath(value.replace("\\", "/"))
    return path.is_absolute() or (len(value) > 2 and value[1] == ":")


def _check_numeric_values(value: Any, errors: list[str]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        errors.append("Report contains a non-finite numeric value.")
    elif isinstance(value, dict):
        for child in value.values():
            _check_numeric_values(child, errors)
    elif isinstance(value, list):
        for child in value:
            _check_numeric_values(child, errors)


def _check_non_negative_counts(report: dict[str, Any], errors: list[str]) -> None:
    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, child_key)
        elif isinstance(value, list):
            for child in value:
                walk(child, key)
        elif "count" in key and isinstance(value, int) and value < 0:
            errors.append(f"Negative count found: {key}.")

    walk(report)


def _check_analysis_times(report: dict[str, Any], errors: list[str]) -> None:
    analysis = report.get("analysis", {})
    try:
        started = datetime.fromisoformat(analysis["started_at"])
        completed = datetime.fromisoformat(analysis["completed_at"])
    except (KeyError, TypeError, ValueError):
        errors.append("Analysis timestamps are missing or invalid.")
        return
    if completed < started:
        errors.append("Analysis completion time is earlier than start time.")
