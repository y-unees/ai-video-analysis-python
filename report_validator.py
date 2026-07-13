from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from config import APP_VERSION, SCHEMA_VERSION
from file_utils import calculate_sha256
from frame_analyzer import summarize_frame_analysis


def validate_report(report: dict[str, Any], base_dir: Path | None = None) -> dict[str, list[str]]:
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

    observation_ids = _collect_observation_ids(report.get("observations", {}), errors)
    if len(observation_ids) != len(set(observation_ids)):
        errors.append("Observation IDs are not unique.")

    for path_value in _relative_paths(report):
        if _is_absolute_path(path_value):
            errors.append(f"Report contains an absolute path: {path_value}")

    _check_numeric_values(report, errors)
    _check_non_negative_counts(report, errors)
    _check_analysis_times(report, errors)
    _check_temporal_analysis(report.get("temporal_analysis", {}), errors, warnings, base_dir)
    _check_audio_analysis(report.get("audio_analysis", {}), errors, warnings, base_dir)
    _check_observation_scopes(report.get("observations", {}), errors)
    _check_evidence_compatibility(report, errors)

    artifacts = report.get("artifacts", {})
    if artifacts.get("paths_relative_to") != "report_directory":
        warnings.append("Artifact path base is missing or unclear.")
    for artifact_key in ("raw_ffprobe_report", "frames_directory"):
        artifact = artifacts.get(artifact_key)
        if not artifact or _is_absolute_path(str(artifact)):
            errors.append(f"Artifact reference is missing or not relative: {artifact_key}.")

    return {"errors": errors, "warnings": warnings}


def _check_audio_analysis(
    audio: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    base_dir: Path | None,
) -> None:
    if not audio:
        errors.append("Audio analysis section is missing.")
        return
    if audio.get("status") not in {"completed", "partial", "skipped", "failed"}:
        errors.append("Audio analysis has an invalid status.")
    if audio.get("status") in {"failed", "skipped"} and not audio.get("reason_code"):
        errors.append("Audio analysis failure/skip is missing a reason code.")
    cleanup = audio.get("temporary_file_cleanup", {})
    if audio.get("extraction", {}).get("status") == "completed":
        if cleanup.get("attempted") is not True:
            errors.append("Audio temporary-file cleanup was not recorded.")
    decoded = audio.get("decoded_audio", {})
    if decoded:
        if decoded.get("sample_rate_hz", 1) <= 0:
            errors.append("Decoded audio sample rate is not positive.")
        if decoded.get("channels", 1) <= 0:
            errors.append("Decoded audio channel count is not positive.")
        if decoded.get("frame_count", 0) < 0:
            errors.append("Decoded audio frame count is negative.")
    metrics = audio.get("global_metrics", {})
    for key in ("rms_amplitude", "peak_absolute_amplitude", "clipping_ratio", "silence_ratio", "zero_crossing_rate"):
        value = metrics.get(key)
        if value is not None and not (0 <= value <= 1):
            errors.append(f"Audio metric is outside 0-1: {key}.")
    intervals = audio.get("silence_intervals", [])
    previous_end = None
    interval_ids = set()
    minimum = audio.get("configuration", {}).get("minimum_silence_interval_seconds", 0)
    for interval in intervals:
        interval_id = interval.get("interval_id")
        if not interval_id or interval_id in interval_ids:
            errors.append("Audio silence interval IDs are missing or duplicated.")
        interval_ids.add(interval_id)
        if interval.get("duration_seconds", 0) < minimum:
            errors.append("Audio silence interval is shorter than configured minimum.")
        if previous_end is not None and interval["start_timestamp_seconds"] < previous_end:
            errors.append("Audio silence intervals overlap.")
        previous_end = interval["end_timestamp_seconds"]
    transition_ids = set()
    for transition in audio.get("notable_transitions", []):
        transition_id = transition.get("transition_id")
        if not transition_id or transition_id in transition_ids:
            errors.append("Audio transition IDs are missing or duplicated.")
        transition_ids.add(transition_id)
        if transition.get("selection_basis") != "relative_within_audio":
            errors.append("Audio transition selection basis is missing or invalid.")
    artifact = audio.get("artifacts", {}).get("audio_metrics_artifact")
    if artifact:
        _check_artifact(artifact, base_dir, errors, warnings)
    if audio.get("status") == "completed":
        if not metrics:
            errors.append("Completed audio analysis is missing global metrics.")
        if not decoded:
            errors.append("Completed audio analysis is missing decoded audio details.")
        if audio.get("extraction", {}).get("status") != "completed":
            errors.append("Completed audio analysis has incomplete extraction details.")
        if not artifact:
            errors.append("Completed audio analysis is missing audio metrics artifact.")


def _check_observation_scopes(observations: dict[str, Any], errors: list[str]) -> None:
    affirmative = (
        "proves the video is ai-generated",
        "video is definitely manipulated",
        "tampering is confirmed",
        "manipulation is confirmed",
        "ai generation is confirmed",
        "definitely ai-generated",
    )
    for items in observations.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            observation_id = item.get("observation_id", "unknown-observation")
            if item.get("supports_authenticity_verdict") is True:
                errors.append(f"{observation_id}: observation claims support for an authenticity verdict.")
            text = " ".join(str(item.get(key, "")).lower() for key in ("description", "interpretation"))
            if any(phrase in text for phrase in affirmative):
                errors.append(f"{observation_id}: unsupported affirmative verdict language.")


def _check_temporal_analysis(
    temporal: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    base_dir: Path | None,
) -> None:
    if not temporal:
        warnings.append("Temporal analysis section is missing.")
        return

    summary = temporal.get("summary", {})
    scenes = temporal.get("scenes", [])
    intervals = temporal.get("notable_intervals", [])
    transitions = temporal.get("notable_transitions", [])
    status = temporal.get("status")
    if status not in {"completed", "partial", "skipped", "failed"}:
        errors.append("Temporal analysis has an invalid status.")

    if summary.get("scene_count") != len(scenes):
        errors.append("Temporal scene count does not match the scene records.")
    if summary.get("sustained_near_static_interval_count") != len(intervals):
        errors.append("Temporal near-static interval count does not match the interval records.")

    scene_indices = [scene.get("scene_index") for scene in scenes]
    if scene_indices != sorted(scene_indices) or len(scene_indices) != len(set(scene_indices)):
        errors.append("Temporal scene indices are not unique and ordered.")
    previous_end = None
    for scene in scenes:
        start = scene.get("start_timestamp_seconds")
        end = scene.get("end_timestamp_seconds")
        duration = scene.get("duration_seconds")
        if not _finite_or_none(start) or not _finite_or_none(end) or not _finite_or_none(duration):
            errors.append("Temporal scene contains a non-finite timestamp or duration.")
            continue
        if start is not None and end is not None and end < start:
            errors.append("Temporal scene duration is negative.")
        if previous_end is not None and start is not None and start < previous_end:
            errors.append("Temporal scenes overlap.")
        previous_end = end

    for interval in intervals:
        start = interval.get("start_timestamp_seconds")
        end = interval.get("end_timestamp_seconds")
        duration = interval.get("duration_seconds")
        if start is None or end is None or end <= start:
            errors.append("Temporal near-static interval has invalid bounds.")
        if duration is not None and duration < 0:
            errors.append("Temporal near-static interval has negative duration.")

    for transition in transitions:
        if transition.get("notable_transition_index") is None:
            errors.append("Notable transition index is missing.")
        notability = transition.get("notability", {})
        reasons = notability.get("reasons", [])
        if len(reasons) != len(set(reasons)):
            errors.append("Notable transition ranking reasons are not unique.")
        if notability.get("reason_count") != len(reasons):
            errors.append("Notable transition reason count does not match reasons.")
        percentile = notability.get("combined_percentile")
        if percentile is not None and not (0 <= percentile <= 1):
            errors.append("Notable transition combined percentile is outside 0-1.")
        flow = transition.get("optical_flow", {})
        for key, value in flow.items():
            if value is not None and (not isinstance(value, (int, float)) or value < 0):
                errors.append(f"Temporal optical-flow metric is invalid: {key}.")
        residual = transition.get("flow_warp_residual", {})
        _check_residual(residual, errors)
        for artifact in transition.get("artifacts", {}).values():
            _check_artifact(artifact, base_dir, errors, warnings)

    artifacts = temporal.get("artifacts", {})
    metrics_artifact = artifacts.get("temporal_metrics_artifact")
    if metrics_artifact:
        path = metrics_artifact.get("path")
        if not path or _is_absolute_path(str(path)):
            errors.append("Temporal metrics artifact path is missing or absolute.")
        if not metrics_artifact.get("sha256") or metrics_artifact.get("size_bytes") is None:
            errors.append("Temporal metrics artifact hash or size is missing.")
        _check_artifact(metrics_artifact, base_dir, errors, warnings)
    for frame in temporal.get("scene_representative_frames", []):
        path = frame.get("artifact_path")
        if not path or _is_absolute_path(str(path)):
            errors.append("Scene representative frame path is missing or absolute.")
        if base_dir and path:
            full_path = base_dir / str(path)
            if not full_path.exists():
                warnings.append(f"Scene representative frame does not exist: {path}")

    coverage = temporal.get("coverage", {})
    if coverage:
        if not coverage.get("coverage_timeline_basis"):
            errors.append("Temporal coverage timeline basis is missing.")
        first = coverage.get("first_sample_timestamp_seconds")
        last = coverage.get("last_sample_timestamp_seconds")
        if first is not None and last is not None and first > last:
            errors.append("Temporal coverage first sample is after the last sample.")
        for key in ("analyzed_time_span_seconds", "source_time_span_seconds"):
            value = coverage.get(key)
            if value is not None and value < 0:
                errors.append(f"Temporal coverage value is negative: {key}.")
        ratio = coverage.get("coverage_ratio")
        if ratio is not None and not (-0.000001 <= ratio <= 1.000001):
            errors.append("Temporal coverage ratio is outside 0-1.")

    forbidden = (
        "proves the video is ai-generated",
        "video is definitely manipulated",
        "tampering is confirmed",
        "manipulation is confirmed",
        "ai generation is confirmed",
        "definitely ai-generated",
    )
    for observation in temporal.get("observations", []):
        observation_id = observation.get("observation_id", "unknown-observation")
        if observation.get("supports_authenticity_verdict") is True:
            errors.append(f"{observation_id}: observation claims support for an authenticity verdict.")
        text = " ".join(
            str(observation.get(key, "")).lower()
            for key in ("description", "interpretation")
        )
        if any(phrase in text for phrase in forbidden):
            errors.append(f"{observation_id}: unsupported affirmative verdict language.")


def _finite_or_none(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _same_value(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 0.001
    return actual == expected


def _check_residual(residual: dict[str, Any], errors: list[str]) -> None:
    for key in (
        "mean_normalized_residual",
        "median_normalized_residual",
        "percentile_95_normalized_residual",
        "high_residual_pixel_ratio",
    ):
        value = residual.get(key)
        if value is not None and not (0 <= value <= 1):
            errors.append(f"Flow-warp residual value is outside 0-1: {key}.")
    if (
        residual.get("percentile_95_normalized_residual") is not None
        and residual.get("median_normalized_residual") is not None
        and residual["percentile_95_normalized_residual"] < residual["median_normalized_residual"]
    ):
        errors.append("Flow-warp residual 95th percentile is below the median.")
    if residual.get("valid_pixel_count") is not None and residual["valid_pixel_count"] < 0:
        errors.append("Flow-warp residual valid-pixel count is negative.")


def _check_artifact(
    artifact: dict[str, Any],
    base_dir: Path | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    path_value = artifact.get("path")
    if not path_value or _is_absolute_path(str(path_value)):
        errors.append("Artifact path is missing or absolute.")
        return
    if not artifact.get("sha256") or artifact.get("size_bytes") is None:
        errors.append(f"Artifact hash or size is missing: {path_value}")
    label = artifact.get("size_human_readable", "")
    if any(unit in label for unit in (" KB", " MB", " GB", " TB")):
        errors.append("Artifact uses decimal size units instead of binary units.")
    if base_dir is None:
        return
    full_path = base_dir / str(path_value)
    if not full_path.exists():
        warnings.append(f"Artifact file does not exist at validation time: {path_value}")
        return
    if full_path.stat().st_size != artifact.get("size_bytes"):
        errors.append(f"Artifact size does not match file on disk: {path_value}")
    if artifact.get("sha256") and calculate_sha256(full_path) != artifact.get("sha256"):
        errors.append(f"Artifact hash does not match file on disk: {path_value}")


def _check_evidence_compatibility(report: dict[str, Any], errors: list[str]) -> None:
    compatibility = report.get("compatibility", {})
    if not compatibility.get("legacy_evidence_view_included"):
        return
    observation_ids = set(_collect_observation_ids(report.get("observations", {}), errors))
    evidence_ids = {
        item.get("observation_id")
        for item in report.get("evidence", [])
        if isinstance(item, dict)
    }
    if observation_ids != evidence_ids:
        errors.append("Legacy evidence view does not match canonical observations.")


def _collect_observation_ids(observations: dict[str, Any], errors: list[str] | None = None) -> list[str]:
    ids: list[str] = []
    for value in observations.values():
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                observation_id = item.get("observation_id")
                if not observation_id and errors is not None:
                    errors.append("Canonical observation is missing an observation_id.")
                ids.append(observation_id)
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
    temporal = report.get("temporal_analysis", {})
    for transition in temporal.get("notable_transitions", []):
        for artifact in transition.get("artifacts", {}).values():
            if artifact.get("path"):
                paths.append(str(artifact["path"]))
    temporal_artifact = temporal.get("artifacts", {}).get("temporal_metrics_artifact")
    if temporal_artifact and temporal_artifact.get("path"):
        paths.append(str(temporal_artifact["path"]))
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
