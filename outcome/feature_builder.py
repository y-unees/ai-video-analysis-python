from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from config import APP_VERSION
from file_utils import artifact_record
from outcome.models import (
    AudioSummaryFeatures,
    D3SummaryFeatures,
    FrameSummaryFeatures,
    MediaFeatures,
    MetadataIndicatorFeatures,
    OutcomeFeatures,
    OutcomeIdentity,
    TemporalSummaryFeatures,
    UnifiedEvidenceSummaryFeatures,
    VisualConsistencySummaryFeatures,
)


OUTCOME_FEATURE_SCHEMA_VERSION = "0.9.0"
OUTCOME_FEATURE_FILENAME = "outcome_features.json"
ALLOWED_EXPECTED_LABELS = {None, "real", "ai_generated"}
RATIO_FIELDS = {"dark_frame_ratio", "white_frame_ratio"}
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


def build_outcome_features(
    report: dict[str, Any],
    unified_evidence: dict[str, Any] | None = None,
    d3_result: dict[str, Any] | None = None,
    expected_label: str | None = None,
    analysis_id: str | None = None,
) -> OutcomeFeatures:
    unified = unified_evidence if unified_evidence is not None else report.get("unified_evidence", {})
    d3 = d3_result if d3_result is not None else report.get("learned_detector_results", {}).get("d3", {})
    if expected_label not in ALLOWED_EXPECTED_LABELS:
        raise ValueError("expected_label must be null, real, or ai_generated.")
    return OutcomeFeatures(
        identity=_identity(report, expected_label, analysis_id),
        media=_media(report),
        metadata_indicators=_metadata_indicators(report),
        frame_summary=_frame_summary(report),
        temporal_summary=_temporal_summary(report),
        audio_summary=_audio_summary(report),
        visual_consistency_summary=_visual_consistency_summary(report),
        unified_evidence_summary=_unified_evidence_summary(unified),
        d3_summary=_d3_summary(d3),
    )


def create_outcome_features_artifact(
    analysis_dir: Path,
    report: dict[str, Any],
    expected_label: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    output_path = analysis_dir / OUTCOME_FEATURE_FILENAME
    temporary_path = analysis_dir / f"{OUTCOME_FEATURE_FILENAME}.tmp"
    try:
        features = build_outcome_features(
            report=report,
            unified_evidence=report.get("unified_evidence", {}),
            d3_result=report.get("learned_detector_results", {}).get("d3", {}),
            expected_label=expected_label,
            analysis_id=analysis_dir.name,
        )
        payload = features.to_dict()
        validation = validate_outcome_features(payload, report.get("source", {}).get("sha256"))
        if validation["errors"]:
            raise ValueError("; ".join(validation["errors"]))
        serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        temporary_path.write_text(serialized, encoding="utf-8")
        temporary_path.replace(output_path)
        artifact = artifact_record(output_path, OUTCOME_FEATURE_FILENAME)
        return {
            "status": "completed",
            "reason_code": None,
            "reason": None,
            "artifact": artifact,
            "schema_version": OUTCOME_FEATURE_SCHEMA_VERSION,
            "purpose": "Stable feature record for future labeled dataset and probability-calibration experiments.",
            "contains_probability": False,
            "contains_classifier_output": False,
            "validation": validation,
        }, validation["warnings"]
    except Exception as error:
        if temporary_path.exists():
            temporary_path.unlink()
        if output_path.exists():
            output_path.unlink()
        return {
            "status": "failed",
            "reason_code": "outcome_feature_generation_failed",
            "reason": "Outcome feature generation failed.",
            "artifact": None,
            "schema_version": OUTCOME_FEATURE_SCHEMA_VERSION,
            "purpose": "Stable feature record for future labeled dataset and probability-calibration experiments.",
            "contains_probability": False,
            "contains_classifier_output": False,
            "diagnostics": {"exception_type": type(error).__name__, "exception_message": str(error)},
        }, [f"Outcome feature generation failed: {type(error).__name__}: {error}"]


def validate_outcome_features(features: dict[str, Any], source_sha256: str | None = None) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    identity = features.get("identity", {})
    for key in ("feature_schema_version", "application_version", "analysis_id", "video_name", "video_sha256", "expected_label"):
        if key not in identity:
            errors.append(f"Outcome identity is missing required field: {key}.")
    if identity.get("expected_label") not in ALLOWED_EXPECTED_LABELS:
        errors.append("Outcome expected_label is invalid.")
    if source_sha256 is not None and identity.get("video_sha256") != source_sha256:
        errors.append("Outcome source SHA-256 does not match the main report.")
    d3 = features.get("d3_summary", {})
    if d3.get("d3_raw_score") is not None and d3.get("d3_calibration_status") != "uncalibrated":
        errors.append("D3 raw score must remain explicitly uncalibrated.")
    _validate_no_nonfinite(features, errors)
    _validate_ratios(features, errors)
    _validate_counts(features, errors)
    _validate_no_absolute_paths(features, errors)
    _validate_no_forbidden_fields(features, errors)
    return {"errors": errors, "warnings": warnings}


def _identity(report: dict[str, Any], expected_label: str | None, analysis_id: str | None) -> OutcomeIdentity:
    return OutcomeIdentity(
        feature_schema_version=OUTCOME_FEATURE_SCHEMA_VERSION,
        application_version=report.get("analysis_environment", {}).get("application_version") or APP_VERSION,
        analysis_id=analysis_id or report.get("analysis", {}).get("analysis_id"),
        video_name=report.get("source", {}).get("filename"),
        video_sha256=report.get("source", {}).get("sha256"),
        expected_label=expected_label,
    )


def _media(report: dict[str, Any]) -> MediaFeatures:
    metadata = report.get("metadata", {})
    video = metadata.get("video", {})
    frame_count = video.get("frame_count", {})
    return MediaFeatures(
        duration_seconds=_number(metadata.get("container", {}).get("duration_seconds")),
        width=_int_or_none(video.get("width")),
        height=_int_or_none(video.get("height")),
        frame_rate=_number(video.get("frame_rate_decimal")),
        frame_count=_int_or_none(frame_count.get("reported_by_stream") if frame_count.get("reported_by_stream") is not None else frame_count.get("estimated_from_duration")),
        audio_present=_bool_or_none(metadata.get("audio", {}).get("present")),
    )


def _metadata_indicators(report: dict[str, Any]) -> MetadataIndicatorFeatures:
    metadata = report.get("metadata", {})
    tags = [
        metadata.get("container", {}).get("tags", {}),
        metadata.get("video", {}).get("tags", {}),
        metadata.get("audio", {}).get("tags", {}),
    ]
    encoding = metadata.get("encoding", {})
    duration_difference = metadata.get("duration_comparison", {}).get("duration_only_difference_seconds")
    return MetadataIndicatorFeatures(
        creation_time_present=any(bool(item.get("creation_time")) for item in tags if isinstance(item, dict)),
        encoder_metadata_present=any(
            encoding.get(key) is not None
            for key in ("container_encoder", "video_stream_encoder", "audio_stream_encoder")
        ),
        duration_metadata_consistent=None if duration_difference is None else float(duration_difference) == 0.0,
    )


def _frame_summary(report: dict[str, Any]) -> FrameSummaryFeatures:
    summary = report.get("frame_analysis", {}).get("summary", {})
    sampled_count = _int_or_none(summary.get("frames_analyzed"))
    return FrameSummaryFeatures(
        sampled_frame_count=sampled_count,
        mean_brightness=_number(summary.get("average_brightness")),
        mean_contrast=_number(summary.get("average_contrast")),
        mean_sharpness=_number(summary.get("average_laplacian_variance")),
        dark_frame_ratio=_ratio_from_count(summary.get("heuristic_near_black_frame_count"), sampled_count),
        white_frame_ratio=_ratio_from_count(summary.get("heuristic_near_white_frame_count"), sampled_count),
    )


def _temporal_summary(report: dict[str, Any]) -> TemporalSummaryFeatures:
    temporal = report.get("temporal_analysis", {})
    summary = temporal.get("summary", {})
    return TemporalSummaryFeatures(
        analyzed_transition_count=_int_or_none(summary.get("transitions_analyzed")),
        notable_transition_count=_int_or_none(summary.get("notable_transition_count", len(temporal.get("notable_transitions", [])) if temporal else None)),
        scene_boundary_count=_int_or_none(summary.get("scene_boundary_candidate_count")),
        near_static_interval_count=_int_or_none(summary.get("sustained_near_static_interval_count")),
        mean_normalized_frame_difference=None,
        maximum_normalized_frame_difference=None,
        mean_flow_warp_residual=_number(summary.get("average_flow_warp_residual")),
        maximum_flow_warp_residual=_number(summary.get("maximum_flow_warp_residual")),
        mean_motion_magnitude=_number(summary.get("average_flow_magnitude")),
        maximum_motion_magnitude=_number(summary.get("maximum_flow_magnitude")),
    )


def _audio_summary(report: dict[str, Any]) -> AudioSummaryFeatures:
    audio = report.get("audio_analysis", {})
    summary = audio.get("summary", {})
    transitions = audio.get("notable_transitions", [])
    return AudioSummaryFeatures(
        audio_window_count=_int_or_none(summary.get("window_count")),
        ranked_audio_transition_count=_int_or_none(summary.get("ranked_energy_transition_count", len(transitions) if audio else None)),
        silence_interval_count=_int_or_none(summary.get("silence_like_interval_count", len(audio.get("silence_intervals", [])) if audio else None)),
        clipping_interval_count=_int_or_none(len(audio.get("clipping_intervals", [])) if audio else None),
        maximum_audio_energy_change=_max_number([item.get("energy_change_ratio", item.get("rms_ratio")) for item in transitions]),
        mean_audio_rms=_number(audio.get("global_metrics", {}).get("rms_amplitude")),
    )


def _visual_consistency_summary(report: dict[str, Any]) -> VisualConsistencySummaryFeatures:
    visual = report.get("visual_consistency_analysis", {})
    summary = visual.get("summary", {})
    transition_summaries = visual.get("transition_summaries", [])
    regional_summaries = [item.get("regional_summary", {}) for item in transition_summaries]
    if not regional_summaries:
        regional_summaries = [item.get("regional_summary", {}) for item in visual.get("ranked_review_transitions", [])]
    return VisualConsistencySummaryFeatures(
        analyzed_region_record_count=_int_or_none(summary.get("consistency_record_count")),
        ranked_consistency_transition_count=_int_or_none(summary.get("ranked_review_transition_count", len(visual.get("ranked_review_transitions", [])) if visual else None)),
        sustained_instability_interval_count=_int_or_none(summary.get("sustained_interval_count", len(visual.get("sustained_intervals", [])) if visual else None)),
        maximum_unstable_region_count=_max_int([item.get("unstable_region_count") for item in regional_summaries]),
        maximum_edge_instability=_max_number([item.get("maximum_edge_instability") for item in regional_summaries]),
        maximum_texture_distance=_max_number([item.get("maximum_texture_distance") for item in regional_summaries]),
        maximum_detail_residual=_max_number([item.get("maximum_regional_detail_residual") for item in regional_summaries]),
    )


def _unified_evidence_summary(unified: dict[str, Any]) -> UnifiedEvidenceSummaryFeatures:
    summary = unified.get("summary", {})
    highlights = unified.get("review_highlights", [])
    priorities = [item.get("review_priority", {}).get("level") for item in highlights]
    cross_modal = [
        item
        for item in highlights
        if item.get("cross_modal_context", {}).get("classification") in {"visual_and_audio_aligned", "multiple_visual_methods"}
    ]
    return UnifiedEvidenceSummaryFeatures(
        timeline_event_count=_int_or_none(summary.get("timeline_event_count")),
        high_priority_event_count=priorities.count("high"),
        moderate_priority_event_count=priorities.count("moderate"),
        cross_modal_event_count=len(cross_modal),
        maximum_independent_evidence_group_count=_max_int([item.get("independent_group_count") for item in highlights]),
    )


def _d3_summary(d3: dict[str, Any]) -> D3SummaryFeatures:
    execution = d3.get("execution", {}) if d3 else {}
    configuration = d3.get("configuration", {}) if d3 else {}
    preprocessing = d3.get("preprocessing", {}) if d3 else {}
    feature = d3.get("feature_summary", {}) if d3 else {}
    native = d3.get("native_output", {}) if d3 else {}
    return D3SummaryFeatures(
        d3_status=execution.get("status", "not_run" if not d3 else None),
        d3_raw_score=_number(native.get("raw_score")),
        d3_encoder=configuration.get("encoder"),
        d3_distance_mode=configuration.get("distance_mode"),
        d3_selected_frame_count=_int_or_none(preprocessing.get("actual_selected_frame_count")),
        d3_first_order_value_count=_int_or_none(feature.get("first_order_value_count")),
        d3_second_order_value_count=_int_or_none(feature.get("second_order_value_count")),
        d3_second_order_mean=_number(feature.get("second_order_mean")),
        d3_second_order_standard_deviation=_number(feature.get("second_order_standard_deviation")),
        d3_calibration_status=native.get("calibration_status"),
        d3_score_direction=native.get("score_direction"),
    )


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _ratio_from_count(count: Any, total: int | None) -> float | None:
    count_value = _int_or_none(count)
    if count_value is None or total is None or total <= 0:
        return None
    return round(count_value / total, 6)


def _max_number(values: list[Any]) -> float | None:
    numbers = [_number(value) for value in values]
    clean = [value for value in numbers if value is not None]
    return max(clean) if clean else None


def _max_int(values: list[Any]) -> int | None:
    numbers = [_int_or_none(value) for value in values]
    clean = [value for value in numbers if value is not None]
    return max(clean) if clean else None


def _validate_no_nonfinite(value: Any, errors: list[str], path: str = "outcome_features") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_no_nonfinite(child, errors, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_nonfinite(child, errors, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"Outcome feature contains a non-finite numeric value: {path}.")


def _validate_ratios(value: Any, errors: list[str], path: str = "outcome_features") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in RATIO_FIELDS and child is not None and not (0.0 <= float(child) <= 1.0):
                errors.append(f"Outcome ratio is outside [0, 1]: {child_path}.")
            _validate_ratios(child, errors, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_ratios(child, errors, f"{path}[{index}]")


def _validate_counts(value: Any, errors: list[str], path: str = "outcome_features") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key.endswith("_count") and child is not None and (not isinstance(child, int) or child < 0):
                errors.append(f"Outcome count is not a non-negative integer: {child_path}.")
            _validate_counts(child, errors, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_counts(child, errors, f"{path}[{index}]")


def _validate_no_absolute_paths(value: Any, errors: list[str], path: str = "outcome_features") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_no_absolute_paths(child, errors, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_absolute_paths(child, errors, f"{path}[{index}]")
    elif isinstance(value, str) and _looks_like_absolute_path(value):
        errors.append(f"Outcome feature contains an absolute path: {path}.")


def _looks_like_absolute_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("/") or value.startswith("\\\\")


def _validate_no_forbidden_fields(value: Any, errors: list[str], path: str = "outcome_features") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_FIELD_NAMES:
                errors.append(f"Outcome feature contains forbidden result field: {path}.{key}.")
            _validate_no_forbidden_fields(child, errors, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_forbidden_fields(child, errors, f"{path}[{index}]")
