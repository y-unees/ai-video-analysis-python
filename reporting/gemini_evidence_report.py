from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from config import APP_VERSION, gemini_compact_report_configuration
from file_utils import artifact_record


COMPACT_REPORT_FILENAME = "gemini_evidence_report.json"
COMPACT_REPORT_SCHEMA_VERSION = "0.8.2"
PRIORITY_RANK = {"high": 0, "moderate": 1, "low": 2}
PREFERRED_ARTIFACT_LABELS = (
    "before_frame",
    "after_frame",
    "absolute_difference",
    "flow_warp_residual",
    "combined_heatmap",
    "detail_residual_heatmap",
)
FORBIDDEN_FIELD_NAMES = {
    "probability",
    "threshold",
    "authenticity_verdict",
    "manipulation_verdict",
    "fake_real_verdict",
    "real_fake_verdict",
    "verdict",
}


def create_gemini_evidence_report(analysis_dir: Path, report: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    config = gemini_compact_report_configuration()
    if not config["enabled"]:
        return _reference("disabled", reason_code="compact_report_disabled", reason="Gemini-ready compact report generation is disabled."), []

    output_path = analysis_dir / COMPACT_REPORT_FILENAME
    temporary_path = analysis_dir / f"{COMPACT_REPORT_FILENAME}.tmp"
    try:
        compact, serialized, size_metadata = _build_sized_report(report, config)
        validation = validate_gemini_evidence_report(compact, int(config["hard_size_limit_bytes"]), len(serialized))
        if validation["errors"]:
            raise ValueError("; ".join(validation["errors"]))
        temporary_path.write_text(serialized, encoding="utf-8")
        temporary_path.replace(output_path)
        artifact = artifact_record(output_path, COMPACT_REPORT_FILENAME)
        main_size = _estimated_main_report_size(report)
        ratio = None if main_size <= 0 else round((artifact["size_bytes"] / main_size) * 100, 2)
        reference = _reference("completed")
        reference.update(
            {
                "artifact": artifact,
                "schema_version": COMPACT_REPORT_SCHEMA_VERSION,
                "purpose": "Recommended compact derivative input for future Gemini-assisted interpretation.",
                "derivative_not_source_of_truth": True,
                "source_of_truth": "Detailed local report.json and referenced forensic artifacts remain authoritative.",
                "compactness": {
                    "compact_report_size_bytes": artifact["size_bytes"],
                    "estimated_main_report_size_bytes": main_size,
                    "ratio_percent": ratio,
                    "calculation": "compact_report_size / estimated_main_report_size * 100",
                    **size_metadata,
                },
                "validation": validation,
            }
        )
        return reference, validation["warnings"]
    except Exception as error:
        if temporary_path.exists():
            temporary_path.unlink()
        if output_path.exists():
            output_path.unlink()
        return _reference(
            "failed",
            reason_code="compact_report_generation_failed",
            reason="Gemini-ready compact report generation failed.",
            diagnostics={"exception_type": type(error).__name__, "exception_message": str(error)},
        ), [f"Gemini-ready compact report generation failed: {type(error).__name__}: {error}"]


def validate_gemini_evidence_report(report: dict[str, Any], hard_size_limit_bytes: int, serialized_size_bytes: int | None = None) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    required = {
        "schema_version",
        "analysis_identity",
        "media_summary",
        "deterministic_summary",
        "key_review_events",
        "d3_summary",
        "limitations",
        "gemini_instructions",
        "future_gemini_output_contract",
        "source_artifacts",
    }
    missing = sorted(required - set(report))
    if missing:
        errors.append(f"Compact report is missing required sections: {missing}")
    try:
        serialized = json.dumps(report, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        errors.append(f"Compact report is not valid strict JSON: {error}")
        serialized = ""
    size = serialized_size_bytes if serialized_size_bytes is not None else len(serialized.encode("utf-8"))
    if size > hard_size_limit_bytes:
        errors.append(f"Compact report exceeds hard size limit: {size} > {hard_size_limit_bytes}")
    identity = report.get("analysis_identity", {})
    if not identity.get("source_sha256"):
        errors.append("Compact report source SHA-256 is missing.")
    events = report.get("key_review_events", [])
    event_ids = [event.get("event_id") for event in events]
    if len(event_ids) != len(set(event_ids)):
        errors.append("Compact report event IDs are duplicated.")
    config = report.get("generation", {}).get("configuration", {})
    max_events = config.get("maximum_key_events")
    if isinstance(max_events, int) and len(events) > max_events:
        errors.append("Compact report exceeds configured key-event count.")
    max_findings = config.get("maximum_findings_per_event")
    max_artifacts = config.get("maximum_artifacts_per_event")
    for event in events:
        if isinstance(max_findings, int) and len(event.get("findings", [])) > max_findings:
            errors.append(f"Compact event exceeds finding cap: {event.get('event_id')}")
        if isinstance(max_artifacts, int) and len(event.get("artifact_references", [])) > max_artifacts:
            errors.append(f"Compact event exceeds artifact cap: {event.get('event_id')}")
    _validate_no_absolute_paths(report, errors)
    _validate_no_nonfinite(report, errors)
    _validate_no_forbidden_fields(report, errors)
    d3 = report.get("d3_summary", {})
    if d3.get("classification") not in {None, "not_assigned"}:
        errors.append("D3 compact classification must remain not_assigned.")
    if d3.get("calibration_status") not in {None, "uncalibrated"}:
        errors.append("D3 compact calibration must remain uncalibrated.")
    if d3.get("score_direction") not in {None, "not_verified"}:
        errors.append("D3 compact score direction must remain not_verified.")
    return {"errors": errors, "warnings": warnings}


def _build_sized_report(report: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    max_events = int(config["maximum_key_events"])
    max_findings = int(config["maximum_findings_per_event"])
    max_artifacts = int(config["maximum_artifacts_per_event"])
    preferred = int(config["preferred_size_bytes"])
    hard = int(config["hard_size_limit_bytes"])
    include_optional_metrics = True
    concise_summaries = False
    reduction_steps: list[str] = []

    while True:
        compact = _build_report(report, config, max_events, max_findings, max_artifacts, include_optional_metrics, concise_summaries)
        compact = _sanitize(compact)
        serialized = json.dumps(compact, indent=2, sort_keys=True, allow_nan=False)
        size = len(serialized.encode("utf-8"))
        if size <= preferred:
            break
        if max_artifacts > 0:
            max_artifacts -= 1
            reduction_steps.append("reduced_artifact_references")
            continue
        if max_findings > 3:
            max_findings -= 1
            reduction_steps.append("reduced_findings_per_event")
            continue
        if max_events > 1:
            max_events -= 1
            reduction_steps.append("reduced_key_events")
            continue
        if not concise_summaries:
            concise_summaries = True
            reduction_steps.append("shortened_summaries")
            continue
        if include_optional_metrics:
            include_optional_metrics = False
            reduction_steps.append("removed_optional_metrics")
            continue
        break

    if size > hard:
        raise ValueError(f"Compact report is {size} bytes, above hard limit {hard} bytes.")
    compact["generation"]["final_size_bytes"] = size
    compact["generation"]["size_reduction_steps"] = reduction_steps
    compact = _sanitize(compact)
    serialized = json.dumps(compact, indent=2, sort_keys=True, allow_nan=False)
    size_metadata = {
        "preferred_size_bytes": preferred,
        "acceptable_size_bytes": int(config["acceptable_size_bytes"]),
        "hard_size_limit_bytes": hard,
        "size_reduction_steps": reduction_steps,
    }
    return compact, serialized, size_metadata


def _build_report(
    report: dict[str, Any],
    config: dict[str, Any],
    max_events: int,
    max_findings: int,
    max_artifacts: int,
    include_optional_metrics: bool,
    concise_summaries: bool,
) -> dict[str, Any]:
    events = _selected_events(report, max_events, max_findings, max_artifacts, include_optional_metrics)
    return {
        "schema_version": COMPACT_REPORT_SCHEMA_VERSION,
        "analysis_identity": _analysis_identity(report),
        "media_summary": _media_summary(report),
        "deterministic_summary": _deterministic_summary(report, concise_summaries),
        "key_review_events": events,
        "d3_summary": _d3_summary(report),
        "limitations": _limitations(),
        "gemini_instructions": _gemini_instructions(),
        "future_gemini_output_contract": _future_output_contract(),
        "source_artifacts": _source_artifacts(report),
        "generation": {
            "configuration": {
                "maximum_key_events": int(config["maximum_key_events"]),
                "maximum_findings_per_event": max_findings,
                "maximum_artifacts_per_event": max_artifacts,
                "preferred_size_bytes": int(config["preferred_size_bytes"]),
                "hard_size_limit_bytes": int(config["hard_size_limit_bytes"]),
            },
            "selection_strategy": "priority, independent evidence groups, audio/visual overlap, multiple visual methods, ranked findings, source percentile, timestamp, event ID",
            "finding_deduplication": "same record, timestamp range, domain, type, and metric keys are collapsed before per-event capping",
            "gemini_api_called": False,
            "network_request_introduced": False,
        },
    }


def _analysis_identity(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report.get("metadata", {})
    source = report.get("source", {})
    analysis = report.get("analysis", {})
    return {
        "application_version": report.get("analysis_environment", {}).get("application_version") or APP_VERSION,
        "analysis_status": analysis.get("status"),
        "source_filename": source.get("filename"),
        "source_sha256": source.get("sha256"),
        "duration_seconds": metadata.get("container", {}).get("duration_seconds"),
        "completed_at": analysis.get("completed_at"),
        "compact_report_schema_version": COMPACT_REPORT_SCHEMA_VERSION,
    }


def _media_summary(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report.get("metadata", {})
    video = metadata.get("video", {})
    audio = metadata.get("audio", {})
    display = metadata.get("display", {})
    return {
        "container_format": metadata.get("container", {}).get("format_name"),
        "width": video.get("width"),
        "height": video.get("height"),
        "orientation": display.get("orientation"),
        "frame_rate": video.get("frame_rate_decimal"),
        "video_codec": video.get("codec_name"),
        "audio_present": audio.get("present"),
        "audio_codec": audio.get("codec_name"),
        "sample_rate": audio.get("sample_rate"),
        "channel_count": audio.get("channels"),
    }


def _deterministic_summary(report: dict[str, Any], concise: bool) -> dict[str, Any]:
    frame = report.get("frame_analysis", {})
    frame_summary = frame.get("summary", {})
    temporal = report.get("temporal_analysis", {})
    audio = report.get("audio_analysis", {})
    visual = report.get("visual_consistency_analysis", {})
    unified = report.get("unified_evidence", {})
    unified_summary = unified.get("summary", {})
    events = unified.get("review_highlights", [])
    priorities = [item.get("review_priority", {}).get("level") for item in events]
    cross_modal_count = sum(1 for item in events if item.get("cross_modal_context", {}).get("classification") in {"visual_and_audio_aligned", "multiple_visual_methods"})
    summary = {
        "representative_frames": {
            "frames_analyzed": frame_summary.get("frames_analyzed"),
            "comparison_count": len(frame.get("comparisons", [])),
            "large_change_pair_count": frame_summary.get("heuristic_large_change_pair_count"),
            "near_duplicate_pair_count": frame_summary.get("heuristic_near_duplicate_pair_count"),
        },
        "temporal_analysis": {
            "status": temporal.get("status"),
            "frames_analyzed": temporal.get("summary", {}).get("temporal_frames_analyzed"),
            "transitions_analyzed": temporal.get("summary", {}).get("transitions_analyzed"),
            "scene_count": temporal.get("summary", {}).get("scene_count"),
            "near_static_interval_count": temporal.get("summary", {}).get("sustained_near_static_interval_count"),
            "notable_transition_count": len(temporal.get("notable_transitions", [])),
        },
        "audio_analysis": {
            "status": audio.get("status"),
            "audio_available": audio.get("summary", {}).get("audio_available"),
            "audio_transition_count": len(audio.get("notable_transitions", [])),
            "silence_interval_count": len(audio.get("silence_intervals", [])),
            "clipping_interval_count": len(audio.get("clipping_intervals", [])),
        },
        "visual_consistency_analysis": {
            "status": visual.get("status"),
            "transitions_analyzed": visual.get("summary", {}).get("transitions_analyzed"),
            "visual_consistency_interval_count": len(visual.get("sustained_intervals", [])),
            "ranked_transition_count": len(visual.get("ranked_review_transitions", [])),
        },
        "unified_evidence_timeline": {
            "status": unified.get("status"),
            "event_count": unified_summary.get("timeline_event_count"),
            "priority_review_event_count": unified_summary.get("priority_review_event_count"),
            "cross_modal_event_count": cross_modal_count,
            "highest_review_priority": _highest_priority(priorities),
            "event_ranking_meaning": "Review priority only; not proof of manipulation or authenticity.",
        },
    }
    if concise:
        return summary
    summary["temporal_analysis"].update(_temporal_metric_summary(temporal))
    summary["audio_analysis"].update(_audio_metric_summary(audio))
    summary["visual_consistency_analysis"].update(_visual_metric_summary(visual))
    return summary


def _selected_events(report: dict[str, Any], max_events: int, max_findings: int, max_artifacts: int, include_optional_metrics: bool) -> list[dict[str, Any]]:
    highlights = list(report.get("unified_evidence", {}).get("review_highlights", []))
    ranked = sorted(highlights, key=_event_sort_key)[:max_events]
    output = []
    for event in ranked:
        findings = _dedupe_findings(event.get("key_findings", event.get("findings", [])))[:max_findings]
        item = {
            "event_id": event.get("event_id"),
            "start_timestamp_seconds": event.get("start_timestamp_seconds"),
            "end_timestamp_seconds": event.get("end_timestamp_seconds"),
            "duration_seconds": _duration(event),
            "review_priority": event.get("review_priority", {}).get("level"),
            "review_priority_basis": event.get("review_priority", {}).get("basis", []),
            "evidence_domains": event.get("evidence_domains", []),
            "evidence_groups_present": event.get("evidence_groups_present", []),
            "independent_group_count": event.get("independent_group_count"),
            "cross_modal_classification": event.get("cross_modal_context", {}).get("classification"),
            "findings": findings,
            "metrics": _event_metrics(event, findings, include_optional_metrics),
            "artifact_references": _limited_artifacts(event.get("artifact_references", []), max_artifacts),
            "ranking_note": "Ranking is for human review and may be relative within the analyzed video.",
        }
        output.append(item)
    return output


def _d3_summary(report: dict[str, Any]) -> dict[str, Any]:
    d3 = report.get("learned_detector_results", {}).get("d3", {})
    if not d3:
        return {
            "detector_id": "d3",
            "execution_status": "not_run",
            "reason_code": "d3_not_present",
            "raw_score_interpretation": "No D3 raw score is available.",
        }
    execution = d3.get("execution", {})
    detector = d3.get("detector", {})
    config = d3.get("configuration", {})
    preprocessing = d3.get("preprocessing", {})
    feature = d3.get("feature_summary", {})
    native = d3.get("native_output", {})
    verification = d3.get("method_verification", {})
    artifacts = d3.get("artifacts", {})
    return {
        "detector_id": detector.get("detector_id"),
        "detector_name": detector.get("detector_name"),
        "execution_status": execution.get("status"),
        "reason_code": execution.get("reason_code"),
        "encoder": config.get("encoder"),
        "distance_mode": config.get("distance_mode"),
        "device_requested": execution.get("device_requested"),
        "device_used": execution.get("device_used"),
        "execution_duration_seconds": execution.get("duration_seconds"),
        "analyzed_window_start_seconds": preprocessing.get("window_start_seconds"),
        "analyzed_window_end_seconds": preprocessing.get("window_end_seconds"),
        "selected_frame_count": preprocessing.get("actual_selected_frame_count"),
        "embedding_dimension": feature.get("embedding_dimension"),
        "first_order_value_count": feature.get("first_order_value_count"),
        "second_order_value_count": feature.get("second_order_value_count"),
        "raw_score_name": native.get("score_name"),
        "raw_score": native.get("raw_score"),
        "score_direction": native.get("score_direction"),
        "calibration_status": native.get("calibration_status"),
        "classification": native.get("classification"),
        "runtime_parity_status": verification.get("runtime_parity"),
        "mathematical_parity_status": verification.get("mathematical_parity"),
        "d3_result_artifact_path": (artifacts.get("d3_detector_result") or {}).get("path"),
        "temporal_feature_artifact_path": (artifacts.get("d3_temporal_features") or {}).get("path"),
        "raw_score_interpretation": "The D3 raw score is an uncalibrated temporal statistic, not an authenticity score.",
        "limitations": [
            "D3 is uncalibrated in this project.",
            "D3 score direction remains unresolved.",
            "No validated D3 operating threshold exists.",
        ],
    }


def _source_artifacts(report: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for key, artifact in sorted((report.get("unified_evidence", {}).get("artifacts") or {}).items()):
        artifacts.append(_artifact_reference(f"unified_evidence.{key}", artifact, "local_source_artifact"))
    d3_artifacts = report.get("learned_detector_results", {}).get("d3", {}).get("artifacts", {})
    for key, artifact in sorted(d3_artifacts.items()):
        artifacts.append(_artifact_reference(f"d3.{key}", artifact, "local_source_artifact"))
    raw = report.get("artifacts", {}).get("raw_ffprobe_report_artifact")
    if raw:
        artifacts.append(_artifact_reference("raw_ffprobe_report", raw, "local_source_artifact_not_recommended_for_gemini"))
    return [item for item in artifacts if item.get("path")]


def _artifact_reference(label: str, artifact: dict[str, Any], purpose: str) -> dict[str, Any]:
    return {
        "label": label,
        "path": artifact.get("path"),
        "size_bytes": artifact.get("size_bytes"),
        "sha256": artifact.get("sha256"),
        "purpose": purpose,
    }


def _limitations() -> list[str]:
    return [
        "Deterministic measurements are observational.",
        "Ranked events indicate review priority, not proof of manipulation.",
        "D3 is uncalibrated and its score direction remains unresolved.",
        "No validated D3 operating threshold exists.",
        "No numerical evidence fusion has been performed.",
        "No authenticity verdict has been assigned.",
        "Results can be affected by compression, editing, frame rate, content, and video duration.",
        "Human review remains necessary.",
    ]


def _gemini_instructions() -> dict[str, Any]:
    return {
        "purpose": "Future external-model interpretation of this compact local evidence artifact.",
        "gemini_api_called_in_v0_8_2": False,
        "tasks": [
            "Summarize the strongest forensic observations.",
            "Identify the most important timestamps.",
            "Explain cross-modal alignment when present.",
            "Explain uncertainty and conflicting evidence.",
            "Distinguish direct observations from interpretations.",
            "Provide a small set of major findings suitable for terminal display.",
        ],
        "constraints": [
            "Remain concise.",
            "Do not invent missing evidence.",
            "Do not treat review priority as proof.",
            "Do not convert D3 raw scores into probabilities.",
            "Do not invent D3 thresholds.",
            "Do not claim D3 establishes authenticity.",
            "Do not declare the video real or fake unless a later validated project policy explicitly permits it.",
        ],
    }


def _future_output_contract() -> dict[str, Any]:
    return {
        "contract_status": "prepared_not_invoked",
        "schema_path": "schemas/gemini_interpretation_response.schema.json",
        "fields": {
            "analysis_summary": "one short overall summary",
            "major_findings": "3 to 5 concise findings",
            "important_timestamps": "3 to 5 timestamp-focused observations",
            "uncertainties": "concise uncertainty list",
            "limitations": "concise limitation list",
            "recommended_human_review": "small set of review prompts",
            "model_disclaimer": "required external-model caveat",
        },
        "limits": {
            "unsupported_numeric_probability": "not_allowed",
            "definitive_real_fake_label": "not_allowed",
        },
    }


def _reference(
    status: str,
    reason_code: str | None = None,
    reason: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": reason_code,
        "reason": reason,
        "artifact": None,
        "schema_version": COMPACT_REPORT_SCHEMA_VERSION,
        "purpose": "Recommended compact derivative input for future Gemini-assisted interpretation.",
        "derivative_not_source_of_truth": True,
        "source_of_truth": "Detailed local report.json and referenced forensic artifacts remain authoritative.",
        "diagnostics": diagnostics,
    }


def _event_sort_key(event: dict[str, Any]) -> tuple[Any, ...]:
    priority = event.get("review_priority", {})
    context = event.get("cross_modal_context", {})
    basis = set(priority.get("basis", []))
    return (
        PRIORITY_RANK.get(priority.get("level"), 9),
        -int(event.get("independent_group_count") or 0),
        0 if "audio_visual_temporal_overlap" in basis or context.get("classification") == "visual_and_audio_aligned" else 1,
        0 if "multiple_visual_methods" in basis or context.get("classification") == "multiple_visual_methods" else 1,
        0 if "ranked_source_findings" in basis else 1,
        -_event_percentile_from_findings(event.get("key_findings", [])),
        event.get("start_timestamp_seconds") or 0,
        event.get("event_id") or "",
    )


def _event_percentile_from_findings(findings: list[dict[str, Any]]) -> float:
    values = []
    for finding in findings:
        metrics = finding.get("metrics", {})
        for key in ("combined_percentile", "source_percentile"):
            if metrics.get(key) is not None:
                value = float(metrics[key])
                if math.isfinite(value):
                    values.append(value)
    return max(values) if values else 0.0


def _dedupe_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for finding in findings:
        metrics = finding.get("metrics", {})
        key = (
            finding.get("source_record_id"),
            finding.get("domain"),
            finding.get("type"),
            tuple(sorted(metrics.keys())),
            finding.get("summary"),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "domain": finding.get("domain"),
                "type": finding.get("type"),
                "summary": finding.get("summary"),
                "source_record_id": finding.get("source_record_id"),
                "source_observation_ids": finding.get("source_observation_ids", []),
                "metrics": _small_metrics(metrics),
            }
        )
    return output


def _limited_artifacts(artifacts: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    seen = set()
    ordered = sorted(artifacts, key=_artifact_sort_key)
    output = []
    for artifact in ordered:
        path = artifact.get("path")
        if not path or path in seen:
            continue
        seen.add(path)
        output.append(
            {
                "label": artifact.get("label"),
                "path": path,
                "size_bytes": artifact.get("size_bytes"),
                "sha256": artifact.get("sha256"),
            }
        )
        if len(output) >= limit:
            break
    return output


def _artifact_sort_key(artifact: dict[str, Any]) -> tuple[Any, ...]:
    label = str(artifact.get("label", ""))
    try:
        label_rank = PREFERRED_ARTIFACT_LABELS.index(label)
    except ValueError:
        label_rank = len(PREFERRED_ARTIFACT_LABELS)
    return (label_rank, label, artifact.get("path") or "")


def _event_metrics(event: dict[str, Any], findings: list[dict[str, Any]], include_optional: bool) -> dict[str, Any]:
    metrics = {
        "finding_count": len(findings),
        "artifact_count": len(event.get("artifact_references", [])),
    }
    if include_optional:
        metrics.update(
            {
                "independent_group_count": event.get("independent_group_count"),
                "visual_method_count": event.get("cross_modal_context", {}).get("visual_method_count"),
                "audio_transition_present": event.get("cross_modal_context", {}).get("audio_transition_present"),
                "maximum_source_percentile": _event_percentile_from_findings(findings),
            }
        )
    return metrics


def _small_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    preferred = (
        "combined_percentile",
        "mean_optical_flow",
        "flow_warp_residual_mean",
        "flow_warp_residual_95th_percentile",
        "energy_change_ratio",
        "absolute_rms_difference",
        "unstable_region_count",
        "maximum_edge_instability",
        "maximum_texture_distance",
        "maximum_detail_residual",
    )
    output = {key: metrics.get(key) for key in preferred if metrics.get(key) is not None}
    if not output:
        for key in sorted(metrics)[:4]:
            output[key] = metrics[key]
    return output


def _temporal_metric_summary(temporal: dict[str, Any]) -> dict[str, Any]:
    transitions = temporal.get("notable_transitions", [])
    flow_values = [item.get("optical_flow", {}).get("mean_magnitude") for item in transitions]
    residual_values = [item.get("flow_warp_residual", {}).get("mean_normalized_residual") for item in transitions]
    return {
        "maximum_mean_motion": _max_number(flow_values),
        "maximum_flow_warp_residual": _max_number(residual_values),
    }


def _audio_metric_summary(audio: dict[str, Any]) -> dict[str, Any]:
    metrics = audio.get("global_metrics", {})
    return {
        "rms_amplitude": metrics.get("rms_amplitude"),
        "peak_absolute_amplitude": metrics.get("peak_absolute_amplitude"),
        "clipping_ratio": metrics.get("clipping_ratio"),
        "silence_ratio": metrics.get("silence_ratio"),
    }


def _visual_metric_summary(visual: dict[str, Any]) -> dict[str, Any]:
    transitions = visual.get("ranked_review_transitions", [])
    detail = [item.get("regional_summary", {}).get("maximum_regional_detail_residual") for item in transitions]
    edge = [item.get("regional_summary", {}).get("maximum_edge_instability") for item in transitions]
    return {
        "maximum_detail_residual": _max_number(detail),
        "maximum_edge_instability": _max_number(edge),
    }


def _max_number(values: list[Any]) -> float | None:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return max(clean) if clean else None


def _highest_priority(priorities: list[Any]) -> str | None:
    clean = [priority for priority in priorities if priority in PRIORITY_RANK]
    if not clean:
        return None
    return sorted(clean, key=lambda item: PRIORITY_RANK[item])[0]


def _duration(event: dict[str, Any]) -> float | None:
    if event.get("duration_seconds") is not None:
        return event.get("duration_seconds")
    start = event.get("start_timestamp_seconds")
    end = event.get("end_timestamp_seconds")
    if start is None or end is None:
        return None
    return round(float(end) - float(start), 6)


def _estimated_main_report_size(report: dict[str, Any]) -> int:
    try:
        return len(json.dumps(_sanitize(report), sort_keys=True, allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _validate_no_absolute_paths(value: Any, errors: list[str], path: str = "compact") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_no_absolute_paths(child, errors, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_absolute_paths(child, errors, f"{path}[{index}]")
    elif isinstance(value, str) and _looks_like_absolute_path(value):
        errors.append(f"Compact report contains an absolute path at {path}.")


def _looks_like_absolute_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("/") or value.startswith("\\\\")


def _validate_no_nonfinite(value: Any, errors: list[str], path: str = "compact") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_no_nonfinite(child, errors, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_nonfinite(child, errors, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"Compact report contains a non-finite number at {path}.")


def _validate_no_forbidden_fields(value: Any, errors: list[str], path: str = "compact") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_FIELD_NAMES:
                errors.append(f"Compact report contains forbidden field {path}.{key}.")
            _validate_no_forbidden_fields(child, errors, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_forbidden_fields(child, errors, f"{path}[{index}]")
