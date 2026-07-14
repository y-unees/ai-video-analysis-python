from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import (
    AI_INPUT_MAX_ARTIFACTS_PER_EVENT,
    AI_INPUT_MAX_FINDINGS_PER_EVENT,
    AI_INPUT_MAX_GLOBAL_FINDINGS,
    AI_INPUT_MAX_TIMELINE_EVENTS,
    AI_INPUT_TARGET_CHARACTER_COUNT,
    EVIDENCE_MAX_ANCHOR_EVENT_SPAN_SECONDS,
    EVIDENCE_MERGE_TOLERANCE_SECONDS,
    REGIONAL_GROUP_MINIMUM_OVERLAP_RATIO,
    SCHEMA_VERSION,
    unified_evidence_configuration,
)
from file_utils import artifact_record


VALID_DOMAINS = {
    "metadata",
    "frame_sampling",
    "visual_temporal",
    "visual_consistency",
    "audio_signal",
    "provenance",
    "external_model_results",
}
VALID_ROLES = {"anchor_event", "supporting_interval", "contextual_interval"}
DOMAIN_GROUPS = {
    "metadata": "metadata",
    "frame_sampling": "visual",
    "visual_temporal": "visual",
    "visual_consistency": "visual",
    "audio_signal": "audio",
    "provenance": "provenance",
    "external_model_results": "learned_model",
}
DOMAIN_FINDING_CAPS = {
    "visual_consistency": 3,
    "visual_temporal": 3,
    "audio_signal": 2,
    "frame_sampling": 1,
}
ANCHOR_TYPES = {
    "sampled_frame_comparison",
    "ranked_review_transition",
    "scene_boundary_candidate",
    "ranked_audio_energy_transition",
    "ranked_visual_consistency_transition",
    "clipping_like_audio_interval",
}


def build_unified_evidence_artifacts(
    analysis_dir: Path,
    report: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    try:
        observation_index = _build_observation_index(report)
        global_evidence = _global_evidence(report)
        candidates = _time_candidates(report, observation_index)
        candidates.sort(key=_candidate_sort_key)
        events, context_intervals = _build_timeline_events(candidates)
        regional_groups = _group_regional_intervals(report.get("visual_consistency_analysis", {}))
        for event in events:
            _finalize_event(event)
        review_highlights = _review_highlights(events)
        bundle = _bundle(
            report,
            global_evidence,
            candidates,
            events,
            context_intervals,
            review_highlights,
            regional_groups,
        )
        ai_input = _ai_input(bundle)
        validation = validate_unified_evidence(bundle, ai_input, observation_index["all_ids"])
        artifacts = _write_artifacts(analysis_dir, bundle, events, ai_input)
        status = "completed" if not validation["errors"] else "failed_validation"
        section = {
            "status": status,
            "reason_code": None if status == "completed" else "validation_failed",
            "reason": None if status == "completed" else "Unified evidence validation failed.",
            "configuration": unified_evidence_configuration(),
            "summary": {
                "global_evidence_count": len(global_evidence),
                "time_based_candidate_count": len(candidates),
                "anchor_candidate_count": _role_count(candidates, "anchor_event"),
                "supporting_interval_count": _role_count(candidates, "supporting_interval"),
                "contextual_interval_count": _role_count(candidates, "contextual_interval"),
                "timeline_event_count": len(events),
                "priority_review_event_count": len(review_highlights),
                "evidence_domains_available": _domains_available(report),
                "evidence_groups_available": sorted({DOMAIN_GROUPS[d] for d in _domains_available(report)}),
                "external_model_result_count": 0,
                "ai_ready_input_character_count": len(json.dumps(ai_input, sort_keys=True)),
            },
            "timeline_configuration": bundle["timeline_configuration"],
            "evidence_domains": bundle["evidence_domains"],
            "review_highlights": review_highlights,
            "ambiguous_findings": bundle["ambiguous_findings"],
            "normal_or_non_supporting_findings": bundle["normal_or_non_supporting_findings"],
            "missing_evidence": bundle["missing_evidence"],
            "artifacts": artifacts,
            "validation": validation,
            "limitations": bundle["analysis_limitations"],
        }
        if validation["warnings"]:
            warnings.extend(validation["warnings"])
        return section, warnings
    except Exception as error:
        warnings.append(f"Unified evidence generation failed: {error}")
        return {
            "status": "failed",
            "reason_code": "unified_evidence_exception",
            "reason": str(error),
            "configuration": unified_evidence_configuration(),
            "summary": {},
            "timeline_configuration": {},
            "evidence_domains": {},
            "review_highlights": [],
            "ambiguous_findings": [],
            "normal_or_non_supporting_findings": [],
            "missing_evidence": [],
            "artifacts": {},
            "validation": {"errors": [], "warnings": []},
            "limitations": ["Unified evidence generation failed; earlier analysis sections may still be available."],
        }, warnings


def validate_unified_evidence(
    bundle: dict[str, Any],
    ai_input: dict[str, Any],
    canonical_observation_ids: set[str] | None = None,
) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    canonical_observation_ids = canonical_observation_ids or set()
    candidates = bundle.get("_time_based_candidates", [])
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    event_ids = [event["event_id"] for event in bundle.get("timeline_events", [])]
    if len(event_ids) != len(set(event_ids)):
        errors.append("Unified evidence event IDs are duplicated.")
    previous_key = None
    primary_assignments = {candidate["candidate_id"]: 0 for candidate in candidates}
    for candidate in candidates:
        if candidate.get("candidate_role") not in VALID_ROLES:
            errors.append(f"Unified evidence candidate has invalid role: {candidate.get('candidate_id')}")
        if candidate["start_timestamp_seconds"] > candidate["end_timestamp_seconds"]:
            errors.append(f"Unified evidence candidate has invalid interval bounds: {candidate['candidate_id']}")
        if candidate.get("observation_reference_status") == "resolved":
            for observation_id in candidate.get("source_observation_ids", []):
                if observation_id not in canonical_observation_ids:
                    errors.append(f"Resolved observation ID is not canonical: {observation_id}")
        if candidate.get("observation_reference_status") == "not_available" and candidate.get("source_observation_ids"):
            errors.append(f"Candidate has observations but is marked unavailable: {candidate['candidate_id']}")
    for event in bundle.get("timeline_events", []):
        if event["start_timestamp_seconds"] > event["end_timestamp_seconds"]:
            errors.append("Unified evidence event start timestamp is after end timestamp.")
        key = (event["start_timestamp_seconds"], event["end_timestamp_seconds"], event["event_id"])
        if previous_key is not None and key < previous_key:
            errors.append("Unified evidence timeline events are not deterministically ordered.")
        previous_key = key
        priority = event.get("review_priority", {})
        if priority.get("level") not in {"low", "moderate", "high"}:
            errors.append("Unified evidence review priority level is invalid.")
        if priority.get("supports_authenticity_verdict") is not False:
            errors.append("Unified evidence review priority must not support authenticity verdicts.")
        if priority.get("supports_ai_probability") is not False:
            errors.append("Unified evidence review priority must not support AI probabilities.")
        anchor_ids = event.get("boundary_basis", {}).get("anchor_candidate_ids", [])
        if not anchor_ids and not event.get("boundary_basis", {}).get("standalone_interval_event"):
            errors.append(f"Unified evidence event has no boundary anchor basis: {event['event_id']}")
        if anchor_ids:
            anchor_start = min(candidate_by_id[cid]["start_timestamp_seconds"] for cid in anchor_ids if cid in candidate_by_id)
            anchor_end = max(candidate_by_id[cid]["end_timestamp_seconds"] for cid in anchor_ids if cid in candidate_by_id)
            if event["start_timestamp_seconds"] != round(anchor_start, 6) or event["end_timestamp_seconds"] != round(anchor_end, 6):
                errors.append("Unified evidence event boundaries are not based on anchor candidates.")
            if event["duration_seconds"] > EVIDENCE_MAX_ANCHOR_EVENT_SPAN_SECONDS and not event.get("boundary_basis", {}).get("maximum_span_override_reason"):
                errors.append("Unified evidence anchor event exceeds maximum span without override reason.")
        for candidate_id in event.get("source_candidate_ids", []):
            if candidate_id not in candidate_by_id:
                errors.append(f"Unified evidence event references an unknown candidate: {candidate_id}")
            else:
                primary_assignments[candidate_id] += 1
        for candidate_id in event.get("context_candidate_ids", []):
            if candidate_id not in candidate_by_id:
                errors.append(f"Unified evidence event references an unknown context candidate: {candidate_id}")
        for domain in event.get("evidence_domains", []):
            if domain not in VALID_DOMAINS:
                errors.append(f"Unified evidence event has an invalid domain: {domain}")
        if event["independent_group_count"] != len(set(event.get("evidence_groups_present", []))):
            errors.append("Unified evidence independent group count is inconsistent.")
        for artifact in event.get("artifact_references", []):
            path = artifact.get("path", "")
            if _is_absolute_path(path):
                errors.append(f"Unified evidence artifact path is absolute: {path}")
        selection = event.get("finding_selection", {})
        if selection:
            if selection.get("included_finding_count") != len(event.get("findings", [])):
                errors.append("Unified evidence finding selection count is inconsistent.")
    for candidate in candidates:
        count = primary_assignments[candidate["candidate_id"]]
        if candidate["candidate_role"] == "anchor_event" and count != 1:
            errors.append(f"Anchor candidate was not assigned to exactly one event: {candidate['candidate_id']}")
        if candidate["candidate_role"] == "supporting_interval" and count > 1:
            errors.append(f"Supporting candidate was assigned to multiple primary events: {candidate['candidate_id']}")
        if candidate["candidate_role"] != "contextual_interval" and count == 0:
            errors.append(f"Non-contextual candidate was discarded: {candidate['candidate_id']}")
    normal_types = {item.get("type") for item in bundle.get("normal_or_non_supporting_findings", [])}
    if "no_provenance_result" in normal_types or "content_provenance" in normal_types:
        errors.append("Unavailable provenance must not be listed as normal or non-supporting evidence.")
    missing_types = {item.get("type") for item in bundle.get("missing_evidence", [])}
    if "content_provenance" not in missing_types:
        errors.append("Missing content provenance is not listed under missing evidence.")
    if ai_input.get("interpretation_constraints", {}).get("must_not_invent_probability") is not True:
        errors.append("AI interpretation input is missing strict probability constraints.")
    if len(ai_input.get("priority_review_events", [])) > AI_INPUT_MAX_TIMELINE_EVENTS:
        errors.append("AI interpretation input exceeds the configured event cap.")
    if ai_input.get("external_model_results") != []:
        errors.append("External model results must be empty in v0.7.")
    for event in ai_input.get("priority_review_events", []):
        if len(event.get("findings", [])) > AI_INPUT_MAX_FINDINGS_PER_EVENT:
            errors.append("AI interpretation input event exceeds finding cap.")
        selection = event.get("finding_selection", {})
        if selection and selection.get("included_finding_count") != len(event.get("findings", [])):
            errors.append("AI interpretation input finding selection metadata is inconsistent.")
    return {"errors": errors, "warnings": warnings}


def _bundle(
    report: dict[str, Any],
    global_evidence: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    events: list[dict[str, Any]],
    context_intervals: list[dict[str, Any]],
    review_highlights: list[dict[str, Any]],
    regional_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "bundle_version": SCHEMA_VERSION,
        "source": {
            "filename": report.get("source", {}).get("filename"),
            "sha256": report.get("source", {}).get("sha256"),
            "duration_seconds": report.get("metadata", {}).get("container", {}).get("duration_seconds"),
        },
        "analysis_scope": {
            "local_only": True,
            "external_model_invoked": False,
            "trained_classifier_invoked": False,
        },
        "timeline_configuration": _timeline_configuration(report),
        "evidence_domains": _evidence_domains(report),
        "global_evidence": global_evidence,
        "timeline_summary": {
            "time_based_candidate_count": len(candidates),
            "anchor_candidate_count": _role_count(candidates, "anchor_event"),
            "supporting_interval_count": _role_count(candidates, "supporting_interval"),
            "contextual_interval_count": _role_count(candidates, "contextual_interval"),
            "timeline_event_count": len(events),
            "regional_interval_group_count": len(regional_groups),
            "merge_tolerance_seconds": EVIDENCE_MERGE_TOLERANCE_SECONDS,
            "maximum_anchor_event_span_seconds": EVIDENCE_MAX_ANCHOR_EVENT_SPAN_SECONDS,
            "merging_strategy": "anchor_based_non_transitive",
        },
        "timeline_events": events,
        "context_intervals": context_intervals,
        "regional_context_groups": regional_groups,
        "regional_interval_groups": regional_groups,
        "priority_review_events": review_highlights,
        "review_highlights": review_highlights,
        "ambiguous_findings": _ambiguous_findings(report),
        "normal_or_non_supporting_findings": _normal_findings(report),
        "missing_evidence": _missing_evidence(),
        "analysis_limitations": _dedupe_strings(
            report.get("limitations", [])
            + [
                "Unified evidence review priority is only an ordering aid for human review.",
                "No external AI model, calibrated detector, or authenticity probability is used in v0.7.1.",
            ]
        ),
        "external_model_results": [],
        "artifact_index": _artifact_index(report),
        "_time_based_candidates": candidates,
    }


def _write_artifacts(
    analysis_dir: Path,
    bundle: dict[str, Any],
    events: list[dict[str, Any]],
    ai_input: dict[str, Any],
) -> dict[str, Any]:
    public_bundle = {key: value for key, value in bundle.items() if not key.startswith("_")}
    unified_path = analysis_dir / "unified_evidence.json"
    unified_path.write_text(json.dumps(public_bundle, indent=2), encoding="utf-8")
    timeline_path = analysis_dir / "evidence_timeline.jsonl"
    with timeline_path.open("w", encoding="utf-8", newline="\n") as file:
        for event in events:
            public_event = {key: value for key, value in event.items() if key not in {"_primary_candidates", "_context_candidates"}}
            file.write(json.dumps(public_event, sort_keys=True) + "\n")
    ai_path = analysis_dir / "ai_interpretation_input.json"
    ai_path.write_text(json.dumps(ai_input, indent=2), encoding="utf-8")
    prompt_path = analysis_dir / "ai_interpretation_prompt.txt"
    prompt_path.write_text(_prompt_template(), encoding="utf-8")
    return {
        "unified_evidence": artifact_record(unified_path, "unified_evidence.json"),
        "evidence_timeline": artifact_record(timeline_path, "evidence_timeline.jsonl"),
        "ai_interpretation_input": artifact_record(ai_path, "ai_interpretation_input.json"),
        "ai_interpretation_prompt": artifact_record(prompt_path, "ai_interpretation_prompt.txt"),
    }


def _ai_input(bundle: dict[str, Any]) -> dict[str, Any]:
    highlight_ids = [item["event_id"] for item in bundle["priority_review_events"]]
    events_by_id = {event["event_id"]: event for event in bundle["timeline_events"]}
    original_events = [events_by_id[event_id] for event_id in highlight_ids if event_id in events_by_id]
    included_events = original_events[:AI_INPUT_MAX_TIMELINE_EVENTS]
    compact_events = []
    for event in included_events:
        selected, selection = _balanced_findings(event)
        compact_events.append(
            {
                "event_id": event["event_id"],
                "start_timestamp_seconds": event["start_timestamp_seconds"],
                "end_timestamp_seconds": event["end_timestamp_seconds"],
                "evidence_domains": event["evidence_domains"],
                "evidence_groups_present": event["evidence_groups_present"],
                "independent_group_count": event["independent_group_count"],
                "cross_modal_context": event["cross_modal_context"],
                "review_priority": event["review_priority"],
                "boundary_basis": event["boundary_basis"],
                "findings": selected,
                "finding_selection": selection,
                "artifact_references": event["artifact_references"][:AI_INPUT_MAX_ARTIFACTS_PER_EVENT],
                "source_candidate_ids": event["source_candidate_ids"],
                "context_candidate_ids": event["context_candidate_ids"],
                "source_record_ids": event["source_record_ids"],
                "source_observation_ids": event["source_observation_ids"],
            }
        )
    output = {
        "bundle_version": bundle["bundle_version"],
        "source": bundle["source"],
        "analysis_scope": bundle["analysis_scope"],
        "analysis_stages": bundle["evidence_domains"],
        "timeline_configuration": bundle["timeline_configuration"],
        "important_global_evidence": bundle["global_evidence"][:AI_INPUT_MAX_GLOBAL_FINDINGS],
        "priority_review_events": compact_events,
        "context_intervals": bundle["context_intervals"][:AI_INPUT_MAX_TIMELINE_EVENTS],
        "ambiguous_findings": bundle["ambiguous_findings"],
        "normal_or_non_supporting_findings": bundle["normal_or_non_supporting_findings"],
        "missing_evidence": bundle["missing_evidence"],
        "limitations": bundle["analysis_limitations"],
        "external_model_results": [],
        "future_response_schema_path": "schemas/ai_interpretation_response.schema.json",
        "interpretation_constraints": {
            "must_not_claim_authenticity": True,
            "must_not_claim_ai_generation_as_fact": True,
            "must_not_invent_probability": True,
            "numeric_probability_requires_calibrated_model": True,
            "must_present_alternative_explanations": True,
            "must_identify_missing_evidence": True,
            "must_treat_context_intervals_separately": True,
            "must_not_assume_overlapping_heuristics_are_independent": True,
        },
    }
    estimated = len(json.dumps(output, sort_keys=True))
    output["compaction"] = {
        "applied": len(bundle["timeline_events"]) > len(compact_events) or estimated > AI_INPUT_TARGET_CHARACTER_COUNT,
        "original_event_count": len(bundle["timeline_events"]),
        "included_event_count": len(compact_events),
        "omitted_event_count": max(0, len(bundle["timeline_events"]) - len(compact_events)),
        "estimated_character_count": estimated,
    }
    return output


def _global_evidence(report: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = report.get("metadata", {})
    encoding = metadata.get("encoding", {})
    duration = metadata.get("duration_comparison", {})
    items = [
        _global("metadata", "source_hash", "source.sha256", f"Source SHA-256: {report.get('source', {}).get('sha256')}", {"sha256": report.get("source", {}).get("sha256")}),
        _global("metadata", "container_format", "metadata.container", f"Container format: {metadata.get('container', {}).get('friendly_format')}", {"format_name": metadata.get("container", {}).get("format_name")}),
        _global("metadata", "video_stream", "metadata.video", f"Video stream codec: {metadata.get('video', {}).get('codec_name')}", {"codec": metadata.get("video", {}).get("codec_name")}),
        _global("metadata", "audio_stream", "metadata.audio", f"Audio stream present: {metadata.get('audio', {}).get('present')}", {"codec": metadata.get("audio", {}).get("codec_name")}),
    ]
    for key, label in (
        ("container_encoder", "Container encoder metadata is absent."),
        ("video_stream_encoder", "Video encoder metadata is absent."),
        ("audio_stream_encoder", "Audio encoder metadata is absent."),
    ):
        if encoding.get(key) is None:
            items.append(_global("metadata", "missing_encoder_metadata", f"metadata.encoding.{key}", label, {"field": key}))
    if duration:
        items.append(_global("metadata", "stream_timing_comparison", "metadata.duration_comparison", "Stream timing differences were recorded without treating small differences as desynchronization.", duration))
    for stage, domain in (
        ("temporal_analysis", "visual_temporal"),
        ("audio_analysis", "audio_signal"),
        ("visual_consistency_analysis", "visual_consistency"),
    ):
        value = report.get(stage, {})
        items.append(_global(domain, "analysis_stage_status", stage, f"{stage} status: {value.get('status')}", {"status": value.get("status"), "reason_code": value.get("reason_code")}))
    return _renumber(items, "global-evidence")


def _time_candidates(report: dict[str, Any], observation_index: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for comparison in report.get("frame_analysis", {}).get("comparisons", []):
        classification = comparison.get("classification", {})
        if classification.get("near_duplicate") or classification.get("large_change"):
            record_id = f"frame-comparison-{comparison['from_sample_index']:05d}-{comparison['to_sample_index']:05d}"
            candidates.append(_candidate("frame_sampling", "sampled_frame_comparison", record_id, comparison["start_timestamp_seconds"], comparison["end_timestamp_seconds"], "Sampled-frame comparison met configured review heuristics.", {"classification": classification}, [], _resolve_observations(observation_index, record_id, "frame_sampling", comparison["start_timestamp_seconds"], comparison["end_timestamp_seconds"])))
    temporal = report.get("temporal_analysis", {})
    for transition in temporal.get("notable_transitions", []):
        record_id = transition["transition_id"]
        candidates.append(_candidate("visual_temporal", "ranked_review_transition", record_id, transition["start_timestamp_seconds"], transition["end_timestamp_seconds"], "Ranked temporal review transition.", _temporal_metrics(transition), _artifacts_from_mapping(transition.get("artifacts", {})), _resolve_observations(observation_index, record_id, "visual_temporal", transition["start_timestamp_seconds"], transition["end_timestamp_seconds"])))
    for scene in temporal.get("scenes", []):
        if scene.get("boundary_transition_id"):
            record_id = scene["boundary_transition_id"]
            candidates.append(_candidate("visual_temporal", "scene_boundary_candidate", record_id, scene["start_timestamp_seconds"], scene["start_timestamp_seconds"], "Scene-boundary candidate from temporal analysis.", {}, [], _resolve_observations(observation_index, record_id, "visual_temporal", scene["start_timestamp_seconds"], scene["start_timestamp_seconds"])))
    for index, interval in enumerate(temporal.get("notable_intervals", []), start=1):
        interval_id = interval.get("interval_id") or f"temporal-near-static-interval-{index:03d}"
        candidates.append(_candidate("visual_temporal", "sustained_near_static_interval", interval_id, interval["start_timestamp_seconds"], interval["end_timestamp_seconds"], "Sustained near-static visual interval.", interval.get("supporting_metrics", interval.get("metrics", {})), [], _resolve_observations(observation_index, interval_id, "visual_temporal", interval["start_timestamp_seconds"], interval["end_timestamp_seconds"])))
    audio = report.get("audio_analysis", {})
    for transition in audio.get("notable_transitions", []):
        record_id = transition["transition_id"]
        candidates.append(_candidate("audio_signal", "ranked_audio_energy_transition", record_id, transition["start_timestamp_seconds"], transition["end_timestamp_seconds"], "Ranked audio-energy transition.", _audio_metrics(transition), [], _resolve_observations(observation_index, record_id, "audio_signal", transition["start_timestamp_seconds"], transition["end_timestamp_seconds"])))
    for interval in audio.get("silence_intervals", []):
        role = "contextual_interval" if interval.get("duration_seconds", 0) > EVIDENCE_MAX_ANCHOR_EVENT_SPAN_SECONDS else "supporting_interval"
        candidates.append(_candidate("audio_signal", "silence_like_audio_interval", interval["interval_id"], interval["start_timestamp_seconds"], interval["end_timestamp_seconds"], "Silence-like audio interval.", {"average_rms_amplitude": interval.get("average_rms_amplitude")}, [], _resolve_observations(observation_index, interval["interval_id"], "audio_signal", interval["start_timestamp_seconds"], interval["end_timestamp_seconds"]), role))
    for interval in audio.get("clipping_intervals", []):
        candidates.append(_candidate("audio_signal", "clipping_like_audio_interval", interval["interval_id"], interval["start_timestamp_seconds"], interval["end_timestamp_seconds"], "Clipping-like audio interval.", interval, [], _resolve_observations(observation_index, interval["interval_id"], "audio_signal", interval["start_timestamp_seconds"], interval["end_timestamp_seconds"]), "anchor_event"))
    visual = report.get("visual_consistency_analysis", {})
    for transition in visual.get("ranked_review_transitions", []):
        record_id = transition["transition_id"]
        candidates.append(_candidate("visual_consistency", "ranked_visual_consistency_transition", record_id, transition["start_timestamp_seconds"], transition["end_timestamp_seconds"], "Ranked regional visual-consistency transition.", _visual_consistency_metrics(transition), _artifacts_from_mapping(transition.get("artifacts", {})), _resolve_observations(observation_index, record_id, "visual_consistency", transition["start_timestamp_seconds"], transition["end_timestamp_seconds"])))
    for interval in visual.get("sustained_intervals", []):
        candidates.append(_candidate("visual_consistency", "sustained_regional_visual_variation", interval["interval_id"], interval["start_timestamp_seconds"], interval["end_timestamp_seconds"], "Sustained regional visual variation.", interval.get("supporting_metrics", {}) | {"affected_regions": interval.get("affected_regions", [])}, [], _resolve_observations(observation_index, interval["interval_id"], "visual_consistency", interval["start_timestamp_seconds"], interval["end_timestamp_seconds"]), "supporting_interval"))
    return _renumber(candidates, "evidence-candidate")


def _build_timeline_events(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchors = [candidate for candidate in candidates if candidate["candidate_role"] == "anchor_event"]
    supporting = [candidate for candidate in candidates if candidate["candidate_role"] == "supporting_interval"]
    contextual = [candidate for candidate in candidates if candidate["candidate_role"] == "contextual_interval"]
    events = _anchor_events(anchors)
    standalone_supporting: list[dict[str, Any]] = []
    for candidate in supporting:
        relation = _best_event_relation(candidate, events)
        if relation is None:
            standalone_supporting.append(candidate)
            continue
        event = relation["event"]
        event["_primary_candidates"].append(candidate)
        event["supporting_assignments"].append({key: value for key, value in relation.items() if key != "event"})
        candidate["primary_event_id"] = event["event_id"]
        candidate["primary_assignment"] = {key: value for key, value in relation.items() if key != "event"}
        for other in _contextual_event_relations(candidate, events, exclude_event_id=event["event_id"]):
            other["event"]["_context_candidates"].append(candidate)
            other["event"]["context_assignments"].append({key: value for key, value in other.items() if key != "event"})
            candidate.setdefault("context_event_ids", []).append(other["event"]["event_id"])
    for candidate in standalone_supporting:
        events.append(_standalone_interval_event(candidate))
    events.sort(key=lambda event: (event["start_timestamp_seconds"], event["end_timestamp_seconds"], event["event_id"]))
    for index, event in enumerate(events, start=1):
        old_id = event["event_id"]
        event["event_id"] = f"evidence-event-{index:03d}"
        for candidate in event["_primary_candidates"]:
            if candidate.get("primary_event_id") == old_id:
                candidate["primary_event_id"] = event["event_id"]
    for candidate in contextual:
        candidate["related_event_ids"] = []
        for relation in _contextual_event_relations(candidate, events):
            event = relation["event"]
            event["_context_candidates"].append(candidate)
            event["context_assignments"].append({key: value for key, value in relation.items() if key != "event"})
            candidate["related_event_ids"].append(event["event_id"])
    context_intervals = [_context_interval_view(candidate) for candidate in contextual]
    return events, context_intervals


def _anchor_events(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for anchor in sorted(anchors, key=_candidate_sort_key):
        if not current:
            current = [anchor]
            continue
        start = min(item["start_timestamp_seconds"] for item in current)
        end = max(item["end_timestamp_seconds"] for item in current)
        merged_start = min(start, anchor["start_timestamp_seconds"])
        merged_end = max(end, anchor["end_timestamp_seconds"])
        if _intervals_close(start, end, anchor["start_timestamp_seconds"], anchor["end_timestamp_seconds"]) and (merged_end - merged_start) <= EVIDENCE_MAX_ANCHOR_EVENT_SPAN_SECONDS:
            current.append(anchor)
        else:
            events.append(_event_from_anchors(current))
            current = [anchor]
    if current:
        events.append(_event_from_anchors(current))
    for index, event in enumerate(events, start=1):
        event["event_id"] = f"evidence-event-{index:03d}"
        for candidate in event["_primary_candidates"]:
            candidate["primary_event_id"] = event["event_id"]
    return events


def _event_from_anchors(anchors: list[dict[str, Any]]) -> dict[str, Any]:
    start = min(candidate["start_timestamp_seconds"] for candidate in anchors)
    end = max(candidate["end_timestamp_seconds"] for candidate in anchors)
    return _event(start, end, anchors, standalone_interval=False)


def _standalone_interval_event(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate["primary_assignment"] = {
        "primary_event_id": "",
        "relationship": "standalone_interval_event",
        "overlap_seconds": 0.0,
        "overlap_ratio": 0.0,
    }
    event = _event(candidate["start_timestamp_seconds"], candidate["end_timestamp_seconds"], [candidate], standalone_interval=True)
    return event


def _event(start: float, end: float, primary_candidates: list[dict[str, Any]], standalone_interval: bool) -> dict[str, Any]:
    candidate_ids = [candidate["candidate_id"] for candidate in primary_candidates]
    anchor_ids = [candidate["candidate_id"] for candidate in primary_candidates if candidate["candidate_role"] == "anchor_event"]
    supporting_ids = [candidate["candidate_id"] for candidate in primary_candidates if candidate["candidate_role"] == "supporting_interval"]
    return {
        "event_id": "",
        "start_timestamp_seconds": round(start, 6),
        "end_timestamp_seconds": round(end, 6),
        "duration_seconds": round(max(0.0, end - start), 6),
        "source_candidate_ids": candidate_ids,
        "context_candidate_ids": [],
        "source_observation_ids": [],
        "source_record_ids": [],
        "evidence_domains": [],
        "domain_count": 0,
        "evidence_groups_present": [],
        "independent_group_count": 0,
        "findings": [],
        "artifact_references": [],
        "cross_modal_context": {},
        "review_priority": {},
        "boundary_basis": {
            "anchor_candidate_ids": anchor_ids,
            "supporting_candidate_ids": supporting_ids,
            "context_candidate_ids": [],
            "standalone_interval_event": standalone_interval,
        },
        "supporting_assignments": [],
        "context_assignments": [],
        "limitations": [],
        "_primary_candidates": primary_candidates,
        "_context_candidates": [],
    }


def _finalize_event(event: dict[str, Any]) -> None:
    primary = _unique_candidates(event["_primary_candidates"])
    context = _unique_candidates(event["_context_candidates"])
    all_for_context = primary + context
    domains = sorted({candidate["source_domain"] for candidate in primary})
    context_domains = sorted({candidate["source_domain"] for candidate in context})
    groups = sorted({DOMAIN_GROUPS[domain] for domain in set(domains) | set(context_domains)})
    event["source_candidate_ids"] = [candidate["candidate_id"] for candidate in primary]
    event["context_candidate_ids"] = [candidate["candidate_id"] for candidate in context]
    event["boundary_basis"]["supporting_candidate_ids"] = [candidate["candidate_id"] for candidate in primary if candidate["candidate_role"] == "supporting_interval"]
    event["boundary_basis"]["context_candidate_ids"] = event["context_candidate_ids"]
    event["source_observation_ids"] = sorted({item for candidate in all_for_context for item in candidate.get("source_observation_ids", [])})
    event["source_record_ids"] = sorted({candidate["source_record_id"] for candidate in all_for_context})
    event["evidence_domains"] = domains
    event["domain_count"] = len(domains)
    event["evidence_groups_present"] = groups
    event["independent_group_count"] = len(groups)
    event["artifact_references"] = _unique_artifacts([artifact for candidate in all_for_context for artifact in candidate.get("artifact_references", [])])
    findings = [_finding(candidate) for candidate in primary]
    if context:
        findings.extend(_context_finding(candidate) for candidate in context)
    event["findings"] = findings
    visual_methods = len((set(domains) | set(context_domains)) & {"frame_sampling", "visual_temporal", "visual_consistency"})
    audio_present = "audio_signal" in set(domains) | set(context_domains)
    visual_present = visual_methods > 0
    source_types = {candidate["source_type"] for candidate in all_for_context}
    if visual_present and audio_present:
        classification = "visual_and_audio_aligned"
    elif "sustained_near_static_interval" in source_types and audio_present:
        classification = "near_static_with_audio_activity"
    elif visual_methods >= 2:
        classification = "multiple_visual_methods"
    elif visual_present:
        classification = "visual_only"
    elif audio_present:
        classification = "audio_only"
    else:
        classification = "other"
    event["cross_modal_context"] = {
        "classification": classification,
        "visual_evidence_present": visual_present,
        "audio_evidence_present": audio_present,
        "visual_method_count": visual_methods,
        "audio_transition_present": any(candidate["source_type"] == "ranked_audio_energy_transition" for candidate in all_for_context),
        "contextual_interval_count": len(context),
    }
    basis: list[str] = []
    if event["independent_group_count"] >= 2:
        basis.append("multiple_independent_evidence_groups")
    if visual_methods >= 2:
        basis.append("multiple_visual_methods")
    if visual_present and audio_present:
        basis.append("audio_visual_temporal_overlap")
    if any("ranked" in candidate["source_type"] for candidate in primary):
        basis.append("ranked_source_findings")
    if event["artifact_references"]:
        basis.append("review_artifacts_available")
    if context:
        basis.append("contextual_interval_available")
    if event["independent_group_count"] >= 2 and ("audio_visual_temporal_overlap" in basis or "multiple_visual_methods" in basis):
        level = "high"
    elif len(basis) >= 2 or "ranked_source_findings" in basis:
        level = "moderate"
    else:
        level = "low"
    event["review_priority"] = {
        "level": level,
        "basis": basis or ["single_source_review_context"],
        "supports_authenticity_verdict": False,
        "supports_ai_probability": False,
    }
    selected, selection = _balanced_findings(event)
    event["finding_selection"] = selection
    event["findings"] = selected


def _review_highlights(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    level_rank = {"high": 0, "moderate": 1, "low": 2}
    ranked = sorted(
        events,
        key=lambda event: (
            level_rank[event["review_priority"]["level"]],
            -event["independent_group_count"],
            -_event_percentile(event),
            -len(event["artifact_references"]),
            event["start_timestamp_seconds"],
            event["event_id"],
        ),
    )[:AI_INPUT_MAX_TIMELINE_EVENTS]
    output = []
    for index, event in enumerate(ranked, start=1):
        output.append(
            {
                "priority_review_index": index,
                "event_id": event["event_id"],
                "start_timestamp_seconds": event["start_timestamp_seconds"],
                "end_timestamp_seconds": event["end_timestamp_seconds"],
                "review_priority": event["review_priority"],
                "evidence_domains": event["evidence_domains"],
                "evidence_groups_present": event["evidence_groups_present"],
                "independent_group_count": event["independent_group_count"],
                "cross_modal_context": event["cross_modal_context"],
                "source_observation_ids": event["source_observation_ids"],
                "boundary_basis": event["boundary_basis"],
                "context_candidate_ids": event["context_candidate_ids"],
                "key_findings": [
                    {
                        "domain": finding["domain"],
                        "type": finding["type"],
                        "summary": finding["summary"],
                        "source_observation_ids": finding.get("source_observation_ids", []),
                    }
                    for finding in event.get("findings", [])[:4]
                ],
                "ranking_logic": "priority level, independent group count, source percentile, artifact availability, start timestamp, event ID",
            }
        )
    return output


def _candidate(
    domain: str,
    source_type: str,
    record_id: str,
    start: float,
    end: float,
    summary: str,
    metrics: dict[str, Any],
    artifacts: list[dict[str, Any]],
    observation_ids: list[str] | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    candidate_role = role or _role_for_source_type(source_type)
    source_observation_ids = sorted(set(observation_ids or []))
    return {
        "candidate_id": "",
        "scope": "time_interval",
        "candidate_role": candidate_role,
        "source_domain": domain,
        "source_type": source_type,
        "source_record_id": record_id,
        "source_observation_ids": source_observation_ids,
        "observation_reference_status": "resolved" if source_observation_ids else "not_available",
        "start_timestamp_seconds": round(float(start), 6),
        "end_timestamp_seconds": round(float(end), 6),
        "summary": summary,
        "metrics": _compact_metrics(metrics),
        "artifact_references": artifacts,
        "primary_event_id": None,
        "context_event_ids": [],
    }


def _global(domain: str, source_type: str, record_id: str, summary: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": "",
        "scope": "global",
        "source_domain": domain,
        "source_type": source_type,
        "source_record_id": record_id,
        "summary": summary,
        "metrics": _compact_metrics(metrics),
        "artifact_references": [],
    }


def _role_for_source_type(source_type: str) -> str:
    if source_type in ANCHOR_TYPES:
        return "anchor_event"
    if source_type in {"sustained_near_static_interval"}:
        return "contextual_interval"
    return "supporting_interval"


def _renumber(items: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    for index, item in enumerate(items, start=1):
        item["candidate_id"] = f"{prefix}-{index:05d}"
    return items


def _timeline_configuration(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report.get("metadata", {})
    audio_start = metadata.get("audio", {}).get("start_time")
    video_start = metadata.get("video", {}).get("start_time")
    offset = None if audio_start is None or video_start is None else round(audio_start - video_start, 6)
    return {
        "timeline_basis": "selected_video_stream_normalized",
        "merge_tolerance_seconds": EVIDENCE_MERGE_TOLERANCE_SECONDS,
        "maximum_anchor_event_span_seconds": EVIDENCE_MAX_ANCHOR_EVENT_SPAN_SECONDS,
        "merging_strategy": "anchor_based_non_transitive",
        "audio_mapping": {
            "method": "audio timestamps are reported in their analysis timeline; source stream start offset is recorded and not silently corrected",
            "audio_start_offset_seconds": offset,
        },
    }


def _evidence_domains(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": {"status": "completed"},
        "frame_sampling": {"status": "completed" if report.get("frame_analysis", {}).get("frames") else "skipped"},
        "visual_temporal": {"status": report.get("temporal_analysis", {}).get("status")},
        "visual_consistency": {"status": report.get("visual_consistency_analysis", {}).get("status")},
        "audio_signal": {"status": report.get("audio_analysis", {}).get("status")},
        "provenance": {"status": "not_implemented"},
        "external_model_results": [],
    }


def _domains_available(report: dict[str, Any]) -> list[str]:
    domains = ["metadata"]
    if report.get("frame_analysis", {}).get("frames"):
        domains.append("frame_sampling")
    if report.get("temporal_analysis", {}).get("status") in {"completed", "partial"}:
        domains.append("visual_temporal")
    if report.get("visual_consistency_analysis", {}).get("status") in {"completed", "partial"}:
        domains.append("visual_consistency")
    if report.get("audio_analysis", {}).get("status") in {"completed", "partial"}:
        domains.append("audio_signal")
    return domains


def _build_observation_index(report: dict[str, Any]) -> dict[str, Any]:
    all_ids: set[str] = set()
    by_record: dict[str, list[str]] = {}
    by_domain_time: list[dict[str, Any]] = []
    observations = report.get("observations", {})
    for category, items in observations.items():
        if not isinstance(items, list):
            continue
        for item in items:
            observation_id = item.get("observation_id")
            if not observation_id:
                continue
            all_ids.add(observation_id)
            domain = _domain_for_observation_type(item.get("type", ""), category)
            start = item.get("timestamp_start")
            end = item.get("timestamp_end", start)
            if start is not None:
                by_domain_time.append({"domain": domain, "start": round(float(start), 6), "end": round(float(end), 6), "observation_id": observation_id})
    _index_ordered_records(by_record, report.get("temporal_analysis", {}).get("notable_transitions", []), observations.get("temporal_heuristics", []), "temporal.ranked_notable_transition", "transition_id")
    _index_ordered_records(by_record, report.get("audio_analysis", {}).get("notable_transitions", []), observations.get("audio_observations", []), "audio.ranked_energy_transition", "transition_id")
    _index_ordered_records(by_record, report.get("visual_consistency_analysis", {}).get("ranked_review_transitions", []), observations.get("visual_consistency_observations", []), "visual_consistency.ranked_review_transition", "transition_id")
    _index_ordered_records(by_record, report.get("visual_consistency_analysis", {}).get("sustained_intervals", []), observations.get("visual_consistency_observations", []), "visual_consistency.sustained_regional_variation", "interval_id")
    temporal_intervals = report.get("temporal_analysis", {}).get("notable_intervals", [])
    temporal_interval_observations = [item for item in observations.get("temporal_heuristics", []) if item.get("type") == "temporal.sustained_near_static_interval"]
    for index, interval in enumerate(temporal_intervals, start=1):
        record_id = interval.get("interval_id") or f"temporal-near-static-interval-{index:03d}"
        if index <= len(temporal_interval_observations):
            by_record.setdefault(record_id, []).append(temporal_interval_observations[index - 1]["observation_id"])
    return {"all_ids": all_ids, "by_record": by_record, "by_domain_time": by_domain_time}


def _index_ordered_records(
    by_record: dict[str, list[str]],
    records: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    observation_type: str,
    record_key: str,
) -> None:
    matching = [item for item in observations if item.get("type") == observation_type]
    for index, record in enumerate(records):
        record_id = record.get(record_key)
        if record_id and index < len(matching):
            by_record.setdefault(record_id, []).append(matching[index]["observation_id"])


def _resolve_observations(
    observation_index: dict[str, Any],
    record_id: str,
    domain: str,
    start: float,
    end: float,
) -> list[str]:
    resolved = list(observation_index["by_record"].get(record_id, []))
    for item in observation_index["by_domain_time"]:
        if item["domain"] != domain:
            continue
        if abs(item["start"] - round(float(start), 6)) <= 0.001 and abs(item["end"] - round(float(end), 6)) <= 0.001:
            resolved.append(item["observation_id"])
    return sorted(set(resolved))


def _domain_for_observation_type(observation_type: str, category: str) -> str:
    if observation_type.startswith("audio.") or category == "audio_observations":
        return "audio_signal"
    if observation_type.startswith("visual_consistency.") or category == "visual_consistency_observations":
        return "visual_consistency"
    if observation_type.startswith("temporal."):
        return "visual_temporal"
    if observation_type.startswith("metadata."):
        return "metadata"
    return "frame_sampling"


def _group_regional_intervals(visual: dict[str, Any]) -> list[dict[str, Any]]:
    intervals = sorted(visual.get("sustained_intervals", []), key=lambda item: (item["start_timestamp_seconds"], item["end_timestamp_seconds"], item["interval_id"]))
    groups: list[list[dict[str, Any]]] = []
    seeds: list[dict[str, Any]] = []
    for interval in intervals:
        placed = False
        for index, seed in enumerate(seeds):
            if _regional_groupable(seed, interval):
                groups[index].append(interval)
                placed = True
                break
        if not placed:
            seeds.append(interval)
            groups.append([interval])
    output = []
    for index, group in enumerate(groups, start=1):
        regions = sorted({region for interval in group for region in interval.get("affected_regions", [])})
        metrics = [interval.get("supporting_metrics", {}) for interval in group]
        output.append(
            {
                "regional_group_id": f"regional-context-{index:03d}",
                "start_timestamp_seconds": min(interval["start_timestamp_seconds"] for interval in group),
                "end_timestamp_seconds": max(interval["end_timestamp_seconds"] for interval in group),
                "source_interval_ids": [interval["interval_id"] for interval in group],
                "affected_regions": regions,
                "affected_region_count": len(regions),
                "neighboring_region_relationships": "grouped when regions are identical or adjacent and intervals substantially overlap with the group seed",
                "minimum_overlap_ratio": REGIONAL_GROUP_MINIMUM_OVERLAP_RATIO,
                "supporting_metrics": {
                    "maximum_average_edge_instability": _max_metric(metrics, "average_edge_instability"),
                    "average_texture_distance": _avg_metric(metrics, "average_texture_distance"),
                    "average_detail_residual": _avg_metric(metrics, "average_detail_residual"),
                },
            }
        )
    return output


def _regional_groupable(seed: dict[str, Any], interval: dict[str, Any]) -> bool:
    overlap = _overlap_seconds(seed, interval)
    shorter = max(1e-9, min(_duration(seed), _duration(interval)))
    if overlap / shorter < REGIONAL_GROUP_MINIMUM_OVERLAP_RATIO:
        return False
    return any(_regions_neighbor(a, b) for a in seed.get("affected_regions", []) for b in interval.get("affected_regions", []))


def _regions_neighbor(first: str, second: str) -> bool:
    first_pos = _region_position(first)
    second_pos = _region_position(second)
    if first_pos is None or second_pos is None:
        return first == second
    return abs(first_pos[0] - second_pos[0]) <= 1 and abs(first_pos[1] - second_pos[1]) <= 1


def _region_position(region_id: str) -> tuple[int, int] | None:
    try:
        row = int(region_id.split("-r", 1)[1].split("-c", 1)[0])
        column = int(region_id.split("-c", 1)[1])
        return row, column
    except (IndexError, ValueError):
        return None


def _ambiguous_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    encoding = report.get("metadata", {}).get("encoding", {})
    if any(encoding.get(key) is None for key in ("container_encoder", "video_stream_encoder", "audio_stream_encoder")):
        items.append({"type": "missing_encoder_metadata", "description": "Encoder metadata is absent for one or more streams; this has many normal explanations."})
    if report.get("audio_analysis", {}).get("notable_transitions"):
        items.append({"type": "audio_energy_changes", "description": "Audio-energy changes may reflect speech, music, edits, effects, microphone behavior, or compression."})
    if report.get("visual_consistency_analysis", {}).get("ranked_review_transitions"):
        items.append({"type": "regional_visual_variation", "description": "Regional edge, texture, brightness, or detail variation may reflect motion, lighting, focus, compression, animation, or ordinary edits."})
    return items


def _normal_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    audio = report.get("audio_analysis", {})
    temporal = report.get("temporal_analysis", {})
    findings = []
    if audio.get("global_metrics", {}).get("clipping_ratio") == 0:
        findings.append({"type": "no_clipping_like_samples", "description": "No clipping-like samples were measured; this is not proof of authenticity."})
    if not temporal.get("notable_intervals"):
        findings.append({"type": "no_sustained_near_static_intervals", "description": "No sustained near-static intervals were reported by temporal analysis."})
    if report.get("metadata", {}).get("duration_comparison", {}).get("duration_only_difference_seconds") in (0, None):
        findings.append({"type": "duration_metadata_not_inconsistent", "description": "Duration metadata did not show a duration-only discrepancy; this is not proof of authenticity."})
    return findings


def _missing_evidence() -> list[dict[str, Any]]:
    return [
        {"type": "content_provenance", "status": "not_available", "importance": "high", "description": "No signed provenance or generator declaration was available."},
        {"type": "trained_ai_video_detector", "status": "not_implemented", "importance": "high"},
        {"type": "generator_watermark_detection", "status": "not_implemented", "importance": "medium"},
        {"type": "dataset_calibrated_probability", "status": "not_available", "importance": "high"},
    ]


def _artifact_index(report: dict[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for section in (
        report.get("artifacts", {}).get("raw_ffprobe_report_artifact"),
        report.get("temporal_analysis", {}).get("artifacts", {}).get("temporal_metrics_artifact"),
        report.get("audio_analysis", {}).get("artifacts", {}).get("audio_metrics_artifact"),
        report.get("visual_consistency_analysis", {}).get("artifacts", {}).get("visual_consistency_metrics_artifact"),
    ):
        if section and section.get("path"):
            artifacts[section["path"]] = section
    return artifacts


def _prompt_template() -> str:
    return """You are given a compact local video-analysis evidence bundle.

Analyze only the provided evidence. Distinguish measured facts from heuristic observations. Identify supporting findings, weak or ambiguous findings, non-supporting findings, important timestamps, plausible normal alternative explanations, and missing evidence.

Treat long contextual intervals separately from localized review events, and do not assume that overlapping heuristic findings are independent confirmations.

Do not claim that the video is authentic, fake, AI-generated, manipulated, or tampered with. Do not invent a percentage or probability. A numeric probability must remain null unless a calibrated trained model result is explicitly provided in the evidence.

Return JSON matching schemas/ai_interpretation_response.schema.json. Keep numeric_probability null for v0.7 evidence bundles because external_model_results is empty.
"""


def _balanced_findings(event: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings = event.get("findings", [])
    domains_present = {finding["domain"] for finding in findings}
    caps_exceeded = any(
        sum(1 for finding in findings if finding["domain"] == domain) > cap
        for domain, cap in DOMAIN_FINDING_CAPS.items()
    )
    if len(findings) <= AI_INPUT_MAX_FINDINGS_PER_EVENT and (len(domains_present) <= 1 or not caps_exceeded):
        return findings, {
            "strategy": "domain_balanced",
            "available_finding_count": len(findings),
            "included_finding_count": len(findings),
            "omitted_finding_count": 0,
        }
    selected: list[dict[str, Any]] = []
    remaining = list(findings)
    domains = sorted({finding["domain"] for finding in findings})
    for domain in domains:
        domain_items = [finding for finding in remaining if finding["domain"] == domain]
        if domain_items:
            chosen = sorted(domain_items, key=_finding_rank_key)[0]
            selected.append(chosen)
            remaining.remove(chosen)
    while len(selected) < AI_INPUT_MAX_FINDINGS_PER_EVENT and remaining:
        domain_counts = {domain: sum(1 for finding in selected if finding["domain"] == domain) for domain in domains}
        eligible = [
            finding
            for finding in remaining
            if domain_counts.get(finding["domain"], 0) < DOMAIN_FINDING_CAPS.get(finding["domain"], AI_INPUT_MAX_FINDINGS_PER_EVENT)
            or len(domains) == 1
        ]
        if not eligible:
            break
        chosen = sorted(eligible, key=_finding_rank_key)[0]
        selected.append(chosen)
        remaining.remove(chosen)
    return selected, {
        "strategy": "domain_balanced",
        "available_finding_count": len(findings),
        "included_finding_count": len(selected),
        "omitted_finding_count": max(0, len(findings) - len(selected)),
    }


def _finding(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "domain": candidate["source_domain"],
        "candidate_role": candidate["candidate_role"],
        "type": candidate["source_type"],
        "summary": candidate["summary"],
        "metrics": candidate.get("metrics", {}),
        "source_record_id": candidate["source_record_id"],
        "source_observation_ids": candidate.get("source_observation_ids", []),
        "observation_reference_status": candidate.get("observation_reference_status"),
    }


def _context_finding(candidate: dict[str, Any]) -> dict[str, Any]:
    finding = _finding(candidate)
    finding["relationship"] = "contextual_reference"
    return finding


def _finding_rank_key(finding: dict[str, Any]) -> tuple[Any, ...]:
    role_rank = {"anchor_event": 0, "supporting_interval": 1, "contextual_interval": 2}
    ranked = 0 if "ranked" in finding.get("type", "") else 1
    return (role_rank.get(finding.get("candidate_role"), 9), ranked, finding.get("domain", ""), finding.get("source_record_id", ""))


def _best_event_relation(candidate: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    relations = [_event_relation(candidate, event) for event in events]
    relations = [relation for relation in relations if relation["overlap_seconds"] > 0 or relation["distance_seconds"] <= EVIDENCE_MERGE_TOLERANCE_SECONDS]
    if not relations:
        return None
    return sorted(
        relations,
        key=lambda item: (
            -item["overlap_seconds"],
            -item["overlap_ratio"],
            item["midpoint_distance_seconds"],
            item["event"]["start_timestamp_seconds"],
            item["event"]["event_id"],
        ),
    )[0]


def _contextual_event_relations(candidate: dict[str, Any], events: list[dict[str, Any]], exclude_event_id: str | None = None) -> list[dict[str, Any]]:
    output = []
    for event in events:
        if exclude_event_id and event["event_id"] == exclude_event_id:
            continue
        relation = _event_relation(candidate, event)
        if relation["overlap_seconds"] > 0 or relation["distance_seconds"] <= EVIDENCE_MERGE_TOLERANCE_SECONDS:
            output.append(relation)
    return output


def _event_relation(candidate: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    overlap = _overlap_seconds(candidate, event)
    duration = max(1e-9, _duration(candidate))
    distance = _interval_distance(candidate["start_timestamp_seconds"], candidate["end_timestamp_seconds"], event["start_timestamp_seconds"], event["end_timestamp_seconds"])
    relationship = "overlaps" if overlap > 0 else "within_tolerance"
    return {
        "event": event,
        "candidate_id": candidate["candidate_id"],
        "primary_event_id": event["event_id"],
        "relationship": relationship,
        "overlap_seconds": round(overlap, 6),
        "overlap_ratio": round(overlap / duration, 6),
        "distance_seconds": round(distance, 6),
        "midpoint_distance_seconds": round(abs(_midpoint(candidate) - _midpoint(event)), 6),
    }


def _context_interval_view(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "type": candidate["source_type"],
        "source_domain": candidate["source_domain"],
        "source_record_id": candidate["source_record_id"],
        "source_observation_ids": candidate.get("source_observation_ids", []),
        "observation_reference_status": candidate.get("observation_reference_status"),
        "start_timestamp_seconds": candidate["start_timestamp_seconds"],
        "end_timestamp_seconds": candidate["end_timestamp_seconds"],
        "duration_seconds": round(candidate["end_timestamp_seconds"] - candidate["start_timestamp_seconds"], 6),
        "related_event_ids": candidate.get("related_event_ids", []),
        "summary": candidate["summary"],
        "metrics": candidate.get("metrics", {}),
    }


def _temporal_metrics(transition: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(transition.get("metrics", {}))
    flow = transition.get("optical_flow", {})
    residual = transition.get("flow_warp_residual", {})
    notability = transition.get("notability", {})
    metrics.update(
        {
            "combined_percentile": notability.get("combined_percentile"),
            "mean_optical_flow": flow.get("mean_magnitude"),
            "flow_warp_residual_mean": residual.get("mean_normalized_residual"),
            "flow_warp_residual_95th_percentile": residual.get("percentile_95_normalized_residual"),
        }
    )
    return metrics


def _audio_metrics(transition: dict[str, Any]) -> dict[str, Any]:
    return {
        "energy_change_ratio": transition.get("energy_change_ratio", transition.get("rms_ratio")),
        "absolute_rms_difference": transition.get("absolute_rms_difference"),
        "rms_before": transition.get("rms_before"),
        "rms_after": transition.get("rms_after"),
        "transition_rank": transition.get("notability", {}).get("rank"),
    }


def _visual_consistency_metrics(transition: dict[str, Any]) -> dict[str, Any]:
    regional = transition.get("regional_summary", {})
    return {
        "combined_percentile": transition.get("combined_percentile"),
        "unstable_region_count": regional.get("unstable_region_count"),
        "maximum_edge_instability": regional.get("maximum_edge_instability"),
        "maximum_texture_distance": regional.get("maximum_texture_distance"),
        "maximum_detail_residual": regional.get("maximum_regional_detail_residual"),
    }


def _artifacts_from_mapping(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for label, artifact in sorted(artifacts.items()):
        if isinstance(artifact, dict) and artifact.get("path"):
            output.append({"label": label, **artifact})
    return output


def _unique_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for artifact in artifacts:
        path = artifact.get("path")
        if not path or path in seen:
            continue
        seen.add(path)
        output.append(artifact)
    return output


def _unique_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        output.append(candidate)
    return output


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["start_timestamp_seconds"],
        candidate["end_timestamp_seconds"],
        candidate["source_domain"],
        candidate["source_record_id"],
    )


def _event_percentile(event: dict[str, Any]) -> float:
    values = []
    for finding in event.get("findings", []):
        metrics = finding.get("metrics", {})
        if metrics.get("combined_percentile") is not None:
            values.append(float(metrics["combined_percentile"]))
    return max(values) if values else 0.0


def _compact_metrics(value: Any, depth: int = 0) -> Any:
    if depth > 2:
        return None
    if isinstance(value, dict):
        output = {}
        for key, child in value.items():
            compact = _compact_metrics(child, depth + 1)
            if compact is not None:
                output[str(key)] = compact
        return output
    if isinstance(value, list):
        return value[:8]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _role_count(candidates: list[dict[str, Any]], role: str) -> int:
    return sum(1 for candidate in candidates if candidate.get("candidate_role") == role)


def _overlap_seconds(first: dict[str, Any], second: dict[str, Any]) -> float:
    return max(0.0, min(first["end_timestamp_seconds"], second["end_timestamp_seconds"]) - max(first["start_timestamp_seconds"], second["start_timestamp_seconds"]))


def _duration(item: dict[str, Any]) -> float:
    return max(0.0, item["end_timestamp_seconds"] - item["start_timestamp_seconds"])


def _midpoint(item: dict[str, Any]) -> float:
    return (item["start_timestamp_seconds"] + item["end_timestamp_seconds"]) / 2.0


def _intervals_close(first_start: float, first_end: float, second_start: float, second_end: float) -> bool:
    return _interval_distance(first_start, first_end, second_start, second_end) <= EVIDENCE_MERGE_TOLERANCE_SECONDS


def _interval_distance(first_start: float, first_end: float, second_start: float, second_end: float) -> float:
    if first_end < second_start:
        return second_start - first_end
    if second_end < first_start:
        return first_start - second_end
    return 0.0


def _max_metric(metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [item.get(key) for item in metrics if isinstance(item.get(key), (int, float))]
    return round(max(values), 6) if values else None


def _avg_metric(metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [item.get(key) for item in metrics if isinstance(item.get(key), (int, float))]
    return round(sum(values) / len(values), 6) if values else None


def _dedupe_strings(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        normalized = " ".join(str(item).lower().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(item)
    return output


def _is_absolute_path(path: str) -> bool:
    return path.startswith("/") or (len(path) > 2 and path[1] == ":")
