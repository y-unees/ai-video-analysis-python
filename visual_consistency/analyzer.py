from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from config import visual_consistency_configuration
from file_utils import format_file_size
from visual_consistency.artifact_writer import write_review_artifacts
from visual_consistency.grid import build_region_grid
from visual_consistency.interval_detection import detect_sustained_intervals
from visual_consistency.region_metrics import (
    calculate_transition_region_records,
    transition_work_images,
)
from visual_consistency.transition_ranking import RANKING_METRICS, rank_review_transitions


LIMITATIONS = [
    "Visual consistency measurements are local review metrics, not authenticity or manipulation verdicts.",
    "Measurements may be affected by camera movement, zoom, stabilization, motion blur, compression, lighting changes, focus changes, occlusion, object movement, shadows, animation, and ordinary editing.",
    "Motion compensation is approximate and may be less reliable in high-motion or low-texture regions.",
]


def analyze_visual_consistency(
    analysis_dir: Path,
    temporal_analysis: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    started = perf_counter()
    configuration = visual_consistency_configuration()
    runtime = temporal_analysis.get("_runtime", {})
    samples = runtime.get("samples") or []
    transitions = runtime.get("transitions") or []
    if temporal_analysis.get("status") not in {"completed", "partial"} or len(samples) < 2:
        return _skipped_result(configuration, "insufficient_temporal_frames"), warnings

    try:
        first_gray = samples[0]["gray"]
        regions = build_region_grid(
            first_gray.shape[1],
            first_gray.shape[0],
            int(configuration["grid_rows"]),
            int(configuration["grid_columns"]),
        )
        transition_records: dict[str, list[dict[str, Any]]] = {}
        transition_images: dict[str, dict[str, Any]] = {}
        all_records: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []

        artifact = _write_metrics_jsonl(
            analysis_dir,
            samples,
            transitions,
            regions,
            transition_records,
            transition_images,
            all_records,
            summaries,
        )
        sustained = detect_sustained_intervals(all_records)
        ranked = rank_review_transitions(summaries)
        artifacts, artifact_warnings = write_review_artifacts(
            analysis_dir / "consistency_frames",
            ranked,
            transition_images,
            transition_records,
            regions,
        )
        warnings.extend(artifact_warnings)
        for ranked_transition in ranked:
            artifact_record = artifacts.get(ranked_transition["transition_id"], {})
            ranked_transition["artifact_status"] = artifact_record.get("status", "missing")
            ranked_transition["artifacts"] = artifact_record.get("artifacts", {})
        duration = round(perf_counter() - started, 3)
        result = _base_result("completed", configuration)
        result.update(
            {
                "reason_code": None,
                "reason": None,
                "summary": {
                    "frames_used": len(samples),
                    "transitions_analyzed": len(transitions),
                    "grid_rows": configuration["grid_rows"],
                    "grid_columns": configuration["grid_columns"],
                    "region_count_per_transition": len(regions),
                    "consistency_record_count": len(all_records),
                    "sustained_interval_count": len(sustained),
                    "ranked_review_transition_count": len(ranked),
                    "motion_compensation_used": True,
                    "metrics_available": True,
                    "processing_duration_seconds": duration,
                },
                "transition_summaries": summaries,
                "sustained_intervals": sustained,
                "ranked_review_transitions": ranked,
                "observations": _observations(sustained, ranked),
                "limitations": LIMITATIONS,
                "artifacts": {
                    "visual_consistency_metrics_artifact": artifact,
                    "consistency_frames_directory": "consistency_frames",
                    "ranked_transition_artifacts": artifacts,
                },
            }
        )
        return result, warnings
    except Exception as error:
        warnings.append(f"Visual consistency analysis failed: {error}")
        result = _base_result("failed", configuration)
        result.update(
            {
                "reason_code": "visual_consistency_exception",
                "reason": str(error),
                "summary": {"metrics_available": False},
                "observations": [
                    _observation(
                        "visual-consistency-failed-001",
                        "visual_consistency.analysis_failed",
                        "Visual consistency analysis failed before metrics were completed.",
                        "Other analysis stages may still be available.",
                    )
                ],
                "limitations": LIMITATIONS,
            }
        )
        return result, warnings


def _write_metrics_jsonl(
    analysis_dir: Path,
    samples: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    transition_records: dict[str, list[dict[str, Any]]],
    transition_images: dict[str, dict[str, Any]],
    all_records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_by_index = {sample["sample"]["temporal_sample_index"]: sample for sample in samples}
    path = analysis_dir / "visual_consistency_metrics.jsonl"
    sha256 = hashlib.sha256()
    size_bytes = 0
    with path.open("wb") as file:
        for transition in transitions:
            previous = sample_by_index[transition["from_sample_index"]]
            current = sample_by_index[transition["to_sample_index"]]
            records = calculate_transition_region_records(
                previous["gray"],
                current["gray"],
                transition,
                regions,
            )
            transition_records[transition["transition_id"]] = records
            transition_images[transition["transition_id"]] = transition_work_images(
                previous["gray"],
                current["gray"],
            )
            all_records.extend(records)
            summaries.append(_transition_summary(transition, records, len(regions)))
            for record in records:
                line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
                file.write(line)
                sha256.update(line)
                size_bytes += len(line)
    return {
        "path": "visual_consistency_metrics.jsonl",
        "size_bytes": size_bytes,
        "size_human_readable": format_file_size(size_bytes),
        "sha256": sha256.hexdigest(),
    }


def _transition_summary(
    transition: dict[str, Any],
    records: list[dict[str, Any]],
    region_count: int,
) -> dict[str, Any]:
    edge = [record["ranking_data"]["edge_instability"] for record in records]
    texture = [record["ranking_data"]["texture_distance"] for record in records]
    detail = [record["ranking_data"]["detail_residual"] for record in records]
    brightness = [record["ranking_data"]["brightness_normalized_difference"] for record in records]
    unstable_count = sum(1 for record in records if record["classification"]["regional_visual_variation"])
    high_motion_count = sum(
        1
        for record in records
        if record["motion_context"]["classification"] == "high_motion_region"
    )
    regional = {
        "valid_region_count": len(records),
        "total_region_count": region_count,
        "unstable_region_count": unstable_count,
        "high_motion_region_count": high_motion_count,
        "average_edge_instability": _average(edge),
        "maximum_edge_instability": max(edge) if edge else 0.0,
        "average_texture_distance": _average(texture),
        "maximum_texture_distance": max(texture) if texture else 0.0,
        "average_regional_detail_residual": _average(detail),
        "maximum_regional_detail_residual": max(detail) if detail else 0.0,
        "average_regional_brightness_change": _average(brightness),
        "maximum_regional_brightness_change": max(brightness) if brightness else 0.0,
        "regional_brightness_change_concentration": max(brightness) - _average(brightness) if brightness else 0.0,
    }
    return {
        "transition_id": transition["transition_id"],
        "from_sample_index": transition["from_sample_index"],
        "to_sample_index": transition["to_sample_index"],
        "start_timestamp_seconds": transition["start_timestamp_seconds"],
        "end_timestamp_seconds": transition["end_timestamp_seconds"],
        "scene_index": transition.get("scene_index"),
        "regional_summary": {key: round(value, 6) if isinstance(value, float) else value for key, value in regional.items()},
    }


def _observations(
    intervals: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observations = [
        _observation(
            "visual-consistency-completed-001",
            "visual_consistency.analysis_completed",
            "Visual consistency analysis completed and produced regional review measurements.",
            "These measurements are non-conclusive observations for human review.",
        )
    ]
    for index, interval in enumerate(intervals, start=1):
        observations.append(
            _localized_observation(
                f"visual-consistency-interval-{index:03d}",
                "visual_consistency.sustained_regional_variation",
                interval["start_timestamp_seconds"],
                interval["end_timestamp_seconds"],
                "Repeated regional visual variation was measured across consecutive samples.",
                interval["interpretation"],
            )
        )
    for index, transition in enumerate(ranked, start=1):
        observations.append(
            _localized_observation(
                f"visual-consistency-ranked-{index:03d}",
                "visual_consistency.ranked_review_transition",
                transition["start_timestamp_seconds"],
                transition["end_timestamp_seconds"],
                "A transition ranked highly within this video for regional visual consistency measurements.",
                transition["interpretation"],
            )
        )
    return observations


def _base_result(status: str, configuration: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": None,
        "reason": None,
        "configuration": configuration,
        "summary": {},
        "transition_summaries": [],
        "sustained_intervals": [],
        "ranked_review_transitions": [],
        "observations": [],
        "limitations": [],
        "artifacts": {},
    }


def _skipped_result(configuration: dict[str, Any], reason_code: str) -> dict[str, Any]:
    result = _base_result("skipped", configuration)
    result["reason_code"] = reason_code
    result["reason"] = "Visual consistency analysis requires at least two temporal samples."
    result["summary"] = {"metrics_available": False}
    result["limitations"] = LIMITATIONS
    result["observations"] = [
        _observation(
            "visual-consistency-skipped-001",
            "visual_consistency.partial_analysis",
            "Visual consistency analysis was skipped because temporal samples were unavailable.",
            "This is a pipeline limitation for this run and not evidence about the source video.",
        )
    ]
    return result


def _observation(observation_id: str, observation_type: str, description: str, interpretation: str) -> dict[str, Any]:
    return {
        "observation_id": observation_id,
        "type": observation_type,
        "severity": "info",
        "conclusion_scope": "non_conclusive_observation",
        "supports_authenticity_verdict": False,
        "description": description,
        "interpretation": interpretation,
    }


def _localized_observation(
    observation_id: str,
    observation_type: str,
    start: float,
    end: float,
    description: str,
    interpretation: str,
) -> dict[str, Any]:
    observation = _observation(observation_id, observation_type, description, interpretation)
    observation["timestamp_start"] = start
    observation["timestamp_end"] = end
    return observation


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0
