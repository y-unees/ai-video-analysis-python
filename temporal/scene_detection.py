from __future__ import annotations

from typing import Any


def construct_scenes(
    samples: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not samples:
        return []

    scene_start_sample = samples[0]["sample"]
    scenes: list[dict[str, Any]] = []
    scene_transitions: list[dict[str, Any]] = []
    boundary_reason: dict[str, Any] | None = None

    for transition in transitions:
        if transition["classification"]["scene_boundary_candidate"]:
            scenes.append(
                _build_scene(
                    len(scenes),
                    scene_start_sample,
                    transition["end_timestamp_seconds"],
                    scene_transitions,
                    boundary_reason,
                )
            )
            scene_start_sample = _sample_by_index(samples, transition["to_sample_index"])
            boundary_reason = {
                "transition": {
                    "from_sample_index": transition["from_sample_index"],
                    "to_sample_index": transition["to_sample_index"],
                },
                "metrics": {
                    "normalized_mean_absolute_difference": transition["normalized_mean_absolute_difference"],
                    "histogram_correlation": transition["histogram_correlation"],
                    "perceptual_hash_distance": transition["perceptual_hash_distance"],
                },
                "triggered_rules": transition["classification"]["triggered_rules"],
            }
            scene_transitions = []
        else:
            scene_transitions.append(transition)

    scenes.append(
        _build_scene(
            len(scenes),
            scene_start_sample,
            samples[-1]["sample"]["timestamp_seconds"],
            scene_transitions,
            boundary_reason,
        )
    )
    return scenes


def _build_scene(
    scene_index: int,
    start_sample: dict[str, Any],
    end_timestamp: float,
    transitions: list[dict[str, Any]],
    boundary_reason: dict[str, Any] | None,
) -> dict[str, Any]:
    start_timestamp = start_sample["timestamp_seconds"]
    end_timestamp = max(start_timestamp, end_timestamp)
    flow_means = [
        transition["optical_flow"]["mean_magnitude"]
        for transition in transitions
        if transition["optical_flow"]["mean_magnitude"] is not None
    ]
    stationary = [
        transition["optical_flow"]["stationary_pixel_ratio"]
        for transition in transitions
        if transition["optical_flow"]["stationary_pixel_ratio"] is not None
    ]
    return {
        "scene_index": scene_index,
        "start_timestamp_seconds": round(start_timestamp, 6),
        "end_timestamp_seconds": round(end_timestamp, 6),
        "duration_seconds": round(end_timestamp - start_timestamp, 6),
        "temporal_sample_count": len(transitions) + 1,
        "boundary_reason": boundary_reason,
        "motion_summary": {
            "average_flow_magnitude": _average(flow_means),
            "median_flow_magnitude": _median(flow_means),
            "maximum_flow_magnitude": max(flow_means) if flow_means else None,
            "average_stationary_pixel_ratio": _average(stationary),
        },
    }


def _sample_by_index(samples: list[dict[str, Any]], sample_index: int) -> dict[str, Any]:
    for sample in samples:
        if sample["sample"]["temporal_sample_index"] == sample_index:
            return sample["sample"]
    return samples[-1]["sample"]


def _average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return round((ordered[middle - 1] + ordered[middle]) / 2, 3)
