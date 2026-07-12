from __future__ import annotations

from typing import Any

from config import STATIC_MINIMUM_DURATION_SECONDS


def detect_static_intervals(transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    for transition in transitions:
        if transition["classification"]["near_static_transition"]:
            current.append(transition)
        else:
            _append_interval(intervals, current)
            current = []
    _append_interval(intervals, current)
    return intervals


def _append_interval(
    intervals: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
) -> None:
    if not transitions:
        return
    start = transitions[0]["start_timestamp_seconds"]
    end = transitions[-1]["end_timestamp_seconds"]
    duration = round(end - start, 6)
    if duration < STATIC_MINIMUM_DURATION_SECONDS:
        return
    stationary_values = [
        transition["optical_flow"]["stationary_pixel_ratio"]
        for transition in transitions
        if transition["optical_flow"]["stationary_pixel_ratio"] is not None
    ]
    intervals.append(
        {
            "interval_type": "sustained_near_static_visual_interval",
            "start_timestamp_seconds": start,
            "end_timestamp_seconds": end,
            "duration_seconds": duration,
            "transition_count": len(transitions),
            "metrics": {
                "maximum_perceptual_hash_distance": max(
                    transition["perceptual_hash_distance"] for transition in transitions
                ),
                "maximum_normalized_mean_absolute_difference": max(
                    transition["normalized_mean_absolute_difference"] for transition in transitions
                ),
                "average_stationary_pixel_ratio": _average(stationary_values),
            },
        }
    )


def _average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None
