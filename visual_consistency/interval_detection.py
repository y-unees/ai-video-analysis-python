from __future__ import annotations

from typing import Any

from config import (
    VISUAL_CONSISTENCY_MINIMUM_INTERVAL_DURATION_SECONDS,
    VISUAL_CONSISTENCY_MINIMUM_INTERVAL_TRANSITIONS,
)


def detect_sustained_intervals(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_region: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_region.setdefault(record["region"]["region_id"], []).append(record)

    intervals: list[dict[str, Any]] = []
    for region_id, region_records in sorted(by_region.items()):
        run: list[dict[str, Any]] = []
        for record in sorted(region_records, key=lambda item: item["from_sample_index"]):
            if record["classification"]["regional_visual_variation"]:
                if run and record["from_sample_index"] != run[-1]["to_sample_index"]:
                    _append_interval(intervals, region_id, run)
                    run = []
                run.append(record)
            elif run:
                _append_interval(intervals, region_id, run)
                run = []
        if run:
            _append_interval(intervals, region_id, run)

    for index, interval in enumerate(intervals, start=1):
        interval["interval_id"] = f"visual-consistency-interval-{index:03d}"
    return intervals


def _append_interval(
    intervals: list[dict[str, Any]],
    region_id: str,
    run: list[dict[str, Any]],
) -> None:
    if len(run) < VISUAL_CONSISTENCY_MINIMUM_INTERVAL_TRANSITIONS:
        return
    duration = run[-1]["end_timestamp_seconds"] - run[0]["start_timestamp_seconds"]
    if duration < VISUAL_CONSISTENCY_MINIMUM_INTERVAL_DURATION_SECONDS:
        return
    intervals.append(
        {
            "interval_id": "",
            "interval_type": "sustained_regional_visual_instability",
            "start_timestamp_seconds": run[0]["start_timestamp_seconds"],
            "end_timestamp_seconds": run[-1]["end_timestamp_seconds"],
            "duration_seconds": round(duration, 6),
            "transition_count": len(run),
            "transition_ids": [record["transition_id"] for record in run],
            "affected_regions": [region_id],
            "supporting_metrics": {
                "average_edge_instability": _average(
                    [record["ranking_data"]["edge_instability"] for record in run]
                ),
                "average_texture_distance": _average(
                    [record["ranking_data"]["texture_distance"] for record in run]
                ),
                "average_detail_residual": _average(
                    [record["ranking_data"]["detail_residual"] for record in run]
                ),
            },
            "interpretation": "Repeated regional visual variation was measured across consecutive samples. Possible causes include movement, occlusion, focus change, compression, lighting, animation, local deformation, or generated instability.",
        }
    )


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0
