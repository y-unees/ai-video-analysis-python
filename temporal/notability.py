from __future__ import annotations

from typing import Any

from config import MAX_NOTABLE_TRANSITIONS, NOTABLE_TRANSITION_MINIMUM_METRIC_REASONS


METRICS = (
    ("normalized_mean_absolute_difference", "highest_normalized_pixel_difference", "high_normalized_pixel_difference", True),
    ("perceptual_hash_distance", "highest_perceptual_hash_distance", "high_perceptual_hash_distance", True),
    ("absolute_brightness_difference", "highest_brightness_difference", "high_brightness_difference", True),
    ("optical_flow.mean_magnitude", "highest_mean_optical_flow", "high_mean_optical_flow", True),
    ("optical_flow.percentile_95_magnitude", "highest_95th_percentile_optical_flow", "high_95th_percentile_optical_flow", True),
    ("histogram_correlation", "lowest_histogram_correlation", "low_histogram_correlation", False),
    ("flow_warp_residual.mean_normalized_residual", "highest_flow_warp_residual", "high_flow_warp_residual", True),
    ("flow_warp_residual.percentile_95_normalized_residual", "highest_95th_percentile_flow_warp_residual", "high_95th_percentile_flow_warp_residual", True),
)


def rank_notable_transitions(transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not transitions:
        return []

    buckets: dict[str, dict[str, Any]] = {
        transition["transition_id"]: {
            "transition": transition,
            "reasons": [],
            "metric_ranks": {},
            "percentiles": [],
        }
        for transition in transitions
    }
    metric_count = max(1, len(transitions))
    top_k = min(MAX_NOTABLE_TRANSITIONS, metric_count)

    for metric_path, highest_reason, high_reason, descending in METRICS:
        ranked = _rank_metric(transitions, metric_path, descending)
        ranked_ids = {transition["transition_id"] for transition, _value in ranked}
        for transition in transitions:
            if transition["transition_id"] not in ranked_ids:
                buckets[transition["transition_id"]]["metric_ranks"][metric_path] = {
                    "rank": None,
                    "value": None,
                    "percentile": None,
                    "available": False,
                    "sort": "descending" if descending else "ascending",
                }
        if not ranked:
            continue
        for rank_index, (transition, value) in enumerate(ranked, start=1):
            percentile = _percentile(rank_index, len(ranked))
            bucket = buckets[transition["transition_id"]]
            bucket["metric_ranks"][metric_path] = {
                "rank": rank_index,
                "value": value,
                "percentile": percentile,
                "available": True,
                "sort": "descending" if descending else "ascending",
            }
            bucket["percentiles"].append(percentile)
        for rank_index, (transition, value) in enumerate(ranked[:top_k], start=1):
            bucket = buckets[transition["transition_id"]]
            reason = highest_reason if rank_index == 1 else high_reason
            if reason not in bucket["reasons"]:
                bucket["reasons"].append(reason)

    ranked_buckets = [
        bucket
        for bucket in buckets.values()
        if len(bucket["reasons"]) >= NOTABLE_TRANSITION_MINIMUM_METRIC_REASONS
    ]
    ranked_buckets.sort(
        key=lambda bucket: (
            -_average(bucket["percentiles"]),
            -len(bucket["reasons"]),
            bucket["transition"]["from_sample_index"],
            bucket["transition"]["transition_id"],
        )
    )

    notable: list[dict[str, Any]] = []
    for notable_index, bucket in enumerate(ranked_buckets[:MAX_NOTABLE_TRANSITIONS]):
        transition = bucket["transition"]
        transition["classification"]["ranked_notable_transition"] = True
        transition["notability"] = {
            "reason_count": len(bucket["reasons"]),
            "reasons": sorted(bucket["reasons"]),
            "combined_percentile": round(_average(bucket["percentiles"]), 6),
            "combined_percentile_metric_count": len(METRICS),
            "available_metric_count": len(bucket["percentiles"]),
            "selection_basis": "relative_within_video",
            "absolute_significance_assessed": False,
            "metric_ranks": bucket["metric_ranks"],
        }
        transition["notable_transition_index"] = notable_index
        notable.append(transition)

    notable_ids = {transition["transition_id"] for transition in notable}
    for transition in transitions:
        transition["classification"].setdefault("ranked_notable_transition", False)
        transition.setdefault(
            "notability",
            {
                "reason_count": 0,
                "reasons": [],
                "combined_percentile": None,
                "combined_percentile_metric_count": len(METRICS),
                "available_metric_count": 0,
                "selection_basis": "relative_within_video",
                "absolute_significance_assessed": False,
                "metric_ranks": {},
            },
        )
        if transition["transition_id"] not in notable_ids:
            transition["classification"]["ranked_notable_transition"] = False
    return notable


def _rank_metric(
    transitions: list[dict[str, Any]],
    metric_path: str,
    descending: bool,
) -> list[tuple[dict[str, Any], float]]:
    values: list[tuple[dict[str, Any], float]] = []
    for transition in transitions:
        value = _get_path(transition, metric_path)
        if isinstance(value, (int, float)):
            values.append((transition, float(value)))
    values.sort(
        key=lambda item: (
            -item[1] if descending else item[1],
            item[0]["from_sample_index"],
        )
    )
    return values


def _get_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _percentile(rank_index: int, count: int) -> float:
    if count <= 1:
        return 1.0
    return round(1.0 - ((rank_index - 1) / (count - 1)), 6)


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
