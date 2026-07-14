from __future__ import annotations

from typing import Any

from config import VISUAL_CONSISTENCY_MAX_RANKED_INTERVALS


RANKING_METRICS = [
    ("maximum_regional_detail_residual", True),
    ("average_regional_detail_residual", True),
    ("maximum_edge_instability", True),
    ("average_edge_instability", True),
    ("maximum_texture_distance", True),
    ("unstable_region_count", True),
    ("regional_brightness_change_concentration", True),
]


def rank_review_transitions(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_ranks = _metric_ranks(summaries)
    ranked: list[dict[str, Any]] = []
    for summary in summaries:
        ranks = metric_ranks.get(summary["transition_id"], {})
        percentiles = [item["percentile"] for item in ranks.values()]
        combined = sum(percentiles) / len(RANKING_METRICS) if percentiles else None
        ranked.append(
            {
                "ranked_review_index": 0,
                "transition_id": summary["transition_id"],
                "from_sample_index": summary["from_sample_index"],
                "to_sample_index": summary["to_sample_index"],
                "start_timestamp_seconds": summary["start_timestamp_seconds"],
                "end_timestamp_seconds": summary["end_timestamp_seconds"],
                "scene_index": summary.get("scene_index"),
                "selection_basis": "relative_within_video",
                "absolute_significance_assessed": False,
                "combined_percentile_metric_count": len(RANKING_METRICS),
                "available_metric_count": len(ranks),
                "combined_percentile": round(combined, 6) if combined is not None else None,
                "metric_ranks": ranks,
                "regional_summary": summary["regional_summary"],
                "ranking_reasons": _ranking_reasons(summary),
                "interpretation": "This transition ranked highly within this video across visual consistency measurements. It is not an objective suspicion score or proof of manipulation.",
            }
        )
    ranked.sort(
        key=lambda item: (
            -(item["combined_percentile"] or 0.0),
            item["start_timestamp_seconds"],
            item["transition_id"],
        )
    )
    for index, item in enumerate(ranked[:VISUAL_CONSISTENCY_MAX_RANKED_INTERVALS], start=1):
        item["ranked_review_index"] = index
    return ranked[:VISUAL_CONSISTENCY_MAX_RANKED_INTERVALS]


def _metric_ranks(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {summary["transition_id"]: {} for summary in summaries}
    total = max(1, len(summaries) - 1)
    for metric_name, descending in RANKING_METRICS:
        ordered = sorted(
            summaries,
            key=lambda item: (
                item["regional_summary"].get(metric_name, 0),
                -item["start_timestamp_seconds"],
                item["transition_id"],
            ),
            reverse=descending,
        )
        for rank, summary in enumerate(ordered, start=1):
            percentile = 1.0 - ((rank - 1) / total) if len(summaries) > 1 else 1.0
            output[summary["transition_id"]][metric_name] = {
                "rank": rank,
                "value": summary["regional_summary"].get(metric_name, 0),
                "percentile": round(percentile, 6),
                "direction": "descending" if descending else "ascending",
            }
    return output


def _ranking_reasons(summary: dict[str, Any]) -> list[str]:
    regional = summary["regional_summary"]
    candidates = [
        ("maximum_regional_detail_residual", "maximum regional detail residual"),
        ("maximum_edge_instability", "maximum edge instability"),
        ("maximum_texture_distance", "maximum texture distance"),
        ("unstable_region_count", "unstable region count"),
    ]
    return [
        label
        for key, label in candidates
        if regional.get(key, 0) > 0
    ] or ["relative visual consistency measurements"]
