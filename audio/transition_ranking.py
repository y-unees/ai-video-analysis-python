from __future__ import annotations

from typing import Any

from config import AUDIO_ENERGY_CHANGE_RATIO_THRESHOLD, AUDIO_MAX_REVIEW_TRANSITIONS


def rank_audio_transitions(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for index in range(len(windows) - 1):
        before = windows[index]
        after = windows[index + 1]
        before_rms = before["rms_amplitude"]
        after_rms = after["rms_amplitude"]
        ratio = _ratio(before_rms, after_rms)
        transition = {
            "transition_id": f"audio-transition-{index:05d}-{index + 1:05d}",
            "from_window_index": before["window_index"],
            "to_window_index": after["window_index"],
            "start_timestamp_seconds": before["start_timestamp_seconds"],
            "end_timestamp_seconds": after["start_timestamp_seconds"],
            "rms_before": before_rms,
            "rms_after": after_rms,
            "rms_ratio": ratio,
            "energy_change_ratio": ratio,
            "rms_increase_ratio": round(after_rms / max(before_rms, 1e-9), 6),
            "rms_decrease_ratio": round(before_rms / max(after_rms, 1e-9), 6),
            "absolute_rms_difference": round(abs(after_rms - before_rms), 6),
            "peak_difference": round(abs(after["peak_absolute_amplitude"] - before["peak_absolute_amplitude"]), 6),
            "spectral_centroid_difference_hz": _difference(after.get("spectral_centroid_hz"), before.get("spectral_centroid_hz")),
            "ranked_review_transition": False,
            "selection_basis": "relative_within_audio",
            "absolute_significance_assessed": False,
            "notability": {"rank": None, "reason": None},
            "interpretation": "This audio transition ranked highly in energy change relative to this file. It may reflect speech onset, music, sound effects, a cut, microphone changes, compression, or other normal audio behavior.",
        }
        transitions.append(transition)

    ranked = sorted(
        transitions,
        key=lambda item: (-item["rms_ratio"], -item["absolute_rms_difference"], -item["peak_difference"], item["start_timestamp_seconds"], item["transition_id"]),
    )
    for rank, transition in enumerate(ranked[:AUDIO_MAX_REVIEW_TRANSITIONS], start=1):
        if (
            transition["rms_ratio"] >= AUDIO_ENERGY_CHANGE_RATIO_THRESHOLD
            or transition["absolute_rms_difference"] > 0
        ):
            transition["ranked_review_transition"] = True
            transition["notability"] = {
                "rank": rank,
                "reason": "highest_absolute_rms_difference" if rank == 1 else "high_absolute_rms_difference",
            }
    return [transition for transition in transitions if transition["ranked_review_transition"]]


def _ratio(first: float, second: float) -> float:
    smaller = max(min(abs(first), abs(second)), 1e-9)
    larger = max(abs(first), abs(second))
    return round(larger / smaller, 6)


def _difference(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return round(abs(first - second), 6)
