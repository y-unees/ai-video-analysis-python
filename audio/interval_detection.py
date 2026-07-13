from __future__ import annotations

from typing import Any

from config import AUDIO_MINIMUM_SILENCE_INTERVAL_SECONDS


def detect_silence_intervals(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for window in windows:
        if window["silence_like"]:
            current.append(window)
        else:
            _append_interval(intervals, current)
            current = []
    _append_interval(intervals, current)
    return intervals


def _append_interval(intervals: list[dict[str, Any]], windows: list[dict[str, Any]]) -> None:
    if not windows:
        return
    start = windows[0]["start_timestamp_seconds"]
    end = windows[-1]["end_timestamp_seconds"]
    duration = round(end - start, 6)
    if duration < AUDIO_MINIMUM_SILENCE_INTERVAL_SECONDS:
        return
    intervals.append(
        {
            "interval_id": f"audio-silence-{len(intervals) + 1:03d}",
            "interval_type": "silence_like_interval",
            "start_timestamp_seconds": start,
            "end_timestamp_seconds": end,
            "duration_seconds": duration,
            "window_count": len(windows),
            "average_rms_amplitude": round(
                sum(window["rms_amplitude"] for window in windows) / len(windows),
                6,
            ),
        }
    )
