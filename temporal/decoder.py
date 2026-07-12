from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from config import (
    TEMPORAL_MAX_ANALYZED_FRAMES,
    TEMPORAL_REQUESTED_ANALYSIS_FPS,
    TEMPORAL_RESIZE_MAX_WIDTH,
)


def decode_temporal_samples(
    video_path: Path,
    duration_seconds: float | None,
    source_fps: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    import cv2

    warnings: list[str] = []
    effective_fps = _effective_fps(duration_seconds)
    interval_seconds = 1.0 / effective_fps if effective_fps > 0 else 1.0
    configuration = {
        "requested_analysis_fps": TEMPORAL_REQUESTED_ANALYSIS_FPS,
        "effective_analysis_fps": round(effective_fps, 6),
        "maximum_analyzed_frames": TEMPORAL_MAX_ANALYZED_FRAMES,
        "resize_max_width": TEMPORAL_RESIZE_MAX_WIDTH,
    }
    if effective_fps < TEMPORAL_REQUESTED_ANALYSIS_FPS:
        warnings.append("Temporal analysis FPS was reduced to respect the configured frame cap.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], configuration, ["OpenCV could not open the selected video for temporal analysis."]

    samples: list[dict[str, Any]] = []
    next_timestamp = 0.0
    source_frame_index = -1
    last_timestamp: float | None = None

    try:
        while len(samples) < TEMPORAL_MAX_ANALYZED_FRAMES:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            source_frame_index += 1
            timestamp, timestamp_source = _timestamp_for_frame(
                capture,
                source_frame_index,
                source_fps,
                last_timestamp,
            )
            if timestamp is None:
                warnings.append("A temporal frame had no finite timestamp and was skipped.")
                continue
            last_timestamp = timestamp

            if timestamp + 1e-9 < next_timestamp and samples:
                continue
            gray = _resize_to_gray(frame)
            height, width = gray.shape[:2]
            samples.append(
                {
                    "sample": {
                        "temporal_sample_index": len(samples),
                        "source_frame_index": source_frame_index,
                        "timestamp_seconds": round(timestamp, 6),
                        "timestamp_source": timestamp_source,
                        "width": int(width),
                        "height": int(height),
                    },
                    "gray": gray,
                }
            )
            next_timestamp = timestamp + interval_seconds
    finally:
        capture.release()

    return samples, configuration, warnings


def _effective_fps(duration_seconds: float | None) -> float:
    if duration_seconds is None or duration_seconds <= 0:
        return TEMPORAL_REQUESTED_ANALYSIS_FPS
    capped_fps = TEMPORAL_MAX_ANALYZED_FRAMES / duration_seconds
    return max(0.001, min(TEMPORAL_REQUESTED_ANALYSIS_FPS, capped_fps))


def _timestamp_for_frame(
    capture: Any,
    source_frame_index: int,
    source_fps: float | None,
    last_timestamp: float | None,
) -> tuple[float | None, str]:
    import cv2

    position_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
    if math.isfinite(position_msec) and position_msec >= 0:
        timestamp = position_msec / 1000.0
        if last_timestamp is None or timestamp >= last_timestamp:
            return timestamp, "opencv_position_msec"
    if source_fps is not None and source_fps > 0:
        return source_frame_index / source_fps, "frame_index_divided_by_source_fps"
    if last_timestamp is not None:
        return last_timestamp, "previous_timestamp_fallback"
    return 0.0, "zero_timestamp_fallback"


def _resize_to_gray(frame: Any) -> Any:
    import cv2

    height, width = frame.shape[:2]
    if width > TEMPORAL_RESIZE_MAX_WIDTH:
        scale = TEMPORAL_RESIZE_MAX_WIDTH / width
        frame = cv2.resize(
            frame,
            (TEMPORAL_RESIZE_MAX_WIDTH, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
