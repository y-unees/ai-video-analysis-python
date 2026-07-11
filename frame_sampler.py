from __future__ import annotations

from pathlib import Path
from typing import Any

from config import (
    END_SEEK_SAFETY_SECONDS,
    JPEG_QUALITY,
    LONG_VIDEO_SAMPLE_COUNT,
    SEEK_ERROR_WARNING_SECONDS,
    SHORT_VIDEO_MAX_SECONDS,
    SHORT_VIDEO_SAMPLE_COUNT,
)
from frame_analyzer import analyze_frame, compare_frames, summarize_frame_analysis


def sample_video_frames(
    video_path: Path,
    frames_dir: Path,
    duration_seconds: float | None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    import cv2
    import numpy as np

    warnings: list[str] = []
    target_count = _target_sample_count(duration_seconds)
    requested_timestamps = _build_timestamps(duration_seconds, target_count)
    frames_dir.mkdir(parents=True, exist_ok=True)

    sampling = {
        "strategy": "uniform",
        "target_frame_count": target_count,
        "requested_timestamps_seconds": requested_timestamps,
        "decoded_frame_count": 0,
        "failed_timestamps_seconds": [],
        "maximum_absolute_seek_error_seconds": None,
        "average_absolute_seek_error_seconds": None,
        "repeated_decoded_timestamps_seconds": [],
    }
    frame_analysis = {
        "summary": summarize_frame_analysis([], []),
        "frames": [],
        "comparisons": [],
    }

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        warnings.append("OpenCV could not open or decode the selected video.")
        return sampling, frame_analysis, warnings

    raw_frames: list[Any] = []
    try:
        for sample_index, requested_timestamp in enumerate(requested_timestamps):
            ok, frame, decoded_timestamp = _read_frame_near_timestamp(
                capture,
                requested_timestamp,
            )
            if not ok or frame is None:
                sampling["failed_timestamps_seconds"].append(requested_timestamp)
                warnings.append(
                    f"Frame decoding failed at requested timestamp {requested_timestamp:.3f} seconds."
                )
                continue
            if not _is_finite(decoded_timestamp):
                sampling["failed_timestamps_seconds"].append(requested_timestamp)
                warnings.append(
                    f"Frame decoding returned a non-finite timestamp near {requested_timestamp:.3f} seconds."
                )
                continue

            artifact_name = f"frame_{len(raw_frames):03d}_{decoded_timestamp:.3f}s.jpg"
            artifact_path = frames_dir / artifact_name
            relative_artifact = Path("frames") / artifact_name
            # Numerical metrics are calculated from the decoded in-memory frame;
            # JPEG artifacts are written afterward for human review only.
            frame_entry = analyze_frame(
                frame=frame,
                sample_index=len(raw_frames),
                requested_timestamp=requested_timestamp,
                decoded_timestamp=decoded_timestamp,
                artifact_path=relative_artifact,
            )
            saved = cv2.imwrite(
                str(artifact_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            if not saved:
                warnings.append(f"Could not write sampled frame artifact {artifact_name}.")
                continue
            if (
                abs(frame_entry["seek_error_seconds"]) > SEEK_ERROR_WARNING_SECONDS
                if frame_entry["seek_error_seconds"] is not None
                else False
            ):
                warnings.append(
                    "Decoded frame timestamp differed from the requested timestamp "
                    f"by {abs(frame_entry['seek_error_seconds']):.3f} seconds. "
                    "This is a sampling/decoder limitation."
                )
            frame_analysis["frames"].append(frame_entry)
            raw_frames.append(np.array(frame, copy=True))
    finally:
        capture.release()

    sampling["decoded_frame_count"] = len(frame_analysis["frames"])
    _update_sampling_seek_summary(sampling, frame_analysis["frames"])

    if len(raw_frames) >= 2:
        frame_analysis["comparisons"] = compare_frames(
            frame_analysis["frames"],
            raw_frames,
        )

    frame_analysis["summary"] = summarize_frame_analysis(
        frame_analysis["frames"],
        frame_analysis["comparisons"],
    )
    return sampling, frame_analysis, warnings


def _update_sampling_seek_summary(
    sampling: dict[str, Any],
    frames: list[dict[str, Any]],
) -> None:
    seek_errors = [
        abs(frame["seek_error_seconds"])
        for frame in frames
        if frame.get("seek_error_seconds") is not None
    ]
    if seek_errors:
        sampling["maximum_absolute_seek_error_seconds"] = round(max(seek_errors), 3)
        sampling["average_absolute_seek_error_seconds"] = round(
            sum(seek_errors) / len(seek_errors),
            3,
        )

    seen: set[float] = set()
    repeated: list[float] = []
    for frame in frames:
        decoded = frame.get("decoded_timestamp_seconds")
        if decoded is None:
            continue
        if decoded in seen and decoded not in repeated:
            repeated.append(decoded)
        seen.add(decoded)
    sampling["repeated_decoded_timestamps_seconds"] = repeated


def _read_frame_near_timestamp(capture: Any, requested_timestamp: float) -> tuple[bool, Any, float]:
    import cv2

    fallback_offsets = (0.0, 0.1, 0.25, 0.5)
    for offset in fallback_offsets:
        seek_timestamp = max(0.0, requested_timestamp - offset)
        capture.set(cv2.CAP_PROP_POS_MSEC, seek_timestamp * 1000.0)
        ok, frame = capture.read()
        if ok and frame is not None:
            decoded_timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            return True, frame, decoded_timestamp
    return False, None, requested_timestamp


def _is_finite(value: float | None) -> bool:
    import math

    return value is not None and math.isfinite(value)


def _target_sample_count(duration_seconds: float | None) -> int:
    if duration_seconds is not None and duration_seconds <= SHORT_VIDEO_MAX_SECONDS:
        return SHORT_VIDEO_SAMPLE_COUNT
    return LONG_VIDEO_SAMPLE_COUNT


def _build_timestamps(duration_seconds: float | None, target_count: int) -> list[float]:
    if duration_seconds is None or duration_seconds <= 0:
        return [0.0]

    safe_end = max(0.0, duration_seconds - END_SEEK_SAFETY_SECONDS)
    if target_count <= 1 or safe_end <= 0:
        return [0.0]

    step = safe_end / (target_count - 1)
    timestamps = [round(step * index, 3) for index in range(target_count)]
    timestamps[0] = 0.0
    timestamps[-1] = round(safe_end, 3)

    deduped: list[float] = []
    for timestamp in timestamps:
        if timestamp not in deduped:
            deduped.append(timestamp)
    return deduped
