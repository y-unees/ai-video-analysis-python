from __future__ import annotations

from pathlib import Path
from typing import Any

from config import (
    BLACK_FRAME_RATIO_THRESHOLD,
    BRIGHT_PIXEL_THRESHOLD,
    COMPARISON_MAX_WIDTH,
    DARK_PIXEL_THRESHOLD,
    HISTOGRAM_BINS,
    LARGE_CHANGE_HASH_DISTANCE,
    LARGE_CHANGE_MEAN_ABS_DIFF,
    NEAR_DUPLICATE_HASH_DISTANCE,
    NEAR_DUPLICATE_MEAN_ABS_DIFF,
    WHITE_FRAME_RATIO_THRESHOLD,
)


def analyze_frame(
    frame: Any,
    sample_index: int,
    requested_timestamp: float,
    decoded_timestamp: float | None,
    artifact_path: Path,
) -> dict[str, Any]:
    import cv2
    import imagehash
    import numpy as np
    from PIL import Image

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_ratio = float(np.mean(gray < DARK_PIXEL_THRESHOLD))
    bright_ratio = float(np.mean(gray > BRIGHT_PIXEL_THRESHOLD))
    mean_rgb_values = np.mean(rgb, axis=(0, 1))
    perceptual_hash = imagehash.phash(Image.fromarray(rgb))

    seek_error = (
        decoded_timestamp - requested_timestamp
        if decoded_timestamp is not None
        else None
    )
    return {
        "sample_index": sample_index,
        "requested_timestamp_seconds": _round(requested_timestamp),
        "decoded_timestamp_seconds": _round(decoded_timestamp),
        "seek_error_seconds": _round(seek_error),
        "artifact_path": artifact_path.as_posix(),
        "width": int(width),
        "height": int(height),
        "brightness_mean": _round(brightness),
        "contrast_stddev": _round(contrast),
        "laplacian_variance": _round(laplacian_variance),
        "dark_pixel_ratio": _round(dark_ratio, 6),
        "bright_pixel_ratio": _round(bright_ratio, 6),
        "likely_near_black_frame": dark_ratio >= BLACK_FRAME_RATIO_THRESHOLD,
        "likely_near_white_frame": bright_ratio >= WHITE_FRAME_RATIO_THRESHOLD,
        "mean_rgb": {
            "red": _round(float(mean_rgb_values[0])),
            "green": _round(float(mean_rgb_values[1])),
            "blue": _round(float(mean_rgb_values[2])),
        },
        "perceptual_hash": str(perceptual_hash),
    }


def compare_frames(frame_entries: list[dict[str, Any]], raw_frames: list[Any]) -> list[dict[str, Any]]:
    import cv2
    import imagehash
    import numpy as np
    from PIL import Image

    comparisons: list[dict[str, Any]] = []
    for index in range(len(raw_frames) - 1):
        first_frame = raw_frames[index]
        second_frame = raw_frames[index + 1]
        first_entry = frame_entries[index]
        second_entry = frame_entries[index + 1]

        first_gray = _resize_for_comparison(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY))
        second_gray = _resize_for_comparison(cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY))
        common_size = (
            min(first_gray.shape[1], second_gray.shape[1]),
            min(first_gray.shape[0], second_gray.shape[0]),
        )
        first_gray = cv2.resize(first_gray, common_size, interpolation=cv2.INTER_AREA)
        second_gray = cv2.resize(second_gray, common_size, interpolation=cv2.INTER_AREA)

        mean_abs_diff = float(
            np.mean(cv2.absdiff(first_gray, second_gray)) / 255.0
        )
        first_hist = cv2.calcHist([first_gray], [0], None, [HISTOGRAM_BINS], [0, 256])
        second_hist = cv2.calcHist([second_gray], [0], None, [HISTOGRAM_BINS], [0, 256])
        cv2.normalize(first_hist, first_hist)
        cv2.normalize(second_hist, second_hist)
        histogram_correlation = float(
            cv2.compareHist(first_hist, second_hist, cv2.HISTCMP_CORREL)
        )

        first_hash = imagehash.hex_to_hash(first_entry["perceptual_hash"])
        second_hash = imagehash.hex_to_hash(second_entry["perceptual_hash"])

        hash_distance = int(first_hash - second_hash)
        rounded_mean_abs_diff = _round(mean_abs_diff, 6)
        comparisons.append(
            {
                "from_sample_index": first_entry["sample_index"],
                "to_sample_index": second_entry["sample_index"],
                "start_timestamp_seconds": first_entry["decoded_timestamp_seconds"],
                "end_timestamp_seconds": second_entry["decoded_timestamp_seconds"],
                "perceptual_hash_distance": hash_distance,
                "normalized_mean_absolute_difference": rounded_mean_abs_diff,
                "histogram_correlation": _round(histogram_correlation, 6),
                "histogram_method": f"OpenCV HISTCMP_CORREL on normalized {HISTOGRAM_BINS}-bin grayscale histograms",
                "classification": classify_comparison(hash_distance, rounded_mean_abs_diff),
            }
        )

    return comparisons


def summarize_frame_analysis(
    frames: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    from config import LARGE_CHANGE_HASH_DISTANCE, NEAR_DUPLICATE_HASH_DISTANCE

    brightness_values = [frame["brightness_mean"] for frame in frames]
    contrast_values = [frame["contrast_stddev"] for frame in frames]
    laplacian_values = [frame["laplacian_variance"] for frame in frames]
    seek_errors = [
        abs(frame["seek_error_seconds"])
        for frame in frames
        if frame.get("seek_error_seconds") is not None
    ]

    return {
        "frames_analyzed": len(frames),
        "average_brightness": _average(brightness_values),
        "minimum_brightness": min(brightness_values) if brightness_values else None,
        "maximum_brightness": max(brightness_values) if brightness_values else None,
        "average_contrast": _average(contrast_values),
        "average_laplacian_variance": _average(laplacian_values),
        "minimum_laplacian_variance": min(laplacian_values) if laplacian_values else None,
        "maximum_laplacian_variance": max(laplacian_values) if laplacian_values else None,
        "heuristic_near_black_frame_count": sum(1 for frame in frames if frame["likely_near_black_frame"]),
        "heuristic_near_white_frame_count": sum(1 for frame in frames if frame["likely_near_white_frame"]),
        "heuristic_large_change_pair_count": sum(
            1
            for comparison in comparisons
            if comparison["classification"]["large_change"]
        ),
        "heuristic_near_duplicate_pair_count": sum(
            1
            for comparison in comparisons
            if comparison["classification"]["near_duplicate"]
        ),
        "maximum_absolute_seek_error_seconds": max(seek_errors) if seek_errors else None,
        "average_absolute_seek_error_seconds": _average(seek_errors),
    }


def classify_comparison(hash_distance: int, normalized_mean_absolute_difference: float) -> dict[str, Any]:
    near_duplicate_rules: list[str] = []
    large_change_rules: list[str] = []

    if hash_distance <= NEAR_DUPLICATE_HASH_DISTANCE:
        near_duplicate_rules.append("perceptual_hash_distance_within_threshold")
    if normalized_mean_absolute_difference <= NEAR_DUPLICATE_MEAN_ABS_DIFF:
        near_duplicate_rules.append("normalized_pixel_difference_within_threshold")
    if hash_distance >= LARGE_CHANGE_HASH_DISTANCE:
        large_change_rules.append("perceptual_hash_distance_at_or_above_threshold")
    if normalized_mean_absolute_difference >= LARGE_CHANGE_MEAN_ABS_DIFF:
        large_change_rules.append("normalized_pixel_difference_at_or_above_threshold")

    near_duplicate = len(near_duplicate_rules) == 2
    large_change = bool(large_change_rules)
    return {
        "near_duplicate": near_duplicate,
        "large_change": large_change,
        "triggered_rules": [
            *(near_duplicate_rules if near_duplicate else []),
            *(large_change_rules if large_change else []),
        ],
    }


def _resize_for_comparison(gray_frame: Any) -> Any:
    import cv2

    height, width = gray_frame.shape[:2]
    if width <= COMPARISON_MAX_WIDTH:
        return gray_frame
    scale = COMPARISON_MAX_WIDTH / width
    new_size = (COMPARISON_MAX_WIDTH, max(1, int(height * scale)))
    return cv2.resize(gray_frame, new_size, interpolation=cv2.INTER_AREA)


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return _round(sum(values) / len(values))


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None
