from __future__ import annotations

from typing import Any

from config import (
    HISTOGRAM_BINS,
    SCENE_CUT_HISTOGRAM_CORRELATION_THRESHOLD,
    SCENE_CUT_PHASH_DISTANCE_THRESHOLD,
    SCENE_CUT_PIXEL_DIFFERENCE_THRESHOLD,
    STATIC_PHASH_DISTANCE_THRESHOLD,
    STATIC_PIXEL_DIFFERENCE_THRESHOLD,
)
from temporal.optical_flow import calculate_flow_warp_residual, calculate_optical_flow


def calculate_transition(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    import cv2
    import imagehash
    import numpy as np
    from PIL import Image

    previous_gray = previous["gray"]
    current_gray = current["gray"]
    if previous_gray.shape != current_gray.shape:
        current_gray = cv2.resize(
            current_gray,
            (previous_gray.shape[1], previous_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    mean_abs_diff = float(np.mean(cv2.absdiff(previous_gray, current_gray)) / 255.0)
    previous_hist = cv2.calcHist([previous_gray], [0], None, [HISTOGRAM_BINS], [0, 256])
    current_hist = cv2.calcHist([current_gray], [0], None, [HISTOGRAM_BINS], [0, 256])
    cv2.normalize(previous_hist, previous_hist)
    cv2.normalize(current_hist, current_hist)
    histogram_correlation = float(
        cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_CORREL)
    )
    previous_hash = imagehash.phash(Image.fromarray(previous_gray))
    current_hash = imagehash.phash(Image.fromarray(current_gray))
    phash_distance = int(previous_hash - current_hash)
    brightness_difference = abs(
        float(np.mean(current_gray)) - float(np.mean(previous_gray))
    )
    optical_flow = calculate_optical_flow(previous_gray, current_gray)
    flow_warp_residual, residual_image = calculate_flow_warp_residual(
        previous_gray,
        current_gray,
    )
    classification = classify_transition(
        phash_distance,
        mean_abs_diff,
        histogram_correlation,
    )

    return {
        "transition_id": f"temporal-transition-{previous['sample']['temporal_sample_index']:05d}-{current['sample']['temporal_sample_index']:05d}",
        "from_sample_index": previous["sample"]["temporal_sample_index"],
        "to_sample_index": current["sample"]["temporal_sample_index"],
        "start_timestamp_seconds": previous["sample"]["timestamp_seconds"],
        "end_timestamp_seconds": current["sample"]["timestamp_seconds"],
        "perceptual_hash_distance": phash_distance,
        "normalized_mean_absolute_difference": _round(mean_abs_diff, 6),
        "histogram_correlation": _round(histogram_correlation, 6),
        "absolute_brightness_difference": _round(brightness_difference),
        "optical_flow": optical_flow,
        "flow_warp_residual": flow_warp_residual,
        "classification": classification,
        "_diagnostics": {
            "previous_gray": previous_gray,
            "current_gray": current_gray,
            "absolute_difference_image": cv2.absdiff(previous_gray, current_gray),
            "flow_warp_residual_image": residual_image,
        },
    }


def classify_transition(
    phash_distance: int,
    normalized_difference: float,
    histogram_correlation: float,
) -> dict[str, Any]:
    scene_rules: list[str] = []
    static_rules: list[str] = []

    pixel_cut = normalized_difference >= SCENE_CUT_PIXEL_DIFFERENCE_THRESHOLD
    hist_cut = histogram_correlation <= SCENE_CUT_HISTOGRAM_CORRELATION_THRESHOLD
    phash_cut = phash_distance >= SCENE_CUT_PHASH_DISTANCE_THRESHOLD
    if pixel_cut:
        scene_rules.append("normalized_pixel_difference_at_or_above_scene_threshold")
    if hist_cut:
        scene_rules.append("histogram_correlation_at_or_below_scene_threshold")
    if phash_cut:
        scene_rules.append("perceptual_hash_distance_at_or_above_scene_threshold")

    near_static_phash = phash_distance <= STATIC_PHASH_DISTANCE_THRESHOLD
    near_static_pixel = normalized_difference <= STATIC_PIXEL_DIFFERENCE_THRESHOLD
    if near_static_phash:
        static_rules.append("perceptual_hash_distance_within_static_threshold")
    if near_static_pixel:
        static_rules.append("normalized_pixel_difference_within_static_threshold")

    scene_boundary = pixel_cut and (hist_cut or phash_cut)
    near_static = near_static_phash and near_static_pixel
    return {
        "scene_boundary_candidate": scene_boundary,
        "near_static_transition": near_static,
        "triggered_rules": [
            *(scene_rules if scene_boundary else []),
            *(static_rules if near_static else []),
        ],
    }


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None
