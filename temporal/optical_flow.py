from __future__ import annotations

from typing import Any

from config import (
    FARNEBACK_ITERATIONS,
    FARNEBACK_LEVELS,
    FARNEBACK_POLY_N,
    FARNEBACK_POLY_SIGMA,
    FARNEBACK_PYR_SCALE,
    FARNEBACK_WINSIZE,
    FLOW_WARP_HIGH_RESIDUAL_THRESHOLD,
    FLOW_STATIONARY_MAGNITUDE_THRESHOLD,
)


def calculate_optical_flow(previous_gray: Any, current_gray: Any) -> dict[str, float | None]:
    import cv2
    import numpy as np

    if previous_gray is None or current_gray is None:
        return _empty_flow()
    if previous_gray.shape != current_gray.shape:
        current_gray = cv2.resize(
            current_gray,
            (previous_gray.shape[1], previous_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    try:
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            FARNEBACK_PYR_SCALE,
            FARNEBACK_LEVELS,
            FARNEBACK_WINSIZE,
            FARNEBACK_ITERATIONS,
            FARNEBACK_POLY_N,
            FARNEBACK_POLY_SIGMA,
            0,
        )
        magnitude, _angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    except cv2.error:
        return _empty_flow()

    return {
        "mean_magnitude": _round(float(np.mean(magnitude))),
        "median_magnitude": _round(float(np.median(magnitude))),
        "percentile_95_magnitude": _round(float(np.percentile(magnitude, 95))),
        "stationary_pixel_ratio": _round(
            float(np.mean(magnitude <= FLOW_STATIONARY_MAGNITUDE_THRESHOLD)),
            6,
        ),
    }


def calculate_flow_warp_residual(previous_gray: Any, current_gray: Any) -> tuple[dict[str, Any], Any]:
    import cv2
    import numpy as np

    if previous_gray is None or current_gray is None:
        return _empty_residual(), None
    if previous_gray.shape != current_gray.shape:
        current_gray = cv2.resize(
            current_gray,
            (previous_gray.shape[1], previous_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    height, width = current_gray.shape[:2]
    try:
        # Backward flow maps each pixel in the current frame to where it came
        # from in the previous frame. cv2.remap samples the previous frame at
        # those source coordinates, yielding previous warped into current time.
        backward_flow = cv2.calcOpticalFlowFarneback(
            current_gray,
            previous_gray,
            None,
            FARNEBACK_PYR_SCALE,
            FARNEBACK_LEVELS,
            FARNEBACK_WINSIZE,
            FARNEBACK_ITERATIONS,
            FARNEBACK_POLY_N,
            FARNEBACK_POLY_SIGMA,
            0,
        )
        x_coords, y_coords = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        map_x = x_coords + backward_flow[..., 0]
        map_y = y_coords + backward_flow[..., 1]
        valid_mask = (
            (map_x >= 0)
            & (map_x <= width - 1)
            & (map_y >= 0)
            & (map_y <= height - 1)
        )
        warped_previous = cv2.remap(
            previous_gray,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        residual = cv2.absdiff(warped_previous, current_gray).astype(np.float32) / 255.0
        valid_residual = residual[valid_mask]
        if valid_residual.size == 0:
            return _empty_residual(), None
        residual_image = (residual * 255.0).clip(0, 255).astype(np.uint8)
        return {
            "mean_normalized_residual": _round(float(np.mean(valid_residual))),
            "median_normalized_residual": _round(float(np.median(valid_residual))),
            "percentile_95_normalized_residual": _round(float(np.percentile(valid_residual, 95))),
            "high_residual_pixel_ratio": _round(
                float(np.mean(valid_residual >= FLOW_WARP_HIGH_RESIDUAL_THRESHOLD)),
                6,
            ),
            "high_residual_threshold": FLOW_WARP_HIGH_RESIDUAL_THRESHOLD,
            "valid_pixel_count": int(valid_residual.size),
        }, residual_image
    except cv2.error:
        return _empty_residual(), None


def _empty_flow() -> dict[str, None]:
    return {
        "mean_magnitude": None,
        "median_magnitude": None,
        "percentile_95_magnitude": None,
        "stationary_pixel_ratio": None,
    }


def _empty_residual() -> dict[str, Any]:
    return {
        "mean_normalized_residual": None,
        "median_normalized_residual": None,
        "percentile_95_normalized_residual": None,
        "high_residual_pixel_ratio": None,
        "high_residual_threshold": FLOW_WARP_HIGH_RESIDUAL_THRESHOLD,
        "valid_pixel_count": 0,
    }


def _round(value: float, digits: int = 3) -> float:
    return round(value, digits)
