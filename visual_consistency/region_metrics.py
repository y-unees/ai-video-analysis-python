from __future__ import annotations

from typing import Any

from config import (
    FARNEBACK_ITERATIONS,
    FARNEBACK_LEVELS,
    FARNEBACK_POLY_N,
    FARNEBACK_POLY_SIGMA,
    FARNEBACK_PYR_SCALE,
    FARNEBACK_WINSIZE,
    VISUAL_CONSISTENCY_EDGE_CANNY_THRESHOLD1,
    VISUAL_CONSISTENCY_EDGE_CANNY_THRESHOLD2,
    VISUAL_CONSISTENCY_HIGH_DETAIL_RESIDUAL_THRESHOLD,
    VISUAL_CONSISTENCY_HIGH_MOTION_FLOW_THRESHOLD,
    VISUAL_CONSISTENCY_STATIONARY_FLOW_THRESHOLD,
    VISUAL_CONSISTENCY_TEXTURE_HISTOGRAM_BINS,
    VISUAL_CONSISTENCY_UNSTABLE_BRIGHTNESS_THRESHOLD,
    VISUAL_CONSISTENCY_UNSTABLE_DETAIL_THRESHOLD,
    VISUAL_CONSISTENCY_UNSTABLE_EDGE_THRESHOLD,
    VISUAL_CONSISTENCY_UNSTABLE_TEXTURE_THRESHOLD,
)
from visual_consistency.grid import region_slice


def calculate_transition_region_records(
    previous_gray: Any,
    current_gray: Any,
    transition: dict[str, Any],
    regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import cv2
    import numpy as np

    if previous_gray.shape != current_gray.shape:
        current_gray = cv2.resize(
            current_gray,
            (previous_gray.shape[1], previous_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    flow = _backward_flow(previous_gray, current_gray)
    magnitude = _flow_magnitude(flow)
    warped_previous_gray = _warp(previous_gray, flow)
    previous_edges = _edges(previous_gray)
    current_edges = _edges(current_gray)
    warped_previous_edges = _edges(warped_previous_gray)
    previous_detail = _detail_map(previous_gray)
    current_detail = _detail_map(current_gray)
    warped_previous_detail = _warp(previous_detail, flow)

    records: list[dict[str, Any]] = []
    for region in regions:
        y_slice, x_slice = region_slice(region)
        prev = previous_gray[y_slice, x_slice]
        curr = current_gray[y_slice, x_slice]
        flow_region = magnitude[y_slice, x_slice]
        edge_metrics = _edge_metrics(
            previous_edges[y_slice, x_slice],
            current_edges[y_slice, x_slice],
            warped_previous_edges[y_slice, x_slice],
        )
        texture_metrics = _texture_metrics(prev, curr)
        detail_metrics = _detail_metrics(
            warped_previous_detail[y_slice, x_slice],
            current_detail[y_slice, x_slice],
        )
        brightness = _brightness_metrics(prev, curr)
        motion = _motion_context(flow_region)
        classification = _classification(brightness, edge_metrics, texture_metrics, detail_metrics)
        records.append(
            {
                "consistency_record_id": (
                    f"consistency-t{transition['from_sample_index']:05d}-"
                    f"{transition['to_sample_index']:05d}-r{region['row']:02d}-c{region['column']:02d}"
                ),
                "transition_id": transition["transition_id"],
                "from_sample_index": transition["from_sample_index"],
                "to_sample_index": transition["to_sample_index"],
                "start_timestamp_seconds": transition["start_timestamp_seconds"],
                "end_timestamp_seconds": transition["end_timestamp_seconds"],
                "scene_index": transition.get("scene_index"),
                "region": {
                    **region,
                    "valid_pixel_ratio": _round(1.0, 6),
                },
                "motion_context": motion,
                "brightness_consistency": brightness,
                "edge_stability": edge_metrics,
                "texture_stability": texture_metrics,
                "detail_persistence": detail_metrics,
                "classification": classification,
                "ranking_data": {
                    "edge_instability": _round(1.0 - edge_metrics["edge_iou"], 6),
                    "texture_distance": texture_metrics["histogram_distance"],
                    "detail_residual": detail_metrics["mean_normalized_residual"],
                    "brightness_normalized_difference": brightness["normalized_difference"],
                },
            }
        )
    return records


def transition_work_images(previous_gray: Any, current_gray: Any) -> dict[str, Any]:
    import cv2

    flow = _backward_flow(previous_gray, current_gray)
    previous_detail = _detail_map(previous_gray)
    current_detail = _detail_map(current_gray)
    residual = cv2.absdiff(_warp(previous_detail, flow), current_detail)
    return {
        "previous_gray": previous_gray,
        "current_gray": current_gray,
        "detail_residual": residual,
    }


def _backward_flow(previous_gray: Any, current_gray: Any) -> Any:
    import cv2
    import numpy as np

    try:
        return cv2.calcOpticalFlowFarneback(
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
    except cv2.error:
        return np.zeros((*current_gray.shape[:2], 2), dtype=np.float32)


def _warp(image: Any, backward_flow: Any) -> Any:
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    x_coords, y_coords = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return cv2.remap(
        image,
        x_coords + backward_flow[..., 0],
        y_coords + backward_flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _flow_magnitude(flow: Any) -> Any:
    import cv2

    magnitude, _angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude


def _edges(gray: Any) -> Any:
    import cv2

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(
        blurred,
        VISUAL_CONSISTENCY_EDGE_CANNY_THRESHOLD1,
        VISUAL_CONSISTENCY_EDGE_CANNY_THRESHOLD2,
    ) > 0


def _detail_map(gray: Any) -> Any:
    import cv2
    import numpy as np

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    return np.clip(np.abs(laplacian), 0, 255).astype("uint8")


def _brightness_metrics(previous: Any, current: Any) -> dict[str, float]:
    import numpy as np

    previous_mean = float(np.mean(previous))
    current_mean = float(np.mean(current))
    absolute = abs(current_mean - previous_mean)
    return {
        "previous_mean": _round(previous_mean),
        "current_mean": _round(current_mean),
        "absolute_difference": _round(absolute),
        "normalized_difference": _round(absolute / 255.0, 6),
    }


def _edge_metrics(previous_edges: Any, current_edges: Any, warped_previous_edges: Any) -> dict[str, Any]:
    import numpy as np

    previous_density = float(np.mean(previous_edges))
    current_density = float(np.mean(current_edges))
    intersection = int(np.logical_and(warped_previous_edges, current_edges).sum())
    previous_count = int(warped_previous_edges.sum())
    current_count = int(current_edges.sum())
    union = int(np.logical_or(warped_previous_edges, current_edges).sum())
    overlap = intersection / previous_count if previous_count else (1.0 if current_count == 0 else 0.0)
    iou = intersection / union if union else 1.0
    return {
        "previous_edge_density": _round(previous_density, 6),
        "current_edge_density": _round(current_density, 6),
        "absolute_density_difference": _round(abs(current_density - previous_density), 6),
        "edge_overlap_ratio": _round(overlap, 6),
        "edge_iou": _round(iou, 6),
        "comparison_mode": "motion_compensated",
    }


def _texture_metrics(previous: Any, current: Any) -> dict[str, float]:
    import cv2
    import numpy as np

    previous_hist = cv2.calcHist([previous], [0], None, [VISUAL_CONSISTENCY_TEXTURE_HISTOGRAM_BINS], [0, 256])
    current_hist = cv2.calcHist([current], [0], None, [VISUAL_CONSISTENCY_TEXTURE_HISTOGRAM_BINS], [0, 256])
    cv2.normalize(previous_hist, previous_hist)
    cv2.normalize(current_hist, current_hist)
    correlation = float(cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_CORREL))
    correlation = max(-1.0, min(1.0, correlation))
    previous_variance = float(np.var(previous))
    current_variance = float(np.var(current))
    previous_gradient = _gradient_energy(previous)
    current_gradient = _gradient_energy(current)
    return {
        "histogram_correlation": _round(correlation, 6),
        "histogram_distance": _round((1.0 - correlation) / 2.0, 6),
        "previous_local_variance": _round(previous_variance),
        "current_local_variance": _round(current_variance),
        "variance_difference": _round(abs(current_variance - previous_variance)),
        "previous_gradient_energy": _round(previous_gradient),
        "current_gradient_energy": _round(current_gradient),
        "gradient_energy_difference": _round(abs(current_gradient - previous_gradient)),
    }


def _gradient_energy(gray: Any) -> float:
    import cv2
    import numpy as np

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(cv2.magnitude(gx, gy)))


def _detail_metrics(warped_previous_detail: Any, current_detail: Any) -> dict[str, float]:
    import cv2
    import numpy as np

    residual = cv2.absdiff(warped_previous_detail, current_detail).astype("float32") / 255.0
    previous_energy = float(np.mean(warped_previous_detail))
    current_energy = float(np.mean(current_detail))
    return {
        "mean_normalized_residual": _round(float(np.mean(residual)), 6),
        "median_normalized_residual": _round(float(np.median(residual)), 6),
        "percentile_95_normalized_residual": _round(float(np.percentile(residual, 95)), 6),
        "high_residual_pixel_ratio": _round(
            float(np.mean(residual >= VISUAL_CONSISTENCY_HIGH_DETAIL_RESIDUAL_THRESHOLD)),
            6,
        ),
        "previous_laplacian_energy": _round(previous_energy),
        "current_laplacian_energy": _round(current_energy),
        "laplacian_energy_difference": _round(abs(current_energy - previous_energy)),
    }


def _motion_context(flow_region: Any) -> dict[str, Any]:
    import numpy as np

    mean = float(np.mean(flow_region))
    median = float(np.median(flow_region))
    stationary = float(np.mean(flow_region <= VISUAL_CONSISTENCY_STATIONARY_FLOW_THRESHOLD))
    if mean >= VISUAL_CONSISTENCY_HIGH_MOTION_FLOW_THRESHOLD:
        classification = "high_motion_region"
    elif stationary >= 0.75:
        classification = "mostly_stationary_region"
    else:
        classification = "moderate_motion_region"
    return {
        "mean_flow_magnitude": _round(mean),
        "median_flow_magnitude": _round(median),
        "stationary_pixel_ratio": _round(stationary, 6),
        "classification": classification,
    }


def _classification(
    brightness: dict[str, float],
    edge: dict[str, Any],
    texture: dict[str, float],
    detail: dict[str, float],
) -> dict[str, Any]:
    triggered: list[str] = []
    edge_instability = 1.0 - edge["edge_iou"]
    if edge_instability >= VISUAL_CONSISTENCY_UNSTABLE_EDGE_THRESHOLD:
        triggered.append("edge_instability_at_or_above_configured_threshold")
    if texture["histogram_distance"] >= VISUAL_CONSISTENCY_UNSTABLE_TEXTURE_THRESHOLD:
        triggered.append("texture_distance_at_or_above_configured_threshold")
    if detail["mean_normalized_residual"] >= VISUAL_CONSISTENCY_UNSTABLE_DETAIL_THRESHOLD:
        triggered.append("detail_residual_at_or_above_configured_threshold")
    if brightness["normalized_difference"] >= VISUAL_CONSISTENCY_UNSTABLE_BRIGHTNESS_THRESHOLD:
        triggered.append("brightness_change_at_or_above_configured_threshold")
    return {
        "regional_visual_variation": bool(triggered),
        "triggered_rules": triggered,
        "interpretation": "Regional visual variation is a review measurement and is not proof of editing, manipulation, or AI generation.",
    }


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(float(value), digits) if value is not None else None
