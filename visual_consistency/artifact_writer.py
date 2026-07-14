from __future__ import annotations

from pathlib import Path
from typing import Any

from config import JPEG_QUALITY
from file_utils import artifact_record
from visual_consistency.grid import region_slice


def write_review_artifacts(
    output_dir: Path,
    ranked_transitions: list[dict[str, Any]],
    transition_images: dict[str, dict[str, Any]],
    transition_records: dict[str, list[dict[str, Any]]],
    regions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    import cv2

    warnings: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for ranked in ranked_transitions:
        transition_id = ranked["transition_id"]
        images = transition_images.get(transition_id)
        if not images:
            records[transition_id] = {"status": "failed", "artifacts": {}}
            warnings.append(f"Missing visual-consistency diagnostic images for {transition_id}.")
            continue
        prefix = f"consistency_{ranked['ranked_review_index']:03d}"
        before_name = f"{prefix}_before_grid_{ranked['start_timestamp_seconds']:.3f}s.jpg"
        after_name = f"{prefix}_after_grid_{ranked['end_timestamp_seconds']:.3f}s.jpg"
        detail_name = f"{prefix}_detail_residual.png"
        heatmap_name = f"{prefix}_combined_heatmap.png"
        artifacts: dict[str, Any] = {}
        specs = {
            "before_grid": (before_name, _grid_overlay(images["previous_gray"], regions)),
            "after_grid": (after_name, _grid_overlay(images["current_gray"], regions)),
            "detail_residual_heatmap": (detail_name, _heatmap(images["detail_residual"])),
            "combined_consistency_heatmap": (
                heatmap_name,
                _combined_heatmap(
                    images["current_gray"],
                    transition_records.get(transition_id, []),
                    regions,
                ),
            ),
        }
        failed = False
        for label, (filename, image) in specs.items():
            path = output_dir / filename
            params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY] if filename.endswith(".jpg") else []
            if not cv2.imwrite(str(path), image, params):
                failed = True
                warnings.append(f"Could not write visual-consistency artifact {filename}.")
                continue
            artifacts[label] = artifact_record(path, f"consistency_frames/{filename}")
        records[transition_id] = {"status": "partial" if failed else "completed", "artifacts": artifacts}
    return records, warnings


def _grid_overlay(gray: Any, regions: list[dict[str, Any]]) -> Any:
    import cv2

    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for region in regions:
        bounds = region["pixel_bounds"]
        x0 = bounds["x"]
        y0 = bounds["y"]
        x1 = x0 + bounds["width"]
        y1 = y0 + bounds["height"]
        cv2.rectangle(image, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (0, 255, 255), 1)
    return image


def _heatmap(gray: Any) -> Any:
    import cv2

    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def _combined_heatmap(
    base_gray: Any,
    records: list[dict[str, Any]],
    regions: list[dict[str, Any]],
) -> Any:
    import cv2
    import numpy as np

    heat = np.zeros_like(base_gray, dtype=np.uint8)
    record_by_region = {record["region"]["region_id"]: record for record in records}
    for region in regions:
        record = record_by_region.get(region["region_id"])
        if not record:
            continue
        value = max(
            record["ranking_data"]["edge_instability"],
            record["ranking_data"]["texture_distance"],
            record["ranking_data"]["detail_residual"],
            record["ranking_data"]["brightness_normalized_difference"],
        )
        y_slice, x_slice = region_slice(region)
        heat[y_slice, x_slice] = int(max(0.0, min(1.0, value)) * 255)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    base = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 0.55, colored, 0.45, 0)
