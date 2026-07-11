from __future__ import annotations

from typing import Any

from config import (
    BRIGHTNESS_JUMP_THRESHOLD,
    LARGE_CHANGE_HASH_DISTANCE,
    LARGE_CHANGE_MEAN_ABS_DIFF,
    LOW_SHARPNESS_HEURISTIC,
    NEAR_DUPLICATE_HASH_DISTANCE,
    NEAR_DUPLICATE_MEAN_ABS_DIFF,
)


def build_temporal_evidence(
    frames: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    failed_timestamps: list[float],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    next_id = 1

    for frame in frames:
        if frame["likely_near_black_frame"]:
            observations.append(
                _observation(
                    next_id,
                    "likely_near_black_sampled_frame",
                    frame.get("decoded_timestamp_seconds"),
                    None,
                    "A sampled frame is likely black based on the configured dark-pixel ratio threshold.",
                    "This may be normal content, a fade, a transition, camera cover, or another ordinary visual condition.",
                    {"dark_pixel_ratio": frame["dark_pixel_ratio"]},
                )
            )
            next_id += 1
        if frame["likely_near_white_frame"]:
            observations.append(
                _observation(
                    next_id,
                    "likely_near_white_sampled_frame",
                    frame.get("decoded_timestamp_seconds"),
                    None,
                    "A sampled frame is likely white based on the configured bright-pixel ratio threshold.",
                    "This may be normal content, a flash, a transition, overexposure, or another ordinary visual condition.",
                    {"bright_pixel_ratio": frame["bright_pixel_ratio"]},
                )
            )
            next_id += 1
        if frame["laplacian_variance"] < LOW_SHARPNESS_HEURISTIC:
            observations.append(
                _observation(
                    next_id,
                    "low_sampled_frame_laplacian_variance",
                    frame.get("decoded_timestamp_seconds"),
                    None,
                    "A sampled frame had low sharpness according to the configured Laplacian-variance heuristic.",
                    "This may reflect blur, compression, motion, defocus, or naturally low-detail content.",
                    {"laplacian_variance": frame["laplacian_variance"]},
                )
            )
            next_id += 1

    frame_by_index = {frame["sample_index"]: frame for frame in frames}
    for comparison in comparisons:
        hash_distance = comparison["perceptual_hash_distance"]
        mean_abs_diff = comparison["normalized_mean_absolute_difference"]
        from_frame = frame_by_index.get(comparison["from_sample_index"])
        to_frame = frame_by_index.get(comparison["to_sample_index"])

        if comparison["classification"]["near_duplicate"]:
            observations.append(
                _observation(
                    next_id,
                    "near_duplicate_sampled_frames",
                    comparison["start_timestamp_seconds"],
                    comparison["end_timestamp_seconds"],
                    "Two consecutive sampled frames are nearly identical by the configured comparison heuristics.",
                    "This may represent a static shot, low motion, duplicated content, or normal recording behavior.",
                    {
                        "perceptual_hash_distance": hash_distance,
                        "normalized_mean_absolute_difference": mean_abs_diff,
                    },
                )
            )
            next_id += 1

        if comparison["classification"]["large_change"]:
            observations.append(
                _observation(
                    next_id,
                    "large_sampled_frame_change",
                    comparison["start_timestamp_seconds"],
                    comparison["end_timestamp_seconds"],
                    "A large visual difference was found between two sampled frames.",
                    "This may represent normal camera motion, a scene cut, an editing transition, or another visual change.",
                    {
                        "perceptual_hash_distance": hash_distance,
                        "normalized_mean_absolute_difference": mean_abs_diff,
                    },
                )
            )
            next_id += 1

        if from_frame and to_frame:
            brightness_jump = abs(
                to_frame["brightness_mean"] - from_frame["brightness_mean"]
            )
            if brightness_jump >= BRIGHTNESS_JUMP_THRESHOLD:
                observations.append(
                    _observation(
                        next_id,
                        "significant_brightness_jump",
                        comparison["start_timestamp_seconds"],
                        comparison["end_timestamp_seconds"],
                        "A significant brightness change was found between two sampled frames.",
                        "This may represent normal lighting changes, exposure changes, a flash, a scene transition, or camera motion.",
                        {"brightness_difference": round(brightness_jump, 3)},
                    )
                )
                next_id += 1

    for failed_timestamp in failed_timestamps:
        observations.append(
            _observation(
                next_id,
                "frame_decoding_failed",
                failed_timestamp,
                None,
                "Frame decoding failed at a requested timestamp.",
                "This can happen with damaged media, unsupported codecs, keyframe seeking behavior, or end-of-file seeking.",
                {"requested_timestamp_seconds": failed_timestamp},
            )
        )
        next_id += 1

    return observations


def _observation(
    number: int,
    evidence_type: str,
    timestamp_start: float | None,
    timestamp_end: float | None,
    description: str,
    interpretation: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "observation_id": f"temporal-{number:03d}",
        "type": evidence_type,
        "severity": "info",
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "description": description,
        "interpretation": interpretation,
        "metrics": metrics,
    }
