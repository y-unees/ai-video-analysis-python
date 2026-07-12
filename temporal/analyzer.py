from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from config import (
    JPEG_QUALITY,
    MAX_SCENE_REPRESENTATIVE_FRAMES,
    SCENE_REPRESENTATIVE_FRAMES_PER_SCENE,
    temporal_configuration,
)
from file_utils import artifact_record, format_file_size
from temporal.decoder import decode_temporal_samples
from temporal.interval_detection import detect_static_intervals
from temporal.metrics import calculate_transition
from temporal.notability import rank_notable_transitions
from temporal.scene_detection import construct_scenes


TEMPORAL_LIMITATIONS = [
    "Temporal analysis uses bounded low-resolution sequential sampling.",
    "Scene-boundary candidates are substantial visual transitions, not proof of editing or manipulation.",
    "Optical-flow metrics are affected by camera motion, zoom, stabilization, blur, compression, object motion, and lighting changes.",
    "Sustained near-static intervals may reflect static camera, still subjects, low motion, intentional freeze frames, repeated content, or normal recording behavior.",
]


def analyze_temporal(
    video_path: Path,
    analysis_dir: Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    video_metadata = metadata.get("video", {})
    if not video_metadata.get("present"):
        return _skipped_result("no_valid_video_stream"), warnings

    duration = (
        metadata.get("container", {}).get("duration_seconds")
        or video_metadata.get("duration_seconds")
    )
    source_fps = video_metadata.get("frame_rate_decimal")

    samples, runtime_config, decode_warnings = decode_temporal_samples(
        video_path,
        duration,
        source_fps,
    )
    warnings.extend(decode_warnings)
    configuration = temporal_configuration()
    configuration.update(runtime_config)

    if not samples:
        result = _base_result("failed", configuration)
        result["reason"] = "zero_temporal_frames_decoded"
        result["summary"] = _summary([], [], [], [], configuration)
        result["limitations"] = TEMPORAL_LIMITATIONS
        result["observations"].append(_observation("temporal.insufficient_frames", "No temporal frames were decoded for sequential analysis.", "This may happen with unsupported, damaged, or partially readable media."))
        return result, warnings

    transitions = [
        calculate_transition(samples[index - 1], samples[index])
        for index in range(1, len(samples))
    ]
    scenes = construct_scenes(samples, transitions)
    _attach_scene_indices(transitions, scenes)
    intervals = detect_static_intervals(transitions)
    notable_transitions_full = rank_notable_transitions(transitions)
    transition_artifacts, artifact_warnings = write_notable_transition_artifacts(
        analysis_dir / "transition_frames",
        notable_transitions_full,
    )
    warnings.extend(artifact_warnings)
    for transition in notable_transitions_full:
        transition["artifact_status"] = transition_artifacts.get(
            transition["transition_id"],
            {},
        ).get("status", "missing")
        transition["artifacts"] = transition_artifacts.get(
            transition["transition_id"],
            {},
        ).get("artifacts", {})
    scene_frames, scene_frame_warnings = extract_scene_representative_frames(
        video_path,
        analysis_dir / "scene_frames",
        scenes,
    )
    warnings.extend(scene_frame_warnings)
    temporal_metrics_artifact = write_temporal_metrics(analysis_dir, transitions)
    notable_transitions = _notable_transitions(notable_transitions_full)
    coverage = _coverage(samples, metadata)
    observations = _observations(transitions, intervals, configuration, notable_transitions)
    status = "completed" if len(samples) >= 2 else "partial"
    reason = None if len(samples) >= 2 else "insufficient_frames_for_temporal_comparison"

    result = _base_result(status, configuration)
    result.update(
        {
            "reason": reason,
            "summary": _summary(samples, transitions, scenes, intervals, configuration),
            "coverage": coverage,
            "scenes": scenes,
            "scene_representative_frames": scene_frames,
            "notable_intervals": intervals,
            "notable_transitions": notable_transitions,
            "observations": observations,
            "limitations": TEMPORAL_LIMITATIONS,
            "artifacts": {
                "temporal_metrics_artifact": temporal_metrics_artifact,
                "scene_frames_directory": "scene_frames",
                "transition_frames_directory": "transition_frames",
            },
        }
    )
    return result, warnings


def write_temporal_metrics(
    analysis_dir: Path,
    transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    path = analysis_dir / "temporal_metrics.jsonl"
    sha256 = hashlib.sha256()
    size_bytes = 0
    with path.open("wb") as file:
        for transition in transitions:
            line = (json.dumps(_public_transition(transition), sort_keys=True) + "\n").encode("utf-8")
            file.write(line)
            sha256.update(line)
            size_bytes += len(line)
    return {
        "path": "temporal_metrics.jsonl",
        "size_bytes": size_bytes,
        "size_human_readable": format_file_size(size_bytes),
        "sha256": sha256.hexdigest(),
    }


def write_notable_transition_artifacts(
    output_dir: Path,
    notable_transitions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    import cv2

    warnings: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for display_index, transition in enumerate(notable_transitions, start=1):
        diagnostics = transition.get("_diagnostics", {})
        timestamp = transition["start_timestamp_seconds"]
        prefix = f"transition_{display_index:03d}"
        artifact_specs = {
            "before_frame": (f"{prefix}_before_{timestamp:.3f}s.jpg", diagnostics.get("previous_gray")),
            "after_frame": (f"{prefix}_after_{transition['end_timestamp_seconds']:.3f}s.jpg", diagnostics.get("current_gray")),
            "absolute_difference": (f"{prefix}_absolute_difference.png", diagnostics.get("absolute_difference_image")),
            "flow_warp_residual": (f"{prefix}_flow_warp_residual.png", diagnostics.get("flow_warp_residual_image")),
        }
        artifacts: dict[str, Any] = {}
        failed = False
        for key, (filename, image) in artifact_specs.items():
            if image is None:
                failed = True
                warnings.append(f"Missing diagnostic image for notable transition artifact: {key}.")
                continue
            path = output_dir / filename
            params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY] if filename.endswith(".jpg") else []
            if not cv2.imwrite(str(path), image, params):
                failed = True
                warnings.append(f"Could not write notable transition artifact {filename}.")
                continue
            artifacts[key] = artifact_record(path, f"transition_frames/{filename}")
        records[transition["transition_id"]] = {
            "status": "partial" if failed else "completed",
            "artifacts": artifacts,
        }
    return records, warnings


def extract_scene_representative_frames(
    video_path: Path,
    output_dir: Path,
    scenes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    import cv2

    warnings: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], ["OpenCV could not open the selected video for scene representative frames."]

    records: list[dict[str, Any]] = []
    seen_timestamps: set[float] = set()
    try:
        for scene in scenes:
            if len(records) >= MAX_SCENE_REPRESENTATIVE_FRAMES:
                break
            for label, timestamp in _scene_timestamps(scene):
                if len(records) >= MAX_SCENE_REPRESENTATIVE_FRAMES:
                    break
                rounded_timestamp = round(timestamp, 3)
                if rounded_timestamp in seen_timestamps:
                    continue
                seen_timestamps.add(rounded_timestamp)
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
                ok, frame = capture.read()
                if not ok or frame is None:
                    warnings.append(
                        f"Scene representative frame decode failed at {timestamp:.3f} seconds."
                    )
                    continue
                decoded_timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                name = f"scene_{scene['scene_index']:03d}_{label}_{decoded_timestamp:.3f}s.jpg"
                path = output_dir / name
                saved = cv2.imwrite(
                    str(path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                )
                if not saved:
                    warnings.append(f"Could not write scene representative frame {name}.")
                    continue
                records.append(
                    {
                        "scene_index": scene["scene_index"],
                        "position": label,
                        "requested_timestamp_seconds": round(timestamp, 3),
                        "decoded_timestamp_seconds": round(decoded_timestamp, 3),
                        "artifact_path": f"scene_frames/{name}",
                    }
                )
    finally:
        capture.release()
    return records, warnings


def _scene_timestamps(scene: dict[str, Any]) -> list[tuple[str, float]]:
    start = scene["start_timestamp_seconds"]
    end = scene["end_timestamp_seconds"]
    duration = max(0.0, end - start)
    if SCENE_REPRESENTATIVE_FRAMES_PER_SCENE <= 1 or duration < 0.2:
        return [("middle", start + duration / 2)]
    return [
        ("start", start),
        ("middle", start + duration / 2),
        ("end", max(start, end - 0.05)),
    ][:SCENE_REPRESENTATIVE_FRAMES_PER_SCENE]


def _summary(
    samples: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    flow_values = [
        transition["optical_flow"]["mean_magnitude"]
        for transition in transitions
        if transition["optical_flow"]["mean_magnitude"] is not None
    ]
    residual_values = [
        transition["flow_warp_residual"]["mean_normalized_residual"]
        for transition in transitions
        if transition["flow_warp_residual"]["mean_normalized_residual"] is not None
    ]
    return {
        "temporal_frames_analyzed": len(samples),
        "transitions_analyzed": len(transitions),
        "scene_count": len(scenes),
        "scene_boundary_candidate_count": sum(
            1
            for transition in transitions
            if transition["classification"]["scene_boundary_candidate"]
        ),
        "sustained_near_static_interval_count": len(intervals),
        "motion_summary_available": bool(flow_values),
        "average_flow_magnitude": _average(flow_values),
        "maximum_flow_magnitude": max(flow_values) if flow_values else None,
        "notable_transition_count": sum(
            1
            for transition in transitions
            if transition["classification"].get("ranked_notable_transition")
        ),
        "average_flow_warp_residual": _average(residual_values),
        "maximum_flow_warp_residual": max(residual_values) if residual_values else None,
        "requested_analysis_fps": configuration["requested_analysis_fps"],
        "effective_analysis_fps": configuration["effective_analysis_fps"],
    }


def _notable_transitions(transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _notable_transition_view(index, transition)
        for index, transition in enumerate(transitions)
    ]


def _notable_transition_view(index: int, transition: dict[str, Any]) -> dict[str, Any]:
    return {
        "notable_transition_index": index,
        "transition_id": transition["transition_id"],
        "from_sample_index": transition["from_sample_index"],
        "to_sample_index": transition["to_sample_index"],
        "scene_index": transition.get("scene_index"),
        "start_timestamp_seconds": transition["start_timestamp_seconds"],
        "end_timestamp_seconds": transition["end_timestamp_seconds"],
        "metrics": {
            "perceptual_hash_distance": transition["perceptual_hash_distance"],
            "normalized_mean_absolute_difference": transition["normalized_mean_absolute_difference"],
            "histogram_correlation": transition["histogram_correlation"],
            "absolute_brightness_difference": transition["absolute_brightness_difference"],
        },
        "optical_flow": transition["optical_flow"],
        "flow_warp_residual": transition["flow_warp_residual"],
        "classification": transition["classification"],
        "notability": transition["notability"],
        "artifact_status": transition.get("artifact_status", "missing"),
        "artifacts": transition.get("artifacts", {}),
        "interpretation": "This transition ranked highly across one or more temporal-change measurements. It is provided for review and is not proof of editing, manipulation, or AI generation.",
    }


def _observations(
    transitions: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    configuration: dict[str, Any],
    notable_transitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    if configuration["effective_analysis_fps"] < configuration["requested_analysis_fps"]:
        observations.append(_observation("temporal-reduced-sampling-001", "temporal.reduced_sampling_rate", "Temporal analysis FPS was reduced to stay within the configured frame cap.", "This limits temporal resolution but keeps processing bounded."))
    scene_count = 1
    for transition in transitions:
        if transition["classification"]["scene_boundary_candidate"]:
            observations.append(_localized_observation(f"temporal-scene-boundary-{scene_count:03d}", "temporal.scene_boundary_candidate", transition["start_timestamp_seconds"], transition["end_timestamp_seconds"], "A substantial visual transition consistent with a possible scene boundary was detected.", "This may represent a normal scene cut, rapid camera movement, exposure change, editing transition, or another visual change. It is not proof of manipulation."))
            scene_count += 1
    for index, transition in enumerate(notable_transitions, start=1):
        observations.append(_localized_observation(f"temporal-notable-transition-{index:03d}", "temporal.ranked_notable_transition", transition["start_timestamp_seconds"], transition["end_timestamp_seconds"], "This transition ranked highly across one or more visual-change measurements.", "The transition is provided for review because it contains notable visual change relative to this video. It may reflect normal motion, lighting change, a scene transition, editing, compression, or another temporal event. It is not proof of manipulation or AI generation."))
    for index, interval in enumerate(intervals, start=1):
        observations.append(_localized_observation(f"temporal-near-static-{index:03d}", "temporal.sustained_near_static_interval", interval["start_timestamp_seconds"], interval["end_timestamp_seconds"], "Visual content remained nearly unchanged during this sampled interval.", "This may represent a static shot, low motion, an intentional freeze frame, repeated content, or normal recording behavior."))
    return observations


def _observation(observation_id: str, observation_type: str, description: str, interpretation: str) -> dict[str, Any]:
    return {
        "observation_id": observation_id,
        "type": observation_type,
        "severity": "info",
        "description": description,
        "interpretation": interpretation,
    }


def _localized_observation(
    observation_id: str,
    observation_type: str,
    start: float,
    end: float,
    description: str,
    interpretation: str,
) -> dict[str, Any]:
    observation = _observation(observation_id, observation_type, description, interpretation)
    observation["timestamp_start"] = start
    observation["timestamp_end"] = end
    return observation


def _base_result(status: str, configuration: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": status,
        "configuration": configuration,
        "summary": {},
        "coverage": {},
        "scenes": [],
        "scene_representative_frames": [],
        "notable_intervals": [],
        "notable_transitions": [],
        "observations": [],
        "limitations": [],
        "artifacts": {},
    }


def _skipped_result(reason: str) -> dict[str, Any]:
    result = _base_result("skipped", temporal_configuration())
    result["reason"] = reason
    result["limitations"] = TEMPORAL_LIMITATIONS
    result["summary"] = _summary([], [], [], [], result["configuration"])
    return result


def _average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _attach_scene_indices(
    transitions: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
) -> None:
    for transition in transitions:
        transition["scene_index"] = _scene_index_for_transition(
            transition["start_timestamp_seconds"],
            scenes,
        )


def _scene_index_for_transition(timestamp: float, scenes: list[dict[str, Any]]) -> int | None:
    for scene in scenes:
        if scene["start_timestamp_seconds"] <= timestamp <= scene["end_timestamp_seconds"]:
            return scene["scene_index"]
    return scenes[-1]["scene_index"] if scenes else None


def _coverage(samples: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    if not samples:
        return {
            "coverage_timeline_basis": "selected_video_stream",
            "first_sample_timestamp_seconds": None,
            "last_sample_timestamp_seconds": None,
            "source_video_start_seconds": None,
            "source_video_end_seconds": None,
            "analyzed_time_span_seconds": None,
            "source_time_span_seconds": None,
            "coverage_ratio": None,
        }
    first = samples[0]["sample"]["timestamp_seconds"]
    last = samples[-1]["sample"]["timestamp_seconds"]
    video = metadata.get("video", {})
    source_start = video.get("start_time")
    source_duration = video.get("duration_seconds") or metadata.get("container", {}).get("duration_seconds")
    source_end = (
        round(source_start + source_duration, 6)
        if source_start is not None and source_duration is not None
        else None
    )
    analyzed_span = max(0.0, last - first)
    source_span = source_duration if source_duration is not None else None
    ratio = (
        max(0.0, min(1.0, analyzed_span / source_span))
        if source_span and source_span > 0
        else None
    )
    return {
        "coverage_timeline_basis": "selected_video_stream",
        "first_sample_timestamp_seconds": first,
        "last_sample_timestamp_seconds": last,
        "source_video_start_seconds": source_start,
        "source_video_end_seconds": source_end,
        "analyzed_time_span_seconds": round(analyzed_span, 6),
        "source_time_span_seconds": round(source_span, 6) if source_span is not None else None,
        "coverage_ratio": round(ratio, 6) if ratio is not None else None,
    }


def _public_transition(transition: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in transition.items()
        if key != "_diagnostics"
    }
