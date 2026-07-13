from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from file_utils import format_bit_rate, format_duration


class FFprobeError(Exception):
    """Raised when ffprobe cannot return usable metadata."""


def extract_metadata(video_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    raw_metadata = _run_ffprobe(video_path)
    metadata = parse_metadata(raw_metadata)
    evidence = build_metadata_evidence(metadata)
    return raw_metadata, metadata, evidence


def _run_ffprobe(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
    except OSError as error:
        raise FFprobeError(str(error)) from error

    if result.returncode != 0:
        message = result.stderr.strip() or "ffprobe returned a non-zero exit code."
        raise FFprobeError(message)

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise FFprobeError("ffprobe returned malformed JSON output.") from error

    if not isinstance(parsed, dict):
        raise FFprobeError("ffprobe returned an unexpected JSON structure.")

    return parsed


def parse_metadata(raw_metadata: dict[str, Any]) -> dict[str, Any]:
    streams = _safe_list(raw_metadata.get("streams"))
    format_info = _safe_dict(raw_metadata.get("format"))
    format_tags = _safe_dict(format_info.get("tags"))

    video_stream, video_selection_reason = _select_video_stream(streams)
    audio_stream, audio_selection_reason = _select_audio_stream(streams)

    container_duration = _to_float(format_info.get("duration"))
    video_start = _to_float(video_stream.get("start_time")) if video_stream else None
    audio_start = _to_float(audio_stream.get("start_time")) if audio_stream else None
    video_duration = _to_float(video_stream.get("duration")) if video_stream else None
    audio_duration = _to_float(audio_stream.get("duration")) if audio_stream else None

    video = _build_video_metadata(video_stream, container_duration, video_selection_reason)
    audio = _build_audio_metadata(audio_stream, audio_selection_reason)

    return {
        "container": {
            "duration_seconds": container_duration,
            "duration_readable": format_duration(container_duration),
            "format_name": format_info.get("format_name"),
            "format_long_name": format_info.get("format_long_name"),
            "friendly_format": _friendly_container(format_info.get("format_name")),
            "start_time": _to_float(format_info.get("start_time")),
            "bit_rate": _to_int(format_info.get("bit_rate")),
            "bit_rate_readable": format_bit_rate(_to_int(format_info.get("bit_rate"))),
            "probe_score": _to_int(format_info.get("probe_score")),
            "stream_count": len(streams),
            "tags": format_tags,
        },
        "video": video,
        "audio": audio,
        "duration_comparison": {
            "container_duration_seconds": container_duration,
            "video_start_time_seconds": video_start,
            "video_duration_seconds": video_duration,
            "video_end_time_seconds": _end_time(video_start, video_duration),
            "audio_start_time_seconds": audio_start,
            "audio_duration_seconds": audio_duration,
            "audio_end_time_seconds": _end_time(audio_start, audio_duration),
            "duration_only_difference_seconds": _duration_difference(audio_duration, video_duration),
            "start_time_difference_seconds": _duration_difference(audio_start, video_start),
            "end_time_difference_seconds": _duration_difference(
                _end_time(audio_start, audio_duration),
                _end_time(video_start, video_duration),
            ),
        },
        "encoding": {
            "container_encoder": format_tags.get("encoder"),
            "video_stream_encoder": _tag(video_stream, "encoder"),
            "audio_stream_encoder": _tag(audio_stream, "encoder"),
            "video_handler_name": _tag(video_stream, "handler_name"),
            "audio_handler_name": _tag(audio_stream, "handler_name"),
            "container_vendor_id": format_tags.get("vendor_id"),
            "video_vendor_id": _tag(video_stream, "vendor_id"),
            "audio_vendor_id": _tag(audio_stream, "vendor_id"),
        },
        "display": _build_display_metadata(video_stream),
    }


def _build_video_metadata(
    video_stream: dict[str, Any] | None,
    container_duration: float | None,
    selection_reason: str | None,
) -> dict[str, Any]:
    if not video_stream:
        return {
            "present": False,
            "selected_stream_index": None,
            "selection_reason": selection_reason or "no_video_stream_found",
            "index": None,
            "codec_name": None,
            "codec_long_name": None,
            "codec_profile": None,
            "codec_tag": None,
            "width": None,
            "height": None,
            "coded_width": None,
            "coded_height": None,
            "pixel_format": None,
            "codec_level_raw": None,
            "codec_level_readable": None,
            "average_frame_rate_raw": None,
            "nominal_frame_rate_raw": None,
            "frame_rate_decimal": None,
            "time_base": None,
            "start_time": None,
            "duration_seconds": None,
            "duration_readable": None,
            "frame_count": {
                "reported_by_stream": None,
                "estimated_from_duration": None,
                "difference": None,
            },
            "bit_rate": None,
            "bit_rate_readable": None,
            "sample_aspect_ratio": None,
            "display_aspect_ratio": None,
            "field_order": None,
            "color_range": None,
            "color_space": None,
            "color_transfer": None,
            "color_primaries": None,
            "rotation": None,
            "tags": {},
            "disposition": {},
        }

    avg_frame_rate = video_stream.get("avg_frame_rate")
    frame_rate_decimal = _parse_frame_rate(avg_frame_rate)
    stream_duration = _to_float(video_stream.get("duration")) or container_duration
    reported_count = _to_int(video_stream.get("nb_frames"))
    estimated_count = (
        round(stream_duration * frame_rate_decimal)
        if stream_duration is not None and frame_rate_decimal is not None
        else None
    )
    difference = (
        abs(reported_count - estimated_count)
        if reported_count is not None and estimated_count is not None
        else None
    )
    bit_rate = _to_int(video_stream.get("bit_rate"))
    codec_level_raw = _to_int(video_stream.get("level"))

    return {
        "present": True,
        "selected_stream_index": _to_int(video_stream.get("index")),
        "selection_reason": selection_reason,
        "index": _to_int(video_stream.get("index")),
        "codec_name": video_stream.get("codec_name"),
        "codec_long_name": video_stream.get("codec_long_name"),
        "codec_profile": video_stream.get("profile"),
        "codec_tag": video_stream.get("codec_tag_string") or video_stream.get("codec_tag"),
        "width": _to_int(video_stream.get("width")),
        "height": _to_int(video_stream.get("height")),
        "coded_width": _to_int(video_stream.get("coded_width")),
        "coded_height": _to_int(video_stream.get("coded_height")),
        "pixel_format": video_stream.get("pix_fmt"),
        "codec_level_raw": codec_level_raw,
        "codec_level_readable": _codec_level_readable(codec_level_raw),
        "average_frame_rate_raw": avg_frame_rate,
        "nominal_frame_rate_raw": video_stream.get("r_frame_rate"),
        "frame_rate_decimal": _round(frame_rate_decimal),
        "time_base": video_stream.get("time_base"),
        "start_time": _to_float(video_stream.get("start_time")),
        "duration_seconds": _to_float(video_stream.get("duration")),
        "duration_readable": format_duration(_to_float(video_stream.get("duration"))),
        "frame_count": {
            "reported_by_stream": reported_count,
            "estimated_from_duration": estimated_count,
            "difference": difference,
        },
        "bit_rate": bit_rate,
        "bit_rate_readable": format_bit_rate(bit_rate),
        "sample_aspect_ratio": video_stream.get("sample_aspect_ratio"),
        "display_aspect_ratio": video_stream.get("display_aspect_ratio"),
        "field_order": video_stream.get("field_order"),
        "color_range": video_stream.get("color_range"),
        "color_space": video_stream.get("color_space"),
        "color_transfer": video_stream.get("color_transfer"),
        "color_primaries": video_stream.get("color_primaries"),
        "rotation": _extract_rotation(video_stream),
        "tags": _safe_dict(video_stream.get("tags")),
        "disposition": _safe_dict(video_stream.get("disposition")),
    }


def _build_audio_metadata(audio_stream: dict[str, Any] | None, selection_reason: str | None) -> dict[str, Any]:
    if not audio_stream:
        return {
            "present": False,
            "selected_stream_index": None,
            "selection_reason": selection_reason or "no_audio_stream_found",
            "index": None,
            "codec_name": None,
            "codec_long_name": None,
            "codec_profile": None,
            "sample_format": None,
            "sample_rate": None,
            "channels": None,
            "channel_layout": None,
            "time_base": None,
            "start_time": None,
            "duration_seconds": None,
            "duration_readable": None,
            "bit_rate": None,
            "bit_rate_readable": None,
            "tags": {},
            "disposition": {},
        }

    bit_rate = _to_int(audio_stream.get("bit_rate"))
    duration = _to_float(audio_stream.get("duration"))
    return {
        "present": True,
        "selected_stream_index": _to_int(audio_stream.get("index")),
        "selection_reason": selection_reason,
        "index": _to_int(audio_stream.get("index")),
        "codec_name": audio_stream.get("codec_name"),
        "codec_long_name": audio_stream.get("codec_long_name"),
        "codec_profile": audio_stream.get("profile"),
        "sample_format": audio_stream.get("sample_fmt"),
        "sample_rate": _to_int(audio_stream.get("sample_rate")),
        "channels": _to_int(audio_stream.get("channels")),
        "channel_layout": audio_stream.get("channel_layout"),
        "time_base": audio_stream.get("time_base"),
        "start_time": _to_float(audio_stream.get("start_time")),
        "duration_seconds": duration,
        "duration_readable": format_duration(duration),
        "bit_rate": bit_rate,
        "bit_rate_readable": format_bit_rate(bit_rate),
        "tags": _safe_dict(audio_stream.get("tags")),
        "disposition": _safe_dict(audio_stream.get("disposition")),
    }


def _build_display_metadata(video_stream: dict[str, Any] | None) -> dict[str, Any]:
    if not video_stream:
        return {
            "orientation": None,
            "width": None,
            "height": None,
            "rotation": None,
            "sample_aspect_ratio": None,
            "display_aspect_ratio": None,
        }

    width = _to_int(video_stream.get("width"))
    height = _to_int(video_stream.get("height"))
    rotation = _extract_rotation(video_stream)
    display_width, display_height = width, height
    if rotation in {90, 270} and width and height:
        display_width, display_height = height, width

    return {
        "orientation": _orientation(display_width, display_height),
        "width": display_width,
        "height": display_height,
        "rotation": rotation,
        "sample_aspect_ratio": video_stream.get("sample_aspect_ratio"),
        "display_aspect_ratio": video_stream.get("display_aspect_ratio"),
    }


def build_metadata_evidence(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    audio = metadata["audio"]
    video = metadata["video"]
    encoding = metadata["encoding"]
    container = metadata["container"]

    if audio["present"]:
        facts.append(_observation("metadata-fact-001", "audio_stream_present", "The video contains an audio stream."))
    else:
        missing.append(_observation("missing-metadata-001", "audio_stream_missing", "No audio stream was found."))

    encoder_values = [
        encoding.get("container_encoder"),
        encoding.get("video_stream_encoder"),
        encoding.get("audio_stream_encoder"),
    ]
    if any(encoder_values):
        facts.append(_observation("metadata-fact-002", "encoder_tag_present", "At least one explicit encoder tag was found."))
    else:
        missing.append(_observation("missing-metadata-002", "encoder_tag_missing", "No explicit encoder tag was found in the inspected container or stream metadata."))

    tags = container.get("tags") if isinstance(container.get("tags"), dict) else {}
    if tags.get("creation_time") or video.get("tags", {}).get("creation_time") or audio.get("tags", {}).get("creation_time"):
        facts.append(_observation("metadata-fact-003", "creation_time_present", "Creation-time metadata was present."))
    else:
        missing.append(_observation("missing-metadata-003", "creation_time_missing", "Creation-time metadata was missing."))

    if _looks_unusual_frame_rate(video.get("average_frame_rate_raw")):
        facts.append(_observation("metadata-fact-004", "unusual_frame_rate_value", "The video uses a variable-looking or unusual frame-rate value."))

    if container.get("stream_count", 0) > 1:
        facts.append(_observation("metadata-fact-005", "multiple_streams", "The file contains multiple streams."))

    codec = video.get("codec_name")
    if codec == "h264":
        facts.append(_observation("metadata-fact-006", "h264_video_codec", "The video stream codec is H.264."))
    elif codec:
        facts.append(_observation("metadata-fact-006", "video_codec_present", f"The video stream codec is {codec}."))
    else:
        missing.append(_observation("missing-metadata-004", "video_codec_missing", "No video codec metadata was found."))

    return {"metadata_facts": facts, "missing_metadata": missing, "temporal_heuristics": []}


def _observation(observation_id: str, observation_type: str, description: str) -> dict[str, Any]:
    return {
        "observation_id": observation_id,
        "type": observation_type,
        "severity": "info",
        "conclusion_scope": "non_conclusive_observation",
        "supports_authenticity_verdict": False,
        "description": description,
        "interpretation": "This is a factual metadata observation and is not proof of editing or tampering.",
        "metrics": {},
    }


def _select_video_stream(streams: list[Any]) -> tuple[dict[str, Any] | None, str]:
    candidates = [
        _safe_dict(stream)
        for stream in streams
        if _safe_dict(stream).get("codec_type") == "video"
        and _safe_dict(stream).get("disposition", {}).get("attached_pic") != 1
    ]
    if not candidates:
        return None, "no_non_attached_video_stream_found"
    for stream in candidates:
        if _safe_dict(stream.get("disposition")).get("default") == 1:
            return stream, "default_non_attached_video_stream"
    return candidates[0], "first_non_attached_video_stream"


def _select_audio_stream(streams: list[Any]) -> tuple[dict[str, Any] | None, str]:
    candidates = [
        _safe_dict(stream)
        for stream in streams
        if _safe_dict(stream).get("codec_type") == "audio"
    ]
    if not candidates:
        return None, "no_audio_stream_found"
    for stream in candidates:
        if _safe_dict(stream.get("disposition")).get("default") == 1:
            return stream, "default_audio_stream"
    return candidates[0], "first_audio_stream"


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _parse_frame_rate(value: Any) -> float | None:
    if not isinstance(value, str) or value in {"", "0/0"}:
        return None
    try:
        frame_rate = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None
    return frame_rate if frame_rate > 0 else None


def _looks_unusual_frame_rate(value: Any) -> bool:
    if not isinstance(value, str) or value in {"", "0/0"}:
        return True
    frame_rate = _parse_frame_rate(value)
    if frame_rate is None:
        return True
    common_rates = (23.976, 24, 25, 29.97, 30, 50, 59.94, 60)
    return all(abs(frame_rate - common_rate) > 0.05 for common_rate in common_rates)


def _extract_rotation(video_stream: dict[str, Any]) -> int | None:
    tags = _safe_dict(video_stream.get("tags"))
    side_data = _safe_list(video_stream.get("side_data_list"))
    rotation = _to_int(tags.get("rotate"))
    if rotation is not None:
        return rotation % 360
    for item in side_data:
        item_dict = _safe_dict(item)
        rotation = _to_int(item_dict.get("rotation"))
        if rotation is not None:
            return rotation % 360
        matrix_rotation = _to_float(item_dict.get("displaymatrix_rotation"))
        if matrix_rotation is not None:
            return int(round(matrix_rotation)) % 360
    return None


def _tag(stream: dict[str, Any] | None, tag_name: str) -> Any:
    if not stream:
        return None
    return _safe_dict(stream.get("tags")).get(tag_name)


def _friendly_container(format_name: Any) -> str | None:
    if not isinstance(format_name, str):
        return None
    parts = {part.strip() for part in format_name.split(",")}
    if {"mov", "mp4"} & parts:
        return "MP4/MOV family"
    if "matroska" in parts or "webm" in parts:
        return "Matroska/WebM family"
    if "avi" in parts:
        return "AVI"
    return format_name


def _duration_difference(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return round(abs(first - second), 6)


def _end_time(start: float | None, duration: float | None) -> float | None:
    if start is None or duration is None:
        return None
    return round(start + duration, 6)


def _codec_level_readable(level: int | None) -> str | None:
    if level is None:
        return None
    if 10 <= level <= 999:
        return f"{level / 10:.1f}"
    return str(level)


def _orientation(width: int | None, height: int | None) -> str | None:
    if width is None or height is None:
        return None
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"
