from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config import SCHEMA_VERSION, heuristic_configuration
from file_utils import calculate_sha256_bytes, format_file_size
from report_validator import validate_report


LIMITATIONS = [
    "Metadata and representative-frame inspection are implemented.",
    "Only a subset of video frames is inspected.",
    "Scene-change heuristics are not proof of editing or tampering.",
    "No AI-generation classifier is implemented.",
    "No deepfake detector is implemented.",
    "No face analysis is implemented.",
    "No audio-signal analysis is implemented yet.",
    "No semantic-content analysis is implemented.",
    "Compressed or damaged videos may reduce reliability.",
]


def create_analysis_directory(
    selected_video: Path,
    reports_dir: Path,
    timestamp: str,
) -> Path:
    safe_stem = _safe_filename_stem(selected_video.stem)
    analysis_dir = reports_dir / f"{safe_stem}_{timestamp}"
    analysis_dir.mkdir(parents=True, exist_ok=False)
    (analysis_dir / "frames").mkdir(exist_ok=True)
    return analysis_dir


def write_raw_ffprobe(analysis_dir: Path, raw_ffprobe: dict[str, Any]) -> dict[str, Any]:
    raw_path = analysis_dir / "ffprobe_raw.json"
    raw_bytes = json.dumps(raw_ffprobe, indent=2).encode("utf-8")
    raw_path.write_bytes(raw_bytes)
    return {
        "path": "ffprobe_raw.json",
        "size_bytes": len(raw_bytes),
        "size_human_readable": format_file_size(len(raw_bytes)),
        "sha256": calculate_sha256_bytes(raw_bytes),
    }


def write_reports(
    analysis_dir: Path,
    report: dict[str, Any],
) -> dict[str, Path]:
    validation = validate_report(report)
    report["validation"] = validation
    if validation["warnings"]:
        report["warnings"].extend(validation["warnings"])
    if validation["errors"]:
        report["analysis"]["status"] = "failed_validation"
    elif report["warnings"] and report["analysis"]["status"] == "completed":
        report["analysis"]["status"] = "completed_with_warnings"

    json_path = analysis_dir / "report.json"
    txt_path = analysis_dir / "report.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    txt_path.write_text(_build_text_report(report), encoding="utf-8")
    return {"json": json_path, "txt": txt_path, "directory": analysis_dir}


def build_report(
    analysis: dict[str, Any],
    analysis_environment: dict[str, Any],
    source: dict[str, Any],
    metadata: dict[str, Any],
    sampling: dict[str, Any],
    frame_analysis: dict[str, Any],
    evidence: list[dict[str, Any]],
    warnings: list[str],
    observations: dict[str, list[dict[str, Any]]] | None = None,
    raw_ffprobe_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_observations = observations or {
        "metadata_facts": [],
        "missing_metadata": [],
        "temporal_heuristics": evidence,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": analysis,
        "analysis_environment": analysis_environment,
        "source": source,
        "metadata": metadata,
        "sampling": sampling,
        "frame_analysis": frame_analysis,
        "heuristic_configuration": heuristic_configuration(),
        "observations": final_observations,
        "evidence": evidence,
        "warnings": warnings,
        "limitations": LIMITATIONS,
        "artifacts": {
            "paths_relative_to": "report_directory",
            "raw_ffprobe_report": "ffprobe_raw.json",
            "raw_ffprobe_report_artifact": raw_ffprobe_artifact,
            "frames_directory": "frames",
        },
        "validation": {"errors": [], "warnings": []},
    }


def _build_text_report(report: dict[str, Any]) -> str:
    analysis = report["analysis"]
    source = report["source"]
    metadata = report["metadata"]
    container = metadata["container"]
    video = metadata["video"]
    audio = metadata["audio"]
    display = metadata["display"]
    duration = metadata["duration_comparison"]
    encoding = metadata["encoding"]
    sampling = report["sampling"]
    frame_summary = report["frame_analysis"]["summary"]

    lines = [
        "VIDEO ANALYSIS REPORT",
        "=====================",
        "",
        "Analysis information",
        "--------------------",
        f"Status: {analysis['status']}",
        f"Started: {analysis['started_at']}",
        f"Completed: {analysis['completed_at']}",
        f"Processing duration: {analysis['processing_duration_seconds']} seconds",
        f"Schema version: {report['schema_version']}",
        "",
        "Source file",
        "-----------",
        f"Filename: {source['filename']}",
        f"Path: {source['path']}",
        f"Extension: {source['extension']}",
        f"Size: {source['size_bytes']} bytes ({source['size_human_readable']})",
        f"SHA-256: {source['sha256']}",
        "",
        "Container metadata",
        "------------------",
        f"Format: {_format(container['format_name'])}",
        f"Friendly format: {_format(container['friendly_format'])}",
        f"Format long name: {_format(container['format_long_name'])}",
        f"Duration: {_format(container['duration_readable'])}",
        f"Start time: {_format(container['start_time'])}",
        f"Total bit rate: {_format(container['bit_rate_readable'])}",
        f"Probe score: {_format(container['probe_score'])}",
        f"Stream count: {_format(container['stream_count'])}",
        "",
        "Video stream metadata",
        "---------------------",
        f"Present: {video['present']}",
        f"Stream index: {_format(video['index'])}",
        f"Codec: {_format(video['codec_name'])}",
        f"Codec long name: {_format(video['codec_long_name'])}",
        f"Profile: {_format(video['codec_profile'])}",
        f"Dimensions: {_format(video['width'])}x{_format(video['height'])}",
        f"Coded dimensions: {_format(video['coded_width'])}x{_format(video['coded_height'])}",
        f"Pixel format: {_format(video['pixel_format'])}",
        f"Average frame rate raw: {_format(video['average_frame_rate_raw'])}",
        f"Nominal frame rate raw: {_format(video['nominal_frame_rate_raw'])}",
        f"Frame-rate decimal: {_format(video['frame_rate_decimal'])}",
        f"Codec level: {_format(video['codec_level_readable'])} (raw: {_format(video['codec_level_raw'])})",
        f"Frame count reported: {_format(video['frame_count']['reported_by_stream'])}",
        f"Frame count estimated: {_format(video['frame_count']['estimated_from_duration'])}",
        f"Frame count difference: {_format(video['frame_count']['difference'])}",
        f"Video bit rate: {_format(video['bit_rate_readable'])}",
        "",
        "Audio stream metadata",
        "---------------------",
        f"Present: {audio['present']}",
        f"Stream index: {_format(audio['index'])}",
        f"Codec: {_format(audio['codec_name'])}",
        f"Codec long name: {_format(audio['codec_long_name'])}",
        f"Sample rate: {_format(audio['sample_rate'])}",
        f"Channels: {_format(audio['channels'])}",
        f"Channel layout: {_format(audio['channel_layout'])}",
        f"Audio bit rate: {_format(audio['bit_rate_readable'])}",
        "",
        "Display information",
        "-------------------",
        f"Orientation: {_format(display['orientation'])}",
        f"Display width: {_format(display['width'])}",
        f"Display height: {_format(display['height'])}",
        f"Rotation: {_format(display['rotation'])}",
        f"Sample aspect ratio: {_format(display['sample_aspect_ratio'])}",
        f"Display aspect ratio: {_format(display['display_aspect_ratio'])}",
        "",
        "Duration comparison",
        "-------------------",
        f"Container duration seconds: {_format(duration['container_duration_seconds'])}",
        f"Video start seconds: {_format(duration['video_start_time_seconds'])}",
        f"Video duration seconds: {_format(duration['video_duration_seconds'])}",
        f"Video end seconds: {_format(duration['video_end_time_seconds'])}",
        f"Audio start seconds: {_format(duration['audio_start_time_seconds'])}",
        f"Audio duration seconds: {_format(duration['audio_duration_seconds'])}",
        f"Audio end seconds: {_format(duration['audio_end_time_seconds'])}",
        f"Duration-only difference seconds: {_format(duration['duration_only_difference_seconds'])}",
        f"Start-time difference seconds: {_format(duration['start_time_difference_seconds'])}",
        f"End-time difference seconds: {_format(duration['end_time_difference_seconds'])}",
        "",
        "Encoding information",
        "--------------------",
        f"Container encoder tag: {_format(encoding['container_encoder'])}",
        f"Video encoder tag: {_format(encoding['video_stream_encoder'])}",
        f"Audio encoder tag: {_format(encoding['audio_stream_encoder'])}",
        f"Video handler: {_format(encoding['video_handler_name'])}",
        f"Audio handler: {_format(encoding['audio_handler_name'])}",
        "",
        "Frame sampling",
        "--------------",
        f"Strategy: {sampling['strategy']}",
        f"Target frame count: {sampling['target_frame_count']}",
        f"Decoded frame count: {sampling['decoded_frame_count']}",
        f"Failed timestamps: {sampling['failed_timestamps_seconds']}",
        f"Maximum absolute seek error: {_format(sampling['maximum_absolute_seek_error_seconds'])} seconds",
        f"Average absolute seek error: {_format(sampling['average_absolute_seek_error_seconds'])} seconds",
        "",
        "Frame analysis summary",
        "----------------------",
        f"Frames analyzed: {frame_summary['frames_analyzed']}",
        f"Average brightness: {_format(frame_summary['average_brightness'])}",
        f"Minimum brightness: {_format(frame_summary['minimum_brightness'])}",
        f"Maximum brightness: {_format(frame_summary['maximum_brightness'])}",
        f"Average contrast: {_format(frame_summary['average_contrast'])}",
        f"Average Laplacian variance: {_format(frame_summary['average_laplacian_variance'])}",
        f"Minimum Laplacian variance: {_format(frame_summary['minimum_laplacian_variance'])}",
        f"Maximum Laplacian variance: {_format(frame_summary['maximum_laplacian_variance'])}",
        "Laplacian variance is an uncalibrated edge-detail measurement affected by resolution, compression, noise, resizing, and scene content.",
        f"Heuristic near-black frame count: {frame_summary['heuristic_near_black_frame_count']}",
        f"Heuristic near-white frame count: {frame_summary['heuristic_near_white_frame_count']}",
        f"Heuristic large-change pair count: {frame_summary['heuristic_large_change_pair_count']}",
        f"Heuristic near-duplicate pair count: {frame_summary['heuristic_near_duplicate_pair_count']}",
        "",
        "Sampled frame details",
        "---------------------",
    ]

    for frame in report["frame_analysis"]["frames"]:
        lines.extend(
            [
                f"Frame {frame['sample_index'] + 1}",
                f"Requested timestamp: {_format(frame['requested_timestamp_seconds'])} s",
                f"Decoded timestamp: {_format(frame['decoded_timestamp_seconds'])} s",
                f"Brightness: {_format(frame['brightness_mean'])}",
                f"Contrast: {_format(frame['contrast_stddev'])}",
                f"Seek error: {_format(frame['seek_error_seconds'])} s",
                f"Laplacian variance: {_format(frame['laplacian_variance'])}",
                f"Near-black frame heuristic: {_yes_no(frame['likely_near_black_frame'])}",
                f"Near-white frame heuristic: {_yes_no(frame['likely_near_white_frame'])}",
                f"Perceptual hash: {_format(frame['perceptual_hash'])}",
                f"Artifact: {frame['artifact_path']}",
                "",
            ]
        )

    lines.extend(["Temporal comparisons", "--------------------"])
    for comparison in report["frame_analysis"]["comparisons"]:
        lines.extend(
            [
                f"{comparison['from_sample_index']} -> {comparison['to_sample_index']}",
                f"Timestamps: {_format(comparison['start_timestamp_seconds'])} s to {_format(comparison['end_timestamp_seconds'])} s",
                f"Hash distance: {comparison['perceptual_hash_distance']}",
                f"Normalized mean absolute difference: {comparison['normalized_mean_absolute_difference']}",
                f"Histogram correlation: {comparison['histogram_correlation']}",
                f"Classification: near_duplicate={comparison['classification']['near_duplicate']}, large_change={comparison['classification']['large_change']}",
                f"Triggered rules: {comparison['classification']['triggered_rules']}",
                "",
            ]
        )

    lines.extend(["Observations", "------------"])
    for label, items in (
        ("Metadata facts", report["observations"]["metadata_facts"]),
        ("Missing metadata", report["observations"]["missing_metadata"]),
        ("Temporal heuristics", report["observations"]["temporal_heuristics"]),
    ):
        lines.append(label)
        if not items:
            lines.append("- None")
        for item in items:
            lines.append(f"- [{item['type']}] {item['description']}")
            if item.get("interpretation"):
                lines.append(f"  Interpretation: {item['interpretation']}")

    config = report["heuristic_configuration"]
    lines.extend(
        [
            "",
            "Heuristic configuration",
            "-----------------------",
            f"Perceptual hash: {config['perceptual_hash_algorithm']} size {config['perceptual_hash_size']}",
            f"Near-black threshold: {config['near_black_frame_rules']}",
            f"Near-white threshold: {config['near_white_frame_rules']}",
            f"Near-duplicate pair rules: {config['near_duplicate_pair_rules']}",
            f"Large-change pair rules: {config['large_change_pair_rules']}",
            f"Pixel difference normalization: {config['pixel_difference_normalization']}",
            f"Histogram method: {config['histogram_comparison_method']} with {config['histogram_bins']} bins",
        ]
    )

    lines.extend(["", "Warnings", "--------"])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- None")

    lines.extend(["", "Limitations", "-----------"])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])

    lines.extend(
        [
            "",
            "Generated artifacts",
            "-------------------",
            f"JSON report: report.json",
            f"TXT report: report.txt",
            f"Raw ffprobe report: {report['artifacts']['raw_ffprobe_report']}",
            f"Raw ffprobe SHA-256: {_format((report['artifacts'].get('raw_ffprobe_report_artifact') or {}).get('sha256'))}",
            f"Frames directory: {report['artifacts']['frames_directory']}",
            f"Artifact paths relative to: {report['artifacts']['paths_relative_to']}",
            "",
        ]
    )

    return "\n".join(lines)


def _format(value: Any) -> str:
    return "Not available" if value is None else str(value)


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _safe_filename_stem(stem: str) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return safe_stem or "video"
    lines.extend(["", "Report validation", "-----------------"])
    if report["validation"]["errors"]:
        lines.extend(f"- ERROR: {error}" for error in report["validation"]["errors"])
    if report["validation"]["warnings"]:
        lines.extend(f"- WARNING: {warning}" for warning in report["validation"]["warnings"])
    if not report["validation"]["errors"] and not report["validation"]["warnings"]:
        lines.append("- Passed")
