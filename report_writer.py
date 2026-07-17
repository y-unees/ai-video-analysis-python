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
    "Basic audio-signal analysis is implemented when an audio stream is available; synthetic-speech detection is not implemented.",
    "Visual consistency analysis provides regional review measurements only and does not produce authenticity, manipulation, or AI-generation verdicts.",
    "Unified evidence review priority is an ordering aid for human review and is not an AI probability or authenticity score.",
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
    validation = validate_report(report, analysis_dir)
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
    temporal_analysis: dict[str, Any],
    audio_analysis: dict[str, Any],
    evidence: list[dict[str, Any]],
    warnings: list[str],
    observations: dict[str, list[dict[str, Any]]] | None = None,
    raw_ffprobe_artifact: dict[str, Any] | None = None,
    visual_consistency_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_observations = observations or {
        "metadata_facts": [],
        "missing_metadata": [],
        "temporal_heuristics": evidence,
    }
    compatibility = {
        "legacy_evidence_view_included": True,
        "legacy_evidence_view_source": "observations",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": analysis,
        "analysis_environment": analysis_environment,
        "source": source,
        "metadata": metadata,
        "sampling": sampling,
        "frame_analysis": frame_analysis,
        "temporal_analysis": temporal_analysis,
        "audio_analysis": audio_analysis,
        "visual_consistency_analysis": visual_consistency_analysis or {
            "status": "skipped",
            "reason_code": "not_run",
            "reason": "Visual consistency analysis was not run.",
            "configuration": {},
            "summary": {"metrics_available": False},
            "transition_summaries": [],
            "sustained_intervals": [],
            "ranked_review_transitions": [],
            "observations": [],
            "limitations": [],
            "artifacts": {},
        },
        "unified_evidence": {
            "status": "pending",
            "reason_code": "generated_after_base_report",
            "reason": "Unified evidence artifacts are generated after the base report object is assembled.",
            "configuration": {},
            "summary": {},
            "timeline_configuration": {},
            "evidence_domains": {},
            "review_highlights": [],
            "ambiguous_findings": [],
            "normal_or_non_supporting_findings": [],
            "missing_evidence": [],
            "artifacts": {},
            "validation": {"errors": [], "warnings": []},
            "limitations": [],
        },
        "learned_detector_results": {
            "status": "not_run",
            "d3": {},
            "standalone_not_fused_with_unified_evidence": True,
            "limitations": [],
        },
        "heuristic_configuration": heuristic_configuration(),
        "observations": final_observations,
        "evidence": _legacy_evidence_from_observations(final_observations),
        "compatibility": compatibility,
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
    audio_metadata = metadata["audio"]
    display = metadata["display"]
    duration = metadata["duration_comparison"]
    encoding = metadata["encoding"]
    sampling = report["sampling"]
    frame_summary = report["frame_analysis"]["summary"]
    temporal = report.get("temporal_analysis", {})
    audio_analysis = report.get("audio_analysis", {})
    visual_consistency = report.get("visual_consistency_analysis", {})
    unified = report.get("unified_evidence", {})
    learned = report.get("learned_detector_results", {})

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
        f"Application version: {report['analysis_environment'].get('application_version')}",
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
        f"Present: {audio_metadata['present']}",
        f"Stream index: {_format(audio_metadata['index'])}",
        f"Codec: {_format(audio_metadata['codec_name'])}",
        f"Codec long name: {_format(audio_metadata['codec_long_name'])}",
        f"Sample rate: {_format(audio_metadata['sample_rate'])}",
        f"Channels: {_format(audio_metadata['channels'])}",
        f"Channel layout: {_format(audio_metadata['channel_layout'])}",
        f"Audio bit rate: {_format(audio_metadata['bit_rate_readable'])}",
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

    _append_temporal_report(lines, temporal)
    _append_audio_report(lines, audio_analysis)
    _append_visual_consistency_report(lines, visual_consistency)
    _append_unified_evidence_report(lines, unified)
    _append_learned_detector_report(lines, learned)

    lines.extend(["Observations", "------------"])
    for label, items in (
        ("Metadata facts", report["observations"]["metadata_facts"]),
        ("Missing metadata", report["observations"]["missing_metadata"]),
        ("Temporal heuristics", report["observations"]["temporal_heuristics"]),
        ("Audio observations", report["observations"].get("audio_observations", [])),
        ("Visual consistency observations", report["observations"].get("visual_consistency_observations", [])),
    ):
        lines.append(label)
        if not items:
            lines.append("- None")
        for item in items:
            lines.append(f"- {item.get('observation_id', 'no-id')} [{item['type']}] {item['description']}")
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

    lines.extend(["", "Report validation", "-----------------"])
    if report["validation"]["errors"]:
        lines.extend(f"- ERROR: {error}" for error in report["validation"]["errors"])
    if report["validation"]["warnings"]:
        lines.extend(f"- WARNING: {warning}" for warning in report["validation"]["warnings"])
    if not report["validation"]["errors"] and not report["validation"]["warnings"]:
        lines.append("- Passed")

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
            f"Legacy evidence view included: {report['compatibility']['legacy_evidence_view_included']}",
            f"Legacy evidence view source: {report['compatibility']['legacy_evidence_view_source']}",
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


def _append_temporal_report(lines: list[str], temporal: dict[str, Any]) -> None:
    summary = temporal.get("summary", {})
    configuration = temporal.get("configuration", {})
    lines.extend(
        [
            "Sequential temporal analysis",
            "----------------------------",
            f"Status: {_format(temporal.get('status'))}",
            f"Reason: {_format(temporal.get('reason'))}",
            f"Requested analysis FPS: {_format(summary.get('requested_analysis_fps'))}",
            f"Effective analysis FPS: {_format(summary.get('effective_analysis_fps'))}",
            f"Temporal frames analyzed: {_format(summary.get('temporal_frames_analyzed'))}",
            f"Transitions analyzed: {_format(summary.get('transitions_analyzed'))}",
            f"Scene count: {_format(summary.get('scene_count'))}",
            f"Scene-boundary candidates: {_format(summary.get('scene_boundary_candidate_count'))}",
            f"Sustained near-static intervals: {_format(summary.get('sustained_near_static_interval_count'))}",
            f"Ranked notable transitions: {_format(summary.get('notable_transition_count'))}",
            f"Motion summary available: {_format(summary.get('motion_summary_available'))}",
            f"Average flow-warp residual: {_format(summary.get('average_flow_warp_residual'))}",
            f"Maximum flow-warp residual: {_format(summary.get('maximum_flow_warp_residual'))}",
            f"Coverage timeline basis: {_format(temporal.get('coverage', {}).get('coverage_timeline_basis'))}",
            f"Coverage ratio: {_format(temporal.get('coverage', {}).get('coverage_ratio'))}",
            f"First/last temporal sample: {_format(temporal.get('coverage', {}).get('first_sample_timestamp_seconds'))} s to {_format(temporal.get('coverage', {}).get('last_sample_timestamp_seconds'))} s",
            f"Frame cap: {_format(configuration.get('maximum_analyzed_frames'))}",
            f"Resize max width: {_format(configuration.get('resize_max_width'))}",
            "",
            "Scene summaries",
            "---------------",
        ]
    )
    for scene in temporal.get("scenes", []):
        lines.append(
            f"Scene {scene['scene_index']}: {_format(scene['start_timestamp_seconds'])} s to "
            f"{_format(scene['end_timestamp_seconds'])} s, samples={scene['temporal_sample_count']}"
        )
    if not temporal.get("scenes"):
        lines.append("- None")

    lines.extend(["", "Sustained near-static intervals", "-------------------------------"])
    for interval in temporal.get("notable_intervals", []):
        lines.append(
            f"- {_format(interval['start_timestamp_seconds'])} s to "
            f"{_format(interval['end_timestamp_seconds'])} s "
            f"({interval['transition_count']} transitions)"
        )
    if not temporal.get("notable_intervals"):
        lines.append("- None")

    lines.extend(["", "Notable temporal transitions", "----------------------------"])
    for transition in temporal.get("notable_transitions", []):
        lines.append(
            f"- Transition {transition['notable_transition_index'] + 1}: {_format(transition['start_timestamp_seconds'])} s to "
            f"{_format(transition['end_timestamp_seconds'])} s, "
            f"scene_boundary={transition['classification']['scene_boundary_candidate']}"
        )
        lines.append("  Ranking reasons:")
        for reason in transition["notability"]["reasons"]:
            lines.append(f"  - {reason}")
        lines.append(f"  Flow-warp residual mean: {_format(transition['flow_warp_residual']['mean_normalized_residual'])}")
        artifacts = transition.get("artifacts", {})
        if artifacts:
            lines.append("  Review artifacts:")
            for label, artifact in artifacts.items():
                lines.append(f"  - {label}: {artifact['path']}")
        lines.append(f"  Interpretation: {transition['interpretation']}")
    if not temporal.get("notable_transitions"):
        lines.append("- None")

    artifacts = temporal.get("artifacts", {})
    metrics = artifacts.get("temporal_metrics_artifact") or {}
    lines.extend(
        [
            "",
            "Temporal artifacts",
            "------------------",
            f"Temporal metrics JSONL: {_format(metrics.get('path'))}",
            f"Temporal metrics SHA-256: {_format(metrics.get('sha256'))}",
            f"Temporal metrics size: {_format(metrics.get('size_human_readable'))}",
            f"Scene frames directory: {_format(artifacts.get('scene_frames_directory'))}",
            "",
            "Temporal limitations",
            "--------------------",
        ]
    )
    lines.extend(f"- {item}" for item in temporal.get("limitations", []))


def _append_audio_report(lines: list[str], audio: dict[str, Any]) -> None:
    summary = audio.get("summary", {})
    decoded = audio.get("decoded_audio", {})
    global_metrics = audio.get("global_metrics", {})
    artifact = audio.get("artifacts", {}).get("audio_metrics_artifact") or {}
    lines.extend(
        [
            "",
            "Audio analysis",
            "--------------",
            f"Status: {_format(audio.get('status'))}",
            f"Reason code: {_format(audio.get('reason_code'))}",
            f"Reason: {_format(audio.get('reason'))}",
            f"Extraction status: {_format(audio.get('extraction', {}).get('status'))}",
            f"Selected audio stream: {_format(audio.get('extraction', {}).get('selected_stream_index'))}",
            f"Temporary cleanup: {_format(audio.get('temporary_file_cleanup'))}",
            f"Audio available: {_format(summary.get('audio_available'))}",
            f"Decoded duration: {_format(summary.get('decoded_duration_seconds'))}",
            f"Window count: {_format(summary.get('window_count'))}",
            f"Silence-like intervals: {_format(summary.get('silence_like_interval_count'))}",
            f"Ranked energy transitions: {_format(summary.get('ranked_energy_transition_count'))}",
            f"Sample rate: {_format(decoded.get('sample_rate_hz'))}",
            f"Channels: {_format(decoded.get('channels'))}",
            f"RMS amplitude: {_format(global_metrics.get('rms_amplitude'))}",
            f"Peak absolute amplitude: {_format(global_metrics.get('peak_absolute_amplitude'))}",
            f"Clipping ratio: {_format(global_metrics.get('clipping_ratio'))}",
            f"Silence ratio: {_format(global_metrics.get('silence_ratio'))}",
            f"Zero-crossing rate: {_format(global_metrics.get('zero_crossing_rate'))}",
            f"Diagnostic stage: {_format(audio.get('diagnostics', {}).get('stage'))}",
            "",
            "Audio channel metrics",
            "---------------------",
        ]
    )
    channel_metrics = global_metrics.get("channel_metrics") or []
    if not channel_metrics:
        lines.append("- None")
    for channel in channel_metrics:
        lines.append(
            f"- Channel {channel['channel_index']}: RMS={channel['rms_amplitude']}, Peak={channel['peak_absolute_amplitude']}"
        )
    lines.extend(
        [
            "",
            "Audio silence-like intervals",
            "----------------------------",
        ]
    )
    intervals = audio.get("silence_intervals", [])
    if not intervals:
        lines.append("- None")
    for interval in intervals:
        lines.append(
            f"- {interval['interval_id']}: {_format(interval['start_timestamp_seconds'])} s to {_format(interval['end_timestamp_seconds'])} s"
        )
    lines.extend(["", "Ranked audio energy transitions", "-------------------------------"])
    transitions = audio.get("notable_transitions", [])
    if not transitions:
        lines.append("- None")
    for transition in transitions:
        lines.append(
            f"- {transition['transition_id']}: {_format(transition['start_timestamp_seconds'])} s, RMS {transition['rms_before']} -> {transition['rms_after']}"
        )
    lines.extend(
        [
            "",
            "Audio artifacts",
            "---------------",
            f"Audio metrics JSONL: {_format(artifact.get('path'))}",
            f"Audio metrics SHA-256: {_format(artifact.get('sha256'))}",
            f"Audio metrics size: {_format(artifact.get('size_human_readable'))}",
            "",
            "Audio limitations",
            "-----------------",
        ]
    )
    lines.extend(f"- {item}" for item in audio.get("limitations", []))


def _append_visual_consistency_report(lines: list[str], visual: dict[str, Any]) -> None:
    summary = visual.get("summary", {})
    configuration = visual.get("configuration", {})
    artifact = visual.get("artifacts", {}).get("visual_consistency_metrics_artifact") or {}
    lines.extend(
        [
            "",
            "Visual consistency analysis",
            "---------------------------",
            f"Status: {_format(visual.get('status'))}",
            f"Reason code: {_format(visual.get('reason_code'))}",
            f"Reason: {_format(visual.get('reason'))}",
            f"Analysis FPS: {_format(configuration.get('analysis_fps'))}",
            f"Resize max width: {_format(configuration.get('resize_max_width'))}",
            f"Region grid: {_format(summary.get('grid_rows'))} x {_format(summary.get('grid_columns'))}",
            f"Frames used: {_format(summary.get('frames_used'))}",
            f"Transitions analyzed: {_format(summary.get('transitions_analyzed'))}",
            f"Consistency records: {_format(summary.get('consistency_record_count'))}",
            f"Motion compensation used: {_format(summary.get('motion_compensation_used'))}",
            f"Sustained visual-consistency intervals: {_format(summary.get('sustained_interval_count'))}",
            f"Ranked review transitions: {_format(summary.get('ranked_review_transition_count'))}",
            f"Processing duration: {_format(summary.get('processing_duration_seconds'))} seconds",
            "",
            "Visual consistency metrics used",
            "-------------------------------",
            f"Edge method: {_format(configuration.get('edge_method'))}",
            f"Texture method: {_format(configuration.get('texture_method'))}",
            f"Fine-detail method: {_format(configuration.get('detail_method'))}",
            f"Ranking basis: {_format(configuration.get('ranking', {}).get('selection_basis'))}",
            "",
            "Visual consistency transition summaries",
            "---------------------------------------",
        ]
    )
    summaries = visual.get("transition_summaries", [])[:10]
    if not summaries:
        lines.append("- None")
    for summary_item in summaries:
        regional = summary_item.get("regional_summary", {})
        lines.append(
            f"- {summary_item['transition_id']}: {_format(summary_item['start_timestamp_seconds'])} s to "
            f"{_format(summary_item['end_timestamp_seconds'])} s, "
            f"unstable_regions={_format(regional.get('unstable_region_count'))}, "
            f"max_detail_residual={_format(regional.get('maximum_regional_detail_residual'))}, "
            f"max_edge_instability={_format(regional.get('maximum_edge_instability'))}"
        )
    if len(visual.get("transition_summaries", [])) > len(summaries):
        lines.append("- Additional transition summaries are available in report.json.")

    lines.extend(["", "Sustained regional visual variation", "------------------------------------"])
    intervals = visual.get("sustained_intervals", [])
    if not intervals:
        lines.append("- None")
    for interval in intervals:
        lines.append(
            f"- {interval['interval_id']}: {_format(interval['start_timestamp_seconds'])} s to "
            f"{_format(interval['end_timestamp_seconds'])} s, regions={interval['affected_regions']}"
        )

    lines.extend(["", "Ranked consistency review transitions", "-------------------------------------"])
    ranked = visual.get("ranked_review_transitions", [])
    if not ranked:
        lines.append("- None")
    for transition in ranked:
        lines.append(
            f"- Rank {transition['ranked_review_index']}: {transition['transition_id']} "
            f"({_format(transition['start_timestamp_seconds'])} s to {_format(transition['end_timestamp_seconds'])} s)"
        )
        lines.append(f"  Combined percentile: {_format(transition.get('combined_percentile'))}")
        lines.append(f"  Ranking reasons: {transition.get('ranking_reasons', [])}")
        artifacts = transition.get("artifacts", {})
        if artifacts:
            lines.append("  Review artifacts:")
            for label, record in artifacts.items():
                lines.append(
                    f"  - {label}: {record['path']} ({record['size_human_readable']}, SHA-256 {record['sha256']})"
                )
        lines.append(f"  Interpretation: {transition.get('interpretation')}")

    lines.extend(
        [
            "",
            "Visual consistency artifacts",
            "----------------------------",
            f"Visual consistency metrics JSONL: {_format(artifact.get('path'))}",
            f"Visual consistency metrics SHA-256: {_format(artifact.get('sha256'))}",
            f"Visual consistency metrics size: {_format(artifact.get('size_human_readable'))}",
            f"Consistency frames directory: {_format(visual.get('artifacts', {}).get('consistency_frames_directory'))}",
            "",
            "Visual consistency limitations",
            "------------------------------",
        ]
    )
    lines.extend(f"- {item}" for item in visual.get("limitations", []))


def _append_unified_evidence_report(lines: list[str], unified: dict[str, Any]) -> None:
    summary = unified.get("summary", {})
    artifacts = unified.get("artifacts", {})
    lines.extend(
        [
            "",
            "Unified evidence",
            "----------------",
            f"Status: {_format(unified.get('status'))}",
            f"Reason code: {_format(unified.get('reason_code'))}",
            f"Timeline basis: {_format(unified.get('timeline_configuration', {}).get('timeline_basis'))}",
            f"Merge tolerance seconds: {_format(unified.get('timeline_configuration', {}).get('merge_tolerance_seconds'))}",
            f"Maximum anchor event span seconds: {_format(unified.get('timeline_configuration', {}).get('maximum_anchor_event_span_seconds'))}",
            f"Merging strategy: {_format(unified.get('timeline_configuration', {}).get('merging_strategy'))}",
            f"Timeline events: {_format(summary.get('timeline_event_count'))}",
            f"Anchor candidates: {_format(summary.get('anchor_candidate_count'))}",
            f"Supporting intervals: {_format(summary.get('supporting_interval_count'))}",
            f"Contextual intervals: {_format(summary.get('contextual_interval_count'))}",
            f"Priority review events: {_format(summary.get('priority_review_event_count'))}",
            f"Time-based candidates: {_format(summary.get('time_based_candidate_count'))}",
            f"Evidence domains available: {_format(summary.get('evidence_domains_available'))}",
            f"Evidence groups available: {_format(summary.get('evidence_groups_available'))}",
            f"External model results: {_format(summary.get('external_model_result_count'))}",
            "",
            "Priority review events",
            "----------------------",
        ]
    )
    highlights = unified.get("review_highlights", [])
    if not highlights:
        lines.append("- None")
    for highlight in highlights:
        context = highlight.get("cross_modal_context", {})
        lines.append(
            f"- {highlight['event_id']}: {_format(highlight['start_timestamp_seconds'])} s to "
            f"{_format(highlight['end_timestamp_seconds'])} s, priority={highlight['review_priority']['level']}, "
            f"context={_format(context.get('classification'))}"
        )
        lines.append(f"  Domains: {highlight.get('evidence_domains', [])}")
        lines.append(f"  Evidence groups: {highlight.get('evidence_groups_present', [])}")
        lines.append(f"  Basis: {highlight.get('review_priority', {}).get('basis', [])}")
        boundary = highlight.get("boundary_basis", {})
        lines.append(f"  Anchor candidates: {boundary.get('anchor_candidate_ids', [])}")
        lines.append(f"  Supporting candidates: {boundary.get('supporting_candidate_ids', [])}")
        lines.append(f"  Context candidates: {highlight.get('context_candidate_ids', [])}")
        lines.append(f"  Observation IDs: {highlight.get('source_observation_ids', [])}")
        findings = highlight.get("key_findings", [])
        if findings:
            lines.append("  Key findings:")
            for finding in findings:
                lines.append(
                    f"  - {finding.get('domain')}/{finding.get('type')}: {finding.get('summary')}"
                )

    lines.extend(["", "Ambiguous findings", "------------------"])
    ambiguous = unified.get("ambiguous_findings", [])
    if not ambiguous:
        lines.append("- None")
    for item in ambiguous:
        lines.append(f"- {item.get('type')}: {item.get('description')}")

    lines.extend(["", "Normal or non-supporting findings", "---------------------------------"])
    normal = unified.get("normal_or_non_supporting_findings", [])
    if not normal:
        lines.append("- None")
    for item in normal:
        lines.append(f"- {item.get('type')}: {item.get('description')}")

    lines.extend(["", "Missing evidence", "----------------"])
    missing = unified.get("missing_evidence", [])
    if not missing:
        lines.append("- None")
    for item in missing:
        lines.append(
            f"- {item.get('type')}: status={item.get('status')}, importance={item.get('importance')}"
        )

    def artifact_line(label: str, key: str) -> str:
        artifact = artifacts.get(key) or {}
        return (
            f"{label}: {_format(artifact.get('path'))} "
            f"({_format(artifact.get('size_human_readable'))}, SHA-256 {_format(artifact.get('sha256'))})"
        )

    lines.extend(
        [
            "",
            "Unified evidence artifacts",
            "--------------------------",
            artifact_line("Unified evidence JSON", "unified_evidence"),
            artifact_line("Evidence timeline JSONL", "evidence_timeline"),
            artifact_line("AI-ready input JSON", "ai_interpretation_input"),
            artifact_line("AI prompt template", "ai_interpretation_prompt"),
        ]
    )


def _append_learned_detector_report(lines: list[str], learned: dict[str, Any]) -> None:
    d3 = learned.get("d3", {})
    execution = d3.get("execution", {})
    detector = d3.get("detector", {})
    configuration = d3.get("configuration", {})
    preprocessing = d3.get("preprocessing", {})
    native = d3.get("native_output", {})
    summary = d3.get("feature_summary", {})
    artifacts = d3.get("artifacts", {})
    verification = d3.get("method_verification", {})
    lines.extend(
        [
            "",
            "LEARNED DETECTOR - D3",
            "---------------------",
            f"Execution status: {_format(execution.get('status'))}",
            f"Reason code: {_format(execution.get('reason_code'))}",
            f"Standalone from unified evidence: {_format(learned.get('standalone_not_fused_with_unified_evidence'))}",
            f"Method: {_format(detector.get('detector_name'))}",
            f"Upstream commit: {_format(detector.get('upstream_commit'))}",
            f"Upstream license: {_format(detector.get('upstream_license'))}",
            f"Encoder: {_format(configuration.get('encoder'))}",
            f"Distance mode: {_format(configuration.get('distance_mode'))}",
            f"Device requested: {_format(execution.get('device_requested'))}",
            f"Device used: {_format(execution.get('device_used'))}",
            f"Timeout seconds: {_format(execution.get('timeout_seconds'))}",
            f"Preprocessing window: {_format(preprocessing.get('window_start_seconds'))} s to {_format(preprocessing.get('window_end_seconds'))} s",
            f"Selected frame count: {_format(preprocessing.get('actual_selected_frame_count'))}",
            f"Temporary frames preserved: {_format(preprocessing.get('temporary_frames', {}).get('preserved'))}",
            f"Raw score: {_format(native.get('raw_score'))}",
            f"Score name: {_format(native.get('score_name'))}",
            f"Score direction: {_format(native.get('score_direction'))}",
            f"Probability: {_format(native.get('probability'))}",
            f"Calibration: {_format(native.get('calibration_status'))}",
            f"Classification: {_format(native.get('classification'))}",
            f"First-order count: {_format(summary.get('first_order_value_count'))}",
            f"Second-order count: {_format(summary.get('second_order_value_count'))}",
            f"Second-order mean: {_format(summary.get('second_order_mean'))}",
            f"Second-order standard deviation: {_format(summary.get('second_order_standard_deviation'))}",
            f"Preprocessing parity: {_format(verification.get('preprocessing_parity'))}",
            f"Mathematical parity: {_format(verification.get('mathematical_parity'))}",
            f"Runtime parity: {_format(verification.get('runtime_parity'))}",
            f"Score-direction verification: {_format(verification.get('score_direction_status'))}",
            "This raw D3 value is an uncalibrated second-order temporal-feature statistic. It is not an authenticity probability and no classification threshold has been applied.",
            "",
            "D3 artifacts",
            "------------",
        ]
    )
    if not artifacts:
        lines.append("- None")
    for label, artifact in artifacts.items():
        lines.append(
            f"- {label}: {artifact.get('path')} ({artifact.get('size_human_readable')}, SHA-256 {artifact.get('sha256')})"
        )
    lines.extend(["", "D3 limitations", "--------------"])
    limitations = d3.get("limitations") or learned.get("limitations", [])
    if not limitations:
        lines.append("- None")
    for limitation in limitations:
        lines.append(f"- {limitation}")


def _legacy_evidence_from_observations(
    observations: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for category, items in observations.items():
        for item in items:
            evidence.append(
                {
                    "observation_id": item.get("observation_id"),
                    "evidence_id": item.get("observation_id"),
                    "category": category,
                    "type": item.get("type"),
                    "severity": item.get("severity", "info"),
                    "description": item.get("description"),
                    "interpretation": item.get("interpretation"),
                    "metrics": item.get("metrics", {}),
                    "timestamp_start": item.get("timestamp_start"),
                    "timestamp_end": item.get("timestamp_end"),
                }
            )
    return evidence
