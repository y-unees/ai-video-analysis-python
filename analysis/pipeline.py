from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from audio.analyzer import analyze_audio
from config import REPORTS_DIR_NAME
from environment_info import collect_environment_info
from evidence_builder import build_temporal_evidence
from file_utils import calculate_sha256, get_file_evidence
from frame_analyzer import summarize_frame_analysis
from frame_sampler import sample_video_frames
from learned_detectors import run_learned_detectors
from metadata_extractor import extract_metadata
from outcome import create_outcome_features_artifact
from reporting import create_gemini_evidence_report
from report_writer import build_report, create_analysis_directory, write_raw_ffprobe, write_reports
from temporal.analyzer import analyze_temporal
from unified_evidence import build_unified_evidence_artifacts
from visual_consistency import analyze_visual_consistency


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class AnalysisPipelineResult:
    status: str
    analysis_dir: Path
    report_paths: dict[str, Path]
    warnings: list[str]
    report: dict[str, Any]


def run_analysis(
    video_path: Path,
    reports_dir: Path = Path(REPORTS_DIR_NAME),
    timestamp: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisPipelineResult:
    selected_video = Path(video_path)
    analysis_started_at = datetime.now().astimezone()
    started_perf = perf_counter()
    run_timestamp = timestamp or analysis_started_at.strftime("%Y-%m-%d_%H%M%S")
    warnings: list[str] = []

    _progress(progress_callback, "creating_analysis_directory")
    analysis_dir = create_analysis_directory(selected_video, reports_dir, run_timestamp)

    _progress(progress_callback, "calculating_sha256")
    file_evidence = get_file_evidence(selected_video)
    file_evidence["sha256"] = calculate_sha256(selected_video)

    _progress(progress_callback, "extracting_metadata")
    raw_ffprobe, metadata, metadata_observations = extract_metadata(selected_video)
    raw_ffprobe_artifact = write_raw_ffprobe(analysis_dir, raw_ffprobe)

    _progress(progress_callback, "sampling_frames")
    try:
        sampling, frame_analysis, frame_warnings = sample_video_frames(selected_video, analysis_dir / "frames", metadata["container"]["duration_seconds"])
        warnings.extend(frame_warnings)
    except Exception as error:
        warnings.append(f"Frame sampling failed: {error}")
        sampling = {"strategy": "uniform", "target_frame_count": 0, "requested_timestamps_seconds": [], "decoded_frame_count": 0, "failed_timestamps_seconds": [], "maximum_absolute_seek_error_seconds": None, "average_absolute_seek_error_seconds": None, "repeated_decoded_timestamps_seconds": []}
        frame_analysis = {"summary": summarize_frame_analysis([], []), "frames": [], "comparisons": []}

    _progress(progress_callback, "temporal_analysis")
    try:
        temporal_analysis, temporal_warnings = analyze_temporal(selected_video, analysis_dir, metadata)
        warnings.extend(temporal_warnings)
    except Exception as error:
        warnings.append(f"Temporal analysis failed: {error}")
        temporal_analysis = {"status": "failed", "reason": str(error), "configuration": {}, "summary": {}, "scenes": [], "scene_representative_frames": [], "notable_intervals": [], "notable_transitions": [], "observations": [], "limitations": ["Temporal analysis failed, but metadata and representative-frame analysis may still be available."], "artifacts": {}}

    _progress(progress_callback, "audio_analysis")
    try:
        audio_analysis, audio_warnings = analyze_audio(selected_video, analysis_dir, metadata)
        warnings.extend(audio_warnings)
    except Exception as error:
        warnings.append(f"Audio analysis failed: {error}")
        audio_analysis = {"status": "failed", "reason": str(error), "configuration": {}, "extraction": {}, "decoded_audio": {}, "timeline": {}, "summary": {"audio_available": False, "analysis_metrics_available": False}, "global_metrics": {}, "silence_intervals": [], "clipping_intervals": [], "notable_transitions": [], "observations": [], "limitations": ["Audio analysis failed, but visual analysis may still be available."], "artifacts": {}}

    _progress(progress_callback, "visual_consistency_analysis")
    try:
        visual_consistency_analysis, visual_consistency_warnings = analyze_visual_consistency(analysis_dir, temporal_analysis)
        warnings.extend(visual_consistency_warnings)
    except Exception as error:
        warnings.append(f"Visual consistency analysis failed: {error}")
        visual_consistency_analysis = {"status": "failed", "reason_code": "visual_consistency_unhandled_exception", "reason": str(error), "configuration": {}, "summary": {"metrics_available": False}, "transition_summaries": [], "sustained_intervals": [], "ranked_review_transitions": [], "observations": [], "limitations": ["Visual consistency analysis failed, but earlier analysis stages may still be available."], "artifacts": {}}

    temporal_analysis.pop("_runtime", None)
    temporal_evidence = build_temporal_evidence(frame_analysis["frames"], frame_analysis["comparisons"], sampling["failed_timestamps_seconds"])
    observations = {
        "metadata_facts": metadata_observations["metadata_facts"],
        "missing_metadata": metadata_observations["missing_metadata"],
        "temporal_heuristics": temporal_evidence + temporal_analysis.get("observations", []),
        "audio_observations": audio_analysis.get("observations", []),
        "visual_consistency_observations": visual_consistency_analysis.get("observations", []),
    }
    analysis_completed_at = datetime.now().astimezone()
    analysis = {
        "started_at": analysis_started_at.isoformat(timespec="milliseconds"),
        "completed_at": analysis_completed_at.isoformat(timespec="milliseconds"),
        "processing_duration_seconds": round(perf_counter() - started_perf, 3),
        "status": "completed_with_warnings" if warnings else "completed",
    }
    environment = collect_environment_info(analysis_started_at, analysis_completed_at)
    report = build_report(analysis, environment, file_evidence, metadata, sampling, frame_analysis, temporal_analysis, audio_analysis, temporal_evidence, warnings, observations, raw_ffprobe_artifact, visual_consistency_analysis)

    _progress(progress_callback, "unified_evidence")
    unified_evidence, unified_warnings = build_unified_evidence_artifacts(analysis_dir, report)
    warnings.extend(unified_warnings)
    report["warnings"] = warnings
    report["unified_evidence"] = unified_evidence
    if unified_evidence.get("status") != "completed" and report["analysis"]["status"] == "completed":
        report["analysis"]["status"] = "completed_with_warnings"

    _progress(progress_callback, "d3")
    learned_detector_results, learned_warnings = run_learned_detectors(selected_video, file_evidence["sha256"], analysis_dir, metadata)
    warnings.extend(learned_warnings)
    report["warnings"] = warnings
    report["learned_detector_results"] = learned_detector_results

    _progress(progress_callback, "outcome_features")
    outcome_reference, outcome_warnings = create_outcome_features_artifact(analysis_dir, report)
    warnings.extend(outcome_warnings)
    report["warnings"] = warnings
    report["artifacts"]["outcome_features"] = outcome_reference

    _progress(progress_callback, "gemini_compact_report")
    compact_reference, compact_warnings = create_gemini_evidence_report(analysis_dir, report)
    warnings.extend(compact_warnings)
    report["warnings"] = warnings
    report["artifacts"]["gemini_evidence_report"] = compact_reference

    _progress(progress_callback, "writing_reports")
    report_paths = write_reports(analysis_dir, report)
    return AnalysisPipelineResult(report["analysis"]["status"], analysis_dir, report_paths, warnings, report)


def _progress(callback: ProgressCallback | None, stage: str) -> None:
    if callback is not None:
        callback(stage)
