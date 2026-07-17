from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

from config import REPORTS_DIR_NAME, SOURCE_DIR_NAME
from audio.analyzer import analyze_audio
from dependency_checker import missing_python_dependencies, tool_is_available
from environment_info import collect_environment_info
from evidence_builder import build_temporal_evidence
from file_utils import calculate_sha256, get_file_evidence
from frame_analyzer import summarize_frame_analysis
from frame_sampler import sample_video_frames
from learned_detectors import run_learned_detectors
from metadata_extractor import FFprobeError, extract_metadata
from report_writer import (
    build_report,
    create_analysis_directory,
    write_raw_ffprobe,
    write_reports,
)
from temporal.analyzer import analyze_temporal
from unified_evidence import build_unified_evidence_artifacts
from video_selector import choose_video, find_video_files
from visual_consistency import analyze_visual_consistency


SOURCE_DIR = Path(SOURCE_DIR_NAME)
REPORTS_DIR = Path(REPORTS_DIR_NAME)


def main() -> int:
    print("Scanning source_videos...")

    if not SOURCE_DIR.exists():
        print("Error: source_videos folder was not found.")
        return 1

    if not SOURCE_DIR.is_dir():
        print("Error: source_videos exists but is not a folder.")
        return 1

    videos = find_video_files(SOURCE_DIR)
    if not videos:
        print("No video files were found in source_videos.")
        return 1

    print(f"{len(videos)} video file{'s' if len(videos) != 1 else ''} found.")
    selected_video = choose_video(videos)
    print(f"Selected: {selected_video.name}")

    missing_packages = missing_python_dependencies()
    if missing_packages:
        print(
            "Required Python packages are missing. Activate your virtual "
            "environment and run: pip install -r requirements.txt"
        )
        print(f"Missing packages: {', '.join(missing_packages)}")
        return 1

    if not tool_is_available("ffmpeg"):
        print(
            "ffmpeg was not found. Install FFmpeg and make sure ffmpeg is "
            "available in your system PATH."
        )
        return 1

    if not tool_is_available("ffprobe"):
        print(
            "ffprobe was not found. Install FFmpeg and make sure ffprobe is "
            "available in your system PATH."
        )
        return 1

    analysis_started_at = datetime.now().astimezone()
    started_perf = perf_counter()
    timestamp = analysis_started_at.strftime("%Y-%m-%d_%H%M%S")
    warnings: list[str] = []

    try:
        analysis_dir = create_analysis_directory(
            selected_video=selected_video,
            reports_dir=REPORTS_DIR,
            timestamp=timestamp,
        )

        print("Calculating SHA-256...")
        file_evidence = get_file_evidence(selected_video)
        file_evidence["sha256"] = calculate_sha256(selected_video)

        print("Extracting metadata...")
        raw_ffprobe, metadata, metadata_observations = extract_metadata(selected_video)
        raw_ffprobe_artifact = write_raw_ffprobe(analysis_dir, raw_ffprobe)

        print("Extracting representative frames...")
        try:
            sampling, frame_analysis, frame_warnings = sample_video_frames(
                video_path=selected_video,
                frames_dir=analysis_dir / "frames",
                duration_seconds=metadata["container"]["duration_seconds"],
            )
            warnings.extend(frame_warnings)
        except Exception as error:
            warnings.append(f"Frame sampling failed: {error}")
            sampling = {
                "strategy": "uniform",
                "target_frame_count": 0,
                "requested_timestamps_seconds": [],
                "decoded_frame_count": 0,
                "failed_timestamps_seconds": [],
                "maximum_absolute_seek_error_seconds": None,
                "average_absolute_seek_error_seconds": None,
                "repeated_decoded_timestamps_seconds": [],
            }
            frame_analysis = {
                "summary": summarize_frame_analysis([], []),
                "frames": [],
                "comparisons": [],
            }

        print("Running temporal analysis...")
        try:
            temporal_analysis, temporal_warnings = analyze_temporal(
                video_path=selected_video,
                analysis_dir=analysis_dir,
                metadata=metadata,
            )
            warnings.extend(temporal_warnings)
        except Exception as error:
            warnings.append(f"Temporal analysis failed: {error}")
            temporal_analysis = {
                "status": "failed",
                "reason": str(error),
                "configuration": {},
                "summary": {},
                "scenes": [],
                "scene_representative_frames": [],
                "notable_intervals": [],
                "notable_transitions": [],
                "observations": [],
                "limitations": [
                    "Temporal analysis failed, but metadata and representative-frame analysis may still be available."
                ],
                "artifacts": {},
            }

        print("Running audio analysis...")
        try:
            audio_analysis, audio_warnings = analyze_audio(
                video_path=selected_video,
                analysis_dir=analysis_dir,
                metadata=metadata,
            )
            warnings.extend(audio_warnings)
        except Exception as error:
            warnings.append(f"Audio analysis failed: {error}")
            audio_analysis = {
                "status": "failed",
                "reason": str(error),
                "configuration": {},
                "extraction": {},
                "decoded_audio": {},
                "timeline": {},
                "summary": {"audio_available": False, "analysis_metrics_available": False},
                "global_metrics": {},
                "silence_intervals": [],
                "clipping_intervals": [],
                "notable_transitions": [],
                "observations": [],
                "limitations": [
                    "Audio analysis failed, but visual analysis may still be available."
                ],
                "artifacts": {},
            }

        print("Running visual consistency analysis...")
        try:
            visual_consistency_analysis, visual_consistency_warnings = analyze_visual_consistency(
                analysis_dir=analysis_dir,
                temporal_analysis=temporal_analysis,
            )
            warnings.extend(visual_consistency_warnings)
        except Exception as error:
            warnings.append(f"Visual consistency analysis failed: {error}")
            visual_consistency_analysis = {
                "status": "failed",
                "reason_code": "visual_consistency_unhandled_exception",
                "reason": str(error),
                "configuration": {},
                "summary": {"metrics_available": False},
                "transition_summaries": [],
                "sustained_intervals": [],
                "ranked_review_transitions": [],
                "observations": [],
                "limitations": [
                    "Visual consistency analysis failed, but earlier analysis stages may still be available."
                ],
                "artifacts": {},
            }

        temporal_analysis.pop("_runtime", None)

        temporal_evidence = build_temporal_evidence(
            frames=frame_analysis["frames"],
            comparisons=frame_analysis["comparisons"],
            failed_timestamps=sampling["failed_timestamps_seconds"],
        )
        observations = {
            "metadata_facts": metadata_observations["metadata_facts"],
            "missing_metadata": metadata_observations["missing_metadata"],
            "temporal_heuristics": temporal_evidence,
        }
        observations["temporal_heuristics"].extend(
            temporal_analysis.get("observations", [])
        )
        observations["audio_observations"] = audio_analysis.get("observations", [])
        observations["visual_consistency_observations"] = visual_consistency_analysis.get(
            "observations",
            [],
        )

        analysis_completed_at = datetime.now().astimezone()
        processing_duration = round(perf_counter() - started_perf, 3)
        status = "completed_with_warnings" if warnings else "completed"
        analysis = {
            "started_at": analysis_started_at.isoformat(timespec="milliseconds"),
            "completed_at": analysis_completed_at.isoformat(timespec="milliseconds"),
            "processing_duration_seconds": processing_duration,
            "status": status,
        }
        environment = collect_environment_info(
            analysis_started_at=analysis_started_at,
            analysis_completed_at=analysis_completed_at,
        )
        report = build_report(
            analysis=analysis,
            analysis_environment=environment,
            source=file_evidence,
            metadata=metadata,
            sampling=sampling,
            frame_analysis=frame_analysis,
            temporal_analysis=temporal_analysis,
            audio_analysis=audio_analysis,
            visual_consistency_analysis=visual_consistency_analysis,
            evidence=temporal_evidence,
            observations=observations,
            warnings=warnings,
            raw_ffprobe_artifact=raw_ffprobe_artifact,
        )

        print("Building unified evidence bundle...")
        unified_evidence, unified_warnings = build_unified_evidence_artifacts(
            analysis_dir=analysis_dir,
            report=report,
        )
        warnings.extend(unified_warnings)
        report["warnings"] = warnings
        report["unified_evidence"] = unified_evidence
        if unified_evidence.get("status") != "completed" and report["analysis"]["status"] == "completed":
            report["analysis"]["status"] = "completed_with_warnings"

        print("Optional D3 learned detector stage...")
        learned_detector_results, learned_warnings = run_learned_detectors(
            video_path=selected_video,
            video_sha256=file_evidence["sha256"],
            analysis_dir=analysis_dir,
            metadata=metadata,
        )
        warnings.extend(learned_warnings)
        report["warnings"] = warnings
        report["learned_detector_results"] = learned_detector_results
        d3_status = learned_detector_results.get("d3", {}).get("execution", {}).get("status")
        if d3_status == "completed":
            d3_output = learned_detector_results["d3"].get("native_output", {})
            print(
                "D3 learned detector: completed standalone raw-score analysis "
                f"(raw score: {d3_output.get('raw_score')}, classification: not assigned)"
            )
        else:
            reason = learned_detector_results.get("d3", {}).get("execution", {}).get("reason_code")
            print(f"D3 learned detector: {d3_status} ({reason})")

        print("Creating reports...")
        report_paths = write_reports(analysis_dir=analysis_dir, report=report)
    except FFprobeError as error:
        print(f"Metadata extraction failed: {error}")
        return 1
    except OSError as error:
        print(f"File operation failed: {error}")
        return 1
    except Exception as error:
        print(f"Unexpected error during analysis: {error}")
        return 1

    print("Analysis completed.")
    print()
    print(f"Analysis directory: {report_paths['directory']}")
    print(f"JSON report: {report_paths['json']}")
    print(f"TXT report: {report_paths['txt']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
