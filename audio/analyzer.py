from __future__ import annotations

import hashlib
import json
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from config import audio_configuration
from file_utils import format_file_size
from audio.extractor import extract_audio_to_wav
from audio.interval_detection import detect_silence_intervals
from audio.metrics import calculate_global_metrics
from audio.pcm_reader import read_pcm_wav
from audio.transition_ranking import rank_audio_transitions
from audio.window_analysis import analyze_windows

PathInput = str | PathLike[str] | Path


AUDIO_LIMITATIONS = [
    "Basic signal measurements cannot determine whether a voice is natural, cloned, synthesized, or manipulated.",
    "Silence-like intervals may represent normal quiet sections.",
    "Abrupt energy changes may reflect speech, music, cuts, effects, compression, or microphone behavior.",
    "Clipping-like samples are not evidence of malicious editing.",
    "Windowed analysis may miss events shorter than the configured hop interval.",
    "Audio extraction and PCM conversion may alter some original encoding characteristics.",
]


def analyze_audio(
    video_path: PathInput,
    analysis_dir: PathInput,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    cleanup = {"attempted": False, "successful": None}
    stage = "audio_path_normalization"

    try:
        video_path = Path(video_path)
        analysis_dir = Path(analysis_dir)
        audio_metadata = metadata.get("audio", {})
        if not audio_metadata.get("present"):
            return _skipped("no_audio_stream"), warnings
        selected_stream_index = audio_metadata.get("selected_stream_index") or audio_metadata.get("index")

        stage = "audio_temporary_directory_setup"
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            cleanup["attempted"] = True
            stage = "audio_extraction_setup"
            wav_path, extraction, extraction_warnings = extract_audio_to_wav(
                video_path,
                selected_stream_index,
                temp_dir,
            )
            warnings.extend(extraction_warnings)
            if wav_path is None:
                result = _failure_or_skip(extraction)
                result["extraction"] = extraction
                result["temporary_file_cleanup"] = cleanup
                return result, warnings
            stage = "audio_wav_decode"
            samples, decoded, decode_warnings = read_pcm_wav(wav_path)
            warnings.extend(decode_warnings)
        cleanup["successful"] = True
    except Exception as error:
        cleanup["successful"] = False
        result = _base("failed")
        result.update(
            {
                "reason_code": "internal_audio_error",
                "reason": "Audio analysis encountered an internal error.",
                "diagnostics": {
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "stage": stage,
                },
                "temporary_file_cleanup": cleanup,
                "observations": [_failure_observation()],
            }
        )
        return result, warnings

    if samples is None:
        result = _base("partial")
        result["reason_code"] = decoded.get("reason_code", "wav_decode_failed")
        result["reason"] = decoded.get("reason", "Audio decoding failed.")
        result["extraction"] = extraction
        result["decoded_audio"] = decoded
        result["temporary_file_cleanup"] = cleanup
        result["observations"] = [_failure_observation("audio-partial-decode-001", "audio.partial_decode")]
        return result, warnings

    stage = "audio_metric_calculation"
    global_metrics = calculate_global_metrics(samples)
    windows = analyze_windows(samples, decoded["sample_rate_hz"])
    silence_intervals = detect_silence_intervals(windows)
    transitions = rank_audio_transitions(windows)
    stage = "audio_metrics_artifact_write"
    artifact = write_audio_metrics(analysis_dir, windows)
    observations = _observations(silence_intervals, transitions, global_metrics)
    result = _base("completed")
    result.update(
        {
            "reason": None,
            "reason_code": None,
            "extraction": extraction,
            "decoded_audio": decoded,
            "timeline": _timeline(metadata, decoded),
            "summary": {
                "audio_available": True,
                "decoded_duration_seconds": decoded["decoded_duration_seconds"],
                "window_count": len(windows),
                "silence_like_interval_count": len(silence_intervals),
                "clipping_like_sample_ratio": global_metrics["clipping_ratio"],
                "ranked_energy_transition_count": len(transitions),
                "channel_count": decoded["channels"],
                "analysis_metrics_available": True,
            },
            "global_metrics": global_metrics,
            "silence_intervals": silence_intervals,
            "clipping_intervals": [],
            "notable_transitions": transitions,
            "observations": observations,
            "artifacts": {"audio_metrics_artifact": artifact},
            "temporary_file_cleanup": cleanup,
        }
    )
    return result, warnings


def write_audio_metrics(analysis_dir: PathInput, windows: list[dict[str, Any]]) -> dict[str, Any]:
    analysis_dir = Path(analysis_dir)
    path = analysis_dir / "audio_metrics.jsonl"
    sha256 = hashlib.sha256()
    size_bytes = 0
    with path.open("wb") as file:
        for window in windows:
            line = (json.dumps(window, sort_keys=True) + "\n").encode("utf-8")
            file.write(line)
            sha256.update(line)
            size_bytes += len(line)
    return {
        "path": "audio_metrics.jsonl",
        "size_bytes": size_bytes,
        "size_human_readable": format_file_size(size_bytes),
        "sha256": sha256.hexdigest(),
    }


def _timeline(metadata: dict[str, Any], decoded: dict[str, Any]) -> dict[str, Any]:
    audio = metadata.get("audio", {})
    video = metadata.get("video", {})
    ffprobe_duration = audio.get("duration_seconds")
    decoded_duration = decoded.get("decoded_duration_seconds")
    return {
        "ffprobe_audio_duration_seconds": ffprobe_duration,
        "decoded_audio_duration_seconds": decoded_duration,
        "decoded_vs_ffprobe_difference_seconds": _difference(decoded_duration, ffprobe_duration),
        "audio_vs_video_duration_difference_seconds": _difference(audio.get("duration_seconds"), video.get("duration_seconds")),
        "audio_start_time_seconds": audio.get("start_time"),
        "video_start_time_seconds": video.get("start_time"),
    }


def _observations(
    silence_intervals: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    global_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    observations = [_observation("audio-stream-001", "audio.stream_decoded", "The selected audio stream was decoded to PCM for lightweight signal analysis.")]
    for index, interval in enumerate(silence_intervals, start=1):
        observations.append(_localized_observation(f"audio-silence-{index:03d}", "audio.silence_like_interval", interval["start_timestamp_seconds"], interval["end_timestamp_seconds"], "A low-energy interval consistent with silence or near-silence was detected.", "This may represent actual silence, quiet speech, background noise, fade, microphone behavior, compression, or intentional muting."))
    if global_metrics.get("clipping_ratio", 0) > 0:
        observations.append(_observation("audio-clipping-001", "audio.clipping_like_samples", "Samples near the maximum representable amplitude were detected."))
    for index, transition in enumerate(transitions, start=1):
        observations.append(_localized_observation(f"audio-transition-{index:03d}", "audio.ranked_energy_transition", transition["start_timestamp_seconds"], transition["end_timestamp_seconds"], "This audio transition ranked highly in energy change relative to this file.", transition["interpretation"]))
    return observations


def _base(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": None,
        "reason": None,
        "configuration": audio_configuration(),
        "extraction": {},
        "decoded_audio": {},
        "timeline": {},
        "summary": {"audio_available": False, "analysis_metrics_available": False},
        "global_metrics": {},
        "silence_intervals": [],
        "clipping_intervals": [],
        "notable_transitions": [],
        "observations": [],
        "limitations": AUDIO_LIMITATIONS,
        "artifacts": {},
        "temporary_file_cleanup": {"attempted": False, "successful": None},
        "diagnostics": {},
    }


def _skipped(reason: str) -> dict[str, Any]:
    result = _base("skipped")
    result["reason_code"] = reason
    result["reason"] = reason
    result["observations"] = [_observation("audio-skipped-001", "audio.analysis_skipped", "Audio analysis was skipped.")]
    return result


def _failure_or_skip(extraction: dict[str, Any]) -> dict[str, Any]:
    status = "failed" if extraction.get("status") == "failed" else "skipped"
    result = _base(status)
    result["reason_code"] = extraction.get("reason_code") or extraction.get("reason") or "audio_analysis_skipped"
    result["reason"] = extraction.get("reason") or "Audio analysis did not run."
    result["observations"] = [_failure_observation() if status == "failed" else _observation("audio-skipped-001", "audio.analysis_skipped", "Audio analysis was skipped.")]
    return result


def _failure_observation(
    observation_id: str = "audio-analysis-failed-001",
    observation_type: str = "audio.analysis_failed",
) -> dict[str, Any]:
    return _observation(
        observation_id,
        observation_type,
        "Audio-signal analysis could not be completed.",
        "This is an analysis-stage limitation and is not evidence about the authenticity of the source.",
    )


def _observation(observation_id: str, observation_type: str, description: str, interpretation: str | None = None) -> dict[str, Any]:
    return {
        "observation_id": observation_id,
        "type": observation_type,
        "severity": "info",
        "conclusion_scope": "non_conclusive_observation",
        "supports_authenticity_verdict": False,
        "description": description,
        "interpretation": interpretation or "This is a non-conclusive audio observation and is not an authenticity verdict.",
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


def _difference(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return round(abs(first - second), 6)
