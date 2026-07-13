from __future__ import annotations

import subprocess
from os import PathLike
from pathlib import Path
from typing import Any

from config import AUDIO_EXTRACTION_TIMEOUT_SECONDS

PathInput = str | PathLike[str] | Path


def extract_audio_to_wav(
    video_path: PathInput,
    stream_index: int | None,
    temp_dir: PathInput,
) -> tuple[Path | None, dict[str, Any], list[str]]:
    video_path = Path(video_path)
    temp_dir = Path(temp_dir)
    warnings: list[str] = []
    extraction = {
        "status": "skipped",
        "selected_stream_index": stream_index,
        "output_codec": "pcm_s16le",
        "temporary_artifact_retained": False,
        "ffmpeg_return_code": None,
        "stderr_summary": None,
    }
    if stream_index is None:
        extraction["reason"] = "no_selected_audio_stream"
        return None, extraction, warnings

    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / "selected_audio.wav"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-map",
        f"0:{stream_index}",
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=AUDIO_EXTRACTION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        extraction["status"] = "failed"
        extraction["reason_code"] = "audio_extraction_timeout"
        extraction["reason"] = "Audio extraction timed out."
        warnings.append("FFmpeg audio extraction timed out.")
        return None, extraction, warnings
    except OSError as error:
        extraction["status"] = "failed"
        extraction["reason_code"] = "ffmpeg_not_available"
        extraction["reason"] = "FFmpeg was not available for audio extraction."
        warnings.append(f"FFmpeg audio extraction could not start: {error}")
        return None, extraction, warnings

    extraction["ffmpeg_return_code"] = result.returncode
    extraction["stderr_summary"] = (result.stderr.strip() or None)
    if result.returncode != 0:
        extraction["status"] = "failed"
        extraction["reason_code"] = "audio_extraction_failed"
        extraction["reason"] = "FFmpeg audio extraction failed."
        return None, extraction, warnings
    if not output_path.exists() or output_path.stat().st_size == 0:
        extraction["status"] = "failed"
        extraction["reason_code"] = "empty_audio_data"
        extraction["reason"] = "FFmpeg produced an empty or missing WAV output."
        return None, extraction, warnings

    extraction["status"] = "completed"
    extraction["reason_code"] = None
    extraction["reason"] = None
    return output_path, extraction, warnings
