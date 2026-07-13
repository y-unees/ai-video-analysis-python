from __future__ import annotations

import wave
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

PathInput = str | PathLike[str] | Path


def read_pcm_wav(path: PathInput) -> tuple[np.ndarray | None, dict[str, Any], list[str]]:
    path = Path(path)
    warnings: list[str] = []
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frame_count = wav.getnframes()
            raw = wav.readframes(frame_count)
    except (wave.Error, OSError) as error:
        return None, {"status": "failed", "reason_code": "wav_decode_failed", "reason": "WAV decoding failed.", "diagnostics": {"exception_type": type(error).__name__, "exception_message": str(error)}}, warnings

    decoded = {
        "status": "completed",
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "decoded_duration_seconds": round(frame_count / sample_rate, 6) if sample_rate else None,
        "sample_format": None,
        "normalization_range": "-1.0_to_1.0",
    }
    if sample_rate <= 0 or channels <= 0 or frame_count < 0:
        decoded["status"] = "failed"
        decoded["reason_code"] = "wav_decode_failed"
        decoded["reason"] = "invalid_wav_header"
        return None, decoded, warnings
    if sample_width != 2:
        decoded["status"] = "skipped"
        decoded["reason_code"] = "unsupported_pcm_format"
        decoded["reason"] = "unsupported_pcm_sample_width"
        return None, decoded, warnings

    samples = np.frombuffer(raw, dtype="<i2")
    if samples.size == 0:
        decoded["status"] = "failed"
        decoded["reason_code"] = "empty_audio_data"
        decoded["reason"] = "empty_pcm_data"
        return None, decoded, warnings
    usable = (samples.size // channels) * channels
    if usable != samples.size:
        warnings.append("PCM data ended with a partial frame; trailing samples were ignored.")
        samples = samples[:usable]
    if channels > 1:
        samples = samples.reshape(-1, channels)
    else:
        samples = samples.reshape(-1, 1)
    decoded["sample_format"] = "signed_pcm_16"
    return samples.astype(np.float32) / 32768.0, decoded, warnings
