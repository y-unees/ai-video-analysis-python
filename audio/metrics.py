from __future__ import annotations

from typing import Any

import numpy as np

from config import AUDIO_CLIPPING_AMPLITUDE_THRESHOLD, AUDIO_SILENCE_SAMPLE_AMPLITUDE_THRESHOLD, AUDIO_SILENCE_WINDOW_RMS_THRESHOLD


def calculate_global_metrics(samples: np.ndarray) -> dict[str, Any]:
    if samples.size == 0:
        return _empty_metrics()
    mono = np.mean(samples, axis=1)
    channel_metrics = []
    for channel_index in range(samples.shape[1]):
        channel = samples[:, channel_index]
        channel_metrics.append(
            {
                "channel_index": channel_index,
                "rms_amplitude": _rms(channel),
                "peak_absolute_amplitude": _peak(channel),
            }
        )
    rms_values = [item["rms_amplitude"] for item in channel_metrics]
    max_rms = max(rms_values) if rms_values else 0.0
    min_rms = min(rms_values) if rms_values else 0.0
    return {
        "rms_amplitude": _rms(mono),
        "peak_absolute_amplitude": _peak(mono),
        "mean_amplitude": _round(float(np.mean(mono))),
        "dc_offset": _round(float(np.mean(mono))),
        "clipping_ratio": _round(float(np.mean(np.abs(samples) >= AUDIO_CLIPPING_AMPLITUDE_THRESHOLD)), 6),
        "silence_ratio": _round(float(np.mean(np.abs(mono) < AUDIO_SILENCE_SAMPLE_AMPLITUDE_THRESHOLD)), 6),
        "zero_crossing_rate": zero_crossing_rate(mono),
        "channel_metrics": channel_metrics,
        "channel_rms_difference": _round(max_rms - min_rms),
        "channel_imbalance_ratio": _round((max_rms - min_rms) / max_rms, 6) if max_rms else 0.0,
    }


def window_metrics(samples: np.ndarray, sample_rate: int, start: int, end: int) -> dict[str, Any]:
    window = samples[start:end]
    mono = np.mean(window, axis=1) if window.size else np.array([], dtype=np.float32)
    return {
        "rms_amplitude": _rms(mono),
        "peak_absolute_amplitude": _peak(mono),
        "mean_amplitude": _round(float(np.mean(mono))) if mono.size else 0.0,
        "zero_crossing_rate": zero_crossing_rate(mono),
        "clipping_ratio": _round(float(np.mean(np.abs(window) >= AUDIO_CLIPPING_AMPLITUDE_THRESHOLD)), 6) if window.size else 0.0,
        "silence_like": _rms(mono) <= AUDIO_SILENCE_WINDOW_RMS_THRESHOLD,
        "spectral_centroid_hz": spectral_centroid(mono, sample_rate),
    }


def zero_crossing_rate(signal: np.ndarray) -> float:
    if signal.size < 2:
        return 0.0
    signs = np.signbit(signal)
    return _round(float(np.mean(signs[1:] != signs[:-1])), 6)


def spectral_centroid(signal: np.ndarray, sample_rate: int) -> float | None:
    if signal.size == 0 or sample_rate <= 0:
        return None
    spectrum = np.abs(np.fft.rfft(signal))
    total = float(np.sum(spectrum))
    if total == 0:
        return 0.0
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    return _round(float(np.sum(freqs * spectrum) / total))


def _empty_metrics() -> dict[str, Any]:
    return {
        "rms_amplitude": 0.0,
        "peak_absolute_amplitude": 0.0,
        "mean_amplitude": 0.0,
        "dc_offset": 0.0,
        "clipping_ratio": 0.0,
        "silence_ratio": 1.0,
        "zero_crossing_rate": 0.0,
        "channel_metrics": [],
        "channel_rms_difference": 0.0,
        "channel_imbalance_ratio": 0.0,
    }


def _rms(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return _round(float(np.sqrt(np.mean(np.square(signal)))))


def _peak(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return _round(float(np.max(np.abs(signal))))


def _round(value: float, digits: int = 6) -> float:
    return round(value, digits)
