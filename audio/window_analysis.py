from __future__ import annotations

from typing import Any

import numpy as np

from config import AUDIO_HOP_SECONDS, AUDIO_WINDOW_SECONDS
from audio.metrics import window_metrics


def analyze_windows(samples: np.ndarray, sample_rate: int) -> list[dict[str, Any]]:
    if sample_rate <= 0 or samples.size == 0:
        return []
    window_size = max(1, int(round(AUDIO_WINDOW_SECONDS * sample_rate)))
    hop_size = max(1, int(round(AUDIO_HOP_SECONDS * sample_rate)))
    windows: list[dict[str, Any]] = []
    start = 0
    while start < samples.shape[0]:
        end = min(samples.shape[0], start + window_size)
        metrics = window_metrics(samples, sample_rate, start, end)
        windows.append(
            {
                "audio_window_id": f"audio-window-{len(windows):05d}",
                "window_index": len(windows),
                "start_timestamp_seconds": round(start / sample_rate, 6),
                "end_timestamp_seconds": round(end / sample_rate, 6),
                "actual_duration_seconds": round((end - start) / sample_rate, 6),
                "sample_count": end - start,
                **metrics,
            }
        )
        if end == samples.shape[0]:
            break
        start += hop_size
    return windows
