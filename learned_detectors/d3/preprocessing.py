from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Any


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def select_d3_window(duration_seconds: float | None, seed: int) -> dict[str, float]:
    duration = 3.0
    if duration_seconds is None or duration_seconds <= 3.0:
        start = 0.0
    else:
        rng = random.Random(seed)
        start = float(math.floor(rng.uniform(0.0, duration_seconds - 3.0)))
    return {"window_start_seconds": start, "window_end_seconds": start + duration, "window_duration_seconds": duration}


def extract_upstream_compatible_frames(
    video_path: Path,
    duration_seconds: float | None,
    seed: int,
    output_directory: Path | None = None,
    preserve_temporary_frames: bool = False,
) -> tuple[list[Any], dict[str, Any], list[str]]:
    import cv2

    warnings: list[str] = []
    window = select_d3_window(duration_seconds, seed)
    diagnostic_dir: Path | None = None
    if output_directory is not None:
        diagnostic_dir = output_directory / "d3_temporary_frames"
        if diagnostic_dir.exists():
            shutil.rmtree(diagnostic_dir)
        if preserve_temporary_frames:
            diagnostic_dir.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], _preprocessing_record(window, [], 0, preserve_temporary_frames, diagnostic_dir), ["OpenCV could not open the video for D3 preprocessing."]
    timestamps = [round(window["window_start_seconds"] + index / 8.0, 6) for index in range(24)]
    frames: list[Any] = []
    selected_timestamps: list[float] = []
    try:
        for timestamp in timestamps:
            if timestamp >= window["window_end_seconds"] - 1e-9:
                break
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if preserve_temporary_frames and diagnostic_dir is not None:
                frame_path = diagnostic_dir / f"d3_frame_{len(frames):03d}_{timestamp:.6f}s.jpg"
                cv2.imwrite(str(frame_path), frame_bgr)
            frames.append(preprocess_frame_rgb(frame_rgb))
            selected_timestamps.append(timestamp)
    finally:
        capture.release()

    total = len(frames)
    if total >= 16:
        frames = frames[:16]
        selected_timestamps = selected_timestamps[:16]
    elif total >= 8:
        frames = frames[:8]
        selected_timestamps = selected_timestamps[:8]
    if not preserve_temporary_frames and diagnostic_dir is not None and diagnostic_dir.exists():
        shutil.rmtree(diagnostic_dir)
    record = _preprocessing_record(window, selected_timestamps, total, preserve_temporary_frames, diagnostic_dir)
    return frames, record, warnings


def preprocess_frame_rgb(frame_rgb: Any) -> Any:
    import cv2
    import numpy as np

    cropped = crop_center_by_percentage(frame_rgb, 0.1)
    resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype("float32") / 255.0
    mean = np.array(IMAGENET_MEAN, dtype="float32")
    std = np.array(IMAGENET_STD, dtype="float32")
    return (arr - mean) / std


def crop_center_by_percentage(image: Any, percentage: float) -> Any:
    height, width = image.shape[:2]
    if width > height:
        trim = int(width * percentage)
        return image[:, trim : width - trim]
    trim = int(height * percentage)
    return image[trim : height - trim, :]


def frames_to_torch_tensor(frames: list[Any], device: str) -> Any:
    import numpy as np
    import torch

    stacked = np.stack([frame.transpose(2, 0, 1) for frame in frames], axis=0)
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(device)


def _preprocessing_record(
    window: dict[str, float],
    timestamps: list[float],
    raw_count: int,
    preserve_temporary_frames: bool = False,
    diagnostic_dir: Path | None = None,
) -> dict[str, Any]:
    return {
        **window,
        "requested_frame_rate": 8.0,
        "raw_decoded_frame_count": raw_count,
        "actual_selected_frame_count": len(timestamps),
        "selected_frame_timestamps_seconds": timestamps,
        "frame_count_policy": "8 frames when fewer than 16 are available, otherwise 16 frames",
        "crop_strategy": "center crop by removing 10 percent from each side of the longer dimension",
        "resize_width": 224,
        "resize_height": 224,
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "max_pixel_value": 255.0,
            "color_order": "RGB",
        },
        "temporary_frames": {
            "preserved": preserve_temporary_frames,
            "directory": diagnostic_dir.name if preserve_temporary_frames and diagnostic_dir is not None else None,
            "cleanup_attempted": not preserve_temporary_frames,
        },
    }
