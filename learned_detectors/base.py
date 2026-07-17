from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


STATUS_DEFINITIONS = {
    "not_run": "The detector stage was never attempted, usually because an earlier fatal pipeline error prevented execution.",
    "disabled": "The learned-detector layer or detector was intentionally disabled by configuration.",
    "unavailable": "The detector was enabled, but dependencies, device support, or model assets were unavailable.",
    "skipped": "The detector was available, but the current input could not be analyzed.",
    "completed": "The requested detector processing completed successfully.",
    "failed": "The detector was attempted but encountered an unexpected execution failure.",
    "timed_out": "The detector exceeded the configured timeout and was terminated.",
}

ALLOWED_STATUSES = set(STATUS_DEFINITIONS)

REASON_CODE_DEFINITIONS = {
    "global_detector_layer_disabled": "Global learned detectors are disabled.",
    "detector_disabled": "This detector is disabled.",
    "optional_dependencies_missing": "Required optional Python packages are missing.",
    "encoder_assets_missing": "Required pretrained model assets are absent while downloads are disabled.",
    "cuda_unavailable": "CUDA was requested but is not available.",
    "insufficient_frames": "Fewer than eight valid D3 frames were available.",
    "unsupported_preprocessing_mode": "The configured preprocessing mode is not supported.",
    "detector_timeout": "Detector execution exceeded D3_TIMEOUT_SECONDS.",
    "preprocessing_failure": "Frame preprocessing failed.",
    "model_loading_failure": "Model loading failed.",
    "inference_failure": "Model inference failed.",
    "artifact_generation_failure": "D3 artifact generation failed.",
    "unexpected_detector_failure": "Unexpected detector failure.",
}


@dataclass(frozen=True)
class DetectorAvailability:
    status: str
    reason_code: str | None = None
    message: str | None = None
    details: dict[str, Any] | None = None


class LearnedVideoDetector(Protocol):
    detector_id: str
    detector_version: str

    def check_availability(self) -> DetectorAvailability:
        ...

    def analyze(
        self,
        video_path: Path,
        video_sha256: str,
        output_directory: Path,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        ...
