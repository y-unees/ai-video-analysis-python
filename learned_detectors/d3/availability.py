from __future__ import annotations

import importlib.util
from typing import Any

from learned_detectors.base import DetectorAvailability
from learned_detectors.d3.configuration import SUPPORTED_DISTANCES, SUPPORTED_ENCODERS


def check_d3_availability(config: dict[str, Any], learned_detectors_enabled: bool) -> DetectorAvailability:
    if not learned_detectors_enabled:
        return DetectorAvailability("disabled", "global_detector_layer_disabled", "Global learned detectors are disabled.")
    if not config.get("enabled"):
        return DetectorAvailability("disabled", "detector_disabled", "D3 is disabled.")
    if config.get("preprocessing_mode") != "upstream_compatible":
        return DetectorAvailability(
            "unavailable",
            "unsupported_preprocessing_mode",
            f"Unsupported D3 preprocessing mode: {config.get('preprocessing_mode')}",
        )
    encoder = config.get("encoder")
    distance = config.get("distance_mode")
    if encoder not in SUPPORTED_ENCODERS:
        return DetectorAvailability("unavailable", "invalid_encoder", f"Unsupported D3 encoder: {encoder}")
    if distance not in SUPPORTED_DISTANCES:
        return DetectorAvailability("unavailable", "invalid_distance_mode", f"Unsupported D3 distance mode: {distance}")

    required = ["torch", "torchvision"]
    if encoder in {"CLIP-16", "CLIP-32", "XCLIP-16", "XCLIP-32", "DINO-base", "DINO-large"}:
        required.append("transformers")
    if encoder == "MobileNet-v3":
        required.append("timm")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        return DetectorAvailability(
            "unavailable",
            "optional_dependencies_missing",
            "Required optional D3 dependencies are not installed.",
            {"missing": missing},
        )
    return DetectorAvailability("completed", None, "D3 optional dependencies are available.")
