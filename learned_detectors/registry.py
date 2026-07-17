from __future__ import annotations

from pathlib import Path
from typing import Any

from config import learned_detector_configuration
from learned_detectors.d3.adapter import D3Detector


def run_learned_detectors(
    video_path: Path,
    video_sha256: str,
    analysis_dir: Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    config = learned_detector_configuration()
    warnings: list[str] = []
    d3 = D3Detector(config["d3"], learned_detectors_enabled=bool(config["learned_detectors_enabled"]))
    result = d3.analyze(
        video_path=video_path,
        video_sha256=video_sha256,
        output_directory=analysis_dir,
        metadata=metadata,
    )
    if result.get("execution", {}).get("status") in {"failed", "timed_out"}:
        warnings.append(f"D3 learned detector {result['execution']['status']}: {result['execution'].get('reason_code')}")
    return {
        "status": "completed",
        "d3": result,
        "standalone_not_fused_with_unified_evidence": True,
        "limitations": [
            "Learned detector outputs are reported separately and are not fused with deterministic forensic evidence in v0.8.",
        ],
    }, warnings
