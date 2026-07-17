from __future__ import annotations

import hashlib
import json
import multiprocessing
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from queue import Empty
from time import perf_counter
from typing import Any

from config import APP_VERSION
from file_utils import artifact_record, calculate_sha256_bytes, safe_relative_path
from learned_detectors.base import DetectorAvailability
from learned_detectors.d3.availability import check_d3_availability
from learned_detectors.d3.configuration import (
    D3_DETECTOR_VERSION,
    D3_PAPER_REFERENCE,
    D3_RESULT_SCHEMA_VERSION,
    D3_UPSTREAM_COMMIT,
    D3_UPSTREAM_COMMIT_DATE,
    D3_UPSTREAM_LICENSE,
    D3_UPSTREAM_REPOSITORY,
    ENCODER_REGISTRY,
    METHOD_VERIFICATION,
    SCORE_DIRECTION_RECORD,
    SUPPORTED_ENCODERS,
    TRANSFORMER_ENCODERS,
)
from learned_detectors.d3.inference import torch_forward_score
from learned_detectors.d3.preprocessing import extract_upstream_compatible_frames, frames_to_torch_tensor


D3_LIMITATIONS = [
    "The raw score is uncalibrated in this application.",
    "No local reference distribution has been established.",
    "No operating threshold has been validated.",
    "The result is not a probability.",
    "Score behavior may vary by encoder and distance metric.",
    "Only one upstream-compatible temporal window is analyzed in v0.8.",
    "Results may be affected by frame rate, compression, editing, and duration.",
    "Deterministic forensic observations and D3 are not fused in v0.8.",
    "The detector must not be used as the sole basis for a consequential decision.",
]


class D3Detector:
    detector_id = "d3"
    detector_version = D3_DETECTOR_VERSION

    def __init__(self, config: dict[str, Any], learned_detectors_enabled: bool) -> None:
        self.config = config
        self.learned_detectors_enabled = learned_detectors_enabled

    def check_availability(self) -> DetectorAvailability:
        return check_d3_availability(self.config, self.learned_detectors_enabled)

    def analyze(
        self,
        video_path: Path,
        video_sha256: str,
        output_directory: Path,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        started = datetime.now(timezone.utc)
        start_perf = perf_counter()
        availability = self.check_availability()
        base = self._base_result(started, video_path, video_sha256, metadata)
        if availability.status != "completed":
            base["execution"].update(
                {
                    "status": availability.status,
                    "reason_code": availability.reason_code,
                    "message": availability.message,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "duration_seconds": round(perf_counter() - start_perf, 3),
                }
            )
            if availability.details:
                base["execution"]["details"] = availability.details
            return base
        return self._analyze_with_timeout(video_path, video_sha256, output_directory, metadata)

    def _analyze_with_timeout(
        self,
        video_path: Path,
        video_sha256: str,
        output_directory: Path,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timeout_seconds = int(self.config.get("timeout_seconds", 300))
        started = datetime.now(timezone.utc)
        start_perf = perf_counter()
        base = self._base_result(started, video_path, video_sha256, metadata)
        status, payload = run_worker_with_timeout(
            _d3_available_worker,
            (self.config, str(video_path), video_sha256, str(output_directory), metadata),
            timeout_seconds,
        )
        if status == "timed_out":
            base["execution"].update(
                {
                    "status": "timed_out",
                    "reason_code": "detector_timeout",
                    "timeout_seconds": timeout_seconds,
                    "message": f"D3 exceeded the configured timeout of {timeout_seconds} seconds.",
                    "error": None,
                }
            )
            self._finish(base, start_perf)
            return base
        if status == "empty":
            base["execution"].update(
                {
                    "status": "failed",
                    "reason_code": "unexpected_detector_failure",
                    "timeout_seconds": timeout_seconds,
                    "error": "D3 worker exited without returning a result.",
                }
            )
            self._finish(base, start_perf)
            return base
        if status == "ok":
            return payload
        base["execution"].update(
            {
                "status": "failed",
                "reason_code": "unexpected_detector_failure",
                "timeout_seconds": timeout_seconds,
                "error": _sanitize(str(payload)),
            }
        )
        self._finish(base, start_perf)
        return base

    def _analyze_available(
        self,
        video_path: Path,
        video_sha256: str,
        output_directory: Path,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        started = datetime.now(timezone.utc)
        start_perf = perf_counter()
        base = self._base_result(started, video_path, video_sha256, metadata)
        try:
            try:
                frames, preprocessing, preprocess_warnings = extract_upstream_compatible_frames(
                    video_path,
                    metadata.get("container", {}).get("duration_seconds"),
                    int(self.config["random_seed"]),
                    output_directory=output_directory,
                    preserve_temporary_frames=bool(self.config.get("preserve_temporary_frames")),
                )
            except Exception as error:
                raise RuntimeError(f"preprocessing_failure: {_sanitize(str(error))}") from error
            base["preprocessing"] = preprocessing
            base["warnings"].extend(preprocess_warnings)
            if len(frames) < 8:
                base["execution"].update(
                    {
                        "status": "skipped",
                        "reason_code": "insufficient_frames",
                        "message": "D3 requires at least eight valid frames under the selected preprocessing configuration.",
                    }
                )
                self._finish(base, start_perf)
                return base

            device = self._select_device()
            base["execution"]["device_used"] = device
            try:
                model = self._load_model(device)
            except Exception as error:
                raise RuntimeError(f"model_loading_failure: {_sanitize(str(error))}") from error
            base["reproducibility"]["model_asset_source"] = (
                "download_allowed_or_cache" if self.config.get("allow_model_download") else "local_cache_only"
            )
            base["reproducibility"]["model_cache_status"] = "loader_invoked"
            tensor = frames_to_torch_tensor(frames, device)
            base["reproducibility"]["frame_tensor_sha256"] = _tensor_sha256(tensor)
            base["reproducibility"]["tensor_dtype"] = str(tensor.dtype).replace("torch.", "")
            try:
                embeddings, first_order, second_order, summary = torch_forward_score(model, tensor)
            except Exception as error:
                raise RuntimeError(f"inference_failure: {_sanitize(str(error))}") from error
            if not all(_finite(value) for value in first_order + second_order):
                raise ValueError("D3 produced a non-finite temporal feature value.")
            base["native_output"] = {
                "score_name": "native_d3_second_order_standard_deviation",
                "raw_score": summary["official_torch_second_order_standard_deviation"],
                "score_direction": "not_verified",
                "score_direction_record": SCORE_DIRECTION_RECORD,
                "probability": None,
                "threshold": None,
                "classification": "not_assigned",
                "calibration_status": "uncalibrated",
            }
            base["feature_summary"] = {
                "embedding_dimension": int(embeddings.shape[-1]),
                "first_order_value_count": len(first_order),
                **summary,
            }
            try:
                artifacts = self._write_artifacts(output_directory, base, first_order, second_order)
            except Exception as error:
                raise RuntimeError(f"artifact_generation_failure: {_sanitize(str(error))}") from error
            base["artifacts"] = artifacts
            base["execution"]["status"] = "completed"
            base["execution"]["message"] = "D3 completed. The raw score is uncalibrated and no classification threshold was applied."
            self._populate_runtime_versions(base)
            self._finish(base, start_perf)
            return base
        except Exception as error:
            text = str(error)
            reason_code = "unexpected_detector_failure"
            for candidate in (
                "cuda_unavailable",
                "preprocessing_failure",
                "model_loading_failure",
                "inference_failure",
                "artifact_generation_failure",
            ):
                if text.startswith(candidate):
                    reason_code = candidate
                    text = text.split(":", 1)[-1].strip()
                    break
            base["execution"].update({"status": "failed", "reason_code": reason_code, "error": _sanitize(text)})
            self._finish(base, start_perf)
            return base

    def _base_result(self, started: datetime, video_path: Path, video_sha256: str, metadata: dict[str, Any]) -> dict[str, Any]:
        configuration_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode("utf-8")).hexdigest()
        return {
            "schema_version": "0.8",
            "detector": {
                "detector_id": "d3",
                "detector_name": "Detection by Difference of Differences",
                "method_category": "training_free_second_order_temporal_detector",
                "upstream_repository": D3_UPSTREAM_REPOSITORY,
                "upstream_commit": D3_UPSTREAM_COMMIT,
                "upstream_commit_date": D3_UPSTREAM_COMMIT_DATE,
                "upstream_license": D3_UPSTREAM_LICENSE,
                "paper_reference": D3_PAPER_REFERENCE,
            },
            "execution": {
                "status": "failed",
                "started_at_utc": started.isoformat(timespec="milliseconds"),
                "completed_at_utc": None,
                "duration_seconds": None,
                "device_requested": self.config.get("device"),
                "device_used": None,
                "reason_code": None,
                "message": None,
                "error": None,
                "timeout_seconds": self.config.get("timeout_seconds"),
            },
            "input": {
                "video_sha256": video_sha256,
                "relative_video_path": safe_relative_path(video_path),
                "duration_seconds": metadata.get("container", {}).get("duration_seconds"),
            },
            "configuration": {
                "encoder": self.config.get("encoder"),
                "distance_mode": self.config.get("distance_mode"),
                "random_seed": self.config.get("random_seed"),
                "preprocessing_mode": self.config.get("preprocessing_mode"),
                "allow_model_download": self.config.get("allow_model_download"),
                "timeout_seconds": self.config.get("timeout_seconds"),
                "preserve_temporary_frames": self.config.get("preserve_temporary_frames"),
            },
            "preprocessing": {},
            "native_output": {
                "score_name": "native_d3_second_order_standard_deviation",
                "raw_score": None,
                "score_direction": "not_verified",
                "score_direction_record": SCORE_DIRECTION_RECORD,
                "probability": None,
                "threshold": None,
                "classification": "not_assigned",
                "calibration_status": "uncalibrated",
            },
            "feature_summary": {},
            "artifacts": {},
            "method_verification": METHOD_VERIFICATION,
            "reproducibility": {
                "application_version": APP_VERSION,
                "d3_result_schema_version": D3_RESULT_SCHEMA_VERSION,
                "upstream_repository": D3_UPSTREAM_REPOSITORY,
                "upstream_commit": D3_UPSTREAM_COMMIT,
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "architecture": platform.machine(),
                "torch_version": None,
                "transformers_version": None,
                "torchvision_version": None,
                "timm_version": None,
                "numpy_version": _version_or_none("numpy"),
                "opencv_version": _version_or_none("opencv-python") or _version_or_none("opencv-python-headless"),
                "ffmpeg_version": None,
                "encoder_registry_entry": ENCODER_REGISTRY.get(str(self.config.get("encoder"))),
                "encoder_model_identifier": SUPPORTED_ENCODERS.get(str(self.config.get("encoder"))),
                "model_asset_source": "not_loaded",
                "model_cache_status": "not_checked",
                "model_cache_directory": _sanitize_path(str(self.config.get("model_cache_directory") or "")),
                "configuration_sha256": configuration_hash,
                "random_seed": self.config.get("random_seed"),
                "cuda_available": None,
                "cuda_version": None,
                "cudnn_version": None,
                "deterministic_algorithms": None,
                "preprocessing_configuration_sha256": None,
                "frame_tensor_sha256": None,
                "tensor_dtype": None,
            },
            "warnings": [],
            "limitations": D3_LIMITATIONS,
        }

    def _select_device(self) -> str:
        import torch

        requested = self.config.get("device", "auto")
        if requested == "cpu":
            return "cpu"
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("cuda_unavailable: D3_DEVICE=cuda was requested but CUDA is not available.")
            return "cuda"
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        raise RuntimeError(f"Unsupported D3 device: {requested}")

    def _load_model(self, device: str) -> Any:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        encoder_type = str(self.config["encoder"])
        distance_mode = str(self.config["distance_mode"])
        allow_download = bool(self.config.get("allow_model_download"))
        cache_dir = self.config.get("model_cache_directory") or None

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss_type = distance_mode
                self.encoder_type = encoder_type
                self.encoder = _load_encoder(encoder_type, allow_download, cache_dir)

            def forward(self, x: Any) -> tuple[Any, Any, Any]:
                b, t, _channels, h, w = x.shape
                images = x.reshape(-1, 3, h, w)
                if encoder_type in TRANSFORMER_ENCODERS:
                    outputs = self.encoder(images, output_hidden_states=True).pooler_output
                else:
                    outputs = self.encoder(images)
                outputs = outputs.reshape(b, t, -1)
                vec1 = outputs[:, :-1, :]
                vec2 = outputs[:, 1:, :]
                if distance_mode == "cos":
                    first = F.cosine_similarity(vec1, vec2, dim=-1)
                else:
                    first = torch.norm(vec1 - vec2, p=2, dim=-1)
                second = first[:, 1:] - first[:, :-1]
                return outputs, torch.mean(second, dim=1), torch.std(second, dim=1)

        model = Model().to(device)
        model.eval()
        return model

    def _populate_runtime_versions(self, result: dict[str, Any]) -> None:
        try:
            import torch
            result["reproducibility"]["torch_version"] = torch.__version__
            result["reproducibility"]["cuda_available"] = bool(torch.cuda.is_available())
            result["reproducibility"]["cuda_version"] = torch.version.cuda
            result["reproducibility"]["cudnn_version"] = torch.backends.cudnn.version()
            result["reproducibility"]["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        except Exception:
            pass
        result["reproducibility"]["transformers_version"] = _version_or_none("transformers")
        result["reproducibility"]["torchvision_version"] = _version_or_none("torchvision")
        result["reproducibility"]["timm_version"] = _version_or_none("timm")
        preprocessing_bytes = json.dumps(result.get("preprocessing", {}), sort_keys=True).encode("utf-8")
        result["reproducibility"]["preprocessing_configuration_sha256"] = calculate_sha256_bytes(preprocessing_bytes)

    def _write_artifacts(
        self,
        output_directory: Path,
        result: dict[str, Any],
        first_order: list[float],
        second_order: list[float],
    ) -> dict[str, Any]:
        result_path = output_directory / "d3_detector_result.json"
        trace_path = output_directory / "d3_temporal_features.jsonl"
        timestamps = result["preprocessing"].get("selected_frame_timestamps_seconds", [])
        with trace_path.open("w", encoding="utf-8", newline="\n") as file:
            for index, value in enumerate(first_order):
                file.write(json.dumps({
                    "record_type": "d3_first_order_feature",
                    "sequence_index": index,
                    "previous_frame_timestamp_seconds": timestamps[index],
                    "current_frame_timestamp_seconds": timestamps[index + 1],
                    "first_order_value": value,
                }, sort_keys=True) + "\n")
            for index, value in enumerate(second_order):
                file.write(json.dumps({
                    "record_type": "d3_second_order_feature",
                    "sequence_index": index,
                    "previous_frame_timestamp_seconds": timestamps[index],
                    "center_frame_timestamp_seconds": timestamps[index + 1],
                    "next_frame_timestamp_seconds": timestamps[index + 2],
                    "first_order_previous": first_order[index],
                    "first_order_current": first_order[index + 1],
                    "second_order_difference": value,
                }, sort_keys=True) + "\n")
        trace_record = {
            **artifact_record(trace_path, "d3_temporal_features.jsonl"),
            "content_type": "application/jsonl",
            "producing_stage": "d3_learned_detector",
        }
        result_copy = {key: value for key, value in result.items() if key != "artifacts"}
        result_copy["artifacts"] = {"d3_temporal_features": trace_record}
        result_path.write_text(json.dumps(result_copy, indent=2), encoding="utf-8")
        return {
            "d3_detector_result": {
                **artifact_record(result_path, "d3_detector_result.json"),
                "content_type": "application/json",
                "producing_stage": "d3_learned_detector",
            },
            "d3_temporal_features": trace_record,
        }

    def _finish(self, result: dict[str, Any], start_perf: float) -> None:
        result["execution"]["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        result["execution"]["duration_seconds"] = round(perf_counter() - start_perf, 3)


def _load_encoder(encoder_type: str, allow_download: bool, cache_dir: str | None) -> Any:
    local_only = not allow_download
    if encoder_type == "CLIP-16":
        from transformers import CLIPVisionModel
        return CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir=cache_dir, local_files_only=local_only)
    if encoder_type == "CLIP-32":
        from transformers import CLIPVisionModel
        return CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir, local_files_only=local_only)
    if encoder_type == "XCLIP-16":
        from transformers import XCLIPVisionModel
        return XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16", cache_dir=cache_dir, local_files_only=local_only)
    if encoder_type == "XCLIP-32":
        from transformers import XCLIPVisionModel
        return XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32", cache_dir=cache_dir, local_files_only=local_only)
    if encoder_type == "DINO-base":
        from transformers import AutoModel
        return AutoModel.from_pretrained("facebook/dinov2-base", cache_dir=cache_dir, local_files_only=local_only)
    if encoder_type == "DINO-large":
        from transformers import AutoModel
        return AutoModel.from_pretrained("facebook/dinov2-large", cache_dir=cache_dir, local_files_only=local_only)
    if encoder_type in {"ResNet-18", "VGG-16", "EfficientNet-b4"}:
        import torch
        import torchvision.models as models
        weights = None if not allow_download else "DEFAULT"
        if encoder_type == "ResNet-18":
            model = models.resnet18(weights=weights)
        elif encoder_type == "VGG-16":
            model = models.vgg16(weights=weights)
        else:
            model = models.efficientnet_b4(weights=weights)
        return torch.nn.Sequential(*list(model.children())[:-1]).eval()
    if encoder_type == "MobileNet-v3":
        import torch
        import timm
        model = timm.create_model("mobilenetv3_large_100", pretrained=allow_download)
        return torch.nn.Sequential(*list(model.children())[:-1]).eval()
    raise RuntimeError(f"Unsupported D3 encoder: {encoder_type}")


def _d3_available_worker(
    config: dict[str, Any],
    video_path: str,
    video_sha256: str,
    output_directory: str,
    metadata: dict[str, Any],
    queue: Any,
) -> None:
    try:
        result = D3Detector(config, learned_detectors_enabled=True)._analyze_available(
            Path(video_path),
            video_sha256,
            Path(output_directory),
            metadata,
        )
        queue.put(("ok", result))
    except Exception as error:
        queue.put(("error", _sanitize(str(error))))


def run_worker_with_timeout(target: Any, args_without_queue: tuple[Any, ...], timeout_seconds: int) -> tuple[str, Any]:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(target=target, args=(*args_without_queue, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        return "timed_out", None
    try:
        return queue.get_nowait()
    except Empty:
        return "empty", None


def _version_or_none(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _tensor_sha256(tensor: Any) -> str:
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def _sanitize_path(path: str) -> str | None:
    if not path:
        return None
    return Path(path).name


def _finite(value: float) -> bool:
    import math
    return math.isfinite(float(value))


def _sanitize(message: str) -> str:
    return message.replace("\\", "/").split("\n")[0][:500]
