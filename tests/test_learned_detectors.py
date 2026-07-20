from __future__ import annotations

import math
import json
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from learned_detectors.d3.availability import check_d3_availability
from learned_detectors.d3.adapter import D3Detector, run_worker_with_timeout
from learned_detectors.d3.configuration import ENCODER_REGISTRY, SCORE_DIRECTION_RECORD, SUPPORTED_ENCODERS
from learned_detectors.d3.inference import first_order_values, second_order_values, summarize_second_order
from learned_detectors.d3.preprocessing import crop_center_by_percentage, preprocess_frame_rgb, select_d3_window
from report_validator import validate_report


def _sleeping_worker(seconds: float, queue: object) -> None:
    time.sleep(seconds)
    queue.put(("ok", {"slept": seconds}))


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "enabled": True,
        "encoder": "XCLIP-16",
        "distance_mode": "l2",
        "random_seed": 42,
        "device": "auto",
        "allow_model_download": False,
        "model_cache_directory": "",
        "preprocessing_mode": "upstream_compatible",
    }
    config.update(overrides)
    return config


class LearnedDetectorAvailabilityTests(unittest.TestCase):
    def test_global_disabled_takes_precedence(self) -> None:
        status = check_d3_availability(_config(), learned_detectors_enabled=False)
        self.assertEqual(status.status, "disabled")
        self.assertEqual(status.reason_code, "global_detector_layer_disabled")

    def test_d3_disabled_is_reported(self) -> None:
        status = check_d3_availability(_config(enabled=False), learned_detectors_enabled=True)
        self.assertEqual(status.status, "disabled")
        self.assertEqual(status.reason_code, "detector_disabled")

    def test_invalid_encoder_and_distance_are_unavailable(self) -> None:
        encoder_status = check_d3_availability(_config(encoder="NopeNet"), learned_detectors_enabled=True)
        distance_status = check_d3_availability(_config(distance_mode="probability"), learned_detectors_enabled=True)
        self.assertEqual(encoder_status.reason_code, "invalid_encoder")
        self.assertEqual(distance_status.reason_code, "invalid_distance_mode")

    def test_missing_optional_dependencies_are_explicit(self) -> None:
        with patch("learned_detectors.d3.availability.importlib.util.find_spec", return_value=None):
            status = check_d3_availability(_config(), learned_detectors_enabled=True)
        self.assertEqual(status.status, "unavailable")
        self.assertEqual(status.reason_code, "optional_dependencies_missing")
        self.assertIn("torch", status.details["missing"])
        self.assertIn("transformers", status.details["missing"])

    def test_unsupported_preprocessing_mode_is_unavailable(self) -> None:
        status = check_d3_availability(_config(preprocessing_mode="experimental"), learned_detectors_enabled=True)
        self.assertEqual(status.status, "unavailable")
        self.assertEqual(status.reason_code, "unsupported_preprocessing_mode")

    def test_encoder_registry_matches_supported_names(self) -> None:
        self.assertEqual(set(ENCODER_REGISTRY), set(SUPPORTED_ENCODERS))
        self.assertEqual(ENCODER_REGISTRY["XCLIP-16"]["model_identifier"], "microsoft/xclip-base-patch16")
        self.assertIn(ENCODER_REGISTRY["XCLIP-16"]["support_status"], {"verified", "implemented_unverified"})


class D3MathTests(unittest.TestCase):
    def test_first_and_second_order_l2(self) -> None:
        embeddings = [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0], [6.0, 10.0]]
        first = first_order_values(embeddings, "l2")
        second = second_order_values(first)
        self.assertEqual(first, [5.0, 5.0, 2.0])
        self.assertEqual(second, [0.0, -3.0])

    def test_first_order_cosine_matches_upstream_similarity(self) -> None:
        embeddings = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        first = first_order_values(embeddings, "cos")
        self.assertAlmostEqual(first[0], 0.0)
        self.assertAlmostEqual(first[1], 1 / math.sqrt(2))

    def test_second_order_summary_uses_torch_default_sample_std(self) -> None:
        summary = summarize_second_order([3.0, 5.0])
        self.assertEqual(summary["second_order_mean"], 4.0)
        self.assertAlmostEqual(summary["second_order_standard_deviation"], math.sqrt(2.0))

    def test_score_direction_stays_neutral_when_upstream_path_conflicts(self) -> None:
        self.assertEqual(SCORE_DIRECTION_RECORD["status"], "conflicting")
        self.assertEqual(SCORE_DIRECTION_RECORD["higher_score_indicates"], "unknown")


class D3PreprocessingTests(unittest.TestCase):
    def test_select_window_is_upstream_seeded_three_second_clip(self) -> None:
        self.assertEqual(select_d3_window(2.5, 42)["window_start_seconds"], 0.0)
        self.assertEqual(select_d3_window(10.0, 42)["window_start_seconds"], 4.0)

    def test_crop_removes_ten_percent_from_longer_dimension(self) -> None:
        wide = np.zeros((10, 20, 3), dtype=np.uint8)
        tall = np.zeros((20, 10, 3), dtype=np.uint8)
        self.assertEqual(crop_center_by_percentage(wide, 0.1).shape, (10, 16, 3))
        self.assertEqual(crop_center_by_percentage(tall, 0.1).shape, (16, 10, 3))

    def test_preprocess_returns_upstream_shape_and_float_scale(self) -> None:
        frame = np.ones((20, 30, 3), dtype=np.uint8) * 128
        processed = preprocess_frame_rgb(frame)
        self.assertEqual(processed.shape, (224, 224, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(np.isfinite(processed).all())


class D3TimeoutAndValidationTests(unittest.TestCase):
    def test_worker_timeout_terminates_slow_process(self) -> None:
        status, payload = run_worker_with_timeout(_sleeping_worker, (2.0,), 1)
        self.assertEqual(status, "timed_out")
        self.assertIsNone(payload)

    def test_worker_success_before_timeout(self) -> None:
        status, payload = run_worker_with_timeout(_sleeping_worker, (0.05,), 2)
        self.assertEqual(status, "ok")
        self.assertEqual(payload["slept"], 0.05)

    def test_base_result_uses_explicit_safe_native_output_fields(self) -> None:
        detector = D3Detector(_config(timeout_seconds=300), learned_detectors_enabled=True)
        result = detector._base_result(  # noqa: SLF001 - focused schema regression test.
            __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            __import__("pathlib").Path("source_videos/sample.mp4"),
            "abc",
            {"container": {"duration_seconds": 1.0}},
        )
        native = result["native_output"]
        self.assertIsNone(native["probability"])
        self.assertIsNone(native["threshold"])
        self.assertEqual(native["classification"], "not_assigned")
        self.assertEqual(native["calibration_status"], "uncalibrated")
        self.assertIsNone(native["raw_score"])

    def test_completed_artifact_records_completed_execution(self) -> None:
        detector = D3Detector(_config(timeout_seconds=300), learned_detectors_enabled=True)
        result = detector._base_result(  # noqa: SLF001 - focused artifact regression test.
            datetime.now(timezone.utc),
            Path("source_videos/sample.mp4"),
            "abc",
            {"container": {"duration_seconds": 1.0}},
        )
        result["preprocessing"] = {"selected_frame_timestamps_seconds": [0.0, 0.1, 0.2]}
        result["execution"]["status"] = "completed"
        result["execution"]["message"] = "D3 completed."
        detector._finish(result, time.perf_counter())  # noqa: SLF001 - focused artifact regression test.
        with tempfile.TemporaryDirectory() as directory:
            detector._write_artifacts(Path(directory), result, [1.0, 2.0], [1.0])  # noqa: SLF001
            artifact = json.loads((Path(directory) / "d3_detector_result.json").read_text(encoding="utf-8"))
        self.assertEqual(artifact["execution"]["status"], "completed")
        self.assertIsNotNone(artifact["execution"]["completed_at_utc"])
        self.assertIsNotNone(artifact["execution"]["duration_seconds"])

    def test_validator_rejects_non_completed_raw_score_and_confidence(self) -> None:
        report = {
            "schema_version": "0.7",
            "analysis_environment": {"application_version": "0.9.3"},
            "frame_analysis": {"frames": [], "comparisons": [], "summary": {"frames_analyzed": 0, "heuristic_near_black_frame_count": 0, "heuristic_near_white_frame_count": 0, "heuristic_large_change_pair_count": 0, "heuristic_near_duplicate_pair_count": 0}},
            "sampling": {"decoded_frame_count": 0},
            "observations": {},
            "temporal_analysis": {"status": "skipped", "summary": {}, "scenes": [], "notable_intervals": [], "notable_transitions": [], "artifacts": {}},
            "audio_analysis": {"status": "skipped", "reason_code": "test", "decoded_audio": {}, "global_metrics": {}, "silence_intervals": [], "notable_transitions": [], "artifacts": {}},
            "visual_consistency_analysis": {"status": "skipped", "reason_code": "test", "configuration": {}, "summary": {}, "transition_summaries": [], "sustained_intervals": [], "ranked_review_transitions": [], "artifacts": {}},
            "unified_evidence": {"status": "pending", "validation": {}, "review_highlights": []},
            "learned_detector_results": {
                "status": "completed",
                "standalone_not_fused_with_unified_evidence": True,
                "d3": {
                    "detector": {"detector_id": "d3"},
                    "execution": {"status": "unavailable", "reason_code": "optional_dependencies_missing"},
                    "input": {"video_sha256": "abc", "relative_video_path": "source_videos/sample.mp4"},
                    "configuration": {"encoder": "XCLIP-16", "distance_mode": "l2"},
                    "native_output": {"raw_score": 0.5, "probability": None, "threshold": None, "classification": "not_assigned", "calibration_status": "uncalibrated", "confidence": 0.5},
                    "artifacts": {},
                },
            },
            "source": {"sha256": "abc"},
            "evidence": [],
            "compatibility": {"legacy_evidence_view_included": False},
            "artifacts": {"paths_relative_to": "report_directory", "raw_ffprobe_report": "ffprobe_raw.json", "frames_directory": "frames"},
            "analysis": {"started_at": "2026-07-14T00:00:00+00:00", "completed_at": "2026-07-14T00:00:01+00:00"},
        }
        errors = validate_report(report)["errors"]
        self.assertTrue(any("Non-completed D3 result" in error for error in errors))
        self.assertTrue(any("forbidden verdict/confidence field" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
