from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from outcome import (
    OUTCOME_FEATURE_SCHEMA_VERSION,
    build_outcome_features,
    create_outcome_features_artifact,
    validate_outcome_features,
)
from outcome.feature_builder import OUTCOME_FEATURE_FILENAME
from tests.test_gemini_evidence_report import _completed_d3, _report


class OutcomeFeatureTests(unittest.TestCase):
    def test_complete_input_builds_serializable_features(self) -> None:
        report = _outcome_report()
        features = build_outcome_features(report, analysis_id="analysis-001").to_dict()
        self.assertEqual(features["identity"]["feature_schema_version"], OUTCOME_FEATURE_SCHEMA_VERSION)
        self.assertEqual(features["identity"]["application_version"], "0.9.4")
        self.assertEqual(features["identity"]["analysis_id"], "analysis-001")
        self.assertEqual(features["identity"]["video_name"], "sample.mp4")
        self.assertEqual(features["identity"]["video_sha256"], "abc")
        self.assertIsNone(features["identity"]["expected_label"])
        self.assertEqual(features["media"]["frame_count"], 150)
        self.assertTrue(features["media"]["audio_present"])
        self.assertEqual(features["frame_summary"]["dark_frame_ratio"], 0.0)
        self.assertEqual(features["frame_summary"]["white_frame_ratio"], 0.0)
        self.assertEqual(features["temporal_summary"]["mean_motion_magnitude"], 1.5)
        self.assertEqual(features["audio_summary"]["maximum_audio_energy_change"], 4.0)
        self.assertEqual(features["visual_consistency_summary"]["maximum_unstable_region_count"], 3)
        self.assertEqual(features["unified_evidence_summary"]["high_priority_event_count"], 1)
        self.assertEqual(features["d3_summary"]["d3_status"], "completed")
        self.assertEqual(features["d3_summary"]["d3_calibration_status"], "uncalibrated")
        self.assertEqual(features["d3_summary"]["d3_score_direction"], "not_verified")
        self.assertEqual(validate_outcome_features(features, "abc")["errors"], [])
        json.dumps(features, allow_nan=False)

    def test_missing_d3_and_failed_d3_are_safe(self) -> None:
        report = _outcome_report()
        report["learned_detector_results"]["d3"] = {}
        missing = build_outcome_features(report, analysis_id="analysis-001").to_dict()
        self.assertEqual(missing["d3_summary"]["d3_status"], "not_run")
        self.assertIsNone(missing["d3_summary"]["d3_raw_score"])

        report["learned_detector_results"]["d3"] = {
            "execution": {"status": "failed", "reason_code": "model_loading_failure"},
            "configuration": {"encoder": "XCLIP-16", "distance_mode": "l2"},
            "native_output": {"raw_score": None, "calibration_status": "uncalibrated", "score_direction": "not_verified"},
        }
        failed = build_outcome_features(report, analysis_id="analysis-001").to_dict()
        self.assertEqual(failed["d3_summary"]["d3_status"], "failed")
        self.assertIsNone(failed["d3_summary"]["d3_raw_score"])

    def test_missing_audio_and_optional_sections_use_nulls(self) -> None:
        report = _outcome_report()
        report["metadata"]["audio"]["present"] = False
        report["audio_analysis"] = {"status": "skipped", "summary": {"audio_available": False}, "global_metrics": {}, "notable_transitions": [], "silence_intervals": [], "clipping_intervals": []}
        report.pop("visual_consistency_analysis")
        report["temporal_analysis"] = {"status": "skipped", "summary": {}, "notable_transitions": [], "notable_intervals": []}
        features = build_outcome_features(report, analysis_id="analysis-001").to_dict()
        self.assertFalse(features["media"]["audio_present"])
        self.assertIsNone(features["audio_summary"]["audio_window_count"])
        self.assertIsNone(features["audio_summary"]["mean_audio_rms"])
        self.assertIsNone(features["visual_consistency_summary"]["analyzed_region_record_count"])
        self.assertIsNone(features["temporal_summary"]["mean_motion_magnitude"])

    def test_expected_label_validation(self) -> None:
        report = _outcome_report()
        labeled = build_outcome_features(report, expected_label="real", analysis_id="analysis-001").to_dict()
        self.assertEqual(labeled["identity"]["expected_label"], "real")
        labeled = build_outcome_features(report, expected_label="ai_generated", analysis_id="analysis-001").to_dict()
        self.assertEqual(labeled["identity"]["expected_label"], "ai_generated")
        with self.assertRaises(ValueError):
            build_outcome_features(report, expected_label="fake", analysis_id="analysis-001")

    def test_artifact_hash_size_and_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference, warnings = create_outcome_features_artifact(Path(directory), _outcome_report())
            path = Path(directory) / OUTCOME_FEATURE_FILENAME
            self.assertEqual(reference["status"], "completed")
            self.assertEqual(warnings, [])
            self.assertTrue(path.exists())
            self.assertEqual(reference["artifact"]["path"], OUTCOME_FEATURE_FILENAME)
            self.assertEqual(reference["artifact"]["size_bytes"], path.stat().st_size)
            self.assertEqual(reference["artifact"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["identity"]["analysis_id"], Path(directory).name)
            self.assertNotIn(str(Path(directory)), json.dumps(payload))

    def test_deterministic_output_for_identical_inputs(self) -> None:
        first = build_outcome_features(_outcome_report(), analysis_id="analysis-001").to_dict()
        second = build_outcome_features(_outcome_report(), analysis_id="analysis-001").to_dict()
        self.assertEqual(first, second)

    def test_generation_failure_does_not_leave_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference, warnings = create_outcome_features_artifact(Path(directory), _outcome_report(), expected_label="not_allowed")
            self.assertEqual(reference["status"], "failed")
            self.assertTrue(warnings)
            self.assertFalse((Path(directory) / OUTCOME_FEATURE_FILENAME).exists())
            self.assertFalse((Path(directory) / f"{OUTCOME_FEATURE_FILENAME}.tmp").exists())

    def test_forbidden_result_fields_are_absent(self) -> None:
        features = build_outcome_features(_outcome_report(), analysis_id="analysis-001").to_dict()
        serialized = json.dumps(features)
        for forbidden in ("probability", "threshold", "confidence", "verdict", "classification"):
            self.assertNotIn(forbidden, serialized)

    def test_validation_rejects_bad_counts_ratios_paths_and_d3_calibration(self) -> None:
        features = build_outcome_features(_outcome_report(), analysis_id="analysis-001").to_dict()
        features["frame_summary"]["dark_frame_ratio"] = 1.5
        features["media"]["frame_count"] = -1
        features["identity"]["video_name"] = "C:\\absolute\\sample.mp4"
        features["d3_summary"]["d3_calibration_status"] = "calibrated"
        errors = validate_outcome_features(features, "abc")["errors"]
        self.assertTrue(any("ratio" in error for error in errors))
        self.assertTrue(any("count" in error for error in errors))
        self.assertTrue(any("absolute path" in error for error in errors))
        self.assertTrue(any("uncalibrated" in error for error in errors))


def _outcome_report() -> dict[str, object]:
    report = _report()
    report["analysis_environment"]["application_version"] = "0.9.4"
    report["metadata"]["video"]["frame_count"] = {"reported_by_stream": None, "estimated_from_duration": 150, "difference": None}
    report["metadata"]["container"]["tags"] = {"creation_time": "2026-01-01T00:00:00Z"}
    report["metadata"]["video"]["tags"] = {}
    report["metadata"]["audio"]["tags"] = {}
    report["metadata"]["encoding"]["container_encoder"] = "Lavf"
    report["frame_analysis"]["summary"].update(
        {
            "frames_analyzed": 2,
            "average_brightness": 10.0,
            "average_contrast": 0.0,
            "average_laplacian_variance": 0.0,
            "heuristic_near_black_frame_count": 0,
            "heuristic_near_white_frame_count": 0,
        }
    )
    report["temporal_analysis"]["summary"].update(
        {
            "transitions_analyzed": 3,
            "notable_transition_count": 1,
            "scene_boundary_candidate_count": 1,
            "sustained_near_static_interval_count": 0,
            "average_flow_warp_residual": 0.25,
            "maximum_flow_warp_residual": 0.5,
            "average_flow_magnitude": 1.5,
            "maximum_flow_magnitude": 3.0,
        }
    )
    report["audio_analysis"]["summary"].update({"window_count": 4, "ranked_energy_transition_count": 1, "silence_like_interval_count": 0})
    report["audio_analysis"]["notable_transitions"] = [{"energy_change_ratio": 4.0}]
    report["visual_consistency_analysis"]["summary"].update({"consistency_record_count": 16, "ranked_review_transition_count": 1, "sustained_interval_count": 0})
    report["visual_consistency_analysis"]["transition_summaries"] = [
        {
            "regional_summary": {
                "unstable_region_count": 3,
                "maximum_edge_instability": 0.4,
                "maximum_texture_distance": 0.5,
                "maximum_regional_detail_residual": 0.6,
            }
        }
    ]
    report["learned_detector_results"]["d3"] = _completed_d3()
    return report


if __name__ == "__main__":
    unittest.main()
