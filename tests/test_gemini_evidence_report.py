from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import heuristic_configuration
from learned_detectors.d3.adapter import D3Detector
from report_writer import write_reports
from reporting.gemini_evidence_report import (
    COMPACT_REPORT_FILENAME,
    create_gemini_evidence_report,
    validate_gemini_evidence_report,
)


class GeminiEvidenceReportTests(unittest.TestCase):
    def test_successful_compact_report_is_valid_and_referenced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            analysis_dir = Path(directory)
            report = _report()
            reference, warnings = create_gemini_evidence_report(analysis_dir, report)
            report["artifacts"]["gemini_evidence_report"] = reference
            written = json.loads((analysis_dir / COMPACT_REPORT_FILENAME).read_text(encoding="utf-8"))

            self.assertEqual(reference["status"], "completed")
            self.assertEqual(warnings, [])
            self.assertTrue((analysis_dir / COMPACT_REPORT_FILENAME).exists())
            self.assertEqual(reference["artifact"]["sha256"], hashlib.sha256((analysis_dir / COMPACT_REPORT_FILENAME).read_bytes()).hexdigest())
            self.assertEqual(reference["artifact"]["size_bytes"], (analysis_dir / COMPACT_REPORT_FILENAME).stat().st_size)
            self.assertEqual(validate_gemini_evidence_report(written, 16000)["errors"], [])
            for key in (
                "schema_version",
                "analysis_identity",
                "media_summary",
                "deterministic_summary",
                "key_review_events",
                "d3_summary",
                "limitations",
                "gemini_instructions",
                "future_gemini_output_contract",
                "source_artifacts",
            ):
                self.assertIn(key, written)
            self.assertEqual(written["analysis_identity"]["source_sha256"], "abc")
            serialized = json.dumps(written)
            self.assertNotIn("C:\\", serialized)
            self.assertNotIn("ffprobe_raw", written)
            self.assertNotIn("frames", written.get("frame_analysis", {}))
            self.assertNotIn("timeline_events", written)
            self.assertLessEqual(len(written["key_review_events"]), 5)
            self.assertEqual(written["d3_summary"]["execution_status"], "completed")
            self.assertEqual(written["d3_summary"]["calibration_status"], "uncalibrated")
            self.assertEqual(written["d3_summary"]["classification"], "not_assigned")
            self.assertNotIn("probability", written["d3_summary"])
            self.assertNotIn("threshold", written["d3_summary"])
            self.assertNotIn("authenticity_verdict", serialized)
            self.assertFalse(written["generation"]["gemini_api_called"])
            self.assertFalse(written["generation"]["network_request_introduced"])

            paths = write_reports(analysis_dir, report)
            main_report = json.loads(paths["json"].read_text(encoding="utf-8"))
            txt = paths["txt"].read_text(encoding="utf-8")
            self.assertEqual(main_report["artifacts"]["gemini_evidence_report"]["status"], "completed")
            self.assertIn("Gemini-ready compact evidence report", txt)
            self.assertIn(COMPACT_REPORT_FILENAME, txt)

    def test_event_selection_dedupes_findings_and_preserves_domains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = _report()
            event = report["unified_evidence"]["review_highlights"][0]
            event["key_findings"].append(dict(event["key_findings"][0]))
            event["artifact_references"].append(dict(event["artifact_references"][0]))
            with patch(
                "reporting.gemini_evidence_report.gemini_compact_report_configuration",
                return_value={
                    "enabled": True,
                    "maximum_key_events": 1,
                    "maximum_findings_per_event": 3,
                    "maximum_artifacts_per_event": 1,
                    "preferred_size_bytes": 8000,
                    "acceptable_size_bytes": 12000,
                    "hard_size_limit_bytes": 16000,
                },
            ):
                create_gemini_evidence_report(Path(directory), report)
            compact = json.loads((Path(directory) / COMPACT_REPORT_FILENAME).read_text(encoding="utf-8"))
            selected = compact["key_review_events"][0]
            domains = {finding["domain"] for finding in selected["findings"]}
            self.assertIn("visual_temporal", domains)
            self.assertIn("audio_signal", domains)
            self.assertLessEqual(len(selected["findings"]), 3)
            self.assertLessEqual(len(selected["artifact_references"]), 1)
            self.assertEqual(len({artifact["path"] for artifact in selected["artifact_references"]}), len(selected["artifact_references"]))

    def test_d3_disabled_status_is_compact_and_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = _report()
            report["learned_detector_results"]["d3"] = {
                "detector": {"detector_id": "d3", "detector_name": "Detection by Difference of Differences"},
                "execution": {"status": "disabled", "reason_code": "detector_disabled"},
                "configuration": {"encoder": "XCLIP-16", "distance_mode": "l2"},
                "native_output": {"classification": "not_assigned", "calibration_status": "uncalibrated", "score_direction": "not_verified"},
                "artifacts": {},
            }
            create_gemini_evidence_report(Path(directory), report)
            compact = json.loads((Path(directory) / COMPACT_REPORT_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(compact["d3_summary"]["execution_status"], "disabled")
            self.assertEqual(compact["d3_summary"]["reason_code"], "detector_disabled")
            self.assertNotIn("probability", compact["d3_summary"])
            self.assertNotIn("threshold", compact["d3_summary"])

    def test_nonfinite_values_are_sanitized_and_hard_limit_failure_cleans_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = _report()
            report["unified_evidence"]["review_highlights"][0]["key_findings"][0]["metrics"]["combined_percentile"] = math.inf
            with patch(
                "reporting.gemini_evidence_report.gemini_compact_report_configuration",
                return_value={
                    "enabled": True,
                    "maximum_key_events": 5,
                    "maximum_findings_per_event": 5,
                    "maximum_artifacts_per_event": 4,
                    "preferred_size_bytes": 8000,
                    "acceptable_size_bytes": 12000,
                    "hard_size_limit_bytes": 16000,
                },
            ):
                reference, _warnings = create_gemini_evidence_report(Path(directory), report)
            self.assertEqual(reference["status"], "completed")
            compact_text = (Path(directory) / COMPACT_REPORT_FILENAME).read_text(encoding="utf-8")
            self.assertNotIn("Infinity", compact_text)

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "reporting.gemini_evidence_report.gemini_compact_report_configuration",
                return_value={
                    "enabled": True,
                    "maximum_key_events": 5,
                    "maximum_findings_per_event": 5,
                    "maximum_artifacts_per_event": 4,
                    "preferred_size_bytes": 10,
                    "acceptable_size_bytes": 10,
                    "hard_size_limit_bytes": 10,
                },
            ):
                reference, warnings = create_gemini_evidence_report(Path(directory), _report())
            self.assertEqual(reference["status"], "failed")
            self.assertTrue(warnings)
            self.assertFalse((Path(directory) / COMPACT_REPORT_FILENAME).exists())
            self.assertFalse((Path(directory) / f"{COMPACT_REPORT_FILENAME}.tmp").exists())

    def test_output_is_deterministic_for_identical_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            create_gemini_evidence_report(Path(first), _report())
            create_gemini_evidence_report(Path(second), _report())
            self.assertEqual(
                (Path(first) / COMPACT_REPORT_FILENAME).read_text(encoding="utf-8"),
                (Path(second) / COMPACT_REPORT_FILENAME).read_text(encoding="utf-8"),
            )


class D3ProgressTests(unittest.TestCase):
    def test_d3_progress_messages_are_stage_level_only(self) -> None:
        detector = D3Detector(_d3_config(), learned_detectors_enabled=True)
        preprocessing = {
            "window_start_seconds": 0.0,
            "window_end_seconds": 3.0,
            "actual_selected_frame_count": 8,
            "selected_frame_timestamps_seconds": [float(index) for index in range(8)],
        }
        summary = {
            "second_order_value_count": 6,
            "second_order_mean": 0.2,
            "second_order_standard_deviation": 0.1,
            "official_torch_second_order_standard_deviation": 0.1,
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("learned_detectors.d3.adapter.extract_upstream_compatible_frames", return_value=([object()] * 8, preprocessing, [])),
                patch.object(detector, "_select_device", return_value="cpu"),
                patch.object(detector, "_load_model", return_value=object()),
                patch("learned_detectors.d3.adapter.frames_to_torch_tensor", return_value=_Tensor()),
                patch("learned_detectors.d3.adapter._tensor_sha256", return_value="tensor-hash"),
                patch("learned_detectors.d3.adapter.torch_forward_score", return_value=(_Embeddings(), [1.0] * 7, [0.1] * 6, summary)),
                patch.object(detector, "_populate_runtime_versions"),
                patch("builtins.print") as mocked_print,
            ):
                result = detector._analyze_available(Path("source_videos/sample.mp4"), "abc", Path(directory), {"container": {"duration_seconds": 3.0}})
        messages = [call.args[0] for call in mocked_print.call_args_list]
        self.assertEqual(result["execution"]["status"], "completed")
        self.assertIn("D3 progress: loading D3 model", messages)
        self.assertIn("D3 progress: preparing D3 frame tensor", messages)
        self.assertIn("D3 progress: running XCLIP-16 inference on CPU", messages)
        self.assertIn("D3 progress: computing temporal features", messages)
        self.assertIn("D3 progress: writing D3 artifacts", messages)
        self.assertLessEqual(len(messages), 5)


class _Embeddings:
    shape = (1, 8, 4)


class _Tensor:
    dtype = "torch.float32"


def _report() -> dict[str, object]:
    return {
        "schema_version": "0.7",
        "analysis": {"started_at": "2026-07-18T00:00:00+00:00", "completed_at": "2026-07-18T00:00:01+00:00", "processing_duration_seconds": 1.0, "status": "completed"},
        "analysis_environment": {"application_version": "0.9.4", "report_schema_version": "0.7"},
        "source": {"filename": "sample.mp4", "path": "source_videos/sample.mp4", "absolute_path_included": False, "extension": ".mp4", "size_bytes": 1, "size_human_readable": "1 B", "sha256": "abc"},
        "metadata": {
            "container": {"duration_seconds": 5.0, "format_name": "mov,mp4", "friendly_format": "MP4/MOV family", "format_long_name": "QuickTime / MOV", "duration_readable": "5.00 seconds", "start_time": 0.0, "bit_rate_readable": None, "probe_score": 100, "stream_count": 2},
            "video": {"present": True, "index": 0, "codec_name": "h264", "codec_long_name": None, "codec_profile": None, "width": 1920, "height": 1080, "coded_width": 1920, "coded_height": 1080, "pixel_format": None, "average_frame_rate_raw": "30/1", "nominal_frame_rate_raw": "30/1", "frame_rate_decimal": 30.0, "codec_level_readable": None, "codec_level_raw": None, "frame_count": {"reported_by_stream": None, "estimated_from_duration": None, "difference": None}, "bit_rate_readable": None},
            "audio": {"present": True, "index": 1, "codec_name": "aac", "codec_long_name": None, "sample_rate": 44100, "channels": 2, "channel_layout": "stereo", "bit_rate_readable": None},
            "display": {"orientation": "landscape", "width": 1920, "height": 1080, "rotation": None, "sample_aspect_ratio": None, "display_aspect_ratio": None},
            "duration_comparison": {"container_duration_seconds": 5.0, "video_start_time_seconds": 0.0, "video_duration_seconds": 5.0, "video_end_time_seconds": 5.0, "audio_start_time_seconds": 0.0, "audio_duration_seconds": 5.0, "audio_end_time_seconds": 5.0, "duration_only_difference_seconds": 0.0, "start_time_difference_seconds": 0.0, "end_time_difference_seconds": 0.0},
            "encoding": {"container_encoder": None, "video_stream_encoder": None, "audio_stream_encoder": None, "video_handler_name": None, "audio_handler_name": None},
        },
        "sampling": {"strategy": "uniform", "target_frame_count": 1, "requested_timestamps_seconds": [0.0], "decoded_frame_count": 1, "failed_timestamps_seconds": [], "maximum_absolute_seek_error_seconds": 0.0, "average_absolute_seek_error_seconds": 0.0, "repeated_decoded_timestamps_seconds": []},
        "frame_analysis": {"summary": _empty_frame_summary(), "frames": [], "comparisons": []},
        "temporal_analysis": {"status": "completed", "summary": {"temporal_frames_analyzed": 10, "transitions_analyzed": 9, "scene_count": 2, "sustained_near_static_interval_count": 0}, "notable_transitions": [], "notable_intervals": [], "artifacts": {}, "observations": [], "limitations": []},
        "audio_analysis": {"status": "completed", "summary": {"audio_available": True}, "global_metrics": {"rms_amplitude": 0.1, "peak_absolute_amplitude": 0.2, "clipping_ratio": 0.0, "silence_ratio": 0.0}, "notable_transitions": [], "silence_intervals": [], "clipping_intervals": [], "decoded_audio": {"sample_rate_hz": 44100, "channels": 2}, "artifacts": {}, "observations": [], "limitations": [], "extraction": {"status": "completed"}},
        "visual_consistency_analysis": {"status": "completed", "summary": {"transitions_analyzed": 9}, "ranked_review_transitions": [], "sustained_intervals": [], "artifacts": {}, "observations": [], "limitations": []},
        "unified_evidence": _unified(),
        "learned_detector_results": {"status": "completed", "standalone_not_fused_with_unified_evidence": True, "limitations": [], "d3": _completed_d3()},
        "heuristic_configuration": heuristic_configuration(),
        "observations": {"metadata_facts": [], "missing_metadata": [], "temporal_heuristics": [], "audio_observations": [], "visual_consistency_observations": []},
        "evidence": [],
        "compatibility": {"legacy_evidence_view_included": True, "legacy_evidence_view_source": "observations"},
        "warnings": [],
        "limitations": [],
        "artifacts": {"paths_relative_to": "report_directory", "raw_ffprobe_report": "ffprobe_raw.json", "raw_ffprobe_report_artifact": {"path": "ffprobe_raw.json", "size_bytes": 100, "sha256": "rawhash"}, "frames_directory": "frames"},
        "validation": {"errors": [], "warnings": []},
    }


def _unified() -> dict[str, object]:
    return {
        "status": "completed",
        "summary": {"timeline_event_count": 2, "priority_review_event_count": 2},
        "review_highlights": [
            _event("event-002", 2.0, "high", ["visual_temporal", "audio_signal"], ["visual", "audio"]),
            _event("event-001", 1.0, "moderate", ["visual_temporal"], ["visual"]),
        ],
        "artifacts": {"unified_evidence": {"path": "unified_evidence.json", "size_bytes": 200, "sha256": "unifiedhash"}, "evidence_timeline": {"path": "evidence_timeline.jsonl", "size_bytes": 120, "sha256": "timelinehash"}},
        "limitations": [],
    }


def _empty_frame_summary() -> dict[str, object]:
    return {
        "frames_analyzed": 0,
        "average_brightness": None,
        "minimum_brightness": None,
        "maximum_brightness": None,
        "average_contrast": None,
        "average_laplacian_variance": None,
        "minimum_laplacian_variance": None,
        "maximum_laplacian_variance": None,
        "heuristic_near_black_frame_count": 0,
        "heuristic_near_white_frame_count": 0,
        "heuristic_large_change_pair_count": 0,
        "heuristic_near_duplicate_pair_count": 0,
        "maximum_absolute_seek_error_seconds": None,
        "average_absolute_seek_error_seconds": None,
    }


def _event(event_id: str, start: float, priority: str, domains: list[str], groups: list[str]) -> dict[str, object]:
    return {
        "event_id": event_id,
        "start_timestamp_seconds": start,
        "end_timestamp_seconds": start + 0.2,
        "duration_seconds": 0.2,
        "review_priority": {"level": priority, "basis": ["ranked_source_findings", "audio_visual_temporal_overlap"]},
        "evidence_domains": domains,
        "evidence_groups_present": groups,
        "independent_group_count": len(groups),
        "cross_modal_context": {"classification": "visual_and_audio_aligned" if len(groups) > 1 else "visual_only", "visual_method_count": 1, "audio_transition_present": len(groups) > 1},
        "key_findings": [
            {"domain": "visual_temporal", "type": "ranked_review_transition", "summary": "Ranked visual transition.", "source_record_id": "visual-1", "metrics": {"combined_percentile": 0.9}, "source_observation_ids": ["obs-v"]},
            {"domain": "audio_signal", "type": "ranked_audio_energy_transition", "summary": "Ranked audio transition.", "source_record_id": "audio-1", "metrics": {"energy_change_ratio": 4.0}, "source_observation_ids": ["obs-a"]},
        ],
        "artifact_references": [
            {"label": "before_frame", "path": "transition_frames/before.jpg", "size_bytes": 10, "sha256": "beforehash"},
            {"label": "after_frame", "path": "transition_frames/after.jpg", "size_bytes": 11, "sha256": "afterhash"},
        ],
    }


def _completed_d3() -> dict[str, object]:
    return {
        "detector": {"detector_id": "d3", "detector_name": "Detection by Difference of Differences"},
        "execution": {"status": "completed", "device_requested": "auto", "device_used": "cpu", "duration_seconds": 1.2, "reason_code": None},
        "configuration": {"encoder": "XCLIP-16", "distance_mode": "l2"},
        "preprocessing": {"window_start_seconds": 0.0, "window_end_seconds": 3.0, "actual_selected_frame_count": 8},
        "feature_summary": {"embedding_dimension": 768, "first_order_value_count": 7, "second_order_value_count": 6},
        "native_output": {"score_name": "native_d3_second_order_standard_deviation", "raw_score": 0.123, "score_direction": "not_verified", "classification": "not_assigned", "calibration_status": "uncalibrated"},
        "method_verification": {"runtime_parity": "not_verified", "mathematical_parity": "verified_synthetic_tensor"},
        "artifacts": {"d3_detector_result": {"path": "d3_detector_result.json", "size_bytes": 10, "sha256": "d3hash"}, "d3_temporal_features": {"path": "d3_temporal_features.jsonl", "size_bytes": 20, "sha256": "d3featurehash"}},
        "limitations": [],
    }


def _d3_config() -> dict[str, object]:
    return {
        "enabled": True,
        "encoder": "XCLIP-16",
        "distance_mode": "l2",
        "random_seed": 42,
        "device": "auto",
        "allow_model_download": False,
        "model_cache_directory": "",
        "preprocessing_mode": "upstream_compatible",
        "timeout_seconds": 300,
        "preserve_temporary_frames": False,
    }


if __name__ == "__main__":
    unittest.main()
