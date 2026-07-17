from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from report_validator import validate_report
from unified_evidence.builder import build_unified_evidence_artifacts


class UnifiedEvidenceTests(unittest.TestCase):
    def test_builds_merged_cross_modal_event_and_artifacts(self) -> None:
        report = _report()
        with tempfile.TemporaryDirectory() as tmp:
            section, warnings = build_unified_evidence_artifacts(Path(tmp), report)
            report["unified_evidence"] = section
            parsed = json.loads((Path(tmp) / "unified_evidence.json").read_text(encoding="utf-8"))
            timeline_lines = (Path(tmp) / "evidence_timeline.jsonl").read_text(encoding="utf-8").splitlines()
            ai_input = json.loads((Path(tmp) / "ai_interpretation_input.json").read_text(encoding="utf-8"))
        self.assertEqual(warnings, [])
        self.assertEqual(section["status"], "completed")
        self.assertEqual(section["summary"]["timeline_event_count"], 1)
        self.assertEqual(len(timeline_lines), 1)
        event = parsed["timeline_events"][0]
        self.assertEqual(event["cross_modal_context"]["classification"], "visual_and_audio_aligned")
        self.assertEqual(event["independent_group_count"], 2)
        self.assertFalse(event["review_priority"]["supports_ai_probability"])
        self.assertEqual(ai_input["external_model_results"], [])
        self.assertTrue(ai_input["interpretation_constraints"]["must_not_invent_probability"])

    def test_interval_outside_tolerance_creates_separate_events(self) -> None:
        report = _report()
        report["audio_analysis"]["notable_transitions"][0]["start_timestamp_seconds"] = 3.0
        report["audio_analysis"]["notable_transitions"][0]["end_timestamp_seconds"] = 3.25
        with tempfile.TemporaryDirectory() as tmp:
            section, _warnings = build_unified_evidence_artifacts(Path(tmp), report)
        self.assertEqual(section["summary"]["timeline_event_count"], 2)

    def test_temporal_interval_without_id_gets_stable_reference(self) -> None:
        report = _report()
        report["temporal_analysis"]["notable_transitions"] = []
        report["audio_analysis"]["notable_transitions"] = []
        report["visual_consistency_analysis"]["ranked_review_transitions"] = []
        report["temporal_analysis"]["notable_intervals"] = [
            {
                "interval_type": "sustained_near_static_visual_interval",
                "start_timestamp_seconds": 2.0,
                "end_timestamp_seconds": 2.8,
                "duration_seconds": 0.8,
                "transition_count": 4,
                "metrics": {"average_stationary_pixel_ratio": 0.9},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            section, warnings = build_unified_evidence_artifacts(Path(tmp), report)
            bundle = json.loads((Path(tmp) / "unified_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(warnings, [])
        self.assertEqual(section["status"], "completed")
        self.assertEqual(bundle["timeline_events"], [])
        self.assertIn("temporal-near-static-interval-001", bundle["context_intervals"][0]["source_record_id"])

    def test_ai_input_event_cap_and_compaction(self) -> None:
        report = _report()
        report["temporal_analysis"]["notable_transitions"] = [
            _temporal_transition(index, index * 0.5, index * 0.5 + 0.1)
            for index in range(12)
        ]
        report["audio_analysis"]["notable_transitions"] = []
        report["visual_consistency_analysis"]["ranked_review_transitions"] = []
        with tempfile.TemporaryDirectory() as tmp:
            section, _warnings = build_unified_evidence_artifacts(Path(tmp), report)
            ai_input = json.loads((Path(tmp) / "ai_interpretation_input.json").read_text(encoding="utf-8"))
        self.assertLessEqual(len(ai_input["priority_review_events"]), 10)
        self.assertTrue(ai_input["compaction"]["applied"])
        self.assertEqual(section["summary"]["external_model_result_count"], 0)

    def test_report_validation_accepts_completed_unified_section(self) -> None:
        report = _report()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            section, _warnings = build_unified_evidence_artifacts(base, report)
            report["unified_evidence"] = section
            report["audio_analysis"]["status"] = "skipped"
            report["audio_analysis"]["reason_code"] = "test"
            report["visual_consistency_analysis"]["status"] = "skipped"
            report["visual_consistency_analysis"]["reason_code"] = "test"
            report["visual_consistency_analysis"]["ranked_review_transitions"] = []
            validation = validate_report(report, base)
        self.assertEqual(validation["errors"], [])

    def test_chained_supporting_interval_does_not_merge_anchors(self) -> None:
        report = _report_with_only_unified_candidates()
        report["temporal_analysis"]["notable_transitions"] = [
            _temporal_transition(0, 0.8, 1.0),
            _temporal_transition(1, 2.8, 3.0),
        ]
        report["visual_consistency_analysis"]["sustained_intervals"] = [
            _regional_interval("visual-consistency-interval-001", 0.5, 3.5, "region-r00-c00")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            section, warnings = build_unified_evidence_artifacts(Path(tmp), report)
            bundle = json.loads((Path(tmp) / "unified_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(warnings, [])
        self.assertEqual(section["status"], "completed")
        self.assertEqual(bundle["timeline_summary"]["timeline_event_count"], 2)
        self.assertTrue(all(event["duration_seconds"] < 1.0 for event in bundle["timeline_events"]))
        self.assertTrue(any(event["context_candidate_ids"] or event["source_candidate_ids"] for event in bundle["timeline_events"]))

    def test_transitive_supporting_bridge_does_not_control_segmentation(self) -> None:
        report = _report_with_only_unified_candidates()
        report["temporal_analysis"]["notable_transitions"] = [
            _temporal_transition(0, 0.0, 0.2),
            _temporal_transition(1, 2.7, 2.9),
        ]
        report["visual_consistency_analysis"]["sustained_intervals"] = [
            _regional_interval("visual-consistency-interval-001", 0.15, 1.5, "region-r00-c00"),
            _regional_interval("visual-consistency-interval-002", 1.4, 2.8, "region-r00-c01"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            section, _warnings = build_unified_evidence_artifacts(Path(tmp), report)
        self.assertEqual(section["summary"]["timeline_event_count"], 2)

    def test_anchor_tolerance_and_maximum_span(self) -> None:
        report = _report_with_only_unified_candidates()
        report["temporal_analysis"]["notable_transitions"] = [
            _temporal_transition(0, 0.0, 0.2),
            _temporal_transition(1, 0.4, 0.5),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            section, _warnings = build_unified_evidence_artifacts(Path(tmp), report)
        self.assertEqual(section["summary"]["timeline_event_count"], 1)

        report = _report_with_only_unified_candidates()
        report["temporal_analysis"]["notable_transitions"] = [
            _temporal_transition(index, index * 0.3, index * 0.3 + 0.1)
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            section, _warnings = build_unified_evidence_artifacts(Path(tmp), report)
        self.assertGreater(section["summary"]["timeline_event_count"], 1)

    def test_context_interval_does_not_extend_anchor_event(self) -> None:
        report = _report_with_only_unified_candidates()
        report["temporal_analysis"]["notable_transitions"] = [_temporal_transition(0, 4.0, 4.2)]
        report["temporal_analysis"]["notable_intervals"] = [
            {
                "interval_type": "sustained_near_static_visual_interval",
                "start_timestamp_seconds": 4.2,
                "end_timestamp_seconds": 5.0,
                "duration_seconds": 0.8,
                "transition_count": 4,
                "metrics": {"average_stationary_pixel_ratio": 0.9},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            section, _warnings = build_unified_evidence_artifacts(Path(tmp), report)
            bundle = json.loads((Path(tmp) / "unified_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(section["summary"]["timeline_event_count"], 1)
        self.assertEqual(bundle["timeline_events"][0]["end_timestamp_seconds"], 4.2)
        self.assertEqual(bundle["context_intervals"][0]["end_timestamp_seconds"], 5.0)

    def test_observation_traceability_resolves_known_records(self) -> None:
        report = _report_with_only_unified_candidates()
        report["temporal_analysis"]["notable_transitions"] = [_temporal_transition(0, 1.0, 1.2)]
        report["audio_analysis"]["notable_transitions"] = [_audio_transition()]
        report["visual_consistency_analysis"]["ranked_review_transitions"] = [_visual_transition()]
        report["observations"] = {
            "metadata_facts": [],
            "missing_metadata": [],
            "temporal_heuristics": [
                {"observation_id": "temporal-notable-transition-001", "type": "temporal.ranked_notable_transition", "timestamp_start": 1.0, "timestamp_end": 1.2}
            ],
            "audio_observations": [
                {"observation_id": "audio-transition-001", "type": "audio.ranked_energy_transition", "timestamp_start": 1.05, "timestamp_end": 1.2}
            ],
            "visual_consistency_observations": [
                {"observation_id": "visual-consistency-ranked-001", "type": "visual_consistency.ranked_review_transition", "timestamp_start": 1.0, "timestamp_end": 1.2}
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            _section, _warnings = build_unified_evidence_artifacts(Path(tmp), report)
            bundle = json.loads((Path(tmp) / "unified_evidence.json").read_text(encoding="utf-8"))
        observation_ids = set(bundle["timeline_events"][0]["source_observation_ids"])
        self.assertIn("temporal-notable-transition-001", observation_ids)
        self.assertIn("audio-transition-001", observation_ids)
        self.assertIn("visual-consistency-ranked-001", observation_ids)

    def test_balanced_findings_and_provenance_categorization(self) -> None:
        report = _report_with_only_unified_candidates()
        report["temporal_analysis"]["notable_transitions"] = [_temporal_transition(0, 1.0, 1.2)]
        report["audio_analysis"]["notable_transitions"] = [_audio_transition()]
        report["visual_consistency_analysis"]["ranked_review_transitions"] = [_visual_transition()]
        report["visual_consistency_analysis"]["sustained_intervals"] = [
            _regional_interval(f"visual-consistency-interval-{index:03d}", 0.8, 1.4, f"region-r00-c{index % 4:02d}")
            for index in range(1, 8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _section, _warnings = build_unified_evidence_artifacts(Path(tmp), report)
            ai_input = json.loads((Path(tmp) / "ai_interpretation_input.json").read_text(encoding="utf-8"))
        findings = ai_input["priority_review_events"][0]["findings"]
        domains = {finding["domain"] for finding in findings}
        self.assertIn("visual_temporal", domains)
        self.assertIn("audio_signal", domains)
        self.assertIn("visual_consistency", domains)
        self.assertLessEqual(sum(1 for finding in findings if finding["domain"] == "visual_consistency"), 3)
        self.assertNotIn("no_provenance_result", {item["type"] for item in ai_input["normal_or_non_supporting_findings"]})
        self.assertIn("content_provenance", {item["type"] for item in ai_input["missing_evidence"]})


def _report() -> dict[str, object]:
    return {
        "schema_version": "0.7",
        "analysis": {
            "started_at": "2026-07-14T00:00:00+00:00",
            "completed_at": "2026-07-14T00:00:01+00:00",
            "processing_duration_seconds": 1.0,
            "status": "completed",
        },
        "analysis_environment": {
            "application_version": "0.8.1",
            "report_schema_version": "0.7",
        },
        "source": {
            "filename": "sample.mp4",
            "path": "source_videos/sample.mp4",
            "absolute_path_included": False,
            "extension": ".mp4",
            "size_bytes": 1,
            "size_human_readable": "1 B",
            "sha256": "abc",
        },
        "metadata": _metadata(),
        "sampling": {
            "strategy": "uniform",
            "target_frame_count": 1,
            "requested_timestamps_seconds": [0.0],
            "decoded_frame_count": 1,
            "failed_timestamps_seconds": [],
            "maximum_absolute_seek_error_seconds": 0.0,
            "average_absolute_seek_error_seconds": 0.0,
            "repeated_decoded_timestamps_seconds": [],
        },
        "frame_analysis": {"summary": _frame_summary(), "frames": [_frame()], "comparisons": []},
        "temporal_analysis": {
            "status": "completed",
            "reason_code": None,
            "summary": {
                "scene_count": 0,
                "sustained_near_static_interval_count": 0,
                "temporal_frames_analyzed": 2,
                "transitions_analyzed": 1,
            },
            "scenes": [],
            "notable_intervals": [],
            "notable_transitions": [_temporal_transition(0, 1.0, 1.2)],
            "observations": [],
            "artifacts": {},
        },
        "audio_analysis": {
            "status": "completed",
            "reason_code": None,
            "summary": {"audio_available": True, "analysis_metrics_available": True},
            "decoded_audio": {"sample_rate_hz": 44100, "channels": 2, "frame_count": 44100},
            "global_metrics": {"clipping_ratio": 0.0, "silence_ratio": 0.0, "rms_amplitude": 0.1, "peak_absolute_amplitude": 0.2, "zero_crossing_rate": 0.1},
            "silence_intervals": [],
            "clipping_intervals": [],
            "notable_transitions": [_audio_transition()],
            "observations": [],
            "artifacts": {},
            "extraction": {"status": "completed"},
            "temporary_file_cleanup": {"attempted": True, "successful": True},
        },
        "visual_consistency_analysis": {
            "status": "completed",
            "reason_code": None,
            "configuration": {"grid_rows": 4, "grid_columns": 4, "ranking": {"combined_percentile_metric_count": 7, "metrics": ["test metric descending"]}},
            "summary": {"region_count_per_transition": 16, "transitions_analyzed": 1, "consistency_record_count": 16},
            "transition_summaries": [],
            "sustained_intervals": [],
            "ranked_review_transitions": [_visual_transition()],
            "observations": [],
            "artifacts": {},
        },
        "learned_detector_results": {
            "status": "not_run",
            "d3": {},
            "standalone_not_fused_with_unified_evidence": True,
            "limitations": [],
        },
        "heuristic_configuration": {},
        "observations": {"metadata_facts": [], "missing_metadata": [], "temporal_heuristics": [], "audio_observations": [], "visual_consistency_observations": []},
        "evidence": [],
        "compatibility": {"legacy_evidence_view_included": True, "legacy_evidence_view_source": "observations"},
        "warnings": [],
        "limitations": [],
        "artifacts": {"paths_relative_to": "report_directory", "raw_ffprobe_report": "ffprobe_raw.json", "frames_directory": "frames"},
        "validation": {"errors": [], "warnings": []},
    }


def _report_with_only_unified_candidates() -> dict[str, object]:
    report = _report()
    report["temporal_analysis"]["notable_transitions"] = []
    report["temporal_analysis"]["notable_intervals"] = []
    report["audio_analysis"]["notable_transitions"] = []
    report["audio_analysis"]["silence_intervals"] = []
    report["visual_consistency_analysis"]["ranked_review_transitions"] = []
    report["visual_consistency_analysis"]["sustained_intervals"] = []
    return report


def _metadata() -> dict[str, object]:
    return {
        "container": {"duration_seconds": 5.0, "duration_readable": "5.00 seconds", "format_name": "mov,mp4", "friendly_format": "MP4/MOV family", "format_long_name": "QuickTime / MOV", "start_time": 0.0, "bit_rate_readable": None, "probe_score": 100, "stream_count": 2},
        "video": {"present": True, "index": 0, "selected_stream_index": 0, "selection_reason": "test", "codec_name": "h264", "codec_long_name": None, "codec_profile": None, "codec_tag": None, "width": 10, "height": 10, "coded_width": 10, "coded_height": 10, "pixel_format": None, "codec_level_raw": None, "codec_level_readable": None, "average_frame_rate_raw": "30/1", "nominal_frame_rate_raw": "30/1", "frame_rate_decimal": 30.0, "time_base": None, "start_time": 0.0, "duration_seconds": 5.0, "duration_readable": None, "frame_count": {"reported_by_stream": None, "estimated_from_duration": None, "difference": None}, "bit_rate_readable": None, "sample_aspect_ratio": None, "display_aspect_ratio": None},
        "audio": {"present": True, "index": 1, "selected_stream_index": 1, "selection_reason": "test", "codec_name": "aac", "codec_long_name": None, "codec_profile": None, "sample_rate": 44100, "channels": 2, "channel_layout": "stereo", "start_time": 0.0, "duration_seconds": 5.0, "duration_readable": None, "bit_rate_readable": None},
        "duration_comparison": {"container_duration_seconds": 5.0, "video_start_time_seconds": 0.0, "video_duration_seconds": 5.0, "video_end_time_seconds": 5.0, "audio_start_time_seconds": 0.0, "audio_duration_seconds": 5.0, "audio_end_time_seconds": 5.0, "duration_only_difference_seconds": 0.0, "start_time_difference_seconds": 0.0, "end_time_difference_seconds": 0.0},
        "encoding": {"container_encoder": None, "video_stream_encoder": None, "audio_stream_encoder": None, "video_handler_name": None, "audio_handler_name": None},
        "display": {"orientation": None, "width": 10, "height": 10, "rotation": None, "sample_aspect_ratio": None, "display_aspect_ratio": None},
    }


def _frame() -> dict[str, object]:
    return {"sample_index": 0, "requested_timestamp_seconds": 0.0, "decoded_timestamp_seconds": 0.0, "seek_error_seconds": 0.0, "artifact_path": "frames/a.jpg", "width": 1, "height": 1, "brightness_mean": 0.0, "contrast_stddev": 0.0, "laplacian_variance": 0.0, "dark_pixel_ratio": 1.0, "bright_pixel_ratio": 0.0, "likely_near_black_frame": True, "likely_near_white_frame": False, "mean_rgb": {"red": 0, "green": 0, "blue": 0}, "perceptual_hash": "0"}


def _frame_summary() -> dict[str, object]:
    return {"frames_analyzed": 1, "average_brightness": 0.0, "minimum_brightness": 0.0, "maximum_brightness": 0.0, "average_contrast": 0.0, "average_laplacian_variance": 0.0, "minimum_laplacian_variance": 0.0, "maximum_laplacian_variance": 0.0, "heuristic_near_black_frame_count": 1, "heuristic_near_white_frame_count": 0, "heuristic_large_change_pair_count": 0, "heuristic_near_duplicate_pair_count": 0, "maximum_absolute_seek_error_seconds": 0.0, "average_absolute_seek_error_seconds": 0.0}


def _temporal_transition(index: int, start: float, end: float) -> dict[str, object]:
    return {"notable_transition_index": index, "transition_id": f"temporal-transition-{index:05d}-{index + 1:05d}", "from_sample_index": index, "to_sample_index": index + 1, "scene_index": 0, "start_timestamp_seconds": start, "end_timestamp_seconds": end, "metrics": {"normalized_mean_absolute_difference": 0.5}, "classification": {"scene_boundary_candidate": False}, "notability": {"combined_percentile": 1.0, "reasons": ["test"], "reason_count": 1}, "optical_flow": {}, "flow_warp_residual": {}, "artifacts": {}, "interpretation": "Review only."}


def _audio_transition() -> dict[str, object]:
    return {"transition_id": "audio-transition-00000-00001", "from_window_index": 0, "to_window_index": 1, "start_timestamp_seconds": 1.05, "end_timestamp_seconds": 1.2, "rms_ratio": 4.0, "absolute_rms_difference": 0.2, "peak_difference": 0.2, "selection_basis": "relative_within_audio", "ranked_review_transition": True, "notability": {"rank": 1, "reason": "test"}}


def _visual_transition() -> dict[str, object]:
    return {"ranked_review_index": 1, "transition_id": "temporal-transition-00000-00001", "from_sample_index": 0, "to_sample_index": 1, "start_timestamp_seconds": 1.0, "end_timestamp_seconds": 1.2, "scene_index": 0, "selection_basis": "relative_within_video", "absolute_significance_assessed": False, "combined_percentile_metric_count": 7, "available_metric_count": 7, "combined_percentile": 1.0, "metric_ranks": {}, "regional_summary": {"valid_region_count": 16, "total_region_count": 16, "unstable_region_count": 3, "high_motion_region_count": 0, "average_edge_instability": 0.2, "maximum_edge_instability": 0.5, "average_texture_distance": 0.1, "maximum_texture_distance": 0.3, "average_regional_detail_residual": 0.1, "maximum_regional_detail_residual": 0.2}, "ranking_reasons": ["test"], "artifacts": {}, "interpretation": "Review only."}


def _regional_interval(interval_id: str, start: float, end: float, region_id: str) -> dict[str, object]:
    return {
        "interval_id": interval_id,
        "interval_type": "sustained_regional_visual_instability",
        "start_timestamp_seconds": start,
        "end_timestamp_seconds": end,
        "duration_seconds": round(end - start, 6),
        "transition_count": 3,
        "transition_ids": [],
        "affected_regions": [region_id],
        "supporting_metrics": {
            "average_edge_instability": 0.5,
            "average_texture_distance": 0.25,
            "average_detail_residual": 0.15,
        },
    }


if __name__ == "__main__":
    unittest.main()
