from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from frame_analyzer import (
    analyze_frame,
    classify_comparison,
    compare_frames,
    summarize_frame_analysis,
)
from metadata_extractor import parse_metadata
from report_validator import validate_report
from report_writer import build_report, write_reports


class ReliabilityTests(unittest.TestCase):
    def test_duration_timeline_and_missing_audio(self) -> None:
        metadata = parse_metadata(
            {
                "format": {"duration": "2.0", "format_name": "mov,mp4"},
                "streams": [
                    {
                        "codec_type": "video",
                        "index": 0,
                        "start_time": "0.1",
                        "duration": "1.9",
                        "avg_frame_rate": "30000/1001",
                        "r_frame_rate": "30000/1001",
                        "nb_frames": "57",
                        "width": 1920,
                        "height": 1080,
                    }
                ],
            }
        )
        self.assertFalse(metadata["audio"]["present"])
        self.assertAlmostEqual(metadata["video"]["frame_rate_decimal"], 29.97003)
        self.assertEqual(metadata["duration_comparison"]["video_end_time_seconds"], 2.0)
        self.assertIsNone(metadata["duration_comparison"]["duration_only_difference_seconds"])

    def test_stream_selection_skips_attached_picture(self) -> None:
        metadata = parse_metadata(
            {
                "format": {},
                "streams": [
                    {
                        "codec_type": "video",
                        "index": 0,
                        "disposition": {"attached_pic": 1},
                        "width": 100,
                        "height": 100,
                    },
                    {
                        "codec_type": "video",
                        "index": 1,
                        "disposition": {"default": 1},
                        "width": 1920,
                        "height": 1080,
                    },
                ],
            }
        )
        self.assertEqual(metadata["video"]["selected_stream_index"], 1)
        self.assertEqual(metadata["video"]["selection_reason"], "default_non_attached_video_stream")

    def test_rotation_display_orientation(self) -> None:
        metadata = parse_metadata(
            {
                "format": {},
                "streams": [
                    {
                        "codec_type": "video",
                        "index": 0,
                        "width": 1920,
                        "height": 1080,
                        "tags": {"rotate": "90"},
                    }
                ],
            }
        )
        self.assertEqual(metadata["display"]["width"], 1080)
        self.assertEqual(metadata["display"]["height"], 1920)
        self.assertEqual(metadata["display"]["orientation"], "portrait")

    def test_frame_metrics_and_comparison_classification(self) -> None:
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        white = np.ones((32, 80, 3), dtype=np.uint8) * 255
        first = analyze_frame(black, 0, 0.0, 0.1, Path("frames/a.jpg"))
        second = analyze_frame(white, 1, 1.0, 1.0, Path("frames/b.jpg"))
        comparisons = compare_frames([first, second], [black, white])
        self.assertAlmostEqual(first["seek_error_seconds"], 0.1)
        self.assertTrue(first["likely_near_black_frame"])
        self.assertTrue(second["likely_near_white_frame"])
        self.assertEqual(comparisons[0]["normalized_mean_absolute_difference"], 1.0)
        self.assertIn("classification", comparisons[0])

    def test_summary_recomputation_zero_and_one_frame(self) -> None:
        self.assertEqual(summarize_frame_analysis([], [])["frames_analyzed"], 0)
        frame = {
            "sample_index": 0,
            "brightness_mean": 1.0,
            "contrast_stddev": 2.0,
            "laplacian_variance": 3.0,
            "likely_near_black_frame": False,
            "likely_near_white_frame": False,
            "seek_error_seconds": 0.0,
        }
        summary = summarize_frame_analysis([frame], [])
        self.assertEqual(summary["frames_analyzed"], 1)
        self.assertEqual(summary["heuristic_large_change_pair_count"], 0)

    def test_classification_rules(self) -> None:
        near_duplicate = classify_comparison(0, 0.0)
        large_change = classify_comparison(64, 1.0)
        self.assertTrue(near_duplicate["near_duplicate"])
        self.assertTrue(large_change["large_change"])

    def test_report_validation_and_txt_from_same_object(self) -> None:
        start = datetime.now().astimezone().isoformat(timespec="milliseconds")
        frame = {
            "sample_index": 0,
            "requested_timestamp_seconds": 0.0,
            "decoded_timestamp_seconds": 0.0,
            "seek_error_seconds": 0.0,
            "artifact_path": "frames/frame_000_0.000s.jpg",
            "width": 1,
            "height": 1,
            "brightness_mean": 0.0,
            "contrast_stddev": 0.0,
            "laplacian_variance": 0.0,
            "dark_pixel_ratio": 1.0,
            "bright_pixel_ratio": 0.0,
            "likely_near_black_frame": True,
            "likely_near_white_frame": False,
            "mean_rgb": {"red": 0.0, "green": 0.0, "blue": 0.0},
            "perceptual_hash": "0000000000000000",
        }
        frame_analysis = {"summary": summarize_frame_analysis([frame], []), "frames": [frame], "comparisons": []}
        report = build_report(
            analysis={
                "started_at": start,
                "completed_at": start,
                "processing_duration_seconds": 0.0,
                "status": "completed",
            },
            analysis_environment={
                "application_version": "0.5.1",
                "report_schema_version": "0.5",
            },
            source={
                "filename": "sample.mp4",
                "path": "source_videos/sample.mp4",
                "absolute_path_included": False,
                "extension": ".mp4",
                "size_bytes": 1,
                "size_human_readable": "1 B",
                "sha256": "abc",
            },
            metadata=_minimal_metadata(),
            sampling={
                "strategy": "uniform",
                "target_frame_count": 1,
                "requested_timestamps_seconds": [0.0],
                "decoded_frame_count": 1,
                "failed_timestamps_seconds": [],
                "maximum_absolute_seek_error_seconds": 0.0,
                "average_absolute_seek_error_seconds": 0.0,
                "repeated_decoded_timestamps_seconds": [],
            },
            frame_analysis=frame_analysis,
            temporal_analysis=_minimal_temporal_analysis(),
            audio_analysis=_minimal_audio_analysis(),
            evidence=[],
            observations={"metadata_facts": [], "missing_metadata": [], "temporal_heuristics": []},
            warnings=[],
            raw_ffprobe_artifact={"path": "ffprobe_raw.json", "size_bytes": 2, "size_human_readable": "2 B", "sha256": "abc"},
        )
        self.assertEqual(validate_report(report)["errors"], [])
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_reports(Path(tmp), report)
            parsed = json.loads(paths["json"].read_text(encoding="utf-8"))
            text = paths["txt"].read_text(encoding="utf-8")
        self.assertIn("Heuristic near-black frame count: 1", text)
        self.assertEqual(parsed["frame_analysis"]["summary"]["heuristic_near_black_frame_count"], 1)


def _minimal_metadata() -> dict[str, object]:
    return {
        "container": {"duration_seconds": None, "duration_readable": None, "format_name": None, "friendly_format": None, "format_long_name": None, "start_time": None, "bit_rate": None, "bit_rate_readable": None, "probe_score": None, "stream_count": 0, "tags": {}},
        "video": {"present": False, "index": None, "selected_stream_index": None, "selection_reason": "test", "codec_name": None, "codec_long_name": None, "codec_profile": None, "codec_tag": None, "width": None, "height": None, "coded_width": None, "coded_height": None, "pixel_format": None, "codec_level_raw": None, "codec_level_readable": None, "average_frame_rate_raw": None, "nominal_frame_rate_raw": None, "frame_rate_decimal": None, "time_base": None, "start_time": None, "duration_seconds": None, "duration_readable": None, "frame_count": {"reported_by_stream": None, "estimated_from_duration": None, "difference": None}, "bit_rate": None, "bit_rate_readable": None, "sample_aspect_ratio": None, "display_aspect_ratio": None, "field_order": None, "color_range": None, "color_space": None, "color_transfer": None, "color_primaries": None, "rotation": None, "tags": {}, "disposition": {}},
        "audio": {"present": False, "index": None, "selected_stream_index": None, "selection_reason": "test", "codec_name": None, "codec_long_name": None, "codec_profile": None, "sample_format": None, "sample_rate": None, "channels": None, "channel_layout": None, "time_base": None, "start_time": None, "duration_seconds": None, "duration_readable": None, "bit_rate": None, "bit_rate_readable": None, "tags": {}, "disposition": {}},
        "duration_comparison": {"container_duration_seconds": None, "video_start_time_seconds": None, "video_duration_seconds": None, "video_end_time_seconds": None, "audio_start_time_seconds": None, "audio_duration_seconds": None, "audio_end_time_seconds": None, "duration_only_difference_seconds": None, "start_time_difference_seconds": None, "end_time_difference_seconds": None},
        "encoding": {"container_encoder": None, "video_stream_encoder": None, "audio_stream_encoder": None, "video_handler_name": None, "audio_handler_name": None, "container_vendor_id": None, "video_vendor_id": None, "audio_vendor_id": None},
        "display": {"orientation": None, "width": None, "height": None, "rotation": None, "sample_aspect_ratio": None, "display_aspect_ratio": None},
    }


def _minimal_temporal_analysis() -> dict[str, object]:
    return {
        "status": "skipped",
        "reason_code": "test",
        "reason": "test",
        "configuration": {
            "requested_analysis_fps": 5.0,
            "effective_analysis_fps": 5.0,
            "maximum_analyzed_frames": 1500,
            "resize_max_width": 320,
        },
        "summary": {
            "temporal_frames_analyzed": 0,
            "transitions_analyzed": 0,
            "scene_count": 0,
            "scene_boundary_candidate_count": 0,
            "sustained_near_static_interval_count": 0,
            "motion_summary_available": False,
            "requested_analysis_fps": 5.0,
            "effective_analysis_fps": 5.0,
        },
        "scenes": [],
        "scene_representative_frames": [],
        "notable_intervals": [],
        "notable_transitions": [],
        "observations": [],
        "limitations": [],
        "artifacts": {},
    }


def _minimal_audio_analysis() -> dict[str, object]:
    return {
        "status": "skipped",
        "reason_code": "test",
        "reason": "test",
        "configuration": {},
        "extraction": {},
        "decoded_audio": {},
        "timeline": {},
        "summary": {"audio_available": False, "analysis_metrics_available": False},
        "global_metrics": {},
        "silence_intervals": [],
        "clipping_intervals": [],
        "notable_transitions": [],
        "observations": [],
        "limitations": [],
        "artifacts": {},
    }


if __name__ == "__main__":
    unittest.main()
