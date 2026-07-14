from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from visual_consistency.analyzer import analyze_visual_consistency
from visual_consistency.artifact_writer import write_review_artifacts
from visual_consistency.grid import build_region_grid
from visual_consistency.interval_detection import detect_sustained_intervals
from visual_consistency.region_metrics import calculate_transition_region_records
from visual_consistency.transition_ranking import rank_review_transitions


class VisualConsistencyTests(unittest.TestCase):
    def test_grid_coverage_odd_dimensions(self) -> None:
        regions = build_region_grid(101, 77, 4, 4)
        self.assertEqual(len(regions), 16)
        self.assertEqual(regions[0]["region_id"], "region-r00-c00")
        covered = np.zeros((77, 101), dtype=np.uint8)
        for region in regions:
            bounds = region["pixel_bounds"]
            self.assertGreater(bounds["width"], 0)
            self.assertGreater(bounds["height"], 0)
            covered[
                bounds["y"] : bounds["y"] + bounds["height"],
                bounds["x"] : bounds["x"] + bounds["width"],
            ] += 1
        self.assertEqual(int(covered.min()), 1)
        self.assertEqual(int(covered.max()), 1)

    def test_region_metrics_identical_frames_are_stable(self) -> None:
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[16:48, 16:48] = 180
        records = calculate_transition_region_records(
            gray,
            gray.copy(),
            _transition(0, 1),
            build_region_grid(64, 64, 4, 4),
        )
        self.assertEqual(len(records), 16)
        self.assertTrue(all(record["detail_persistence"]["mean_normalized_residual"] == 0 for record in records))
        self.assertTrue(all(record["edge_stability"]["edge_iou"] == 1 for record in records if record["edge_stability"]["previous_edge_density"] == 0))

    def test_region_metrics_detect_brightness_and_detail_change(self) -> None:
        previous = np.zeros((64, 64), dtype=np.uint8)
        current = previous.copy()
        current[16:32, 16:32] = 255
        records = calculate_transition_region_records(
            previous,
            current,
            _transition(0, 1),
            build_region_grid(64, 64, 4, 4),
        )
        self.assertTrue(any(record["brightness_consistency"]["normalized_difference"] > 0 for record in records))
        self.assertTrue(any(record["classification"]["regional_visual_variation"] for record in records))

    def test_sustained_interval_detection(self) -> None:
        records = []
        for index, active in enumerate([False, True, True, True, False]):
            record = _record(index)
            record["classification"]["regional_visual_variation"] = active
            records.append(record)
        intervals = detect_sustained_intervals(records)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["transition_count"], 3)

    def test_ranking_is_deterministic_and_complete(self) -> None:
        summaries = [_summary(index, value) for index, value in enumerate([0.1, 0.7, 0.3])]
        ranked = rank_review_transitions(summaries)
        self.assertEqual(ranked[0]["transition_id"], "temporal-transition-00001-00002")
        self.assertEqual(ranked[0]["combined_percentile_metric_count"], 7)
        self.assertEqual(ranked[0]["available_metric_count"], 7)
        self.assertEqual(ranked[0]["selection_basis"], "relative_within_video")
        self.assertFalse(ranked[0]["absolute_significance_assessed"])

    def test_artifact_writer_creates_relative_hashed_artifacts(self) -> None:
        gray = np.zeros((32, 32), dtype=np.uint8)
        regions = build_region_grid(32, 32, 4, 4)
        ranked = [
            {
                "ranked_review_index": 1,
                "transition_id": "temporal-transition-00000-00001",
                "start_timestamp_seconds": 0.0,
                "end_timestamp_seconds": 0.2,
            }
        ]
        records = [*_minimal_records(regions)]
        with tempfile.TemporaryDirectory() as tmp:
            artifacts, warnings = write_review_artifacts(
                Path(tmp) / "consistency_frames",
                ranked,
                {
                    "temporal-transition-00000-00001": {
                        "previous_gray": gray,
                        "current_gray": gray,
                        "detail_residual": gray,
                    }
                },
                {"temporal-transition-00000-00001": records},
                regions,
            )
        self.assertEqual(warnings, [])
        record = artifacts["temporal-transition-00000-00001"]["artifacts"]["before_grid"]
        self.assertTrue(record["path"].startswith("consistency_frames/"))
        self.assertEqual(len(record["sha256"]), 64)

    def test_analyzer_writes_expected_jsonl_count(self) -> None:
        samples = [
            _sample(0, 0.0, np.zeros((32, 32), dtype=np.uint8)),
            _sample(1, 0.2, np.ones((32, 32), dtype=np.uint8) * 20),
            _sample(2, 0.4, np.ones((32, 32), dtype=np.uint8) * 40),
        ]
        transitions = [_transition(0, 1), _transition(1, 2)]
        temporal = {"status": "completed", "_runtime": {"samples": samples, "transitions": transitions}}
        with tempfile.TemporaryDirectory() as tmp:
            result, warnings = analyze_visual_consistency(Path(tmp), temporal)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["summary"]["consistency_record_count"], 32)
        self.assertEqual(result["artifacts"]["visual_consistency_metrics_artifact"]["path"], "visual_consistency_metrics.jsonl")
        self.assertEqual(warnings, [])


def _sample(index: int, timestamp: float, gray: np.ndarray) -> dict[str, object]:
    return {
        "sample": {
            "temporal_sample_index": index,
            "source_frame_index": index,
            "timestamp_seconds": timestamp,
            "timestamp_source": "test",
            "width": gray.shape[1],
            "height": gray.shape[0],
        },
        "gray": gray,
    }


def _transition(start_index: int, end_index: int) -> dict[str, object]:
    return {
        "transition_id": f"temporal-transition-{start_index:05d}-{end_index:05d}",
        "from_sample_index": start_index,
        "to_sample_index": end_index,
        "start_timestamp_seconds": round(start_index * 0.2, 6),
        "end_timestamp_seconds": round(end_index * 0.2, 6),
        "scene_index": 0,
    }


def _record(index: int) -> dict[str, object]:
    return {
        "transition_id": f"temporal-transition-{index:05d}-{index + 1:05d}",
        "from_sample_index": index,
        "to_sample_index": index + 1,
        "start_timestamp_seconds": round(index * 0.2, 6),
        "end_timestamp_seconds": round((index + 1) * 0.2, 6),
        "region": {"region_id": "region-r00-c00"},
        "classification": {"regional_visual_variation": False},
        "ranking_data": {
            "edge_instability": 0.5,
            "texture_distance": 0.5,
            "detail_residual": 0.5,
        },
    }


def _summary(index: int, value: float) -> dict[str, object]:
    return {
        "transition_id": f"temporal-transition-{index:05d}-{index + 1:05d}",
        "from_sample_index": index,
        "to_sample_index": index + 1,
        "start_timestamp_seconds": index * 0.2,
        "end_timestamp_seconds": (index + 1) * 0.2,
        "scene_index": 0,
        "regional_summary": {
            "valid_region_count": 16,
            "total_region_count": 16,
            "unstable_region_count": int(value * 10),
            "high_motion_region_count": 0,
            "average_edge_instability": value,
            "maximum_edge_instability": value,
            "average_texture_distance": value,
            "maximum_texture_distance": value,
            "average_regional_detail_residual": value,
            "maximum_regional_detail_residual": value,
            "average_regional_brightness_change": value,
            "maximum_regional_brightness_change": value,
            "regional_brightness_change_concentration": value,
        },
    }


def _minimal_records(regions: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for region in regions:
        output.append(
            {
                "region": region,
                "ranking_data": {
                    "edge_instability": 0.0,
                    "texture_distance": 0.0,
                    "detail_residual": 0.0,
                    "brightness_normalized_difference": 0.0,
                },
            }
        )
    return output


if __name__ == "__main__":
    unittest.main()
