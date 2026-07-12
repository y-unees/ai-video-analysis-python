from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from file_utils import format_file_size
from temporal.analyzer import write_temporal_metrics
from temporal.interval_detection import detect_static_intervals
from temporal.metrics import calculate_transition, classify_transition
from temporal.notability import rank_notable_transitions
from temporal.optical_flow import calculate_optical_flow
from temporal.scene_detection import construct_scenes


class TemporalAnalysisTests(unittest.TestCase):
    def test_identical_frames_are_near_static(self) -> None:
        previous = _sample(0, 0.0, np.zeros((64, 64), dtype=np.uint8))
        current = _sample(1, 0.2, np.zeros((64, 64), dtype=np.uint8))
        transition = calculate_transition(previous, current)
        self.assertTrue(transition["classification"]["near_static_transition"])
        self.assertEqual(transition["normalized_mean_absolute_difference"], 0.0)

    def test_different_frames_can_be_scene_boundary(self) -> None:
        classification = classify_transition(32, 0.5, 0.1)
        self.assertTrue(classification["scene_boundary_candidate"])

    def test_scene_construction_from_hard_cut_sequence(self) -> None:
        samples = [
            _sample(index, index * 0.2, np.zeros((32, 32), dtype=np.uint8))
            for index in range(4)
        ]
        transitions = [
            _transition(0, 1, 0.0, 0.2, False),
            _transition(1, 2, 0.2, 0.4, True),
            _transition(2, 3, 0.4, 0.6, False),
        ]
        scenes = construct_scenes(samples, transitions)
        self.assertEqual(len(scenes), 2)
        self.assertLessEqual(scenes[0]["end_timestamp_seconds"], scenes[1]["start_timestamp_seconds"])

    def test_static_interval_merging(self) -> None:
        transitions = [
            _transition(0, 1, 0.0, 0.2, False, near_static=False),
            _transition(1, 2, 0.2, 0.4, False, near_static=True),
            _transition(2, 3, 0.4, 0.6, False, near_static=True),
            _transition(3, 4, 0.6, 0.9, False, near_static=True),
            _transition(4, 5, 0.9, 1.1, False, near_static=False),
        ]
        intervals = detect_static_intervals(transitions)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["transition_count"], 3)

    def test_optical_flow_metrics_are_finite(self) -> None:
        first = np.zeros((64, 64), dtype=np.uint8)
        second = np.zeros((64, 64), dtype=np.uint8)
        second[20:30, 22:32] = 255
        metrics = calculate_optical_flow(first, second)
        self.assertIsNotNone(metrics["mean_magnitude"])
        self.assertGreaterEqual(metrics["mean_magnitude"], 0)

    def test_flow_warp_residual_identical_frames(self) -> None:
        first = _sample(0, 0.0, np.zeros((64, 64), dtype=np.uint8))
        second = _sample(1, 0.2, np.zeros((64, 64), dtype=np.uint8))
        transition = calculate_transition(first, second)
        self.assertEqual(transition["flow_warp_residual"]["mean_normalized_residual"], 0.0)
        self.assertGreater(transition["flow_warp_residual"]["valid_pixel_count"], 0)

    def test_flow_warp_residual_brightness_change(self) -> None:
        first = _sample(0, 0.0, np.zeros((64, 64), dtype=np.uint8))
        second = _sample(1, 0.2, np.ones((64, 64), dtype=np.uint8) * 128)
        transition = calculate_transition(first, second)
        self.assertGreater(transition["flow_warp_residual"]["mean_normalized_residual"], 0)

    def test_temporal_metrics_jsonl_artifact(self) -> None:
        transition = _transition(0, 1, 0.0, 0.2, False)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = write_temporal_metrics(Path(tmp), [transition])
            self.assertEqual(artifact["path"], "temporal_metrics.jsonl")
            self.assertGreater(artifact["size_bytes"], 0)
            self.assertEqual(len(artifact["sha256"]), 64)

    def test_binary_size_units(self) -> None:
        self.assertEqual(format_file_size(1023), "1023 B")
        self.assertEqual(format_file_size(1024), "1.00 KiB")
        self.assertEqual(format_file_size(1048576), "1.00 MiB")

    def test_ranked_notable_transition_merges_reasons(self) -> None:
        transitions = [
            _transition(0, 1, 0.0, 0.2, False),
            _transition(1, 2, 0.2, 0.4, False),
            _transition(2, 3, 0.4, 0.6, False),
        ]
        transitions[1]["normalized_mean_absolute_difference"] = 0.9
        transitions[1]["absolute_brightness_difference"] = 80.0
        transitions[1]["histogram_correlation"] = 0.05
        transitions[1]["transition_id"] = "temporal-transition-00001-00002"
        for index, transition in enumerate(transitions):
            transition.setdefault("transition_id", f"temporal-transition-{index:05d}-{index + 1:05d}")
            transition["flow_warp_residual"] = {
                "mean_normalized_residual": 0.1 + index * 0.1,
                "median_normalized_residual": 0.1,
                "percentile_95_normalized_residual": 0.2 + index * 0.1,
                "high_residual_pixel_ratio": 0.1,
                "high_residual_threshold": 0.1,
                "valid_pixel_count": 1,
            }
            transition["notability"] = {"reason_count": 0, "reasons": [], "combined_percentile": None, "metric_ranks": {}}
        notable = rank_notable_transitions(transitions)
        self.assertLessEqual(len(notable), 5)
        self.assertTrue(notable[0]["classification"]["ranked_notable_transition"])
        self.assertEqual(notable[0]["notability"]["reason_count"], len(notable[0]["notability"]["reasons"]))
        self.assertFalse(notable[0]["classification"]["scene_boundary_candidate"])


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


def _transition(
    start_index: int,
    end_index: int,
    start: float,
    end: float,
    scene_boundary: bool,
    near_static: bool = False,
) -> dict[str, object]:
    return {
        "transition_id": f"temporal-transition-{start_index:05d}-{end_index:05d}",
        "scene_index": 0,
        "from_sample_index": start_index,
        "to_sample_index": end_index,
        "start_timestamp_seconds": start,
        "end_timestamp_seconds": end,
        "perceptual_hash_distance": 0 if near_static else 20,
        "normalized_mean_absolute_difference": 0.0 if near_static else 0.3,
        "histogram_correlation": 1.0 if near_static else 0.5,
        "absolute_brightness_difference": 0.0,
        "optical_flow": {
            "mean_magnitude": 0.0,
            "median_magnitude": 0.0,
            "percentile_95_magnitude": 0.0,
            "stationary_pixel_ratio": 1.0,
        },
        "flow_warp_residual": {
            "mean_normalized_residual": 0.0,
            "median_normalized_residual": 0.0,
            "percentile_95_normalized_residual": 0.0,
            "high_residual_pixel_ratio": 0.0,
            "high_residual_threshold": 0.1,
            "valid_pixel_count": 1,
        },
        "classification": {
            "scene_boundary_candidate": scene_boundary,
            "near_static_transition": near_static,
            "ranked_notable_transition": False,
            "triggered_rules": [],
        },
        "notability": {
            "reason_count": 0,
            "reasons": [],
            "combined_percentile": None,
            "metric_ranks": {},
        },
    }


if __name__ == "__main__":
    unittest.main()
