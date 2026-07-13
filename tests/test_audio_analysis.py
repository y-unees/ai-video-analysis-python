from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

from audio.analyzer import write_audio_metrics
from audio.analyzer import analyze_audio
from audio.extractor import extract_audio_to_wav
from audio.interval_detection import detect_silence_intervals
from audio.metrics import calculate_global_metrics
from audio.transition_ranking import rank_audio_transitions
from audio.window_analysis import analyze_windows


class AudioAnalysisTests(unittest.TestCase):
    def test_silence_metrics(self) -> None:
        samples = np.zeros((1000, 1), dtype=np.float32)
        metrics = calculate_global_metrics(samples)
        self.assertEqual(metrics["rms_amplitude"], 0.0)
        self.assertEqual(metrics["peak_absolute_amplitude"], 0.0)
        self.assertEqual(metrics["silence_ratio"], 1.0)

    def test_sine_wave_metrics(self) -> None:
        sample_rate = 1000
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        samples = (0.5 * np.sin(2 * np.pi * 10 * t)).reshape(-1, 1).astype(np.float32)
        metrics = calculate_global_metrics(samples)
        self.assertGreater(metrics["rms_amplitude"], 0.3)
        self.assertLess(metrics["peak_absolute_amplitude"], 0.51)

    def test_clipping_and_stereo_imbalance(self) -> None:
        left = np.ones(1000, dtype=np.float32)
        right = np.ones(1000, dtype=np.float32) * 0.1
        metrics = calculate_global_metrics(np.column_stack([left, right]))
        self.assertGreater(metrics["clipping_ratio"], 0)
        self.assertGreater(metrics["channel_imbalance_ratio"], 0)

    def test_windows_and_silence_interval(self) -> None:
        samples = np.zeros((2000, 1), dtype=np.float32)
        windows = analyze_windows(samples, 1000)
        intervals = detect_silence_intervals(windows)
        self.assertGreater(len(windows), 1)
        self.assertIn("actual_duration_seconds", windows[-1])
        self.assertIn("sample_count", windows[-1])
        self.assertEqual(len(intervals), 1)

    def test_analyze_audio_accepts_string_and_path_for_no_audio(self) -> None:
        metadata = {"audio": {"present": False}}
        with tempfile.TemporaryDirectory() as tmp:
            result_string, _ = analyze_audio("source_videos/test_video.mp4", tmp, metadata)
            result_path, _ = analyze_audio(Path("source_videos/test_video.mp4"), Path(tmp), metadata)
        self.assertEqual(result_string["status"], "skipped")
        self.assertEqual(result_path["status"], "skipped")
        self.assertEqual(result_string["reason_code"], "no_audio_stream")

    def test_audio_extractor_accepts_string_and_path_temp_dirs(self) -> None:
        def fake_run(command: list[str], **_: object) -> SimpleNamespace:
            Path(command[-1]).write_bytes(b"fake wav bytes")
            return SimpleNamespace(returncode=0, stderr="")

        for temp_dir_factory in (str, Path):
            with self.subTest(temp_dir_factory=temp_dir_factory):
                with tempfile.TemporaryDirectory() as tmp:
                    with patch("audio.extractor.subprocess.run", side_effect=fake_run):
                        wav_path, extraction, warnings = extract_audio_to_wav(
                            "source_videos/test_video.mp4",
                            1,
                            temp_dir_factory(tmp),
                        )
                self.assertIsNotNone(wav_path)
                self.assertEqual(extraction["status"], "completed")
                self.assertEqual(extraction["reason_code"], None)
                self.assertEqual(warnings, [])

    def test_analyze_audio_path_combinations_reach_extraction_setup(self) -> None:
        metadata = {"audio": {"present": True, "selected_stream_index": 1}}

        def fake_extract(video_path: Path, stream_index: int | None, temp_dir: Path) -> tuple[None, dict[str, object], list[str]]:
            self.assertIsInstance(video_path, Path)
            self.assertEqual(stream_index, 1)
            self.assertIsInstance(temp_dir, Path)
            return None, {"status": "skipped", "reason_code": "mock_skip", "reason": "Mock skip."}, []

        with tempfile.TemporaryDirectory() as tmp:
            combinations = (
                ("source_videos/test_video.mp4", tmp),
                (Path("source_videos/test_video.mp4"), tmp),
                ("source_videos/test_video.mp4", Path(tmp)),
                (Path("source_videos/test_video.mp4"), Path(tmp)),
            )
            for video_path, analysis_dir in combinations:
                with self.subTest(video_path=type(video_path).__name__, analysis_dir=type(analysis_dir).__name__):
                    with patch("audio.analyzer.extract_audio_to_wav", side_effect=fake_extract):
                        result, warnings = analyze_audio(video_path, analysis_dir, metadata)
                    self.assertEqual(result["status"], "skipped")
                    self.assertEqual(result["reason_code"], "mock_skip")
                    self.assertEqual(warnings, [])

    def test_audio_transition_ranking(self) -> None:
        windows = [
            {"window_index": 0, "start_timestamp_seconds": 0.0, "rms_amplitude": 0.01, "peak_absolute_amplitude": 0.1, "spectral_centroid_hz": 100.0},
            {"window_index": 1, "start_timestamp_seconds": 0.25, "rms_amplitude": 0.5, "peak_absolute_amplitude": 0.8, "spectral_centroid_hz": 200.0},
            {"window_index": 2, "start_timestamp_seconds": 0.5, "rms_amplitude": 0.5, "peak_absolute_amplitude": 0.8, "spectral_centroid_hz": 200.0},
        ]
        transitions = rank_audio_transitions(windows)
        self.assertEqual(transitions[0]["selection_basis"], "relative_within_audio")
        self.assertTrue(transitions[0]["ranked_review_transition"])

    def test_audio_metrics_artifact(self) -> None:
        windows = [{"audio_window_id": "audio-window-00000", "window_index": 0}]
        with tempfile.TemporaryDirectory() as tmp:
            artifact = write_audio_metrics(Path(tmp), windows)
            self.assertEqual(artifact["path"], "audio_metrics.jsonl")
            self.assertGreater(artifact["size_bytes"], 0)
            self.assertEqual(len(artifact["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
