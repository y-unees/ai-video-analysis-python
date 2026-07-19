from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from dataset_tools import (
    export_dataset_csv,
    export_dataset_jsonl,
    register_sample,
    summarize_dataset,
    validate_dataset,
)
from dataset_tools.manager import MANIFEST_FILENAME
from outcome import build_outcome_features
from tests.test_outcome_features import _outcome_report


class DatasetToolsTests(unittest.TestCase):
    def test_successful_real_and_ai_generated_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_feature = _write_feature(root / "real_features.json", sha="a" * 64, video_name="real.mp4")
            ai_feature = _write_feature(root / "ai_features.json", sha="b" * 64, video_name="ai.mp4")

            real = register_sample(real_feature, "real", root, source="camera set", generator_or_camera="camera", notes="real notes")
            ai = register_sample(ai_feature, "ai_generated", root, source="generated set", generator_or_camera="generator", notes="ai notes")

            self.assertEqual(real.expected_label, "real")
            self.assertEqual(ai.expected_label, "ai_generated")
            self.assertTrue((root / MANIFEST_FILENAME).exists())
            self.assertTrue((root / real.feature_file).exists())
            self.assertTrue((root / ai.feature_file).exists())
            self.assertEqual(validate_dataset(root)["errors"], [])

    def test_invalid_label_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = _write_feature(root / "features.json")
            with self.assertRaises(ValueError):
                register_sample(feature, "fake", root)

    def test_duplicate_sha_rejected_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write_feature(root / "first.json", sha="c" * 64)
            second = _write_feature(root / "second.json", sha="c" * 64)
            register_sample(first, "real", root)
            with self.assertRaises(ValueError):
                register_sample(second, "ai_generated", root)
            duplicate = register_sample(second, "ai_generated", root, allow_duplicate=True)
            self.assertIn("dup", duplicate.sample_id)

    def test_original_feature_file_remains_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = _write_feature(root / "features.json")
            before = feature.read_text(encoding="utf-8")
            register_sample(feature, "real", root)
            after = feature.read_text(encoding="utf-8")
            copied = json.loads((root / "features" / f"real-{'d' * 16}.json").read_text(encoding="utf-8"))
            self.assertEqual(before, after)
            self.assertIsNone(json.loads(after)["identity"]["expected_label"])
            self.assertEqual(copied["identity"]["expected_label"], "real")

    def test_missing_and_invalid_feature_files_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / MANIFEST_FILENAME).write_text(
                json.dumps(
                    {
                        "sample_id": "missing",
                        "video_name": "missing.mp4",
                        "video_sha256": "e" * 64,
                        "expected_label": "real",
                        "feature_schema_version": "0.9.0",
                        "application_version": "0.9.1",
                        "feature_file": "features/missing.json",
                        "source": None,
                        "generator_or_camera": None,
                        "notes": None,
                        "added_at": "2026-07-19T00:00:00+00:00",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            result = validate_dataset(root)
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("missing" in error for error in result["errors"]))

            bad = root / "bad.json"
            payload = _feature(sha="f" * 64)
            payload["probability"] = 0.5
            bad.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                register_sample(bad, "real", root)

    def test_relative_path_safety(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / MANIFEST_FILENAME).write_text(
                json.dumps(
                    {
                        "sample_id": "escape",
                        "video_name": "escape.mp4",
                        "video_sha256": "a" * 64,
                        "expected_label": "real",
                        "feature_schema_version": "0.9.0",
                        "application_version": "0.9.1",
                        "feature_file": "../outside.json",
                        "source": None,
                        "generator_or_camera": None,
                        "notes": None,
                        "added_at": "2026-07-19T00:00:00+00:00",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            result = validate_dataset(root)
            self.assertTrue(any("escapes" in error for error in result["errors"]))

    def test_deterministic_jsonl_and_csv_exports_with_null_handling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write_feature(root / "b.json", sha="b" * 64, video_name="b.mp4", d3_status="not_run")
            second = _write_feature(root / "a.json", sha="a" * 64, video_name="a.mp4")
            register_sample(first, "real", root)
            register_sample(second, "ai_generated", root)

            jsonl = export_dataset_jsonl(root)
            csv_path = export_dataset_csv(root)
            jsonl_lines = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["sample_id"] for row in jsonl_lines], sorted(row["sample_id"] for row in jsonl_lines))
            self.assertIn("d3_summary.d3_raw_score", jsonl_lines[0])
            self.assertIn(None, [row["d3_summary.d3_raw_score"] for row in jsonl_lines])

            with csv_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertIn("media.duration_seconds", rows[0])
            self.assertIn("", [row["d3_summary.d3_raw_score"] for row in rows])

            second_jsonl = export_dataset_jsonl(root, root / "exports" / "second.jsonl")
            self.assertEqual(jsonl.read_text(encoding="utf-8"), second_jsonl.read_text(encoding="utf-8"))

    def test_dataset_summary_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            register_sample(_write_feature(root / "real.json", sha="1" * 64, d3_status="completed"), "real", root)
            register_sample(_write_feature(root / "ai.json", sha="2" * 64, d3_status="failed"), "ai_generated", root)
            register_sample(_write_feature(root / "unavailable.json", sha="3" * 64, d3_status="unavailable"), "ai_generated", root)
            summary = summarize_dataset(root).to_dict()
            self.assertEqual(summary["total_samples"], 3)
            self.assertEqual(summary["real_sample_count"], 1)
            self.assertEqual(summary["ai_generated_sample_count"], 2)
            self.assertEqual(summary["valid_sample_count"], 3)
            self.assertEqual(summary["invalid_sample_count"], 0)
            self.assertEqual(summary["d3_status_counts"]["completed"], 1)
            self.assertEqual(summary["d3_status_counts"]["failed"], 1)
            self.assertEqual(summary["d3_status_counts"]["unavailable"], 1)
            self.assertIn("0.9.0", summary["feature_schema_versions_present"])
            self.assertGreater(summary["missing_value_count_per_feature"]["temporal_summary.mean_normalized_frame_difference"], 0)

    def test_forbidden_verdict_field_invalidates_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = _write_feature(root / "features.json", sha="4" * 64)
            entry = register_sample(feature_path, "real", root)
            copied = root / entry.feature_file
            payload = json.loads(copied.read_text(encoding="utf-8"))
            payload["verdict"] = "real"
            copied.write_text(json.dumps(payload), encoding="utf-8")
            result = validate_dataset(root)
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("Forbidden field" in error or "forbidden" in error for error in result["errors"]))


def _write_feature(
    path: Path,
    sha: str = "d" * 64,
    video_name: str = "sample.mp4",
    d3_status: str = "completed",
) -> Path:
    path.write_text(json.dumps(_feature(sha, video_name, d3_status), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _feature(sha: str = "d" * 64, video_name: str = "sample.mp4", d3_status: str = "completed") -> dict[str, object]:
    report = _outcome_report()
    report["source"]["sha256"] = sha
    report["source"]["filename"] = video_name
    report["analysis_environment"]["application_version"] = "0.9.2"
    report["learned_detector_results"]["d3"]["execution"]["status"] = d3_status
    if d3_status != "completed":
        report["learned_detector_results"]["d3"]["native_output"]["raw_score"] = None
    return build_outcome_features(report, analysis_id=f"analysis-{sha[:8]}").to_dict()


if __name__ == "__main__":
    unittest.main()
