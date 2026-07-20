from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset_tools.feature_audit import FeatureAuditError, load_feature_export, run_feature_audit
from tools import dataset_tool


class FeatureAuditTests(unittest.TestCase):
    def test_valid_balanced_dataset_outputs_and_model_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = _write_dataset(root / "dataset_features.csv", _balanced_rows())
            output_dir = root / "statistics"
            audit = run_feature_audit(input_path=input_path, output_dir=output_dir)

            profile = audit["dataset_profile"]
            self.assertEqual(profile["total_samples"], 6)
            self.assertEqual(profile["class_counts"], {"real": 3, "ai_generated": 3})
            self.assertEqual(profile["model_readiness_status"], "insufficient_dataset_size")
            self.assertIn("media.duration_seconds", audit["model_feature_schema"]["included_features"])
            self.assertNotIn("identity.video_name", audit["model_feature_schema"]["included_features"])
            self.assertTrue((output_dir / "dataset_profile.json").exists())
            self.assertTrue((output_dir / "column_profile.csv").exists())
            self.assertTrue((output_dir / "feature_statistics.csv").exists())
            self.assertTrue((output_dir / "class_comparison.csv").exists())
            self.assertTrue((output_dir / "correlation_matrix.csv").exists())
            self.assertTrue((output_dir / "statistics_report.txt").exists())

    def test_missing_empty_malformed_and_missing_target_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FeatureAuditError):
                load_feature_export(root / "missing.csv")
            empty = root / "empty.csv"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(FeatureAuditError):
                load_feature_export(empty)
            malformed = root / "bad.jsonl"
            malformed.write_text("{bad json}\n", encoding="utf-8")
            with self.assertRaises(FeatureAuditError):
                load_feature_export(malformed)
            no_target = _write_dataset(root / "no_target.csv", [{"sample_id": "a", "feature": 1}])
            with self.assertRaises(FeatureAuditError):
                run_feature_audit(input_path=no_target, output_dir=root / "out")

    def test_unexpected_and_single_class_labels_are_warnings_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unexpected = _write_dataset(
                root / "unexpected.csv",
                [
                    {"sample_id": "a", "expected_label": "real", "identity.video_sha256": "sha1", "feature": 1},
                    {"sample_id": "b", "expected_label": "synthetic", "identity.video_sha256": "sha2", "feature": 2},
                ],
            )
            audit = run_feature_audit(input_path=unexpected, output_dir=root / "unexpected_out")
            self.assertIn("Unexpected labels found", " ".join(audit["dataset_profile"]["warnings"]))

            single = _write_dataset(
                root / "single.csv",
                [
                    {"sample_id": "a", "expected_label": "real", "identity.video_sha256": "sha1", "feature": 1},
                    {"sample_id": "b", "expected_label": "real", "identity.video_sha256": "sha2", "feature": 2},
                ],
            )
            audit = run_feature_audit(input_path=single, output_dir=root / "single_out")
            self.assertIn("Expected labels missing", " ".join(audit["dataset_profile"]["warnings"]))

    def test_feature_quality_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = _balanced_rows()
            rows[0]["mostly_missing"] = ""
            rows[1]["mostly_missing"] = ""
            rows[2]["mostly_missing"] = ""
            rows[3]["mostly_missing"] = ""
            rows[4]["mostly_missing"] = "1"
            rows[5]["mostly_missing"] = "2"
            rows[0]["mixed_numeric"] = "abc123"
            for row in rows[1:]:
                row["mixed_numeric"] = "3"
            for row in rows:
                row["fully_empty"] = ""
                row["constant_feature"] = "7"
                row["near_constant_feature"] = "same"
            rows[-1]["near_constant_feature"] = "different"
            input_path = _write_dataset(root / "quality.csv", rows)
            audit = run_feature_audit(input_path=input_path, output_dir=root / "out", near_constant_threshold=0.80)
            quality = {item["column"]: item for item in audit["feature_quality"]["features"]}

            self.assertEqual(quality["fully_empty"]["primary_status"], "fully_empty")
            self.assertEqual(quality["constant_feature"]["primary_status"], "constant")
            self.assertEqual(quality["near_constant_feature"]["primary_status"], "near_constant")
            self.assertEqual(quality["mostly_missing"]["primary_status"], "missing_heavy")
            self.assertEqual(quality["mixed_numeric"]["primary_status"], "invalid_numeric")

    def test_leakage_identifier_path_and_duplicate_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = _balanced_rows()
            rows[1]["sample_id"] = rows[0]["sample_id"]
            rows[2]["identity.video_sha256"] = rows[0]["identity.video_sha256"]
            rows[0]["notes"] = "real personal source"
            input_path = _write_dataset(root / "leakage.csv", rows)
            audit = run_feature_audit(input_path=input_path, output_dir=root / "out")
            profile = audit["dataset_profile"]
            leakage_columns = {finding["column"] for finding in audit["leakage_report"]["findings"]}

            self.assertGreaterEqual(profile["duplicate_sample_ids"], 1)
            self.assertGreaterEqual(profile["duplicate_source_hashes"], 1)
            self.assertIn("sample_id", leakage_columns)
            self.assertIn("identity.video_name", leakage_columns)
            self.assertIn("notes", leakage_columns)
            self.assertIn("identity.video_sha256", profile["identifier_columns"])

    def test_statistics_comparison_and_correlation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = _write_dataset(root / "stats.csv", _balanced_rows())
            audit = run_feature_audit(input_path=input_path, output_dir=root / "out")

            comparison = {row["column"]: row for row in audit["class_comparison"]}
            self.assertIn("feature_a", comparison)
            self.assertIsNotNone(comparison["feature_a"]["standardized_effect_size"])
            self.assertIsNotNone(comparison["feature_a"]["point_biserial_correlation"])
            correlated = {(row["feature_a"], row["feature_b"]) for row in audit["high_correlation_pairs"]}
            self.assertIn(("feature_a", "feature_b"), correlated)

    def test_jsonl_input_and_valid_json_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "dataset_features.jsonl"
            jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in _balanced_rows()) + "\n", encoding="utf-8")
            output_dir = root / "out"
            run_feature_audit(input_path=jsonl, output_dir=output_dir)
            payload = json.loads((output_dir / "model_feature_schema.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "0.9.3")
            self.assertNotIn("NaN", (output_dir / "dataset_profile.json").read_text(encoding="utf-8"))

    def test_cli_commands_return_success_and_fatal_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = _write_dataset(root / "dataset_features.csv", _balanced_rows())
            output_dir = root / "out"
            for command in ("statistics", "feature-audit", "model-schema", "audit-features"):
                with patch("sys.argv", ["dataset_tool.py", command, "--input", str(input_path), "--output-dir", str(output_dir)]):
                    self.assertEqual(dataset_tool.main(), 0)
            with patch("sys.argv", ["dataset_tool.py", "audit-features", "--input", str(root / "missing.csv"), "--output-dir", str(output_dir)]):
                self.assertEqual(dataset_tool.main(), 1)


def _balanced_rows() -> list[dict[str, object]]:
    return [
        _row("real-001", "real", "real__camera_001.mp4", "sha001", 10, 100, 100),
        _row("real-002", "real", "real__camera_002.mp4", "sha002", 11, 110, 110),
        _row("real-003", "real", "real__camera_003.mp4", "sha003", 12, 120, 120),
        _row("ai_generated-001", "ai_generated", "ai__gen_001.mp4", "sha004", 30, 300, 300),
        _row("ai_generated-002", "ai_generated", "ai__gen_002.mp4", "sha005", 31, 310, 310),
        _row("ai_generated-003", "ai_generated", "ai__gen_003.mp4", "sha006", 32, 320, 320),
    ]


def _row(sample_id: str, label: str, video_name: str, sha: str, feature_a: int, feature_b: int, duration: int) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "expected_label": label,
        "identity.expected_label": label,
        "identity.video_name": video_name,
        "identity.video_sha256": sha,
        "feature_file": f"features/{sample_id}.json",
        "notes": "",
        "source": "manual_source",
        "media.duration_seconds": duration,
        "feature_a": feature_a,
        "feature_b": feature_b,
        "constant_feature": 1,
    }


def _write_dataset(path: Path, rows: list[dict[str, object]]) -> Path:
    columns = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


if __name__ == "__main__":
    unittest.main()
