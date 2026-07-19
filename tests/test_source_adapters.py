from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_tools.source_adapters import (
    audit_faceforensics_metadata,
    normalize_faceforensics_rows,
    plan_faceforensics_samples,
    validate_plan,
)


class SourceAdapterTests(unittest.TestCase):
    def test_csv_audit_schema_and_trusted_labels(self) -> None:
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            metadata = _metadata(root, delimiter=";")
            audit = audit_faceforensics_metadata(metadata, root)
            self.assertEqual(audit["delimiter"], ";")
            self.assertEqual(audit["row_count"], 6)
            self.assertIn("label", audit["column_names"])
            self.assertEqual(audit["trusted_label_mapping"]["real"], 2)
            self.assertEqual(audit["trusted_label_mapping"]["ai_generated"], 3)
            self.assertGreaterEqual(audit["duplicate_row_count"], 1)
            self.assertEqual(audit["row_representation"], "videos")
            self.assertIn(audit["path_style"], {"unix_relative", "filename_only"})
            records = normalize_faceforensics_rows(metadata)
            self.assertEqual(records[0]["normalized_label"], "real")
            self.assertEqual(records[1]["normalized_label"], "ai_generated")
            self.assertIsNone(records[-1]["normalized_label"])

    def test_video_availability_modes_and_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            videos = root / "videos"
            (videos / "sub").mkdir(parents=True)
            (videos / "real1.mp4").write_bytes(b"real")
            (videos / "sub" / "fake1.mp4").write_bytes(b"fake")
            (videos / "a").mkdir()
            (videos / "b").mkdir()
            (videos / "a" / "dup.mp4").write_bytes(b"dup")
            (videos / "b" / "dup.mp4").write_bytes(b"dup")
            metadata = root / "metadata.csv"
            metadata.write_text(
                "\n".join(
                    [
                        "path,label",
                        "real1.mp4,real",
                        "fake1.mp4,fake",
                        "missing.mp4,real",
                        "dup.mp4,fake",
                        "../escape.mp4,real",
                    ]
                ),
                encoding="utf-8",
            )
            audit = audit_faceforensics_metadata(metadata, root)
            self.assertEqual(audit["video_availability"]["video_found"], 2)
            self.assertEqual(audit["video_availability"]["video_missing"], 1)
            self.assertEqual(audit["video_availability"]["ambiguous_video_match"], 1)
            self.assertEqual(audit["video_availability"]["invalid_metadata"], 1)

    def test_deterministic_balanced_planning_and_shortages(self) -> None:
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            metadata = _metadata(root)
            result = plan_faceforensics_samples(metadata, root, real_count=2, ai_count=2, seed=7)
            plan = root / "plans" / "pilot_plan.jsonl"
            lines = [json.loads(line) for line in plan.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["selected_real_count"], 2)
            self.assertEqual(result["selected_ai_generated_count"], 2)
            self.assertEqual([item["plan_index"] for item in lines], [1, 2, 3, 4])
            self.assertEqual(validate_plan(plan, project_root=Path("."), requested_real_count=2, requested_ai_count=2)["errors"], [])
            second = plan_faceforensics_samples(metadata, root, real_count=2, ai_count=2, seed=7)
            self.assertEqual(result["selected_count"], second["selected_count"])

            shortage = plan_faceforensics_samples(metadata, root, real_count=10, ai_count=10, seed=7)
            self.assertGreater(shortage["shortages"]["real"], 0)
            self.assertGreater(shortage["shortages"]["ai_generated"], 0)

    def test_plan_validation_rejects_forbidden_and_unsafe_fields(self) -> None:
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            metadata = _metadata(root)
            plan_faceforensics_samples(metadata, root, real_count=1, ai_count=1, seed=1)
            plan = root / "plans" / "pilot_plan.jsonl"
            entries = [json.loads(line) for line in plan.read_text(encoding="utf-8").splitlines()]
            entries[0]["probability"] = 0.9
            entries[0]["resolved_video_path"] = "../escape.mp4"
            plan.write_text("\n".join(json.dumps(entry, sort_keys=True) for entry in entries) + "\n", encoding="utf-8")
            result = validate_plan(plan, project_root=Path("."))
            self.assertEqual(result["status"], "invalid")
            self.assertTrue(any("Forbidden" in error for error in result["errors"]))
            self.assertTrue(any("unsafe" in error for error in result["errors"]))


def _metadata(root: Path, delimiter: str = ",") -> Path:
    path = root / "metadata.csv"
    rows = [
        ["path", "label", "method", "compression", "frames", "width", "height", "codec"],
        ["real1.mp4", "real", "original", "c23", "100", "1920", "1080", "h264"],
        ["fake1.mp4", "fake", "DeepFakes", "c23", "100", "1920", "1080", "h264"],
        ["real2.mp4", "original", "original", "c23", "100", "1920", "1080", "h264"],
        ["fake2.mp4", "manipulated", "FaceSwap", "c23", "100", "1920", "1080", "h264"],
        ["fake2.mp4", "manipulated", "FaceSwap", "c23", "100", "1920", "1080", "h264"],
        ["unknown.mp4", "unknown", "", "c23", "", "", "", ""],
    ]
    path.write_text("\n".join(delimiter.join(row) for row in rows), encoding="utf-8")
    videos = root / "videos"
    videos.mkdir()
    for name in ("real1.mp4", "fake1.mp4", "real2.mp4", "fake2.mp4"):
        (videos / name).write_bytes(name.encode("utf-8"))
    return path


if __name__ == "__main__":
    unittest.main()
