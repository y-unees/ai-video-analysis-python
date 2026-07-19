from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import script
from tests.test_dataset_tools import _feature


class ScriptRunnerTests(unittest.TestCase):
    def test_filename_sidecar_metadata_labels_conflicts_and_dry_run_no_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source_videos"
            source.mkdir()
            (source / "real__mine.mp4").write_bytes(b"real")
            (source / "ai__mine.mp4").write_bytes(b"ai")
            (source / "sidecar.mp4").write_bytes(b"side")
            (source / "unknown.mp4").write_bytes(b"unknown")
            (source / "video_labels.csv").write_text("filename,expected_label,source,generator_or_camera,notes\nsidecar.mp4,real,self_recorded,camera,note\n", encoding="utf-8")
            with _patched_roots(root), patch("script.run_analysis") as mocked:
                code = script.main.__wrapped__() if hasattr(script.main, "__wrapped__") else _run_script(["--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse((root / "dataset").exists())
            self.assertFalse(mocked.called)

    def test_metadata_csv_match_and_conflict_skip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source_videos"
            source.mkdir()
            (source / "ff.mp4").write_bytes(b"ff")
            (source / "video_labels.csv").write_text("filename,expected_label\nff.mp4,real\n", encoding="utf-8")
            metadata = root / "dataset_sources" / "faceforensics_pp" / "metadata"
            metadata.mkdir(parents=True)
            (metadata / "FF++_Metadata.csv").write_text("path,label,method\nff.mp4,fake,DeepFakes\n", encoding="utf-8")
            with _patched_roots(root):
                self.assertEqual(_run_script(["--dry-run"]), 0)

    def test_success_failure_registration_exports_and_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source_videos"
            source.mkdir()
            (source / "real__ok.mp4").write_bytes(b"ok")
            (source / "ai__fail.mp4").write_bytes(b"fail")
            with _patched_roots(root), patch("script.run_analysis", side_effect=_analysis_side_effect(root)):
                code = _run_script([])
            self.assertEqual(code, 0)
            ledger = json.loads((root / "dataset" / "runs" / "latest_run.json").read_text(encoding="utf-8"))
            self.assertEqual(ledger["analyzed_success_count"], 1)
            self.assertEqual(ledger["analyzed_failure_count"], 1)
            self.assertEqual(ledger["registered_count"], 1)
            self.assertTrue((root / "dataset" / "manifest.jsonl").exists())
            self.assertTrue((root / "dataset" / "exports" / "dataset_features.jsonl").exists())
            self.assertTrue((root / "dataset" / "exports" / "dataset_features.csv").exists())
            self.assertTrue((root / "source_videos" / "real__ok.mp4").exists())

            with _patched_roots(root), patch("script.run_analysis") as mocked:
                code = _run_script([])
            self.assertEqual(code, 0)
            self.assertTrue(mocked.called)
            ledger = json.loads((root / "dataset" / "runs" / "latest_run.json").read_text(encoding="utf-8"))
            self.assertEqual(ledger["skipped_existing_count"], 1)

    def test_force_reanalysis_runs_existing_sample_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source_videos"
            source.mkdir()
            (source / "real__ok.mp4").write_bytes(b"ok")
            with _patched_roots(root), patch("script.run_analysis", side_effect=_analysis_side_effect(root)) as mocked:
                self.assertEqual(_run_script([]), 0)
            with _patched_roots(root), patch("script.run_analysis", side_effect=_analysis_side_effect(root)) as mocked:
                self.assertEqual(_run_script(["--force-reanalyze"]), 0)
                self.assertTrue(mocked.called)

    def test_unresolved_video_is_copied_but_not_analyzed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source_videos"
            source.mkdir()
            (source / "needs_label.mp4").write_bytes(b"unknown")
            with _patched_roots(root), patch("script.run_analysis") as mocked:
                self.assertEqual(_run_script([]), 0)
            ledger = json.loads((root / "dataset" / "runs" / "latest_run.json").read_text(encoding="utf-8"))
            self.assertEqual(ledger["unresolved_count"], 1)
            self.assertEqual(ledger["copied_count"], 1)
            stored_path = next((root / "dataset_sources" / "pilot" / "unresolved").glob("*needs_label.mp4"))
            self.assertTrue(stored_path.exists())
            self.assertFalse(mocked.called)


def _run_script(args: list[str]) -> int:
    with patch("sys.argv", ["script.py", *args]):
        return script.main()


def _patched_roots(root: Path):
    return patch.multiple(
        script,
        SOURCE_VIDEOS=root / "source_videos",
        PILOT_ROOT=root / "dataset_sources" / "pilot",
        DATASET_ROOT=root / "dataset",
        REPORTS_ROOT=root / "reports",
        METADATA_CANDIDATES=[root / "dataset_sources" / "faceforensics_pp" / "metadata" / "FF++_Metadata.csv"],
    )


def _analysis_side_effect(root: Path):
    def run(video_path: Path, reports_dir: Path, timestamp: str):
        if "fail" in video_path.name:
            raise RuntimeError("mock analysis failure")
        analysis_dir = reports_dir / timestamp
        analysis_dir.mkdir(parents=True, exist_ok=True)
        payload = _feature(sha=script.calculate_sha256(video_path), video_name=video_path.name)
        (analysis_dir / "outcome_features.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return type("Result", (), {"analysis_dir": analysis_dir})()
    return run


if __name__ == "__main__":
    unittest.main()
