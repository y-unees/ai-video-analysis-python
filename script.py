from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis import run_analysis
from config import APP_VERSION
from dataset_tools import export_dataset_csv, export_dataset_jsonl, match_faceforensics_video, register_sample, summarize_dataset, validate_dataset
from file_utils import calculate_sha256


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SOURCE_VIDEOS = Path("source_videos")
PILOT_ROOT = Path("dataset_sources") / "pilot"
DATASET_ROOT = Path("dataset")
REPORTS_ROOT = Path("reports")
METADATA_CANDIDATES = [
    Path("dataset_sources") / "faceforensics_pp" / "metadata" / "FF++_Metadata.csv",
    Path("datasets") / "FF++_Metadata.csv",
    Path("dataset") / "FF++_Metadata.csv",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="One-command local dataset preparation runner.")
    parser.add_argument("--real-limit", type=int, default=None)
    parser.add_argument("--ai-limit", type=int, default=None)
    parser.add_argument("--force-reanalyze", action="store_true")
    parser.add_argument("--allow-label-override", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    metadata_path = _metadata_path()
    sidecar = _read_sidecar(SOURCE_VIDEOS / "video_labels.csv")
    videos = sorted([path for path in SOURCE_VIDEOS.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS])
    ledger: dict[str, Any] = {
        "started_at": started.isoformat(timespec="seconds"),
        "application_version": APP_VERSION,
        "videos_discovered": len(videos),
        "per_video": [],
    }
    counts = {"resolved_real_count": 0, "resolved_ai_count": 0, "unresolved_count": 0, "copied_count": 0, "duplicate_count": 0, "analyzed_success_count": 0, "analyzed_failure_count": 0, "registered_count": 0, "skipped_existing_count": 0}
    seen_hashes: dict[str, Path] = {}
    label_seen = {"real": 0, "ai_generated": 0}
    for index, video in enumerate(videos, start=1):
        record = {"source_video": _relative(video)}
        print(f"[{index}/{len(videos)}] Resolving label")
        resolution = _resolve_label(video, metadata_path, sidecar, args.allow_label_override)
        record.update(resolution)
        label = resolution["expected_label"]
        if label == "unresolved":
            counts["unresolved_count"] += 1
            sha = calculate_sha256(video)
            record["video_sha256"] = sha
            destination = _pilot_destination(video, label, sha)
            record["stored_source_path"] = _relative(destination)
            if not args.dry_run and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video, destination)
                counts["copied_count"] += 1
            record["status"] = "skipped_unresolved"
            print(f"[{index}/{len(videos)}] Skipped unresolved")
            ledger["per_video"].append(record)
            continue
        if label == "real" and args.real_limit is not None and label_seen["real"] >= args.real_limit:
            record["status"] = "skipped_limit"
            ledger["per_video"].append(record)
            continue
        if label == "ai_generated" and args.ai_limit is not None and label_seen["ai_generated"] >= args.ai_limit:
            record["status"] = "skipped_limit"
            ledger["per_video"].append(record)
            continue
        label_seen[label] += 1
        counts["resolved_real_count" if label == "real" else "resolved_ai_count"] += 1
        sha = calculate_sha256(video)
        record["video_sha256"] = sha
        if sha in seen_hashes:
            counts["duplicate_count"] += 1
            record["status"] = "skipped_duplicate_source"
            ledger["per_video"].append(record)
            print(f"[{index}/{len(videos)}] Skipped duplicate")
            continue
        seen_hashes[sha] = video
        destination = _pilot_destination(video, label, sha)
        record["stored_source_path"] = _relative(destination)
        existing = _manifest_has_sha(sha)
        if existing and not args.force_reanalyze:
            counts["skipped_existing_count"] += 1
            record["status"] = "skipped_existing"
            ledger["per_video"].append(record)
            print(f"[{index}/{len(videos)}] Skipped existing")
            continue
        if args.dry_run:
            record["status"] = "dry_run_planned"
            ledger["per_video"].append(record)
            continue
        print(f"[{index}/{len(videos)}] Copying source")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, destination)
            counts["copied_count"] += 1
        print(f"[{index}/{len(videos)}] Running analysis")
        try:
            analysis = run_analysis(destination, reports_dir=REPORTS_ROOT, timestamp=f"{_safe_stem(destination.stem)}_{started.strftime('%Y%m%d%H%M%S')}")
            outcome_path = analysis.analysis_dir / "outcome_features.json"
            if not outcome_path.exists():
                raise RuntimeError("outcome_features.json was not created.")
            counts["analyzed_success_count"] += 1
        except Exception as error:
            counts["analyzed_failure_count"] += 1
            record["status"] = "analysis_failed"
            record["error"] = str(error)
            ledger["per_video"].append(record)
            print(f"[{index}/{len(videos)}] Analysis failed")
            continue
        print(f"[{index}/{len(videos)}] Registering sample")
        try:
            entry = register_sample(outcome_path, label, DATASET_ROOT, source=resolution.get("source"), generator_or_camera=resolution.get("generator_or_camera"), notes=resolution.get("notes"), allow_duplicate=False)
            counts["registered_count"] += 1
            record["sample_id"] = entry.sample_id
            record["status"] = "completed"
            print(f"[{index}/{len(videos)}] Completed")
        except ValueError as error:
            if "Duplicate video SHA-256" in str(error):
                counts["skipped_existing_count"] += 1
                record["status"] = "skipped_existing"
            else:
                record["status"] = "registration_failed"
                record["error"] = str(error)
            print(f"[{index}/{len(videos)}] Registration skipped")
        ledger["per_video"].append(record)
    ledger.update(counts)
    if not args.dry_run:
        validation = validate_dataset(DATASET_ROOT)
        jsonl = export_dataset_jsonl(DATASET_ROOT)
        csv_path = export_dataset_csv(DATASET_ROOT)
        summary = summarize_dataset(DATASET_ROOT).to_dict()
        ledger["dataset_validation"] = validation
        ledger["dataset_summary"] = summary
        ledger["export_paths"] = {"jsonl": _relative(jsonl), "csv": _relative(csv_path)}
        ledger_path = DATASET_ROOT / "runs" / "latest_run.json"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    else:
        validation = {"status": "dry_run"}
        jsonl = None
        csv_path = None
        ledger["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _print_summary(ledger, validation, jsonl, csv_path)
    return 0 if validation.get("status") in {"valid", "dry_run"} else 1


def _resolve_label(video: Path, metadata_path: Path | None, sidecar: dict[str, dict[str, str]], allow_override: bool) -> dict[str, Any]:
    metadata = match_faceforensics_video(video, metadata_path)
    sidecar_entry = sidecar.get(video.name.lower())
    if metadata:
        label = metadata["normalized_label"]
        if sidecar_entry and sidecar_entry.get("expected_label") and sidecar_entry["expected_label"] != label and not allow_override:
            return {"expected_label": "unresolved", "label_source": "unresolved", "notes": "label_conflict_between_metadata_csv_and_sidecar"}
        return {"expected_label": label, "label_source": "metadata_csv", "source": metadata["source_dataset"], "source_row_id": metadata["source_row_id"], "original_label": metadata["original_label"], "manipulation_method": metadata.get("manipulation_method"), "compression": metadata.get("compression"), "metadata_filename": metadata_path.name if metadata_path else None, "generator_or_camera": metadata.get("manipulation_method"), "notes": sidecar_entry.get("notes") if sidecar_entry else None}
    if sidecar_entry and sidecar_entry.get("expected_label") in {"real", "ai_generated"}:
        return {"expected_label": sidecar_entry["expected_label"], "label_source": "sidecar_csv", "source": sidecar_entry.get("source") or "manual_source", "generator_or_camera": sidecar_entry.get("generator_or_camera"), "manipulation_method": sidecar_entry.get("manipulation_method"), "compression": sidecar_entry.get("compression"), "notes": sidecar_entry.get("notes")}
    lowered = video.name.lower()
    if lowered.startswith("real__"):
        return {"expected_label": "real", "label_source": "filename_convention", "source": "self_recorded", "generator_or_camera": None, "notes": None}
    if lowered.startswith("ai__"):
        return {"expected_label": "ai_generated", "label_source": "filename_convention", "source": "manual_source", "generator_or_camera": None, "notes": None}
    return {"expected_label": "unresolved", "label_source": "unresolved", "notes": "no_trusted_label_source"}


def _metadata_path() -> Path | None:
    for candidate in METADATA_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _read_sidecar(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {row.get("filename", "").lower(): row for row in csv.DictReader(file) if row.get("filename")}


def _pilot_destination(video: Path, label: str, sha: str) -> Path:
    return PILOT_ROOT / label / f"{sha[:16]}__{_safe_stem(video.name)}"


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "video"


def _manifest_has_sha(sha: str) -> bool:
    manifest = DATASET_ROOT / "manifest.jsonl"
    if not manifest.exists():
        return False
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("video_sha256") == sha:
            return True
    return False


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _print_summary(ledger: dict[str, Any], validation: dict[str, Any], jsonl: Path | None, csv_path: Path | None) -> None:
    print("Dataset preparation complete.")
    print(f"Total videos discovered: {ledger['videos_discovered']}")
    print(f"Real videos: {ledger['resolved_real_count']}")
    print(f"AI-generated videos: {ledger['resolved_ai_count']}")
    print(f"Unresolved videos: {ledger['unresolved_count']}")
    print(f"Successful analyses: {ledger['analyzed_success_count']}")
    print(f"Failed analyses: {ledger['analyzed_failure_count']}")
    print(f"Newly registered samples: {ledger['registered_count']}")
    print(f"Skipped existing samples: {ledger['skipped_existing_count']}")
    print(f"Dataset validation result: {validation.get('status')}")
    print(f"Manifest path: {DATASET_ROOT / 'manifest.jsonl'}")
    print(f"JSONL export path: {jsonl}")
    print(f"CSV export path: {csv_path}")


if __name__ == "__main__":
    raise SystemExit(main())
