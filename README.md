# Local Video Analysis MVP

Current version: `v0.9.4`

Local Video Analysis MVP is a local-first, terminal-based video analysis project for collecting explainable forensic evidence from videos. It is being built as a cautious foundation for future labeled-dataset testing, not as a finished deepfake detector.

The project helps inspect media metadata, sampled frames, temporal behavior, audio signal changes, regional visual consistency, optional D3/XCLIP temporal features, and dataset-ready feature artifacts. It does not upload videos, call cloud AI services, assign authenticity verdicts, or calculate fake/real probabilities.

Detailed technical version history is maintained in [VERSIONS.md](VERSIONS.md).

## Purpose

AI-generated and manipulated video detection is difficult because a single metric rarely proves anything. This project focuses on measurable, reviewable evidence instead of premature conclusions. The long-term goal is to support probability testing on manually labeled datasets, while keeping the analysis pipeline transparent enough that a developer or reviewer can inspect where every feature came from.

The current system is useful for:

- Local exploratory video analysis.
- Producing structured evidence reports.
- Producing `outcome_features.json` artifacts for future supervised testing.
- Preparing a labeled dataset manifest from multiple completed analyses.
- Auditing FaceForensics++-style metadata and planning small pilot samples.

## Current Scope

Implemented today:

- Terminal video selection with `python main.py`.
- Non-interactive reusable analysis pipeline used by the dataset runner.
- Metadata extraction with FFmpeg/FFprobe.
- Representative frame sampling and frame metrics.
- Sequential temporal analysis.
- Lightweight audio signal analysis.
- Regional visual-consistency analysis.
- Unified evidence timeline and compact AI-ready evidence artifacts.
- Optional local D3/XCLIP-style temporal analysis when explicitly enabled.
- `outcome_features.json` generation for future labeled outcome work.
- Dataset registration, validation, summary, and deterministic JSONL/CSV exports.
- FaceForensics++ metadata audit and deterministic pilot planning.
- One-command local dataset preparation with `python script.py`.
- Dataset statistics, feature-quality auditing, leakage checks, and future model-feature schema generation.
- Versioned feature registry, explainable derived features, standardized feature exports, lineage, and schema compatibility checks.

Not implemented:

- No trained fake/real classifier.
- No probability calculation.
- No threshold selection.
- No model calibration.
- No authenticity, manipulation, or AI-generation verdict.
- No Gemini API call or SDK integration.
- No web API, frontend, database, cloud storage, or mobile app.

## Requirements

- Python 3.10 or newer.
- FFmpeg with both `ffmpeg` and `ffprobe` available in `PATH`.
- Python packages from `requirements.txt`.
- Optional D3 dependencies from `requirements-d3.txt` only if D3 is enabled.

Set up a local environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional D3 dependencies:

```powershell
pip install -r requirements-d3.txt
```

If FFmpeg is installed inside a local environment, activate that environment before running analysis so the project can find `ffmpeg` and `ffprobe`.

## Project Structure

```text
main.py                 Interactive terminal entry point
script.py               One-command local dataset preparation runner
analysis/               Reusable non-interactive analysis pipeline
metadata_extractor.py   FFprobe execution and metadata parsing
frame_sampler.py        Representative-frame extraction
frame_analyzer.py       Per-frame metrics and comparisons
temporal/               Sequential temporal-analysis modules
audio/                  Lightweight audio-signal analysis modules
visual_consistency/     Regional visual-consistency modules
unified_evidence/       Evidence timeline and AI-ready bundle
learned_detectors/      Optional learned-detector adapters, including D3
outcome/                outcome_features.json builder and validation
dataset_tools/          Dataset manifest, validation, export, and source planning
tools/dataset_tool.py   Dataset toolkit CLI
docs/                   Design and workflow notes
schemas/                Future interpretation response contracts
tests/                  Unit and regression tests
```

Local generated or private folders are intentionally ignored:

```text
source_videos/          Local input videos
reports/                Analysis reports and extracted frame artifacts
dataset_sources/        Copied pilot/source dataset videos and source metadata
dataset/                Manifest, copied feature files, exports, and run ledgers
datasets/               Downloaded datasets or archives
```

Only placeholder files such as `dataset/.gitkeep` and `dataset_sources/.gitkeep` are intended to be tracked.

## Interactive Analysis

Place videos in:

```text
source_videos/
```

Run:

```powershell
python main.py
```

Choose a video from the terminal menu. The source video is not moved, renamed, edited, or deleted.

Each analysis creates a timestamped report folder under `reports/`.

## Generated Reports

A typical report directory contains:

```text
reports/
└── video-name_YYYY-MM-DD_HHMMSS/
    ├── report.json
    ├── report.txt
    ├── ffprobe_raw.json
    ├── temporal_metrics.jsonl
    ├── audio_metrics.jsonl
    ├── visual_consistency_metrics.jsonl
    ├── unified_evidence.json
    ├── evidence_timeline.jsonl
    ├── ai_interpretation_input.json
    ├── ai_interpretation_prompt.txt
    ├── gemini_evidence_report.json
    ├── outcome_features.json
    ├── d3_detector_result.json
    ├── d3_temporal_features.jsonl
    ├── frames/
    ├── transition_frames/
    ├── scene_frames/
    └── consistency_frames/
```

Some D3 files appear only when D3 is enabled and completes. `gemini_evidence_report.json` is a compact local artifact for future Gemini use; the project does not call Gemini.

## Forensic Stages

Metadata:

- File size and SHA-256.
- FFprobe stream/container metadata.
- Raw FFprobe JSON for auditability.

Frames:

- Representative frame extraction.
- Brightness, contrast, blur proxy, black/white ratios, mean RGB, perceptual hashes.
- Consecutive frame comparisons.

Temporal:

- Bounded sequential sampling.
- Scene-boundary candidates.
- Sustained near-static intervals.
- Optical-flow summaries and flow-warp residual review artifacts.

Audio:

- Local FFmpeg PCM extraction.
- RMS/energy windows, silence-like intervals, clipping-like samples, and ranked energy transitions.
- No speech authenticity or synthetic-speech detector.

Visual consistency:

- Deterministic 4 x 4 regional grid.
- Regional brightness flicker, edge stability, texture stability, and fine-detail persistence.
- Review overlays and heatmaps.

Unified evidence:

- Domain-separated evidence timeline.
- Cross-modal context.
- Human review priority only, not a risk score or authenticity score.

D3/XCLIP:

- Optional local learned-detector adapter.
- Disabled by default.
- Produces raw uncalibrated temporal features and status artifacts.
- Not fused into a probability or final verdict.

## Outcome Features

`outcome_features.json` is a stable feature artifact for future labeled experiments. It includes identity, media, metadata indicators, frame summary, temporal summary, audio summary, visual consistency, unified evidence, and D3 summary fields.

It intentionally does not contain:

- Probability.
- Confidence.
- Threshold.
- Classification.
- Verdict.

`expected_label` is nullable in normal analysis output. Labels are added manually later by the dataset toolkit or runner.

## Dataset Preparation

v0.9.1, v0.9.2, and v0.9.3 add tools for collecting and auditing labeled `outcome_features.json` artifacts without training a model.

Manifest path:

```text
dataset/manifest.jsonl
```

Feature copies:

```text
dataset/features/
```

Exports:

```text
dataset/exports/dataset_features.jsonl
dataset/exports/dataset_features.csv
```

Run ledger:

```text
dataset/runs/latest_run.json
```

Allowed labels are:

- `real`
- `ai_generated`

Labels must come from manual knowledge, trusted metadata, a sidecar CSV, or explicit filename convention. The toolkit does not infer labels from D3 scores, feature values, folders, or model output.

## One-Command Dataset Runner

Put pilot videos in:

```text
source_videos/
```

Filename conventions:

```text
real__example.mp4
ai__example.mp4
```

Optional sidecar:

```text
source_videos/video_labels.csv
```

The runner resolves labels in this order:

1. FaceForensics++ metadata CSV, when present.
2. Optional sidecar CSV.
3. Filename convention.
4. Unresolved and skipped from analysis.

Dry run:

```powershell
python script.py --dry-run
```

Full local preparation:

```powershell
python script.py
```

Limited pilot run:

```powershell
python script.py --real-limit 10 --ai-limit 10
```

The runner copies resolved videos into `dataset_sources/pilot/real/` or `dataset_sources/pilot/ai_generated/`, runs the analysis pipeline, registers successful samples, validates the dataset, exports JSONL/CSV, and writes `dataset/runs/latest_run.json`.

Unresolved videos are copied into `dataset_sources/pilot/unresolved/` but are not analyzed or registered.

## Dataset Toolkit CLI

Register one completed analysis artifact:

```powershell
python tools\dataset_tool.py register reports\example\outcome_features.json real --source self_recorded --generator-or-camera "phone camera"
```

Validate the dataset:

```powershell
python tools\dataset_tool.py validate
```

Print summary:

```powershell
python tools\dataset_tool.py summary
```

Export flattened features:

```powershell
python tools\dataset_tool.py export-jsonl
python tools\dataset_tool.py export-csv
```

The export uses deterministic row ordering and flattened column names such as:

```text
media.duration_seconds
temporal_summary.maximum_motion_magnitude
d3_summary.d3_raw_score
```

Unavailable values are `null` in JSONL and empty fields in CSV.

## Dataset Statistics and Feature Audit

v0.9.3 adds a local feature-audit workflow that runs before any classifier work. Its job is to inspect the exported dataset, identify usable numeric feature candidates, exclude identifiers and leakage-prone fields, report missing/constant/invalid values, compare classes descriptively, and generate a future model-feature schema.

Run the full audit:

```powershell
python tools\dataset_tool.py audit-features
```

Use an explicit input export:

```powershell
python tools\dataset_tool.py audit-features --input dataset\exports\dataset_features.csv
```

Reusable subcommands:

```powershell
python tools\dataset_tool.py statistics
python tools\dataset_tool.py feature-audit
python tools\dataset_tool.py model-schema
```

Generated outputs:

```text
dataset/statistics/
├── dataset_profile.json
├── column_profile.csv
├── feature_statistics.csv
├── class_comparison.csv
├── missing_values.csv
├── invalid_values.csv
├── constant_features.csv
├── correlation_matrix.csv
├── high_correlation_pairs.csv
├── leakage_report.json
├── feature_quality.json
├── model_feature_schema.json
└── statistics_report.txt
```

The audit explicitly excludes target labels, filenames, paths, sample IDs, hashes, source notes, and other class-revealing metadata from future model inputs. Filename conventions such as `real__...` and `ai__...` are treated as label-resolution aids only, never as model features.

`model_feature_schema.json` is a preparation artifact. It contains included and excluded feature lists plus exclusion reasons, but it does not contain trained parameters, predictions, probabilities, thresholds, or performance scores.

The current 12-video dataset is a pilot dataset. It can test the workflow, but it is not large enough to support reliable generalization claims.

See [docs/FEATURE_AUDIT.md](docs/FEATURE_AUDIT.md) for command defaults, output descriptions, and readiness rules.

## Feature Engineering and Standardization

v0.9.4 adds a registry-controlled feature layer between raw exported features and future model experiments.

The flow is:

```text
dataset/exports/dataset_features.csv
    -> schemas/feature_registry.json
    -> schemas/feature_schema.json
    -> dataset/standardized/standardized_features.csv
```

Run the full preparation workflow:

```powershell
python tools\dataset_tool.py prepare-features
```

Individual commands:

```powershell
python tools\dataset_tool.py feature-registry
python tools\dataset_tool.py validate-features
python tools\dataset_tool.py engineer-features
python tools\dataset_tool.py standardize-features
python tools\dataset_tool.py check-feature-compatibility
python tools\dataset_tool.py generate-feature-docs
```

Generated schema/docs:

```text
schemas/feature_registry.json
schemas/feature_schema.json
schemas/feature_lineage.json
schemas/feature_evolution.json
docs/FEATURES.md
```

Generated standardized dataset outputs:

```text
dataset/standardized/
├── standardized_features.csv
├── standardized_features.jsonl
├── standardized_dataset_profile.json
├── feature_validation.json
├── feature_validation.csv
├── feature_engineering_report.json
├── feature_engineering_report.txt
├── compatibility_report.json
└── compatibility_report.txt
```

Standardization here means canonical names, order, types, null handling, ranges, lineage, derivation rules, and schema fingerprints. It does not mean fitted z-score normalization, min-max scaling, fitted imputation, PCA, target-aware feature selection, train/test splitting, or model training.

See [docs/FEATURES.md](docs/FEATURES.md) for generated feature definitions and [schemas/feature_lineage.json](schemas/feature_lineage.json) for derived-feature lineage.

## FaceForensics++ Metadata Support

Expected local metadata path:

```text
dataset_sources/faceforensics_pp/metadata/FF++_Metadata.csv
```

Expected video root:

```text
dataset_sources/faceforensics_pp/videos/
```

Audit metadata:

```powershell
python tools\dataset_tool.py audit-source --dataset faceforensics_pp --metadata dataset_sources\faceforensics_pp\metadata\FF++_Metadata.csv
```

Create a deterministic balanced pilot plan:

```powershell
python tools\dataset_tool.py plan-source --dataset faceforensics_pp --metadata dataset_sources\faceforensics_pp\metadata\FF++_Metadata.csv --real-count 10 --ai-count 10 --seed 42
```

Validate a plan:

```powershell
python tools\dataset_tool.py validate-plan dataset_sources\faceforensics_pp\plans\pilot_plan.jsonl
```

The adapter inspects the actual CSV delimiter and columns instead of assuming a fixed schema. It normalizes trusted source labels to `real` and `ai_generated`, checks safe relative paths, detects missing or ambiguous videos, and reports label availability.

Downloaded FaceForensics++ files, Kaggle files, metadata copies, source videos, pilot copies, and generated exports are local data and should not be committed unless a future release explicitly adds small public fixtures.

## Validation Rules

Dataset validation checks:

- Manifest entries have corresponding feature files.
- Feature files pass outcome-feature validation.
- Manifest and feature SHA-256 values match.
- Labels are valid.
- Sample IDs are unique.
- Video hashes are unique unless explicitly allowed.
- Feature paths are relative and stay inside `dataset/`.
- Numeric feature values are finite.
- Probability, confidence, threshold, verdict, and classification fields are absent.

## Testing

Run:

```powershell
python -m unittest discover -s tests
```

Useful pre-push checks:

```powershell
python script.py --dry-run
python tools\dataset_tool.py validate
python tools\dataset_tool.py summary
python tools\dataset_tool.py audit-features
python tools\dataset_tool.py prepare-features
git -c safe.directory=E:/Anything/video_analysis status --short --untracked-files=all
```

The `safe.directory` option is only needed when Git reports a Windows user/sandbox ownership mismatch.

## Configuration

Core version and analysis settings live in `config.py`.

Optional local D3 settings can be controlled through environment variables. `.env` is ignored because it may contain local paths or secrets. `.env.example` is safe to commit as documentation.

## Roadmap

Near-term:

- Gather more manually labeled `outcome_features.json` samples.
- Audit and plan small balanced pilot datasets.
- Use v0.9.3 feature-audit reports to remove leakage risks before any classifier experiment.
- Use v0.9.4 standardized features as the stable input contract for future split-safe preprocessing.
- Improve dataset documentation as real dataset sources are added.
- Keep validation strict around labels and forbidden probability/verdict fields.

Later:

- Probability experiments after enough labeled data exists.
- Calibration and threshold research after dataset validation is mature.
- Backend/API reuse only after the analysis service boundary is stable.
- Web or mobile interfaces only after the local pipeline and data contracts are reliable.

## Responsible Use

This project produces evidence for review, not proof. Reported observations can be caused by ordinary camera motion, compression, lighting changes, editing, stabilization, scene cuts, animation, or generated-video instability.

Do not use this project as the sole basis for accusing a person, rejecting media, or making a high-stakes decision. Dataset labels must be assigned from trusted ground truth, and external datasets such as FaceForensics++ or Kaggle datasets must be used according to their own licenses and terms.

## Privacy

Analysis is local, but generated reports may contain extracted frames and metadata. Treat `source_videos/`, `reports/`, `dataset_sources/`, and `dataset/` as potentially sensitive local data.
