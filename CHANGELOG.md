# Changelog

## v0.9.2

- Added FaceForensics++ metadata auditing with CSV encoding/delimiter inspection, schema inference, duplicate/missing-value counts, label counts, normalized records, and video availability checks.
- Added balanced FaceForensics++ pilot sample planning and plan validation.
- Added a reusable non-interactive analysis pipeline under `analysis/`.
- Added root-level `script.py` one-command dataset preparation runner with label resolution from metadata CSV, optional sidecar CSV, filename convention, or unresolved status.
- Added organized pilot source copying into `dataset_sources/pilot/`, automatic analysis/registration for resolved videos, duplicate handling, deterministic JSONL/CSV export refresh, and `dataset/runs/latest_run.json`.
- Added `dataset_sources/` source-data separation and ignore rules.
- Added `docs/FACEFORENSICS_SOURCE_PLANNING.md` and `docs/ONE_COMMAND_DATASET_RUNNER.md`.
- Verified `python -m unittest discover -s tests` with 95 passing tests.
- Kept v0.9.2 non-modeling: no classifier training, probability calculation, thresholds, D3 calibration, Gemini call, API, frontend, or mobile integration.

## v0.9.1

- Added a local dataset preparation toolkit under `dataset_tools/`.
- Added `tools/dataset_tool.py` for sample registration, validation, deterministic JSONL/CSV exports, and dataset summaries.
- Added `dataset/manifest.jsonl` manifest support with manually supplied labels only.
- Added duplicate video SHA-256 detection and safe relative feature-file handling.
- Added `docs/DATASET_TOOLKIT.md`.
- Kept v0.9.1 non-modeling: no classifier training, probability calculation, thresholds, D3 calibration, Gemini call, API, frontend, or mobile integration.

## v0.9.0

- Added a modular outcome feature layer under `outcome/`.
- Added `outcome_features.json` as a stable JSON feature record for future labeled-dataset and probability-calibration experiments.
- Added typed feature groups for identity, media, metadata indicators, frame summary, temporal summary, audio summary, visual consistency, unified evidence, and D3.
- Added outcome feature validation for required identity, finite numbers, valid ratios, non-negative counts, relative paths, manual labels, D3 uncalibrated semantics, and forbidden probability/confidence/verdict fields.
- Added outcome artifact references to main JSON and TXT reports.
- Added `docs/OUTCOME_FEATURES.md`.
- Kept v0.9.0 non-outcome-based: no probability calculation, classifier training, thresholds, D3 calibration, Gemini call, API, frontend, or mobile integration.

## v0.8.2

- Added `gemini_evidence_report.json` as a compact derivative report for future Gemini-assisted interpretation.
- Added deterministic key-event selection, finding deduplication, artifact caps, validation, and size-limit enforcement for compact reports.
- Added compact artifact references to main JSON and TXT reports.
- Added `schemas/gemini_interpretation_response.schema.json` for a future Gemini response contract.
- Added `docs/GEMINI_READY_REPORT.md`.
- Added D3 progress messages for model loading, tensor preparation, inference, temporal-feature handling, and artifact writing.
- Kept v0.8.2 local-only: no Gemini API calls, SDK installation, network requests, media upload, probability generation, thresholds, evidence fusion, or authenticity verdicts.

## v0.8.1

- Hardened the optional D3 learned-detector integration.
- Updated application version to `0.8.1`.
- Added centralized learned-detector status and reason-code definitions.
- Added a centralized D3 encoder registry with concrete model identifiers.
- Added real process-based timeout handling for the heavy D3 execution path.
- Added explicit temporary-frame preservation behavior.
- Added D3 method-verification, score-direction, and reproducibility metadata.
- Kept D3 output standalone from unified evidence and AI-ready inputs.
- Tightened validator checks for null probability, null threshold, `not_assigned` classification, and no confidence/verdict fields.
- Fixed completed D3 artifact execution reporting in `d3_detector_result.json`.
- Documented pinned upstream parity status in `docs/D3_UPSTREAM_PARITY.md`.
- Kept actual pretrained inference marked unverified when optional ML dependencies/assets are absent.

## v0.8.0

- Added optional learned-detector infrastructure and D3 adapter.
- Added D3 availability checks and standalone D3 report output.
- Added D3 raw temporal-feature artifacts for completed runs.
