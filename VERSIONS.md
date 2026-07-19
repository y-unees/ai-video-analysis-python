# Version History

This file keeps the project history that used to live inside the README. The README now focuses on purpose, setup, and current workflows; this file explains how the system grew over time and what each version means technically.

Current application version: `0.9.2`

Current report schema version: `0.7`

Outcome feature schema version: `0.9.0`

Dataset manifest schema version: `0.9.1`

## Version Line

The project is still pre-1.0. Versions describe local workflow maturity, artifact contracts, and analysis-module boundaries. They do not imply a production detector, trained classifier, or validated probability model.

The core principle across all versions remains the same: produce local, explainable evidence artifacts without assigning authenticity verdicts.

## v0.9.2

v0.9.2 adds source-dataset preparation around the existing outcome-feature workflow.

Key additions:

- FaceForensics++ metadata CSV auditing with encoding/delimiter inspection.
- Conservative schema inference instead of assuming a fixed CSV layout.
- Normalized source records with trusted labels mapped to `real` or `ai_generated`.
- Video availability checks for found, missing, ambiguous, and invalid paths.
- Deterministic balanced pilot planning with a seed.
- Plan validation for safe paths, duplicate IDs, labels, and requested counts.
- Reusable non-interactive analysis pipeline under `analysis/`.
- Root-level `python script.py` runner for one-command local dataset preparation.
- Label resolution from metadata CSV, optional sidecar CSV, filename convention, or unresolved status.
- Organized pilot source copying under `dataset_sources/pilot/`.
- Automatic analysis and dataset registration for resolved videos.
- Dataset validation, JSONL/CSV export refresh, and `dataset/runs/latest_run.json`.
- Documentation for FaceForensics++ planning and the one-command runner.

Still intentionally absent:

- No classifier training.
- No probability calculation.
- No thresholds.
- No D3 calibration.
- No Gemini API call.
- No API, frontend, or mobile integration.

Validation status at cleanup: `python -m unittest discover -s tests` passed with 95 tests.

## v0.9.1

v0.9.1 adds a local dataset preparation toolkit for collecting manually labeled `outcome_features.json` artifacts.

Key additions:

- `dataset_tools/` package.
- `tools/dataset_tool.py` CLI.
- JSONL manifest at `dataset/manifest.jsonl`.
- Feature copies under `dataset/features/`.
- Manual labels only: `real` and `ai_generated`.
- Sample registration with duplicate video SHA-256 detection.
- Validation for manifest records, feature files, labels, hashes, safe relative paths, finite numeric values, and forbidden probability/verdict fields.
- Deterministic JSONL and CSV exports under `dataset/exports/`.
- Dataset summary counts, including D3 status counts and missing-value counts.
- Documentation in `docs/DATASET_TOOLKIT.md`.

Important constraint:

Labels are supplied manually or from trusted source metadata. The toolkit does not infer labels from filenames, folders, D3 scores, metadata values, or analysis outputs unless a later explicit workflow documents that source of truth.

Still intentionally absent:

- No classifier.
- No probability.
- No thresholding.
- No Gemini integration.
- No API or frontend.

## v0.9.0

v0.9.0 adds `outcome_features.json`, the first stable feature record intended for future labeled-dataset and probability-calibration experiments.

Key additions:

- `outcome/` package.
- Typed feature groups for:
  - identity
  - media
  - metadata indicators
  - frame summary
  - temporal summary
  - audio summary
  - visual consistency
  - unified evidence
  - D3 summary
- `expected_label` field, nullable during normal analysis.
- Outcome artifact references in the main JSON and TXT reports.
- Validation for required identity fields, finite numbers, valid ratios, non-negative counts, relative paths, manual label values, and D3 uncalibrated semantics.
- Guards against probability, confidence, threshold, classifier output, and verdict fields.
- Documentation in `docs/OUTCOME_FEATURES.md`.

Important constraint:

This version creates features only. It does not create conclusions.

## v0.8.2

v0.8.2 adds a compact Gemini-ready evidence report while remaining local-only.

Key additions:

- `gemini_evidence_report.json` as a compact derivative artifact for future Gemini-assisted interpretation.
- Deterministic key-event selection.
- Finding deduplication.
- Artifact caps and size-limit enforcement.
- Machine-readable instruction block for a future Gemini call.
- Future Gemini response schema.
- Compact artifact references in the main JSON and TXT reports.
- D3 progress messages for long local inference stages.

Important constraint:

No Gemini SDK, API key handling, upload, network request, probability, threshold, evidence fusion, or authenticity verdict is added. The compact report is just a local artifact.

## v0.8.1

v0.8.1 hardens the optional D3 learned-detector integration.

Key additions and fixes:

- D3 upstream parity documentation against pinned commit `c798fbc57fe0c4198d63a73732c2c0f9e4b4816c`.
- Clear statement that the local D3 layer is a documented single-video adaptation, not proven full upstream runtime parity.
- Process-based timeout enforcement for the heavy D3 path.
- Temporary-frame preservation controls for diagnostics.
- Centralized D3 encoder mappings.
- Centralized learned-detector status and reason-code semantics.
- D3 method-verification, score-direction, and reproducibility metadata.
- Validation for null probability, null threshold, `not_assigned` classification, and no confidence/verdict fields.
- Fixed completed D3 artifact execution reporting so `d3_detector_result.json` records `status: completed`, `completed_at_utc`, and `duration_seconds`.

Important constraint:

D3 remains standalone from unified evidence and AI-ready inputs.

## v0.8.0

v0.8.0 introduces the optional learned-detector layer and the first D3 adapter.

Key additions:

- `learned_detectors/` package.
- Modular learned-detector adapter interface.
- Optional D3 integration pinned to upstream commit `c798fbc57fe0c4198d63a73732c2c0f9e4b4816c`.
- Disabled-by-default configuration through `LEARNED_DETECTORS_ENABLED=false` and `D3_ENABLED=false`.
- Single-video D3 preprocessing adaptation.
- First-order and second-order temporal feature tracing.
- D3 report rendering and validation.
- `d3_detector_result.json` and `d3_temporal_features.jsonl` for completed D3 runs.
- `requirements-d3.txt`, `.env.example`, `docs/D3_INTEGRATION.md`, and `docs/LEARNED_DETECTORS.md`.

Important constraint:

D3 raw scores are uncalibrated. They are not probabilities, thresholds, classifications, authenticity verdicts, or manipulation verdicts.

## v0.7.1

v0.7.1 fixes unified timeline over-merging.

Key additions and fixes:

- Replaced chained interval merging with anchor-based, non-transitive timeline segmentation.
- Added evidence candidate roles:
  - `anchor_event`
  - `supporting_interval`
  - `contextual_interval`
- Prevented long supporting/contextual intervals from bridging unrelated anchor events.
- Kept contextual intervals separate from localized review-event boundaries.
- Added regional context grouping with minimum overlap requirements.
- Preserved canonical observation IDs where matching observations exist.
- Balanced AI-ready event findings by evidence domain.
- Kept unavailable provenance only under missing evidence.

Important constraint:

Still local-only. No Gemini call, external model, AI probability, or authenticity verdict.

## v0.7.0

v0.7.0 adds unified evidence.

Key additions:

- `unified_evidence/` package.
- Domain-separated evidence from metadata, frame sampling, temporal, audio, and visual-consistency stages.
- Chronological evidence timeline.
- Deterministic interval merging.
- Evidence-domain grouping so related visual methods are not treated as independent confirmations.
- Cross-modal context such as visual-only, audio-only, multiple visual methods, and visual/audio overlap.
- Review priority levels: `low`, `moderate`, and `high`.
- Grouped regional visual-consistency intervals for timeline-level review.
- Compact AI-ready interpretation input with token-aware limits.
- Model-neutral prompt template for a future interpreter.
- Future response schema requiring nullable probability and restricted assessment labels.

Important constraint:

Review priority is a human review ordering aid only. It is not an authenticity score, AI-generation probability, manipulation score, or risk score.

## v0.6.0

v0.6.0 adds explainable visual-consistency analysis.

Key additions:

- `visual_consistency/` package.
- Deterministic 4 x 4 region grid.
- Regional brightness, edge stability, texture stability, fine-detail persistence, and motion-context measurements.
- Motion-compensated edge and detail comparisons using the existing temporal frame stream.
- `visual_consistency_metrics.jsonl`.
- Transition-level visual consistency summaries.
- Sustained regional visual-variation intervals.
- Ranked visual-consistency review transitions.
- Bounded `consistency_frames/` artifacts:
  - before grid
  - after grid
  - detail residual heatmap
  - combined heatmap
- Report and validation integration.

Important constraint:

Regional variation can be caused by normal camera motion, compression, lighting, focus, occlusion, depth changes, or ordinary editing. It is not proof of manipulation.

## v0.5.1

v0.5.1 fixes audio-analysis startup reliability.

Key fixes:

- Fixed inconsistent string and `pathlib.Path` handling.
- Public audio functions accept both string paths and `Path` objects.
- Audio extraction records stable reason codes, cleanup status, and diagnostics without exposing temporary absolute paths.
- Windowed audio records include actual duration and sample count.
- Temporary PCM WAV files are cleaned after analysis unless debug retention is added in the future.

## v0.5.0

v0.5.0 adds lightweight audio analysis.

Key additions:

- `audio/` package.
- FFmpeg extraction of the selected audio stream into temporary PCM WAV.
- Python `wave` plus NumPy-based signal analysis.
- Decoded PCM details.
- Windowed RMS/energy metrics.
- Silence-like intervals.
- Clipping-like sample counts.
- Ranked energy transitions.
- `audio_metrics.jsonl`.
- Report integration.

Related fixes:

- v0.4.1 validator behavior was adjusted so negative disclaimers such as "not proof" are allowed while affirmative unsupported verdicts remain rejected.
- Temporal coverage uses a normalized selected-video-stream timeline.
- Ranked visual transition percentiles use the full configured metric set when available.

Important constraint:

Audio metrics cannot determine whether speech is natural, cloned, synthesized, edited, or authentic.

## v0.4.1

v0.4.1 improves temporal review output and report semantics.

Key additions and fixes:

- Ranked notable transitions are separate from scene-boundary candidates.
- Transition rankings are relative to the current video.
- Reports explain which metrics selected a transition.
- Flow-warp residual metrics compare a flow-warped previous frame with the actual next frame.
- Temporal coverage reports how much of the selected video-stream timeline was sampled.
- Canonical observations carry stable observation IDs.
- Legacy `evidence` is a compatibility view generated from canonical observations.
- Human-readable file sizes use binary units: `KiB`, `MiB`, and `GiB`.

## v0.4.0

v0.4.0 adds the first sequential temporal-analysis layer.

Key additions:

- Separate `temporal_analysis` report section.
- Bounded sequential temporal sampling with requested and effective FPS.
- Scene-boundary candidates.
- Constructed scenes.
- Sustained near-static intervals.
- Optical-flow metrics.
- `temporal_metrics.jsonl`.
- Scene-aware representative frames.

## Versioning Notes

Application version and schema version are separate:

- `APP_VERSION` identifies the application release.
- `SCHEMA_VERSION` identifies the main report schema.
- `OUTCOME_FEATURE_SCHEMA_VERSION` identifies the `outcome_features.json` contract.
- `DATASET_MANIFEST_SCHEMA_VERSION` identifies the dataset manifest contract.

This means `APP_VERSION` can be `0.9.2` while the main report schema remains `0.7`, because v0.8 and v0.9 added optional artifacts and dataset tooling without replacing the core report schema.

## Stable Non-Goals

Across the current version line, the project does not provide:

- AI-generation probability.
- Fake/real verdict.
- Authenticity verdict.
- Manipulation verdict.
- Threshold-based decision.
- Trained classifier.
- Face identity analysis.
- Synthetic-speech detector.
- C2PA provenance verification.
- Gemini API integration.
- Web/API/frontend/mobile application.

Any future version that adds one of these must document the new artifact contract, validation rules, calibration assumptions, and responsible-use limitations.
