# 🎬 Local Video Analysis MVP

Terminal-only, local-first video-analysis MVP for metadata inspection, representative-frame extraction, and cautious heuristic observations. It does not call external AI services or upload media.

## Requirements

- Python 3.10 or newer
- FFmpeg with both `ffmpeg` and `ffprobe` available in your system PATH
- Python packages from `requirements.txt` for OpenCV/Pillow/ImageHash/NumPy-based local analysis
- Optional D3 learned-detector packages from `requirements-d3.txt` only when D3 inference is enabled

If FFmpeg is installed inside a virtual environment or local tool directory, activate that environment before running `python main.py` so the app can find `ffmpeg` and `ffprobe`.

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place videos in the root-level `source_videos/` folder.
2. Run:

```bash
python main.py
```

3. Choose a video from the numbered terminal menu.

## Project Structure

```text
main.py                 Terminal entry point
video_selector.py       Video discovery and menu selection
metadata_extractor.py   FFprobe execution and normalized metadata parsing
frame_sampler.py        OpenCV representative-frame seeking/extraction
frame_analyzer.py       Per-frame metrics and pairwise comparisons
evidence_builder.py     Cautious temporal heuristic observations
report_writer.py        JSON/TXT report rendering
report_validator.py     Internal report consistency checks
config.py               Version and heuristic configuration constants
temporal/               v0.4 sequential temporal-analysis modules
audio/                  v0.5 lightweight audio-signal analysis modules
visual_consistency/     v0.6 regional visual-consistency analysis modules
unified_evidence/       v0.7 unified evidence timeline and AI-ready bundle
learned_detectors/      v0.8 optional learned-detector adapters
docs/                   Integration notes and optional detector documentation
schemas/                Future AI interpretation response contract
tests/                  Lightweight synthetic/unit tests
```

## Generated Reports

Each analysis creates a timestamped directory:

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
    ├── gemini_evidence_report.json    # compact derivative for future Gemini use
    ├── d3_detector_result.json        # only when D3 completes
    ├── d3_temporal_features.jsonl     # only when D3 completes
    ├── frames/
    ├── transition_frames/
    ├── scene_frames/
    └── consistency_frames/
```

The source video is not moved, renamed, edited, or deleted.
Paths in reports are relative to the report directory or project directory unless absolute paths are explicitly enabled in code.

## What This MVP Does

- Calculates file size and SHA-256.
- Extracts normalized FFprobe metadata.
- Saves the complete raw FFprobe JSON.
- Samples a small number of representative frames with OpenCV.
- Saves sampled frames as JPEG artifacts.
- Calculates brightness, contrast, Laplacian variance, near-black/near-white pixel ratios, mean RGB, and perceptual hashes.
- Compares consecutive sampled frames using perceptual-hash distance, normalized mean absolute difference, and grayscale histogram correlation.
- Embeds the heuristic configuration used for each report.
- Validates the finalized report dictionary before JSON and TXT files are written.
- Runs a bounded low-resolution sequential temporal pass for scene-boundary candidates, sustained near-static intervals, and optical-flow summaries.
- Saves full temporal transition metrics to `temporal_metrics.jsonl` and hashes the artifact.
- Extracts scene-aware representative frames into `scene_frames/`.
- Ranks a bounded set of notable transitions for human review without changing scene-boundary rules.
- Calculates flow-warp residuals to show how well estimated motion explains the next sampled frame.
- Saves before/after/difference/residual artifacts for ranked notable transitions in `transition_frames/`.
- Extracts the selected audio stream with FFmpeg into temporary PCM WAV for local analysis.
- Calculates lightweight audio-signal metrics, silence-like intervals, clipping-like samples, and ranked energy transitions.
- Writes full windowed audio metrics to `audio_metrics.jsonl` and hashes the artifact.
- Runs explainable visual consistency analysis on the reused temporal frame stream.
- Divides each analyzed frame into a deterministic 4 x 4 region grid with stable region IDs.
- Measures local brightness flicker, motion-aware edge stability, texture stability, and fine-detail persistence.
- Records region-level transition metrics to `visual_consistency_metrics.jsonl` and hashes the artifact.
- Builds transition-level regional summaries, sustained regional-variation intervals, and ranked review transitions.
- Saves bounded review overlays and heatmaps in `consistency_frames/`.
- Builds a unified evidence timeline from metadata, frame sampling, temporal, audio, and visual-consistency observations.
- Merges overlapping or nearby time-based evidence with a documented tolerance.
- Records cross-modal context and review priority without producing authenticity, manipulation, or AI-generation scores.
- Writes `unified_evidence.json`, `evidence_timeline.jsonl`, `ai_interpretation_input.json`, and `ai_interpretation_prompt.txt`.
- Defines a future AI response contract in `schemas/ai_interpretation_response.schema.json`.
- Adds an optional modular learned-detector layer under `learned_detector_results`.
- Integrates D3 as an optional local adapter when explicitly enabled and optional dependencies are installed.
- Records D3 availability, configuration, preprocessing trace, raw second-order temporal-feature statistic, and artifacts without fusing it into unified evidence.
- Writes `gemini_evidence_report.json`, a compact derivative artifact for future Gemini-assisted interpretation without calling Gemini.

## Schema Notes

Current schema: `0.7`.

Application version: `0.8.2`.

v0.8 stable status: the optional D3 learned-detector line is stable for standalone local raw-feature reporting, and v0.8.2 adds a compact Gemini-ready derivative report while keeping learned outputs separate from unified evidence and AI-ready inputs.

Important additions in `0.8.2`:

- Adds `gemini_evidence_report.json` as the recommended compact single-input artifact for future Gemini analysis.
- Keeps the full forensic report and raw artifacts as the local sources of truth.
- Adds deterministic key-event selection, finding deduplication, artifact caps, and size-limit enforcement.
- Adds a machine-readable Gemini instruction block and future Gemini response schema.
- Adds compact artifact references to the main JSON and TXT reports.
- Adds D3 progress messages for long CPU inference stages.
- Does not call Gemini, install a Gemini SDK, send media, add network requests, create probabilities, or assign authenticity verdicts.

Important fixes in `0.8.1`:

- Adds D3 upstream parity documentation against pinned commit `c798fbc57fe0c4198d63a73732c2c0f9e4b4816c`.
- Clarifies that the local D3 layer is a documented single-video adaptation, not proven full upstream runtime parity.
- Adds process-based timeout enforcement for the heavy D3 path.
- Adds temporary-frame preservation controls for D3 diagnostics.
- Centralizes D3 encoder mappings and learned-detector status/reason-code semantics.
- Tightens validation for null probability, null threshold, `not_assigned` classification, and no confidence/verdict fields.
- Keeps D3 score direction neutral because upstream label/AP handling does not support a clean synthetic/real direction claim.
- Fixes `d3_detector_result.json` execution reporting so completed D3 artifact files record `status: completed`, `completed_at_utc`, and `duration_seconds`.

Important additions in `0.8.0`:

- Adds the `learned_detectors/` package with a modular adapter interface.
- Adds optional D3 integration pinned to upstream commit `c798fbc57fe0c4198d63a73732c2c0f9e4b4816c`.
- Keeps learned detectors disabled by default through `LEARNED_DETECTORS_ENABLED=false` and `D3_ENABLED=false`.
- Adds documented single-video D3 preprocessing adaptation, first/second-order feature tracing, report rendering, and validation.
- Writes `d3_detector_result.json` and `d3_temporal_features.jsonl` only for completed D3 runs.
- Keeps D3 standalone from unified evidence in v0.8.
- Reports D3 raw score as uncalibrated: no probability, threshold, classification, authenticity verdict, or manipulation verdict.
- Adds `requirements-d3.txt`, `.env.example`, `docs/D3_INTEGRATION.md`, and `docs/LEARNED_DETECTORS.md`.

Important fixes in `0.7.1`:

- Replaces chained interval merging with anchor-based, non-transitive timeline segmentation.
- Adds evidence candidate roles: `anchor_event`, `supporting_interval`, and `contextual_interval`.
- Prevents long supporting or contextual intervals from bridging unrelated anchor events.
- Keeps contextual intervals separate from localized review-event boundaries.
- Adds regional context grouping with minimum overlap requirements.
- Preserves canonical observation IDs where matching observations exist.
- Balances AI-ready event findings by evidence domain so repeated regional findings do not dominate.
- Keeps unavailable provenance only under missing evidence.
- Keeps v0.7.1 local-only: no Gemini call, no external model, no AI probability, and no authenticity verdict.

Important additions in `0.7`:

- Adds `unified_evidence` with domain-separated evidence from metadata, frame sampling, temporal, audio, and visual-consistency stages.
- Adds a chronological evidence timeline with deterministic interval merging.
- Adds evidence-domain grouping so related visual methods are not counted as fully independent confirmations.
- Adds cross-modal context such as visual-only, audio-only, multiple visual methods, and visual/audio overlap.
- Adds review priority levels (`low`, `moderate`, `high`) as a human review ordering aid only.
- Adds grouped regional visual-consistency intervals for timeline-level review while preserving original region records.
- Adds compact AI-ready interpretation input with token-aware limits and strict interpretation constraints.
- Adds a model-neutral prompt template for a future interpreter.
- Adds a future response schema requiring nullable numeric probability and restricted assessment labels.
- Does not call Gemini or any other external model, does not require an API key, and does not produce an AI probability or authenticity verdict.

Important additions in `0.6`:

- Adds `visual_consistency_analysis` with regional brightness, edge, texture, detail-persistence, and motion-context measurements.
- Adds deterministic 4 x 4 region-grid records with normalized and pixel bounds.
- Adds motion-compensated edge and detail comparisons using the existing temporal frame stream and optical-flow approach.
- Adds `visual_consistency_metrics.jsonl` with one region record per temporal transition and hashes the artifact.
- Adds transition-level visual consistency summaries and sustained regional visual-variation intervals.
- Adds ranked visual consistency review transitions using relative-within-video percentile ranking.
- Adds `consistency_frames/` review artifacts: before grid, after grid, detail residual heatmap, and combined heatmap.
- Extends JSON/TXT reports and validation for visual-consistency structure, artifacts, rankings, intervals, and non-conclusive observations.
- Maintains local-only operation and does not add an AI classifier, authenticity verdict, manipulation verdict, fake/real verdict, face analysis, cloud API, database, or frontend.

Important fixes in `0.5.1`:

- Fixed an audio-analysis startup failure caused by inconsistent string and `pathlib.Path` handling.
- Public audio functions accept both string paths and `Path` objects.
- Audio extraction records stable reason codes, cleanup status, and diagnostics without exposing temporary absolute paths.
- Windowed audio records include actual duration and sample count.
- Temporary PCM WAV files are cleaned after analysis unless debug retention is added in the future.

Important additions in `0.5`:

- Adds `audio_analysis` with extraction, decoded PCM details, timeline comparison, global metrics, silence-like intervals, ranked energy transitions, observations, limitations, and artifacts.
- Adds `audio_metrics.jsonl` with one windowed audio record per line.
- Keeps temporary WAV extraction artifacts out of reports and deletes them after analysis.
- Uses Python `wave` plus NumPy for PCM analysis.
- Fixes v0.4.1 validator behavior so negative disclaimers such as "not proof" are allowed, while affirmative unsupported verdicts are rejected.
- Corrects temporal coverage to use a normalized selected-video-stream timeline.
- Makes ranked visual transition percentiles use the complete configured metric set when available.

Important additions in `0.4.1`:

- Ranked notable transitions are separate from scene-boundary candidates.
- Transition rankings are relative to the current video and explain which metrics selected a transition.
- Flow-warp residual metrics compare a flow-warped previous frame with the actual next frame.
- Temporal coverage reports how much of the selected video-stream timeline was sampled.
- Canonical observations carry stable observation IDs.
- Legacy `evidence` is a compatibility view generated from canonical observations.
- Human-readable file sizes use binary units: `KiB`, `MiB`, `GiB`.

Important additions in `0.4`:

- Adds a separate `temporal_analysis` report section.
- Adds bounded sequential temporal sampling with requested and effective FPS.
- Adds scene-boundary candidates, constructed scenes, sustained near-static intervals, and optical-flow metrics.
- Adds `temporal_metrics.jsonl` for complete transition records.
- Adds scene-aware representative frames.

## Changelog

### v0.8.2

- Added compact Gemini-ready evidence report generation without Gemini API calls.
- Added compact artifact references in the main JSON/TXT reports.
- Added future Gemini response contract schema and documentation.
- Added D3 progress feedback for long local inference stages.
- Preserved local source-of-truth forensic artifacts and non-verdict semantics.

### v0.8.1

- Hardened D3 integration with timeout, temporary-frame, status, reason-code, and validator updates.
- Fixed completed D3 artifact status reporting in `d3_detector_result.json`.
- Added `docs/D3_UPSTREAM_PARITY.md`, `CHANGELOG.md`, and `THIRD_PARTY_NOTICES.md`.
- Added deterministic D3 math, timeout, status, and safe-output regression tests.
- Preserved local-only default behavior and no D3 fusion with unified evidence or AI-ready input.

### v0.8.0

- Added optional learned-detector infrastructure and D3 adapter.
- Added D3 availability checks for disabled, unavailable, skipped, failed, timed-out, and completed states.
- Added D3 raw output and temporal feature artifacts with report validation.
- Added focused tests for D3 availability, preprocessing, and second-order feature math.
- Kept the default pipeline local heuristic-only and kept learned outputs unfused from unified evidence.

### v0.7.1

- Fixed unified timeline over-merging where long regional/context intervals could collapse many findings into one full-video event.
- Added anchor-based event boundaries, supporting/context candidate references, and contextual interval separation.
- Added canonical observation traceability in unified events and AI-ready input.
- Added domain-balanced finding selection and compact high-value metrics for AI-ready event summaries.
- Removed unavailable provenance from normal/non-supporting findings and retained it as missing evidence.
- Added focused regression tests for bridge prevention, max-span segmentation, contextual intervals, regional grouping, observation references, and balanced AI input.

### v0.7.0

- Added unified evidence timeline generation and deterministic interval merging.
- Added evidence groups, cross-modal context, review priority, ambiguous findings, non-supporting findings, and missing-evidence sections.
- Added `unified_evidence.json`, `evidence_timeline.jsonl`, `ai_interpretation_input.json`, and `ai_interpretation_prompt.txt`.
- Added token-aware AI input compaction and a future response schema under `schemas/`.
- Kept v0.7 local-only with no Gemini call, no trained model, no AI probability, and no authenticity or manipulation verdict.

### v0.6.0

- Added explainable visual consistency analysis using the existing temporal samples.
- Added deterministic region-grid metrics for brightness, edge stability, texture stability, fine-detail persistence, and motion context.
- Added sustained regional visual-variation intervals, ranked review transitions, and bounded consistency review artifacts.
- Added `visual_consistency_metrics.jsonl`, report integration, validation checks, and focused tests.
- Kept the project local-only and non-verdict-based: no AI classifier, authenticity score, manipulation score, or fake/real conclusion.

## Metric Notes

- Representative-frame sampling inspects only a small subset of frames.
- OpenCV seeking may decode a nearby frame rather than the exact requested timestamp; seek error is reported.
- Laplacian variance is affected by resolution, compression, noise, resizing, and scene content.
- Perceptual-hash distance estimates visual similarity between sampled frames but is not proof of editing.
- Normalized mean absolute difference is computed on resized grayscale decoded frames and divided by 255.
- Histogram correlation compares grayscale distributions but does not preserve spatial layout.
- Sequential temporal sampling is capped to keep long videos bounded; effective FPS may be reduced automatically.
- Optical-flow metrics measure motion between resized grayscale frames and are affected by camera movement, object motion, blur, compression, stabilization, zoom, and lighting changes.
- Scene-boundary candidates are substantial visual transitions, not proof of editing or manipulation.
- Ranked notable transitions are selected because they stand out in one or more measurements within the current video. They are review prompts, not verdicts.
- Flow-warp residuals can increase because of scene changes, occlusion, rapid motion, motion blur, lighting changes, compression, inaccurate optical flow, zoom, camera movement, object deformation, or generated temporal inconsistency.
- Sustained near-static intervals may be ordinary static shots, low motion, intentional freeze frames, repeated content, or normal recording behavior.
- Audio RMS, clipping, silence-like intervals, and energy transitions are basic signal measurements only. They cannot determine whether speech is natural, cloned, synthesized, edited, or authentic.
- Visual consistency metrics are regional measurements for review. They may be affected by camera movement, zoom, stabilization, motion blur, compression, lighting changes, focus changes, occlusion, object movement, depth changes, shadows, animation, ordinary editing, or generated instability.
- Edge stability uses Canny edges and prefers motion-compensated comparison where possible.
- Texture stability uses grayscale histogram distance, local variance difference, and gradient-energy difference.
- Fine-detail persistence compares Laplacian-detail residuals after motion compensation.
- Ranked visual consistency transitions are selected relative to the current video only. They are not anomaly scores, suspicion scores, or objective signs of manipulation.
- Unified evidence events merge time-based findings that overlap or fall within the configured tolerance. A merged event is a review convenience, not a claim that all sources prove the same cause.
- Evidence group counts separate broader sources such as visual, audio, metadata, provenance, and learned model. Multiple visual modules are correlated and are not treated as independent confirmations.
- Review priority means "review this interval first." It is not an authenticity score, AI-generation probability, manipulation score, or risk score.
- The AI-ready input is compact by design and omits raw JSONL records, raw FFprobe JSON, full region records, pixel arrays, and audio arrays.
- D3, when enabled, exports a raw second-order temporal-feature statistic. This value is uncalibrated in this project and must not be interpreted as probability or verdict.

## What It Does Not Do

- No AI-generation classifier.
- No deepfake detector.
- No face analysis.
- No audio authenticity analysis.
- No synthetic-speech detector.
- No video authenticity verdict.
- No manipulation verdict.
- No fake/real verdict.
- No semantic-content analysis.
- No Gemini integration.
- No external model call by default.
- No AI probability.
- No trained model inference unless optional D3 is explicitly enabled and dependencies/model assets are available.
- No C2PA provenance analysis.
- No API, frontend, database, cloud storage, or deployment code.

All temporal, audio, and visual-consistency flags are observations only. They may point to normal camera motion, lighting changes, compression, scene transitions, object motion, or other ordinary video behavior. They are not verdicts and are not proof of editing, deception, tampering, AI generation, or authenticity.

The analyzer may identify visual or audio transitions and intervals that deserve review. v0.8 can optionally run D3 as a standalone learned detector, but it does not contain an audio-authenticity detector, external model interpreter, direct provenance verifier, or calibrated authenticity decision layer.

## 🧪 Tests

Run the focused test suite with:

```bash
python -m unittest discover -s tests
```

## 🔒 Privacy Note

Analysis remains local. Reports may contain extracted frame images, so treat the `reports/` folder as potentially sensitive.
