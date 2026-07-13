# 🎬 Local Video Analysis MVP

Terminal-only, local-first video-analysis MVP for metadata inspection, representative-frame extraction, and cautious heuristic observations. It does not call external AI services or upload media.

## Requirements

- Python 3.10 or newer
- FFmpeg with both `ffmpeg` and `ffprobe` available in your system PATH
- Python packages from `requirements.txt` for OpenCV/Pillow/ImageHash/NumPy-based local analysis

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
    ├── frames/
    ├── transition_frames/
    └── scene_frames/
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

## Schema Notes

Current schema: `0.5`.

Application version: `0.5.1`.

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

## What It Does Not Do

- No AI-generation classifier.
- No deepfake detector.
- No face analysis.
- No audio authenticity analysis.
- No synthetic-speech detector.
- No semantic-content analysis.
- No Gemini integration.
- No C2PA provenance analysis.
- No API, frontend, database, cloud storage, or deployment code.

All temporal flags are heuristic observations only. They may point to normal camera motion, lighting changes, compression, scene transitions, or other ordinary video behavior. They are not verdicts and are not proof of editing, deception, tampering, AI generation, or authenticity.

The analyzer may identify visual or audio transitions and intervals that deserve review, but v0.5.1 does not contain a trained AI-video detector, audio-authenticity detector, or direct provenance verifier.

## 🧪 Tests

Run the focused test suite with:

```bash
python -m unittest discover -s tests
```

## 🔒 Privacy Note

Analysis remains local. Reports may contain extracted frame images, so treat the `reports/` folder as potentially sensitive.
