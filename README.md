# 🎬 Local Video Analysis MVP

Terminal-only, local-first video-analysis MVP for metadata inspection, representative-frame extraction, and cautious heuristic observations. It does not call external AI services or upload media.

## Requirements

- Python 3.10 or newer
- FFmpeg with both `ffmpeg` and `ffprobe` available in your system PATH
- Python packages from `requirements.txt`

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
    └── frames/
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

## Schema Notes

Current schema: `0.3`.

Important changes from `0.2`:

- Analysis timestamps use timezone-aware ISO strings with millisecond precision.
- Stream timing fields are named as timeline observations, not audio-sync measurements.
- Frame records include seek error between requested and decoded timestamps.
- Laplacian variance is reported as an uncalibrated edge-detail measurement, not a universal sharpness verdict.
- Factual metadata observations, missing metadata, and temporal heuristics are separated.
- Artifact paths declare that they are relative to the report directory.

## Metric Notes

- Representative-frame sampling inspects only a small subset of frames.
- OpenCV seeking may decode a nearby frame rather than the exact requested timestamp; seek error is reported.
- Laplacian variance is affected by resolution, compression, noise, resizing, and scene content.
- Perceptual-hash distance estimates visual similarity between sampled frames but is not proof of editing.
- Normalized mean absolute difference is computed on resized grayscale decoded frames and divided by 255.
- Histogram correlation compares grayscale distributions but does not preserve spatial layout.

## What It Does Not Do

- No AI-generation classifier.
- No deepfake detector.
- No face analysis.
- No audio-signal analysis.
- No semantic-content analysis.
- No Gemini integration.
- No API, frontend, database, cloud storage, or deployment code.

All temporal flags are heuristic observations only. They may point to normal camera motion, lighting changes, compression, scene transitions, or other ordinary video behavior. They are not verdicts and are not proof of editing, deception, tampering, AI generation, or authenticity.

## 🧪 Tests

Run the focused test suite with:

```bash
python -m unittest discover -s tests
```

## 🔒 Privacy Note

Analysis remains local. Reports may contain extracted frame images, so treat the `reports/` folder as potentially sensitive.
