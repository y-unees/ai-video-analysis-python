# One-Command Dataset Runner

v0.9.2 adds a temporary root-level runner:

```bash
python script.py
```

The runner prepares labeled source videos for later statistical analysis and model experiments. It does not train a classifier, calculate probability, create thresholds, call Gemini, or add API/frontend/mobile code.

## Source Video Naming

Place pilot videos under:

```text
source_videos/
```

Label resolution priority:

1. FaceForensics++ metadata CSV match
2. optional `source_videos/video_labels.csv`
3. filename convention
4. unresolved and skipped

Filename convention:

```text
real__<name>.mp4
ai__<name>.mp4
```

Ordinary filenames are not guessed.

## Metadata Locations

The runner looks for `FF++_Metadata.csv` in:

```text
dataset_sources/faceforensics_pp/metadata/FF++_Metadata.csv
datasets/FF++_Metadata.csv
dataset/FF++_Metadata.csv
```

## Optional Sidecar

`source_videos/video_labels.csv` may contain:

```text
filename,expected_label,source,generator_or_camera,manipulation_method,compression,notes
```

Sidecar labels do not silently override conflicting trusted FaceForensics++ metadata matches.

## Commands

Default:

```bash
python script.py
```

Limits:

```bash
python script.py --real-limit 5 --ai-limit 3
```

Force reanalysis:

```bash
python script.py --force-reanalyze
```

Dry run:

```bash
python script.py --dry-run
```

## Generated Structure

```text
dataset_sources/pilot/real/
dataset_sources/pilot/ai_generated/
dataset_sources/pilot/unresolved/
dataset/manifest.jsonl
dataset/features/
dataset/exports/dataset_features.jsonl
dataset/exports/dataset_features.csv
dataset/runs/latest_run.json
```

Original files in `source_videos/` are not deleted.

## Resume Behavior

The runner uses SHA-256 duplicate detection and the dataset manifest to skip already registered source videos. Previous failed analyses may be retried on later runs. `--force-reanalyze` reruns analysis even when a matching manifest entry already exists, while duplicate registration remains protected by the dataset toolkit.

## Final Validation

At the end, non-dry runs automatically validate the dataset, generate JSONL and CSV exports, write a run ledger, and exit successfully only when registered samples and exports are valid.
