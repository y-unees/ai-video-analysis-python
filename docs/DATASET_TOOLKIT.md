# Dataset Preparation Toolkit

v0.9.1 adds a small local toolkit for collecting manually labeled `outcome_features.json` artifacts.

The toolkit prepares data for future probability testing. It does not train a model, calculate probability, create thresholds, calibrate D3, call Gemini, or add API/frontend/mobile code.

## Manifest

The dataset manifest is JSONL at:

```text
dataset/manifest.jsonl
```

Each line contains:

- `sample_id`
- `video_name`
- `video_sha256`
- `expected_label`
- `feature_schema_version`
- `application_version`
- `feature_file`
- `source`
- `generator_or_camera`
- `notes`
- `added_at`

Labels are always supplied manually. Allowed labels:

- `real`
- `ai_generated`

Labels are never inferred from filenames, folders, metadata, D3 scores, or analysis values.

## Register A Sample

```bash
python tools/dataset_tool.py register reports/example/outcome_features.json real --source "camera test set" --generator-or-camera "iPhone 15" --notes "handheld daylight clip"
```

```bash
python tools/dataset_tool.py register reports/example_ai/outcome_features.json ai_generated --source "manual collection" --generator-or-camera "example generator" --notes "prompt archived separately"
```

Registration validates the feature artifact, writes a labeled copy under `dataset/features/`, and appends one manifest record. The original analysis artifact is not modified.

Duplicate source video SHA-256 values are rejected unless `--allow-duplicate` is provided.

## Validate

```bash
python tools/dataset_tool.py validate
```

Validation checks manifest records, feature files, labels, duplicate IDs, duplicate video hashes, relative path safety, v0.9 outcome validation, matching video SHA-256 values, finite numeric values, and forbidden probability/confidence/threshold/verdict fields.

## Export

JSONL:

```bash
python tools/dataset_tool.py export-jsonl
```

CSV:

```bash
python tools/dataset_tool.py export-csv
```

Exports are deterministic and flatten feature groups with dotted column names such as:

```text
media.duration_seconds
temporal_summary.maximum_motion_magnitude
d3_summary.d3_raw_score
```

Unavailable values remain `null` in JSONL and empty in CSV. No feature normalization, scaling, selection, or model-specific transformation is performed.

## Summary

```bash
python tools/dataset_tool.py summary
```

The summary reports total samples, label counts, duplicate count, valid/invalid sample counts, schema versions, D3 status counts, and missing-value counts per feature.

## Recording Real And Generated Videos

For real videos, record camera/device, source collection context, and any known processing. For generated videos, record generator name/version when known, generation source, prompt or workflow location if available, and any post-processing notes.

Keep videos themselves out of Git unless a future project policy explicitly says otherwise.
