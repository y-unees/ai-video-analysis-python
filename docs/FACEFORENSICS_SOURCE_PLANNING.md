# FaceForensics++ Source Planning

v0.9.2 adds source metadata auditing and pilot sample planning for FaceForensics++-style CSV metadata.

This layer stops at:

```text
CSV audit -> normalized metadata -> video availability check -> balanced pilot plan
```

It does not process all FaceForensics++ videos, train a model, calculate probability, create thresholds, calibrate D3, call Gemini, add APIs, or add frontend/mobile code.

## Source Layout

External source data belongs outside the generated feature dataset:

```text
dataset_sources/
  faceforensics_pp/
    metadata/
      FF++_Metadata.csv
    videos/

dataset/
  manifest.jsonl
  features/
  exports/
```

The CSV alone cannot produce forensic features. Actual video files are still required before analysis can generate `outcome_features.json`.

## Audit Metadata

```bash
python tools/dataset_tool.py audit-source --dataset faceforensics_pp --metadata dataset_sources/faceforensics_pp/metadata/FF++_Metadata.csv
```

Audit output:

```text
dataset_sources/faceforensics_pp/audits/metadata_audit.json
```

The audit reports encoding, delimiter, columns, inferred types, row counts, duplicate counts, missing values, labels, manipulation/category values, path style, row representation, and video availability.

## Add Videos

Place matching local videos under:

```text
dataset_sources/faceforensics_pp/videos/
```

The matcher supports exact relative paths, normalized path separators, exact filenames, and recursive filename lookup. It refuses ambiguous filename matches.

## Create A Pilot Plan

```bash
python tools/dataset_tool.py plan-source --dataset faceforensics_pp --metadata dataset_sources/faceforensics_pp/metadata/FF++_Metadata.csv --real-count 10 --ai-count 10 --seed 42
```

Plan output:

```text
dataset_sources/faceforensics_pp/plans/pilot_plan.jsonl
```

The planner selects only rows with explicit trusted labels, prefers rows with available local videos, avoids duplicate video names and resolved paths, and preserves manipulation-method diversity where possible.

## Validate A Plan

```bash
python tools/dataset_tool.py validate-plan dataset_sources/faceforensics_pp/plans/pilot_plan.jsonl
```

Validation checks required fields, allowed labels, unique plan indices, unique selected paths, requested class balance when provided, source metadata existence, available-video existence, safe paths, deterministic ordering, and forbidden probability/confidence/threshold/verdict/classifier fields.

## Label Normalization

FaceForensics++ manipulated samples are normalized to the current `ai_generated` dataset label because the project currently has only two manual outcome labels: `real` and `ai_generated`. This is a practical dataset-bucket choice for future experiments, not a claim that every FaceForensics++ manipulation is fully generated video.

## Limitations

FaceForensics++ alone is not enough for robust probability testing. It emphasizes face manipulation methods and may not represent modern text-to-video, image-to-video, camera originals, compression pipelines, platforms, or editing workflows. Licensing and terms-of-use compliance remain the user's responsibility.
