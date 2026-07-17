# Learned Detectors

The learned-detector layer is optional and modular. v0.8.1 ships one adapter, `d3`, and keeps it standalone from deterministic forensic evidence.

## Base Detector Contract

Each detector exposes:

- `detector_id`
- `detector_version`
- `check_availability()`
- `analyze(video_path, video_sha256, output_directory, metadata)`

Detector results must include execution status, input traceability, configuration, native output, artifacts, reproducibility metadata, warnings, and limitations.

## Registry Lifecycle

The registry reads environment configuration, constructs enabled detectors, runs availability checks, executes detectors, collects warnings, and returns `learned_detector_results`.

## Availability Checks

Availability checks are lightweight. They validate global enablement, detector enablement, supported configuration, optional dependencies, and obvious device or preprocessing incompatibilities before heavy execution.

## Execution Lifecycle

For D3, disabled and unavailable states return immediately. Available heavy execution runs in a worker process with a timeout covering preprocessing, model loading, inference, and artifact generation.

## Status Definitions

- `not_run`: The detector stage was never attempted, usually because an earlier fatal pipeline error prevented execution.
- `disabled`: The learned-detector layer or detector was intentionally disabled by configuration.
- `unavailable`: The detector was enabled, but dependencies, device support, or model assets were unavailable.
- `skipped`: The detector was available, but the current input could not be analyzed.
- `completed`: The requested detector processing completed successfully.
- `failed`: The detector was attempted but encountered an unexpected execution failure.
- `timed_out`: The detector exceeded the configured timeout and was terminated.

## Reason Codes

- `global_detector_layer_disabled`
- `detector_disabled`
- `optional_dependencies_missing`
- `encoder_assets_missing`
- `cuda_unavailable`
- `insufficient_frames`
- `unsupported_preprocessing_mode`
- `detector_timeout`
- `preprocessing_failure`
- `model_loading_failure`
- `inference_failure`
- `artifact_generation_failure`
- `unexpected_detector_failure`

## Timeout Model

Timeouts use process isolation. The main analyzer terminates the worker process when `D3_TIMEOUT_SECONDS` is exceeded, records `timed_out/detector_timeout`, and continues writing the main forensic report.

## Artifact Responsibilities

Completed detectors may write artifacts. Failed, skipped, unavailable, disabled, or timed-out detectors must not reference completed artifacts. Artifact paths must be relative and hashes must validate.

## Privacy Requirements

Reports must avoid secrets and private absolute paths. Model cache paths are sanitized. Temporary diagnostic frames are retained only when explicitly requested.

## Result-Schema Expectations

Uncalibrated D3 output must use:

```json
{
  "probability": null,
  "threshold": null,
  "classification": "not_assigned",
  "calibration_status": "uncalibrated"
}
```

The raw value is `raw_score` and must not be renamed to confidence, percentage, fake score, real score, or authenticity score.

## Future Detector Rules

Future adapters must:

- Keep availability checks separate from heavy execution.
- Use explicit statuses and reason codes.
- Record reproducibility metadata.
- Avoid network access unless explicitly configured.
- Avoid verdicts unless a separate validated calibration layer exists.
- Keep artifacts relative and hashed.
- Avoid modifying unified evidence unless a future version explicitly designs that integration.

## Unsupported Verdict Rule

Learned detectors may not produce fake/real, authenticity, manipulation, AI-generation, or risk verdicts without documented calibration, validation, thresholds, and scope.

## Future DeMamba Compatibility

The lifecycle and status model are intended to support future adapters, including possible DeMamba-compatible experiments, but no DeMamba integration exists in v0.8.1.
