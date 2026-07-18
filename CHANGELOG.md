# Changelog

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
